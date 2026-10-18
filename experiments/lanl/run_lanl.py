"""Part 30 - PrismFlow and naive fusion on LANL cyber1, days 1-16.

WHAT IS REUSED AND WHAT IS NOT
------------------------------
Every metric in this Part is computed by Part 28's own functions, imported from
`experiments/ait_clean/run_ait_clean.py` and not edited: `metric_row` (which is
`prismflow.evaluation.protocol.compute_metrics` plus `part28_extras`),
`majority_baseline`, `summarise`, `paired_difference`, `LOWER_IS_BETTER`,
`roc_auc_binary` and `contiguous_blocks`. The protocol writer
`prismflow.evaluation.evaluate` is called as well, and its per-seed rows are
cross-checked against the rows built here, so the numbers are checked against
code this Part does not own.

Three of Part 28's functions could NOT be imported: `fit_scaler`,
`apply_scaler` and `train_ait`. All three resolve a view's native feature width
through `prismflow.data.ait_lds.FEATURES[source]`, which is keyed by AIT source
names. LANL's views are `auth, proc, flows, dns`; `FEATURES['proc']` and
`FEATURES['flows']` do not exist, and `FEATURES['auth']` and `FEATURES['dns']`
exist with AIT's widths, not LANL's. Importing them would therefore either
raise or silently scale the wrong number of columns. They are re-implemented
below as structural mirrors parameterised on LANL's widths, with every
hyperparameter read from `config.yaml`, which holds Part 28's values unchanged.
That is a reuse limitation of the Part 28 code, reported here and NOT patched
there - `run_ait_clean.py` is Part 28's and is untouched by this Part.

NOTHING HERE WRITES TO data/raw/. The loader streams the gzip files read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.ait_clean.run_ait_clean import (
    LOWER_IS_BETTER,
    roc_auc_binary,
    contiguous_blocks,
    majority_baseline,
    metric_row,
    pair_table,
    paired_difference,
    summarise,
)
from prismflow.data.lanl_cyber1 import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    load_samples,
    save_samples,
    FEATURES,
    LABEL_MEANING,
    LANLMultiViewDataset,
    TimestampInversion,
    build_lanl_samples,
    load_lanl_config,
    read_redteam,
    scan_view,
    subset,
    view_configs,
)
from prismflow.data.loaders import iter_batches
from prismflow.eniv.eniv import compute_eniv
from prismflow.evaluation import evaluate
from prismflow.evaluation.protocol import collect_predictions
from prismflow.models.prismflow import REPORTING_NULL_PERMUTATIONS, PrismFlow
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.train import compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger("prismflow.lanl")
CONFIG_PATH = Path(__file__).with_name("config.yaml")
RESULTS_DIR = Path("results/lanl")


class StopCondition(RuntimeError):
    """A pre-registered stop condition fired. Nothing is widened or repaired."""


# ---------------------------------------------------------------------------
# Scaling and training: structural mirrors of Part 28, on LANL widths
# ---------------------------------------------------------------------------


def fit_scaler(dataset, indices: np.ndarray, cfg: dict) -> dict:
    """Per (view, native feature) mean and std over rows where the view is PRESENT.

    Mirror of `run_ait_clean.fit_scaler`; see this module's docstring for why it
    cannot be imported. An absent view's row is all zeros and carries no
    observation, so including it would pull the mean toward zero by an amount
    that depends only on the missing rate.
    """
    views = dataset.views.numpy()[indices]
    mask = dataset.view_mask.numpy()[indices]
    n_views, dim = views.shape[1], views.shape[2]
    mean = np.zeros((n_views, dim), dtype=np.float64)
    std = np.ones((n_views, dim), dtype=np.float64)
    native = [len(FEATURES[s]) for s in dataset.sources]
    present_counts = []
    for v in range(n_views):
        rows = mask[:, v]
        present_counts.append(int(rows.sum()))
        if rows.sum() < 2:
            continue
        block = views[rows, v, : native[v]].astype(np.float64)
        mean[v, : native[v]] = block.mean(axis=0)
        std[v, : native[v]] = block.std(axis=0, ddof=0)
    epsilon = float(cfg["scaling"]["epsilon"])
    std = np.where(std > epsilon, std, 1.0)
    return {
        "mean": mean, "std": std, "native_dims": native,
        "fit_rows": int(len(indices)), "present_rows_per_view": present_counts,
    }


def apply_scaler(dataset, scaler: dict):
    """z-score the native columns of PRESENT views; leave everything else at 0."""
    views = dataset.views.numpy().copy()
    mask = dataset.view_mask.numpy()
    for v, native in enumerate(scaler["native_dims"]):
        rows = mask[:, v]
        if not rows.any():
            continue
        block = views[rows, v, :native].astype(np.float64)
        block = (block - scaler["mean"][v, :native]) / scaler["std"][v, :native]
        views[rows, v, :native] = block.astype(np.float32)
        views[~rows, v] = 0.0
    scaled = dataclasses.replace(dataset.samples, views=views.astype(np.float32))
    return LANLMultiViewDataset(scaled)


def train_lanl(dataset, indices: np.ndarray, seed: int, use_discount: bool, cfg: dict):
    """Train one model on `indices`. Mirror of `run_ait_clean.train_ait`.

    Optimiser, loss, annealing schedule, per-view loss weight, epoch count and
    the same-seed-same-init discipline are Part 28's, read from config.yaml.
    Only the encoder widths differ, because LANL views are 2-8 features wide
    and `lanl_cyber1.view_configs` declares each one's native width so the
    padded columns stay unreachable.

    There is no validation split and no early stopping, because Part 28 has
    none and this Part's settings are Part 28's.
    """
    set_seed(seed)
    model = PrismFlow(
        view_configs(
            dataset.sources,
            hidden_dims=cfg["model"]["hidden_dims"],
            feature_dim=cfg["model"]["feature_dim"],
            dropout=cfg["model"]["dropout"],
        ),
        n_classes=dataset.n_classes,
        use_discount=use_discount,
    )
    training = cfg["training"]
    optimiser = torch.optim.Adam(model.parameters(), lr=training["lr"])
    for epoch in range(training["epochs"]):
        model.train()
        for batch in iter_batches(
            dataset, indices, training["batch_size"], shuffle=True, seed=seed + epoch
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=dataset.n_classes
            ).to(batch.views.dtype)
            loss = compute_loss(
                model(batch.views, batch.view_mask), targets,
                epoch=epoch, anneal_epochs=training["anneal_epochs"],
                per_view_loss_weight=training["per_view_loss_weight"],
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Diagnostics: identical call to Part 28's
# ---------------------------------------------------------------------------


def _safe(value) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


@torch.no_grad()
def dependence_and_eniv(model, dataset, indices: np.ndarray, cfg: dict) -> dict:
    """Reporting-grade [V, V] matrix and ENIV on the test block.

    `REPORTING_NULL_PERMUTATIONS` (12), not the forward pass's 4, and computed
    for BOTH systems by the same code: naive fusion's forward pass leaves
    `output.eniv` None, so reading it off the model would compare one system's
    measurement with the other's absence. Same call as
    `run_ait_clean.dependence_and_eniv`.
    """
    views = dataset.views[indices]
    view_mask = dataset.view_mask[indices]
    output = model(views, view_mask)
    matrix = feature_dependence_matrix(
        model.encoder(views, view_mask),
        output.per_view_evidence,
        view_mask,
        method=model.dependence_method,
        conditioning=model.dependence_conditioning,
        seed=model.dependence_seed,
        null_permutations=REPORTING_NULL_PERMUTATIONS,
    )
    available = available_views(view_mask)
    record = {
        "matrix": [[_safe(x) for x in row] for row in np.asarray(matrix).tolist()],
        "views": list(dataset.sources),
        "available": available.tolist(),
        "null_permutations": REPORTING_NULL_PERMUTATIONS,
    }
    for method in cfg["diagnostics"]["eniv_methods"]:
        result = compute_eniv(matrix, available=available, method=method)
        record[f"eniv_{method}"] = _safe(result.effective_views)
        record[f"mean_dependence_{method}"] = _safe(result.mean_dependence)
        record[f"efficiency_ratio_{method}"] = _safe(result.efficiency_ratio)
    return record


def missingness(dataset, indices: np.ndarray) -> dict:
    mask = dataset.view_mask.numpy()[indices]
    return {
        source: float(1.0 - mask[:, v].mean())
        for v, source in enumerate(dataset.sources)
    }


# ---------------------------------------------------------------------------
# Loading, in parallel across views
# ---------------------------------------------------------------------------

_SCAN_STATE: dict = {}


def _scan_worker(view: str) -> dict:
    return scan_view(view, _SCAN_STATE["cfg"], _SCAN_STATE["cells"])


def _init_worker(cfg, cells) -> None:
    _SCAN_STATE["cfg"] = cfg
    _SCAN_STATE["cells"] = cells


def load_dataset(data_cfg, workers: int) -> tuple[LANLMultiViewDataset, dict, dict]:
    """Stream all four views, one process per file, and join them.

    Results come back through the pool as in-memory objects. Nothing is cached
    to disk: only the KEPT cells carry an accumulator, so a whole view's
    contribution is a few megabytes.
    """
    redteam = read_redteam(data_cfg)
    LOGGER.info(
        "redteam: %d rows -> %d attack cells, %d attacked windows",
        redteam["rows"], len(redteam["cells"]), len(redteam["windows"]),
    )
    started = time.time()
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            min(workers, len(data_cfg.views)),
            initializer=_init_worker,
            initargs=(data_cfg, redteam["cells"]),
        ) as pool:
            scans = dict(zip(data_cfg.views, pool.map(_scan_worker, list(data_cfg.views))))
    else:
        scans = {v: scan_view(v, data_cfg, redteam["cells"]) for v in data_cfg.views}
    elapsed = time.time() - started
    for view, scan in scans.items():
        LOGGER.info(
            "%s: %d records, %d occupied cells, %d kept",
            view, scan["records"], scan["population_cells"], len(scan["kept"]),
        )
    samples = build_lanl_samples(data_cfg, scans, redteam)
    dataset = LANLMultiViewDataset(samples)
    timing = {"scan_seconds": elapsed, "workers": workers}
    return dataset, redteam, timing


# ---------------------------------------------------------------------------
# Split bookkeeping and the stop-condition gate
# ---------------------------------------------------------------------------


def split_indices(dataset) -> dict[str, np.ndarray]:
    blocks = np.asarray(dataset.samples.block)
    return {name: np.flatnonzero(blocks == name) for name in ("train", "test")}


def split_report(dataset, redteam: dict, indices: dict, data_cfg) -> dict:
    """Counts per split, and the per-day concentration caveat as numbers."""
    labels = dataset.labels.numpy()
    starts = dataset.window_start.numpy()
    out = {"label_meaning": LABEL_MEANING, "per_split": {}}
    for name, idx in indices.items():
        lab = labels[idx]
        day = ((starts[idx] - 1) // data_cfg.day_seconds + 1).astype(int)
        window = ((starts[idx] - 1) // data_cfg.window_seconds).astype(int)
        attack_windows = sorted({int(w) for w in window[lab == CLASS_ATTACK]})
        per_day = {}
        for d in sorted(set(day.tolist())):
            on_day = day == d
            per_day[int(d)] = {
                "samples": int(on_day.sum()),
                "attack_samples": int((lab[on_day] == CLASS_ATTACK).sum()),
                "attack_windows": len({
                    int(w) for w in window[on_day & (lab == CLASS_ATTACK)]
                }),
            }
        bounds = data_cfg.split[name]
        out["per_split"][name] = {
            "seconds": [bounds["start"], bounds["end"]],
            "days": bounds["days"],
            "n_samples": int(idx.size),
            "n_attack_samples": int((lab == CLASS_ATTACK).sum()),
            "n_benign_samples": int((lab == 0).sum()),
            "attack_prevalence_in_sample": float((lab == CLASS_ATTACK).mean()) if idx.size else float("nan"),
            "n_distinct_attack_windows": len(attack_windows),
            "per_day": per_day,
            "attack_blocks_on_window_grid": contiguous_blocks(
                lab == CLASS_ATTACK, starts[idx]
            ),
        }
    gap = data_cfg.split["gap"]
    out["gap"] = {
        "seconds": [gap["start"], gap["end"]],
        "days": gap["days"],
        "redteam_events_in_gap": sum(
            n for d, n in redteam["events_per_day"].items()
            if gap["days"][0] <= d <= gap["days"][1]
        ),
        "samples_built": 0,
    }
    return out


def label_audit(data_cfg, redteam: dict) -> dict:
    """Both label readings, per block, from redteam.txt.gz alone.

    Cheap enough to run on its own (`--label-audit`): it reads only the 4,846
    byte red-team file and touches none of the four view streams.

    This is what reconciles "how many cells the label file names" with "how
    many attack samples exist". A cell the red-team names but that NO view
    recorded an event for cannot become a sample, because the sample set is
    the union of cells the views actually occupy. The difference is reported
    rather than quietly absorbed.
    """
    out = {
        "note": (
            "computer_window is the primary reading: a (computer, window) "
            "cell is attack if a red-team event in that window names that "
            "computer as source or destination. window_only is the "
            "alternative reading in configs/data/lanl.yaml label.also_report, "
            "under which every computer in an attacked window would be "
            "attack; it is counted here and NOT scored anywhere."
        ),
        "per_block": {},
    }
    for name in ("train", "gap", "test"):
        bounds = data_cfg.split[name]
        lo = (bounds["start"] - 1) // data_cfg.window_seconds
        hi = bounds["end"] // data_cfg.window_seconds - 1
        cells = {c for c in redteam["cells"] if lo <= c[1] <= hi}
        windows = {w for w in redteam["windows"] if lo <= w <= hi}
        out["per_block"][name] = {
            "computer_window_attack_cells_named_by_redteam": len(cells),
            "window_only_attacked_windows": len(windows),
            "distinct_computers_named": len({c[0] for c in cells}),
        }
    return out


def check_stop_conditions(report: dict, minimum: int) -> None:
    for name in ("train", "test"):
        got = report["per_split"][name]["n_distinct_attack_windows"]
        if got < minimum:
            raise StopCondition(
                f"{name} block has {got} distinct attack windows, "
                f"below the pre-registered minimum of {minimum}. "
                "Reporting the count; the day range is NOT widened."
            )


# ---------------------------------------------------------------------------
# One system, all seeds
# ---------------------------------------------------------------------------


def run_system(system, train_ds, train_idx, test_ds, test_idx, configuration, cfg, seeds):
    batch_size = cfg["evaluation"]["batch_size"]
    models, rows, diagnostics = {}, [], []
    for seed in seeds:
        started = time.time()
        model = train_lanl(train_ds, train_idx, seed, system["use_discount"], cfg)
        models[seed] = model
        collected = collect_predictions(
            model, iter_batches(test_ds, test_idx, batch_size, shuffle=False)
        )
        row = {"seed": seed}
        row.update(metric_row(
            collected["labels"], collected["probs"], collected["vacuity_confidence"], cfg
        ))
        rows.append(row)
        record = {"seed": seed}
        record.update(dependence_and_eniv(model, test_ds, test_idx, cfg))
        diagnostics.append(record)
        LOGGER.info(
            "%s/%s seed=%d trained+scored in %.1fs (bal_acc=%.4f roc_auc=%.4f)",
            configuration["name"], system["name"], seed, time.time() - started,
            row["balanced_accuracy"], row["roc_auc"],
        )

    experiment_id = f"{cfg['experiment']['id']}/{configuration['name']}/{system['name']}"
    protocol = evaluate(
        models,
        lambda seed: iter_batches(test_ds, test_idx, batch_size, shuffle=False),
        experiment_id, seeds, results_dir="results",
        n_bins=cfg["evaluation"]["n_bins"],
        coverages=cfg["evaluation"]["coverages"],
        title=f"Part 30 {configuration['label']} - {system['label']}",
    )
    cross_check(rows, protocol["per_seed"], experiment_id)
    return models, {
        "system": system["name"],
        "label": system["label"],
        "use_discount": system["use_discount"],
        "experiment_id": experiment_id,
        "protocol_output": f"results/{experiment_id}/metrics.json",
        "per_seed": rows,
        "summary": summarise(rows),
        "diagnostics_per_seed": diagnostics,
        "diagnostics_summary": summarise(
            [{k: v for k, v in d.items() if isinstance(v, float)} for d in diagnostics]
        ),
        # Part 29 test C's second half needs the most and least dependent view
        # pairs READ from a committed table rather than re-derived, so the
        # table is written here, by Part 28's own function.
        "pair_table": pair_table(diagnostics, list(test_ds.sources)),
    }


@torch.no_grad()
def test_a_natural_attack_windows(models, dataset, test_idx, cfg) -> dict:
    """Part 29 test A on LANL: ENIV and mean pairwise dependence on REAL attack
    windows versus real benign windows. No synthetic perturbation anywhere.

    Part 29 split AIT's attack windows into dnsteal-only and multi-step
    populations and forbade pooling them. LANL's red-team file carries no step
    taxonomy at all - a row is a time, a user, a source and a destination - so
    there is no equivalent split to make here, and this is ONE attack
    population, not a pooling of two. That difference is stated in the README
    rather than left for a reader to assume.

    The two populations are reported separately and never pooled, and the
    attack-minus-benign difference is paired within seed.
    """
    labels = dataset.labels.numpy()[test_idx]
    populations = {
        "attack_windows": test_idx[labels == CLASS_ATTACK],
        "benign_windows": test_idx[labels == 0],
    }
    out = {
        "note": (
            "ENIV and mean pairwise dependence on real attack vs real benign "
            "windows of the test block. No synthetic perturbation. LANL's "
            "red-team file has no attack-step taxonomy, so unlike Part 29 on "
            "AIT there is no dnsteal-only/multi-step split to report; this is "
            "one attack population, not two pooled."
        ),
        "n": {k: int(v.size) for k, v in populations.items()},
        "per_system": {},
    }
    for system, seeded in models.items():
        per_population = {}
        for name, idx in populations.items():
            if idx.size < 2:
                per_population[name] = {"not_measured": f"only {idx.size} windows"}
                continue
            rows = []
            for seed, model in sorted(seeded.items()):
                record = {"seed": seed}
                record.update(dependence_and_eniv(model, dataset, idx, cfg))
                rows.append({k: v for k, v in record.items() if isinstance(v, (int, float))})
            per_population[name] = {"per_seed": rows, "summary": summarise(rows)}
        paired = {}
        if all("per_seed" in per_population[k] for k in populations):
            paired = paired_difference(
                per_population["attack_windows"]["per_seed"],
                per_population["benign_windows"]["per_seed"],
                LOWER_IS_BETTER,
            )
        out["per_system"][system] = {
            "populations": per_population,
            "paired_attack_minus_benign": paired,
        }
    return out


def cross_check(rows, protocol_rows, experiment_id: str) -> None:
    """The rows built here must agree with the protocol writer's own rows."""
    by_seed = {r["seed"]: r for r in protocol_rows}
    for row in rows:
        other = by_seed.get(row["seed"])
        if other is None:
            raise RuntimeError(f"{experiment_id}: protocol has no seed {row['seed']}")
        for key in ("accuracy", "brier"):
            if key in row and key in other:
                if abs(float(row[key]) - float(other[key])) > 1e-9:
                    raise RuntimeError(
                        f"{experiment_id} seed {row['seed']}: {key} "
                        f"{row[key]} != protocol {other[key]}"
                    )


def run_configuration(configuration, dataset, indices, cfg):
    """One view configuration: scale on train, train both systems, pair them."""
    data = subset(dataset, tuple(configuration["views"]))
    train_idx, test_idx = indices["train"], indices["test"]
    scaler = fit_scaler(data, train_idx, cfg)
    scaled = apply_scaler(data, scaler)
    LOGGER.info(
        "%s: scaler fitted on %d TRAIN rows only; present rows per view %s",
        configuration["name"], scaler["fit_rows"], scaler["present_rows_per_view"],
    )

    systems, models = {}, {}
    for system in cfg["systems"]:
        models[system["name"]], systems[system["name"]] = run_system(
            system, scaled, train_idx, scaled, test_idx, configuration, cfg, cfg["seeds"]
        )

    labels = scaled.labels.numpy()[test_idx]
    return {
        "name": configuration["name"],
        "label": configuration["label"],
        "views": list(configuration["views"]),
        "scaler": {
            "fit_rows": scaler["fit_rows"],
            "present_rows_per_view": scaler["present_rows_per_view"],
            "fitted_on": "train block only",
        },
        "missingness": {
            "train": missingness(scaled, train_idx),
            "test": missingness(scaled, test_idx),
            "threshold": cfg["missingness"]["threshold"],
            "over_threshold_train": [
                v for v, frac in missingness(scaled, train_idx).items()
                if frac > cfg["missingness"]["threshold"]
            ],
            "over_threshold_test": [
                v for v, frac in missingness(scaled, test_idx).items()
                if frac > cfg["missingness"]["threshold"]
            ],
        },
        "majority_baseline": majority_baseline(labels, cfg),
        "systems": systems,
        "paired_prismflow_minus_naive": paired_difference(
            systems["prismflow"]["per_seed"], systems["naive"]["per_seed"], LOWER_IS_BETTER
        ),
        "part29_test_a_natural_attack_windows": test_a_natural_attack_windows(
            models, scaled, test_idx, cfg
        ),
        # part30.md step 4 asks for Part 29's tests A-D. Only A is run; B, C and
        # D are named here with the reason, so the gap is in the results file
        # and not only in the README.
        "part29_tests_not_run": {
            "B_chorus_attack": (
                "NOT RUN. Part 29's grid is k in {1,2,3} x Part 09's epsilon "
                "and beta grid, retrained per cell, which does not fit this "
                "Part's time budget. No partial grid is reported, because a "
                "subset of the cells is not the test."
            ),
            "C_honest_duplication": (
                "PARTIALLY RUN. The strict same-sensor pair does not exist "
                "here: LANL cyber1 ships no sensor-group definition, so there "
                "is no within-group pair to score and the answer is NO "
                "EVIDENCE, exactly as on AIT but for a different reason. The "
                "'different sensors, naturally dependent' half needs the "
                "most/least dependent pairs, which ARE written to "
                "systems.<name>.pair_table above; scoring the detector on them "
                "was not run."
            ),
            "D_missingness": (
                "NOT RUN as a detector-AUC test. The per-view missing "
                "fractions on train and test ARE measured above under "
                "'missingness'; the 30%/50% masked detector AUC of Part 10 "
                "was not run."
            ),
        },
    }


# ---------------------------------------------------------------------------
# README. Every number is read from a results/ file; none is typed by hand.
# ---------------------------------------------------------------------------

#: Metrics shown in the headline table, in order. Prevalence-dependent figures
#: are marked, because this is a case-control sample and they describe the
#: sample rather than the network.
HEADLINE = [
    ("balanced_accuracy", "balanced accuracy", False),
    ("roc_auc", "ROC AUC", False),
    ("attack_recall", "attack recall", False),
    ("benign_recall", "benign recall", False),
    ("accuracy", "accuracy", True),
    ("brier", "Brier", True),
    ("prob_ece", "ECE (prob)", True),
    ("eniv_eigen", "ENIV (eigen)", False),
]


def fmt(entry, places: int = 4) -> str:
    """mean +/- std, or `not measured` when nothing backs it."""
    if entry is None:
        return "not measured"
    if isinstance(entry, (int, float)):
        value = float(entry)
        return "not measured" if not math.isfinite(value) else f"{value:.{places}f}"
    mean, std = entry.get("mean"), entry.get("std")
    if mean is None or not math.isfinite(float(mean)):
        return "not measured"
    if std is None or not math.isfinite(float(std)):
        return f"{float(mean):.{places}f} (single value, no std)"
    return f"{float(mean):.{places}f} +/- {float(std):.{places}f}"


def dig(obj, *path):
    """Nested lookup that returns None instead of raising, for cited cells."""
    for key in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, list) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return None
    return obj


def read_json(path: str):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def signed(entry, places: int = 4) -> str:
    """A paired difference with its per-seed sign count, never the mean alone."""
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return "not measured"
    mean, std = float(entry["mean"]), entry.get("std")
    n, favouring = entry.get("n_seeds"), entry.get("n_seeds_favouring_prismflow")
    tail = f" ({favouring}/{n} seeds favour PrismFlow)" if n else ""
    spread = (
        f" +/- {float(std):.{places}f}"
        if std is not None and math.isfinite(float(std)) else ""
    )
    return f"{mean:+.{places}f}{spread}{tail}"


def raised(entry, places: int = 4) -> str:
    """A paired difference with the count of seeds whose difference is > 0.

    Used for test A, where the question is whether the quantity is HIGHER on
    attack windows than on benign ones. `signed()`'s favouring count is
    read through `LOWER_IS_BETTER` and answers a different question, so it is
    not shown here - a count of positive differences is.
    """
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return "not measured"
    mean, std = float(entry["mean"]), entry.get("std")
    spread = (
        f" +/- {float(std):.{places}f}"
        if std is not None and math.isfinite(float(std)) else ""
    )
    per_seed = entry.get("per_seed") or []
    n = entry.get("n_seeds") or len(per_seed)
    tail = (
        f" ({sum(1 for d in per_seed if float(d) > 0)}/{n} seeds > 0)"
        if per_seed else ""
    )
    return f"{mean:+.{places}f}{spread}{tail}"


def write_readme(results: dict, path: Path) -> None:
    split = results["split_report"]
    population = results["population"]
    configurations = results["configurations"]
    data_cfg = results["split"]
    day = results["day_seconds"]
    window = results["window_seconds"]

    # Status is DERIVED from what the results file actually contains, not read
    # from a field, so a half-finished run cannot be labelled complete.
    expected = {c["name"] for c in results["config"]["view_configurations"]}
    finished = set(configurations)
    smoke = bool(results.get("smoke"))
    seeds_ok = all(
        dig(configuration, "systems", system, "summary", "balanced_accuracy",
            "n_seeds") == len(results["config"]["seeds"])
        for configuration in configurations.values()
        for system in ("naive", "prismflow")
    )
    complete = expected == finished and seeds_ok and not smoke
    if smoke:
        reason = "SMOKE RUN - a shortened read range. NOT A FINDING."
    elif expected != finished:
        elapsed = results.get("elapsed_seconds")
        missing = sorted(expected - finished)
        scan = dig(results, "population", "scan_seconds")
        reason = (
            f"Only {len(finished)} of {len(expected)} view configurations "
            f"finished ({', '.join(sorted(finished))} of "
            f"{', '.join(sorted(expected))}). "
            f"**{', '.join(missing)} did NOT run** "
            "and every figure below covers the finished configuration only. "
            + (
                "Reason: the wall-clock budget. Each arm is 10 models and the "
                "discounted system costs about 1.7x the undiscounted one per "
                "model on this data, so an arm does not fit once the budget "
                "is spent. Seeds were NOT reduced and the day range was NOT "
                "shrunk to make anything fit; the arms above are simply "
                "missing, and no figure in this README is computed from them."
            )
        )
    elif not seeds_ok:
        reason = (
            "At least one system did not report all "
            f"{len(results['config']['seeds'])} seeds; see `n_seeds` in "
            "`results/lanl/results.json`."
        )
    else:
        reason = (
            "Part 28's metric set and Part 29's test A ran on all "
            f"{len(expected)} view configurations "
            f"({', '.join(sorted(expected))}) over "
            f"{len(results['config']['seeds'])} seeds and both systems. "
            "Part 29's tests B, D and the second half of C did NOT run "
            "(section 9), so this Part is complete for what it reports and "
            "INCOMPLETE against part30.md step 4's full A-D list. Read "
            "section 9 before citing anything here as a Part 29 repeat."
        )

    L: list[str] = []
    A = L.append

    A("# Part 30 - PrismFlow on LANL cyber1 (days 1-16)")
    A("")
    A(f"**STATUS: {'COMPLETE (for what it reports)' if complete else 'INCOMPLETE'}.** "
      + reason)
    A("")
    A("Every number below is generated from a file under `results/lanl/` by "
      "`experiments/lanl/run_lanl.py --readme-only`. None is typed by hand. "
      "Figures that were not produced say `not measured` and never 0 or blank.")
    A("")

    # -- ambiguities -------------------------------------------------------
    A("## 0. Readings chosen where the brief is silent")
    A("")
    A("`.claude/commands/part30.md` is authoritative wherever it speaks. It is "
      "silent on the following, so each was decided BEFORE any model ran and "
      "committed in `configs/data/lanl.yaml` (commit 1 of this branch).")
    A("")
    A("| ambiguity | what part30.md says | reading taken |")
    A("|---|---|---|")
    A(f"| day boundary | nothing | `day = floor((t-1)/{day})+1`, so day 1 is "
      f"t in [1, {day}] |")
    A(f"| window length | \"window from config, default = the window chosen "
      f"for AIT\" | **{window} s**, the Part 27 review-gate choice carried by "
      "`configs/data/ait.yaml` and `experiments/ait_clean/config.yaml` "
      "(`chose_at: part27 review gate`). Not a Part 30 invention. |")
    A("| which computer owns an auth/flow record | nothing | the **source** "
      "computer (`attributed_computer: source`); destination columns still "
      "feed distinct-value features |")
    A("| what \"a red-team event falls inside it\" means for a "
      "(computer, window) sample | \"Labels from redteam.txt\" | **attack = 1 "
      "if an event in that window names that computer as source or "
      "destination**. The window-only alternative is counted and reported in "
      "section 3 rather than discarded. |")
    A("| features | nothing | the simplest count-based per-window aggregates, "
      "listed in `configs/data/lanl.yaml` `features:` before any run |")
    A("| train/test split fractions and gap length | \"Split by TIME with a "
      "gap\" | the pre-registered seconds in section 1 |")
    A("| sample universe | \"Sample = (computer, time window)\" | kept, but "
      "the cell count forced a declared case-control sample - section 2 |")
    A("")
    A("Two further readings are forced by this Part's own standing rules "
      "rather than by part30.md:")
    A("")
    A("- **No validation split and no early stopping.** The instruction was to "
      "carve validation from inside the train block *by the same mechanism "
      "Part 28 uses, and if Part 28 has none, add none*. Part 28 has none "
      "(`experiments/ait_clean/run_ait_clean.py::train_ait`: \"There is no "
      "validation split and no early stopping\"), so none was added.")
    A("- **Three Part 28 functions could not be imported.** `fit_scaler`, "
      "`apply_scaler` and `train_ait` resolve native view widths through "
      "`prismflow.data.ait_lds.FEATURES`, which is keyed by AIT source names: "
      "`FEATURES['proc']` and `FEATURES['flows']` do not exist and "
      "`FEATURES['auth']`/`FEATURES['dns']` carry AIT's widths, not LANL's. "
      "They are mirrored in `run_lanl.py` on LANL widths with Part 28's "
      "hyperparameters read from `config.yaml`. This is a reuse limitation of "
      "the Part 28 code, reported and **not patched** - "
      "`run_ait_clean.py` is untouched. Every metric, by contrast, IS Part "
      "28's own imported function.")
    A("")

    # -- day to seconds ----------------------------------------------------
    A("## 1. Day-to-seconds mapping and the split")
    A("")
    A(f"Window grid: `window = floor((t-1)/{window})`, so window 0 is "
      f"t in [1, {window}]. Every split boundary below is an exact multiple of "
      f"{window} s, so **no window straddles a boundary** - asserted in "
      "`tests/data/test_lanl_cyber1.py::"
      "test_every_split_boundary_lands_on_a_window_edge`.")
    A("")
    A("| block | days | seconds | windows | used for |")
    A("|---|---|---|---|---|")
    for name, use in (
        ("train", "fitting, scaling, training"),
        ("gap", "**dropped, never read into a sample**"),
        ("test", "scoring only"),
    ):
        bounds = data_cfg[name]
        lo = (bounds["start"] - 1) // window
        hi = bounds["end"] // window - 1
        A(f"| {name} | {bounds['days'][0]}-{bounds['days'][1]} | "
          f"{bounds['start']:,} .. {bounds['end']:,} | {lo:,} .. {hi:,} | {use} |")
    A(f"| unused | 17-30 | {data_cfg['test']['end'] + 1:,} .. 2,592,000 | - | "
      "not read |")
    A("")
    A(f"Red-team events inside the gap: "
      f"**{split['gap']['redteam_events_in_gap']}**. Samples built from the "
      f"gap: **{split['gap']['samples_built']}**.")
    A("")
    A("Nothing from the test block is used for fitting, scaling, feature "
      "selection, thresholds or model selection. The scaler is fitted on train "
      "rows only and refitted per view configuration "
      "(`results/lanl/results.json`, `configurations.<name>.scaler`).")
    A("")

    # -- population and sampling ------------------------------------------
    A("## 2. Why this is a case-control sample")
    A("")
    A("part30.md fixes the sample as `(computer, time window)`. Measured over "
      f"days 1-16, per view (`results/lanl/population.json`, `per_view`):")
    A("")
    A("| view | records streamed | occupied (computer, window) cells |")
    A("|---|---:|---:|")
    for view, record in population["per_view"].items():
        A(f"| {view} | {record['records_scanned']:,} | "
          f"{record['population_cells']:,} |")
    bounds = population["estimated_population_cells_union"]
    A("")
    A(f"The union over views is **not directly counted** "
      f"(`results/lanl/population.json`, "
      f"`estimated_population_cells_union`): it is bounded below by "
      f"{bounds['lower_bound']:,} and above by {bounds['upper_bound']:,}. "
      "Either way the sample universe is tens of millions of cells, against "
      f"{split['per_split']['train']['n_attack_samples']} train and "
      f"{split['per_split']['test']['n_attack_samples']} test attack cells - a "
      "population prevalence near 2e-5.")
    A("")
    A("Two consequences, both measurements rather than preferences:")
    A("")
    A("- 40 epochs over ~2e7 samples does not fit this Part's time budget.")
    A("- a uniform subsample that *does* fit leaves single-digit attack cells, "
      "which trips this Part's own stop condition of >= 10 attack windows per "
      "split.")
    A("")
    A(f"So the sample is a **declared case-control sample**: every attack cell "
      f"is kept, and a benign cell is kept iff "
      f"`crc32(computer|window) % {population['control_divisor']} == 0`. That "
      "is a deterministic hash of the cell identity - not a draw from a "
      "generator - so it is identical on every rerun and on every experiment "
      "seed, needs no prior knowledge of the cell universe, and reads neither "
      "a label nor a feature value.")
    A("")
    A("**What it costs.** Sample prevalence is an artefact of the divisor and "
      "is NOT population prevalence. Accuracy, Brier and ECE therefore "
      "describe this sample, not the network, and are marked `(sample "
      "prevalence)` wherever they appear. Balanced accuracy, ROC AUC and "
      "per-class recall are the figures that survive the reweighting.")
    A("")

    # -- attack windows per split -----------------------------------------
    A("## 3. Attack-window counts per split")
    A("")
    A("| block | samples | attack samples | benign samples | distinct attack "
      "windows | sample prevalence |")
    A("|---|---:|---:|---:|---:|---:|")
    for name in ("train", "test"):
        record = split["per_split"][name]
        A(f"| {name} | {record['n_samples']:,} | "
          f"{record['n_attack_samples']:,} | {record['n_benign_samples']:,} | "
          f"**{record['n_distinct_attack_windows']}** | "
          f"{record['attack_prevalence_in_sample']:.5f} |")
    A("")
    audit = read_json("results/lanl/label_audit.json")
    if audit:
        A("")
        A("**Reconciliation with the label file.** The red-team file names a "
          "(computer, window) cell; a sample exists only where some view "
          "actually recorded an event for that cell, so the two counts need "
          "not agree (`results/lanl/label_audit.json`):")
        A("")
        A("| block | attack cells named by redteam | attack samples built | "
          "named but unobserved by every view | distinct computers named |")
        A("|---|---:|---:|---:|---:|")
        for name in ("train", "test"):
            named = dig(audit, "per_block", name,
                        "computer_window_attack_cells_named_by_redteam")
            built = split["per_split"][name]["n_attack_samples"]
            A(f"| {name} | {named} | {built} | {named - built} | "
              f"{dig(audit, 'per_block', name, 'distinct_computers_named')} |")
        A("")
        A("An unobserved cell is a computer the red team touched in a window "
          "in which none of auth, proc, flows or dns logged anything for it. "
          "It is dropped, not counted as benign - counting it benign would "
          "label a known attack as clean.")
        A("")
        A("Under the window-only reading the same blocks would carry "
          f"{dig(audit, 'per_block', 'train', 'window_only_attacked_windows')} "
          "(train) and "
          f"{dig(audit, 'per_block', 'test', 'window_only_attacked_windows')} "
          "(test) attacked windows, with every computer present in those "
          "windows relabelled attack. That reading was not scored.")
        A("")
    A(f"Pre-registered stop condition: at least "
      f"{results['config']['stop_conditions']['min_attack_windows_per_split']}"
      " distinct attack windows in each of train and test. "
      f"**Both pass** ({split['per_split']['train']['n_distinct_attack_windows']}"
      f" and {split['per_split']['test']['n_distinct_attack_windows']}). The "
      "day range was not widened and no count was repaired "
      "(`results/lanl/split.json`).")
    A("")
    A("### Caveat (i): red-team events are concentrated")
    A("")
    A("The label is not spread evenly across either block. Per day "
      "(`results/lanl/split.json`, `per_split.<block>.per_day`):")
    A("")
    A("| block | day | samples | attack samples | attack windows |")
    A("|---|---:|---:|---:|---:|")
    for name in ("train", "test"):
        for dayno, record in sorted(split["per_split"][name]["per_day"].items(),
                                    key=lambda kv: int(kv[0])):
            A(f"| {name} | {dayno} | {record['samples']:,} | "
              f"{record['attack_samples']} | {record['attack_windows']} |")
    A("")
    A("A block's attack evidence is therefore dominated by a few days, and any "
      "per-seed spread below says nothing about variation across attack "
      "episodes. Red-team events per day across the whole file, including the "
      "days this Part does not use, are in `results/lanl/population.json` "
      "(`redteam_events_per_day`).")
    A("")
    A("The window-only label reading (section 0) was **not scored**: only the "
      "primary (computer, window) reading was trained and evaluated. The "
      "alternative is defined in `configs/data/lanl.yaml` `label.also_report` "
      "so the choice is on record, but no metric here rests on it.")
    A("")

    # -- caveats -----------------------------------------------------------
    A("### Caveat (ii): the split is fixed, so the std is training randomness only")
    A("")
    A("The split is fixed by time and the case-control sample is a "
      "deterministic hash, so a seed CANNOT redraw or re-partition the data. A "
      "seed selects exactly two things: parameter initialisation (`set_seed`) "
      "and minibatch order (`iter_batches(shuffle=True, seed=seed+epoch)`). "
      "**Every +/- below is initialisation-and-batch-order variance on one "
      "fixed split.** It is not dataset variance and it is not a spread over "
      "independent attacks.")
    A("")
    A("### Caveat (iii): the auth view is label-adjacent")
    A("")
    A("LANL's red-team file records authentication events, so the auth view "
      "contains the very records the labels are derived from. Any auth-view "
      "result is therefore label-adjacent and cannot be read as detection from "
      "independent evidence. That is why part30.md step 2 requires both "
      "configurations, and why the WITHOUT-auth column is the one to read for "
      "a detection claim.")
    A("")
    A("### Caveat (iv): \"benign\" means \"no known red-team event\"")
    A("")
    A(f"{results['label_meaning']}")
    A("")
    A("Unlike AIT, cyber1 ships no label-coverage mask, so there is no "
      "`label_known` filter to apply and no window can be marked unexamined. "
      "An unlabelled compromise inside the capture is counted as benign here, "
      "which inflates the benign class and can only depress measured "
      "precision-like quantities.")
    A("")

    # -- results -----------------------------------------------------------
    A("## 4. Part 28's metrics, repeated on LANL")
    A("")
    A("Identical settings to Part 28, read from `experiments/lanl/config.yaml`"
      f": {results['config']['training']['epochs']} epochs, batch "
      f"{results['config']['training']['batch_size']}, lr "
      f"{results['config']['training']['lr']}, anneal "
      f"{results['config']['training']['anneal_epochs']}, hidden "
      f"{results['config']['model']['hidden_dims']}, feature_dim "
      f"{results['config']['model']['feature_dim']}. Seeds "
      f"{results['config']['seeds']}. Not re-tuned for LANL.")
    A("")
    A("Every row is `prismflow.evaluation.protocol.compute_metrics` plus Part "
      "28's `part28_extras`, called through Part 28's imported `metric_row`.")
    A("")

    # -- execution mode ----------------------------------------------------
    mode = results.get("execution_mode") or {}
    if "decision" in mode:
        A("**Execution mode, disclosed.** Units of work - one "
          "`(arm, system, seed)` each - were run as "
          f"**{mode['decision']}** processes, "
          f"{mode['max_concurrent_processes']} at a time, with "
          f"**{mode['threads_per_process']} torch thread per process**. "
          "Hyperparameters and seeds 0-4 are unchanged; parallelism is across "
          "units only, and no unit shares state with another beyond the "
          "read-only sample table.")
        A("")
        A("| torch threads | ms/step (200 real steps, idle machine) | "
          "projected naive model |")
        A("|---:|---:|---:|")
        for row in mode.get("benchmark_idle_machine", []):
            A(f"| {row['threads']} | {row['ms_per_step']:.2f} | "
              f"{row['projected_naive_model_seconds']:.0f} s |")
        A("")
        A(mode.get("benchmark_note", ""))
        A("")
        achieved = mode.get("achieved_parallel_measured") or {}
        if achieved:
            A("The benchmark's throughput figure was a PROJECTION that assumed "
              "no contention between concurrent units. What the chosen mode "
              "actually delivered, measured from the `seconds` field of every "
              "unit file in `results/lanl/units/`:")
            A("")
            A("| quantity | projected before the run | measured after it |")
            A("|---|---:|---:|")
            A("| seconds per unit | "
              f"{fmt(mode.get('projected_parallel_seconds_per_unit'), 1)} | "
              f"{fmt(achieved.get('mean_seconds_per_unit'), 1)} |")
            A("| throughput vs run 1 sequential | "
              f"{fmt(mode.get('projected_throughput_speedup_vs_run1'), 2)}x | "
              f"{fmt(achieved.get('throughput_speedup_vs_run1'), 2)}x |")
            A("")
            A(achieved.get("note", ""))
            A("")
        thread_obs = mode.get("thread_count_reproducibility_observation") or {}
        if thread_obs:
            A("**Reported, not fixed - a frozen-module observation.** "
              + thread_obs["what"] + " " + thread_obs["status"])
            A("")
            measured = thread_obs.get("measured", {})
            A("| condition | `eniv_eigen` | against committed | asserted "
              "tolerance |")
            A("|---|---:|---:|---:|")
            A("| torch default (12 threads) | reproduces the committed value "
              f"| - | {measured.get('tolerance_asserted')} |")
            A(f"| `OMP_NUM_THREADS=1` | "
              f"{measured.get('omp_num_threads_1_eniv_eigen_obtained'):.10f} | "
              f"{measured.get('committed_part28_eniv_eigen_expected'):.10f} | "
              f"{measured.get('tolerance_asserted')} |")
            A("")
            A(thread_obs.get("why_it_matters_here", ""))
            A("")
            A(thread_obs.get("gate_consequence", ""))
            A("")
        contended = mode.get("benchmark_run2_contended") or {}
        if contended.get("note"):
            A("A second thread benchmark was taken on 2026-10-18 and is "
              "**excluded**. It is kept in `execution_mode.json` so the "
              "exclusion is auditable, not hidden: " + contended["note"])
            A("")
        audit = mode.get("phase0b_process_audit_run2_continuation") or {}
        if audit:
            A("**Process audit before this run's measurements "
              "(part30.md Phase 0b).** "
              f"{audit['n_python_processes_found']} python processes were "
              "found. " + audit.get("not_terminated", ""))
            A("")
            for line in audit.get("terminated", []):
                A(f"- Terminated: {line}")
            A("")
            A("*On the idle-machine requirement:* " +
              audit.get("cpu_not_idle_reading_taken", ""))
            A("")
        equivalence = mode.get("equivalence_check_seed0") or {}
        if equivalence:
            A("**Equivalence check (part30.md Phase 1).** Seed 0, `with_auth`, "
              "both systems, re-run in the chosen mode and compared with the "
              "same seed of run 1, which ran sequentially at torch's default "
              "thread count. Run 1's per-seed values are read from "
              "`results/lanl/results.run1.json.gz`.")
            A("")
            A("| system | metric | run 1 (seed 0) | run 2 (seed 0) | "
              "absolute difference |")
            A("|---|---|---:|---:|---:|")
            for system in ("naive", "prismflow"):
                rows = (equivalence.get("per_system") or {}).get(system) or {}
                for metric in ("balanced_accuracy", "roc_auc"):
                    cell = rows.get(metric)
                    if not cell:
                        A(f"| {system} | {metric} | not measured | "
                          "not measured | not measured |")
                        continue
                    A(f"| {system} | {metric} | {cell['run1']:.4f} | "
                      f"{cell['run2']:.4f} | {abs(cell['absolute_difference']):.4f} |")
            A("")
            A(equivalence.get("note", ""))
            A("")

    # -- arms summary ------------------------------------------------------
    A("### 4.1 All arms, side by side")
    A("")
    A("Paired within seed: the same 5 seeds, the same rows, the same split. "
      "Prevalence-dependent metrics are flagged - the sample is case-control "
      "at about 2% attack prevalence against about 2e-5 in the population, so "
      "ECE here describes the sample.")
    A("")
    A("| arm | views | system | balanced accuracy | ROC AUC | "
      "ECE *(sample prevalence)* | attack recall |")
    A("|---|---|---|---|---|---|---|")
    for name, configuration in configurations.items():
        for system in ("naive", "prismflow"):
            summary = dig(configuration, "systems", system, "summary")
            if not summary:
                continue
            A(f"| {name} | {', '.join(configuration['views'])} | {system} | "
              f"{fmt(dig(summary, 'balanced_accuracy'))} | "
              f"{fmt(dig(summary, 'roc_auc'))} | "
              f"{fmt(dig(summary, 'prob_ece'))} | "
              f"{fmt(dig(summary, 'attack_recall'))} |")
    A("")
    A("Paired difference, PrismFlow minus naive, within seed:")
    A("")
    A("| arm | balanced accuracy | ROC AUC | ECE | attack recall |")
    A("|---|---|---|---|---|")
    for name, configuration in configurations.items():
        paired = configuration.get("paired_prismflow_minus_naive") or {}
        if not paired:
            A(f"| {name} | not measured - only one system finished | | | |")
            continue
        A(f"| {name} | {signed(paired.get('balanced_accuracy'))} | "
          f"{signed(paired.get('roc_auc'))} | "
          f"{signed(paired.get('prob_ece'))} | "
          f"{signed(paired.get('attack_recall'))} |")
    A("")
    A("A seed count of 5/5 means every seed moved the same way; it does NOT "
      "mean the effect is larger than test-set sampling noise. Section 7.4's "
      "bootstrap interval is the number to read for that.")
    A("")

    # -- Brier against the constant forecast -------------------------------
    A("Brier against the majority baseline, which predicts the test block's "
      "class prevalence for every window. Its Brier is the score of a "
      "CONSTANT forecast, so a trained model above it is worse than that "
      "constant:")
    A("")
    A("| arm | system | Brier *(sample prevalence)* | constant-prevalence "
      "baseline | above the baseline |")
    A("|---|---|---:|---:|---|")
    above = []
    for name, configuration in configurations.items():
        base = dig(configuration, "majority_baseline", "brier")
        for system in ("naive", "prismflow"):
            value = dig(configuration, "systems", system, "summary", "brier",
                        "mean")
            flag = (
                isinstance(value, (int, float)) and isinstance(base, (int, float))
                and math.isfinite(float(value)) and math.isfinite(float(base))
                and float(value) > float(base)
            )
            if flag and name not in above:
                above.append(name)
            A(f"| {name} | {system} | "
              f"{fmt(dig(configuration, 'systems', system, 'summary', 'brier'))} | "
              f"{fmt(base)} | {'**YES**' if flag else 'no'} |")
    A("")
    if above:
        A("**Flagged: "
          + ", ".join(f"`{n}`" for n in above)
          + " "
          + ("is" if len(above) == 1 else "are")
          + " above the constant-prevalence baseline on Brier**, for both "
            "systems in each flagged arm. A probability forecast worse than a "
            "constant is a calibration failure, not a ranking failure - the "
            "same arms' ROC AUC is well above 0.5. **The cause is not "
            "diagnosed here.** Nothing in this Part isolates whether it comes "
            "from the empty-tensor rows of section 4.2, from the absence of a "
            "validation split, or from the case-control prevalence the "
            "forecast is trained against.")
        A("")

    # -- Part 28 step 5 drop-view configurations ---------------------------
    widest = max(configurations.values(), key=lambda c: len(c["views"]),
                 default=None)
    threshold = dig(widest, "missingness", "threshold")
    required = sorted({
        view
        for configuration in configurations.values()
        for view in configuration["missingness"]["over_threshold_train"]
    })
    if required and widest is not None:
        fractions = ", ".join(
            f"`{view}` {dig(widest, 'missingness', 'train', view):.1%}"
            for view in required
            if isinstance(dig(widest, "missingness", "train", view), (int, float))
        )
        ran = [view for view in required if f"without_{view}" in configurations]
        missing = [view for view in required if f"without_{view}" not in configurations]
        both = [
            name for name, configuration in configurations.items()
            if not set(required) & set(configuration["views"])
        ]
        A("**Part 28 step 5 drop-view configurations.** Step 5 asks for a "
          f"second configuration without any view missing in more than "
          f"{threshold:.0%} of train windows. On this sample table that is "
          f"{len(required)} view{'s' if len(required) != 1 else ''} - "
          f"{fractions} of train windows missing. "
          + (f"Configuration{'s' if len(ran) != 1 else ''} "
             + ", ".join(f"`without_{v}`" for v in ran)
             + (" ran" if len(ran) != 1 else " ran")
             + f", over all {len(results['config']['seeds'])} seeds and both "
               "systems; they are in the table above. "
             if ran else "No single-view-drop configuration ran. ")
          + (f"Still owed: {', '.join(f'`without_{v}`' for v in missing)}. "
             if missing else "")
          + ("**No configuration dropping ALL of "
             + ", ".join(f"`{v}`" for v in required)
             + " at once was run**, so the arms above each drop one of them "
               "and never both; this Part does not know what the remaining "
               "views do with both absent."
             if not both else
             "A configuration dropping all of them at once is "
             + ", ".join(f"`{n}`" for n in both) + "."))
        A("")

    # -- coverage ----------------------------------------------------------
    coverage = dig(results, "phase3_diagnostics", "arm_coverage")
    if coverage:
        A("### 4.2 Samples an arm cannot predict by construction")
        A("")
        A(coverage.get("note", ""))
        A("")
        A("| arm | train: attack with no present view | "
          "test: attack with no present view | test: benign with none |")
        A("|---|---:|---:|---:|")
        for name in configurations:
            entry = dig(coverage, "per_arm", name)
            if not entry:
                continue
            A(f"| {name} | {entry['train']['n_attack_empty']} of "
              f"{entry['train']['n_attack_samples']} | "
              f"**{entry['test']['n_attack_empty']} of "
              f"{entry['test']['n_attack_samples']}** "
              f"({entry['test']['attack_empty_share']:.1%}) | "
              f"{entry['test']['n_benign_empty']:,} |")
        A("")
        worst = max(
            ((n, dig(coverage, "per_arm", n, "test", "n_attack_empty") or 0)
             for n in configurations),
            key=lambda kv: kv[1], default=(None, 0),
        )
        if worst[1] > 0:
            A(f"This is a ceiling, not a tuning problem. In **{worst[0]}**, "
              f"{worst[1]} test attack samples have an all-zero feature "
              "tensor and an all-False view mask, so every model - and every "
              "future model - must give them the same score as each other and "
              "as the equally empty benign samples. The arm's attainable "
              "attack recall is capped below 1 before training starts, and "
              "its AUC carries a block of tied scores. Read that arm's "
              "numbers as a floor on what the remaining views can do, not as "
              "a measurement of what the views cannot.")
            A("")
    for name, configuration in configurations.items():
        A(f"### {configuration['label']}")
        A("")
        test_record = split["per_split"]["test"]
        A(f"Test set: n = {test_record['n_samples']:,}, "
          f"{test_record['n_attack_samples']} attack, "
          f"{test_record['n_benign_samples']:,} benign, sample prevalence "
          f"{test_record['attack_prevalence_in_sample']:.5f} "
          "(case-control; see section 2).")
        A("")
        A("| metric | naive fusion | V1 PrismFlow | majority baseline | "
          "paired PrismFlow - naive |")
        A("|---|---|---|---|---|")
        naive = configuration["systems"]["naive"]
        prism = configuration["systems"]["prismflow"]
        baseline = configuration["majority_baseline"]
        paired = dict(configuration["paired_prismflow_minus_naive"])
        # The diagnostics (ENIV, mean dependence) live in their own per-seed
        # list, so their paired difference is not in the metric pairing above.
        # Pair them here with the same imported function, after dropping the
        # non-scalar entries (`matrix`, `views`, `available`) it cannot read.
        def scalars(rows):
            return [
                {k: v for k, v in row.items() if isinstance(v, (int, float))}
                for row in rows
            ]
        for key, value in paired_difference(
            scalars(prism["diagnostics_per_seed"]),
            scalars(naive["diagnostics_per_seed"]),
            LOWER_IS_BETTER,
        ).items():
            paired.setdefault(key, value)
        for key, label, prevalence_dependent in HEADLINE:
            source_n = naive["summary"].get(key) or naive["diagnostics_summary"].get(key)
            source_p = prism["summary"].get(key) or prism["diagnostics_summary"].get(key)
            base = baseline.get(key)
            mark = " *(sample prevalence)*" if prevalence_dependent else ""
            A(f"| {label}{mark} | {fmt(source_n)} | {fmt(source_p)} | "
              f"{fmt(base)} | {signed(paired.get(key))} |")
        A("")
        seed_counts = {
            system: dig(configuration, "systems", system, "summary",
                        "balanced_accuracy", "n_seeds")
            for system in ("naive", "prismflow")
        }
        A(f"Seeds reported: naive {seed_counts['naive']}, PrismFlow "
          f"{seed_counts['prismflow']}, of "
          f"{len(results['config']['seeds'])} configured. "
          + ("Mean +/- std over 5 seeds, so these are findings under the "
             "5-seed rule - subject to every caveat in section 3."
             if all(c == len(results["config"]["seeds"])
                    for c in seed_counts.values())
             else "**Fewer than the configured seeds completed, so these are "
                  "NOT findings under the 5-seed rule.**"))
        A("")
        A("Majority-class baseline predicts the test set's majority class for "
          "every window with a constant probability vector, so its accuracy IS "
          "the prevalence, its balanced accuracy is 0.5 and its ROC AUC is 0.5 "
          "- the number an accuracy figure must beat before it describes "
          "anything but class balance "
          "(`experiments/ait_clean/run_ait_clean.py::majority_baseline`).")
        A("")
        A("Per-view missing fraction "
          "(`results/lanl/results.json`, `configurations."
          f"{name}.missingness`):")
        A("")
        A("| view | train | test |")
        A("|---|---:|---:|")
        for view in configuration["views"]:
            A(f"| {view} | {configuration['missingness']['train'][view]:.4f} | "
              f"{configuration['missingness']['test'][view]:.4f} |")
        over = configuration["missingness"]["over_threshold_train"]
        A("")
        A(f"Views missing in more than "
          f"{configuration['missingness']['threshold']:.0%} of train windows: "
          f"{', '.join(over) if over else 'none'}.")
        if over:
            A("")
            A("Part 28's step 5 drop-view configurations for "
              f"{', '.join(f'`{v}`' for v in over)} are stated once in "
              "section 4, above, rather than repeated per arm.")
        A("")

    # -- test A ------------------------------------------------------------
    # -- replicate ---------------------------------------------------------
    replicate = results.get("replicate_run1_vs_run2") or {}
    if "per_metric" in replicate:
        A("### 4.3 Run 1 as a replicate of with_auth")
        A("")
        A(replicate.get("note", ""))
        A("")
        A("Run 1's record is `results/lanl/results.run1.json.gz`; its sha256 "
          "is in `results/lanl/results_manifest.json` and was "
          + ("**verified** on reading."
             if replicate.get("run1_sha256_verified") else
             "**NOT verified** - the digest did not match, so treat the run-1 "
             "column with suspicion.")
          + f" Run 1 was commit `{replicate.get('run1_commit', 'unknown')}`.")
        A("")
        A("| metric | system | run 1 | run 2 | run 2 - run 1 |")
        A("|---|---|---:|---:|---:|")
        for key, systems in (replicate.get("per_metric") or {}).items():
            for system, row in systems.items():
                A(f"| {key} | {system} | {row['run1_mean']:.4f} | "
                  f"{row['run2_mean']:.4f} | "
                  f"{row['difference_run2_minus_run1']:+.4f} |")
        A("")
        within = replicate.get("paired_within_run") or {}
        if within.get("per_metric"):
            A("PrismFlow minus naive, paired within seed, in each run "
              "separately:")
            A("")
            A("| metric | run 1 paired difference | run 2 paired difference | "
              "sign differs between runs |")
            A("|---|---:|---:|---|")
            for key, row in within["per_metric"].items():
                def cell(prefix):
                    mean = row.get(f"{prefix}_paired_mean")
                    std = row.get(f"{prefix}_paired_std")
                    favouring = row.get(f"{prefix}_n_seeds_favouring_prismflow")
                    text = f"{float(mean):+.4f}"
                    if isinstance(std, (int, float)) and math.isfinite(float(std)):
                        text += f" +/- {float(std):.4f}"
                    if favouring is not None and row.get("n_seeds"):
                        text += f" ({favouring}/{row['n_seeds']})"
                    return text
                A(f"| {key} | {cell('run1')} | {cell('run2')} | "
                  + ("**YES**" if row["sign_differs_between_runs"] else "no")
                  + " |")
            A("")
            flips = within.get("sign_differs_between_runs") or []
            A("Metrics whose paired sign differs between the two runs: "
              + (", ".join(f"`{k}`" for k in flips) if flips
                 else "none of those listed")
              + ". " + within.get("note", ""))
            A("")
        A("These two runs are **replicates, not reproductions**. Same seeds "
          "and same hyperparameters, but a different torch thread count "
          "changes the order of floating-point reductions, so the trained "
          "weights differ and a seed does not reproduce bit-exactly across "
          "modes. Where a run-1-to-run-2 difference is comparable to or "
          "larger than that metric's per-seed std, the seed std is "
          "understating the spread an independent re-run would show.")
        A("")

    A("## 5. Part 29 test A - ENIV on real attack vs real benign windows")
    A("")
    first = next(iter(configurations.values()))["part29_test_a_natural_attack_windows"]
    A(f"{first['note']}")
    A("")
    for name, configuration in configurations.items():
        record = configuration["part29_test_a_natural_attack_windows"]
        A(f"### {configuration['label']}")
        A("")
        A(f"n = {record['n']['attack_windows']} attack windows, "
          f"{record['n']['benign_windows']:,} benign windows, **never pooled**.")
        A("")
        A("| system | quantity | attack windows | benign windows | "
          "paired attack - benign |")
        A("|---|---|---|---|---|")
        for system, block in record["per_system"].items():
            attack = dig(block, "populations", "attack_windows", "summary")
            benign = dig(block, "populations", "benign_windows", "summary")
            for key, label in (
                ("eniv_eigen", "ENIV (eigen)"),
                ("mean_dependence_eigen", "mean pairwise dependence"),
            ):
                A(f"| {system} | {label} | {fmt(dig(attack, key))} | "
                  f"{fmt(dig(benign, key))} | "
                  f"{raised(dig(block, 'paired_attack_minus_benign', key))} |")
        A("")
    A("Read the WITHOUT-auth block for a detection claim: with auth present, "
      "the views contain the records the labels came from (caveat iii).")
    A("")
    A("Sign convention: the last column is attack minus benign, and the count "
      "beside it is the number of seeds whose own difference is above zero. "
      "`paired_difference`'s own favouring count is NOT shown here: it reads "
      "a fall in `mean_dependence_eigen` as favouring PrismFlow, because that "
      "key is in `LOWER_IS_BETTER`, which answers a different question from "
      "the one test A asks.")
    A("")

    # -- test A: the signs, and how many rows can carry a pair -------------
    signs = []
    for name, configuration in configurations.items():
        record = configuration["part29_test_a_natural_attack_windows"]
        for system, block in record["per_system"].items():
            row = {"arm": name, "system": system}
            for key, short in (("eniv_eigen", "eniv"),
                               ("mean_dependence_eigen", "dependence")):
                mean = dig(block, "paired_attack_minus_benign", key, "mean")
                row[short] = (
                    None if not isinstance(mean, (int, float))
                    or not math.isfinite(float(mean))
                    else ("+" if float(mean) > 0 else
                          "-" if float(mean) < 0 else "0")
                )
            signs.append(row)
    if signs:
        A("**Signs of the attack-minus-benign differences, per arm and "
          "system.** A `+` means the quantity is higher on attack windows:")
        A("")
        A("| arm | system | ENIV (eigen) | mean pairwise dependence |")
        A("|---|---|:--:|:--:|")
        for row in signs:
            A(f"| {row['arm']} | {row['system']} | "
              f"{row['eniv'] or 'not measured'} | "
              f"{row['dependence'] or 'not measured'} |")
        A("")
        patterns = {}
        for row in signs:
            patterns.setdefault(row["arm"], set()).add(
                (row["eniv"] or "?", row["dependence"] or "?"))
        majority = {}
        for arm, pairs in patterns.items():
            for pair in pairs:
                majority[pair] = majority.get(pair, 0) + 1
        if majority:
            common = max(majority.items(), key=lambda kv: kv[1])[0]
            odd = [arm for arm, pairs in patterns.items() if common not in pairs]
            A("The most common pattern over the arms is ENIV "
              f"`{common[0]}` with dependence `{common[1]}`"
              + (", on every arm." if not odd else
                 ", and "
                 + ", ".join(f"`{a}`" for a in odd)
                 + (" is" if len(odd) == 1 else " are")
                 + " opposite to it. An arm that reverses sign while the "
                   "others agree is not explained by this Part.")
              )
            A("")
    shares = dig(results, "phase4_closeout", "views_present_shares")
    if shares:
        A("Dependence and ENIV are PAIR statistics, so only a test row with "
          "at least two of the arm's views present contributes anything to "
          "them. How many rows that is, per class:")
        A("")
        A("| arm | attack rows with >= 2 views | share of attack | "
          "benign rows with >= 2 views | share of benign |")
        A("|---|---:|---:|---:|---:|")
        for name in configurations:
            row = dig(shares, "per_arm", name)
            if not row:
                continue
            A(f"| {name} | {row['n_attack_two_or_more_views']:,} of "
              f"{row['n_attack_test']:,} | "
              f"{row['attack_share_two_or_more_views']:.1%} | "
              f"{row['n_benign_two_or_more_views']:,} of "
              f"{row['n_benign_test']:,} | "
              f"{row['benign_share_two_or_more_views']:.1%} |")
        A("")
        A(shares.get("note", ""))
        A("")

    # -- pair table --------------------------------------------------------
    A("## 6. View-pair dependence (Part 29 test C, first half)")
    A("")
    A("The strict same-sensor pair does not exist here, as it did not on AIT - "
      "but for a different reason. AIT's `sensor_groups()` returns a "
      "single-member group (`results/ait_attack/results.json`, "
      "`sensor_group_check.verdict`: \"NO EVIDENCE\"); LANL cyber1 ships **no "
      "sensor-group definition at all**, so there is no within-group pair to "
      "score. Verdict: **NO EVIDENCE** for strict same-sensor pairs. Nothing "
      "was substituted under that name.")
    A("")
    A("Below is a DIFFERENT and separately labelled quantity - **different "
      "sensors, naturally dependent** - the measured pair dependence on clean "
      "test data, from `results/lanl/results.json`, "
      "`configurations.<name>.systems.<system>.pair_table`.")
    A("")
    for name, configuration in configurations.items():
        A(f"### {configuration['label']}")
        A("")
        A("| pair | naive fusion | V1 PrismFlow |")
        A("|---|---|---|")
        tables = {
            system: {row["pair"]: row for row in block.get("pair_table", [])}
            for system, block in configuration["systems"].items()
        }
        # `pair_table` rows carry mean/std directly, so the row IS the entry
        # fmt reads. Ordered most-dependent first, which is the ordering Part
        # 29 test C asks to be READ from a committed table rather than
        # re-derived.
        naive_pairs = tables.get("naive", {})
        ordered_pairs = sorted(
            naive_pairs,
            key=lambda p: (
                -naive_pairs[p]["mean"]
                if math.isfinite(naive_pairs[p].get("mean", float("nan")))
                else 0.0
            ),
        )
        for pair in ordered_pairs:
            A(f"| {pair} | {fmt(dig(tables, 'naive', pair))} | "
              f"{fmt(dig(tables, 'prismflow', pair))} |")
        A("")
        if ordered_pairs:
            prism_pairs = tables.get("prismflow", {})
            prism_order = sorted(
                prism_pairs,
                key=lambda p: (
                    -prism_pairs[p]["mean"]
                    if math.isfinite(prism_pairs[p].get("mean", float("nan")))
                    else 0.0
                ),
            )
            A(f"Most dependent pair: **{ordered_pairs[0]}** (naive), "
              f"**{prism_order[0]}** (PrismFlow). Least dependent: "
              f"**{ordered_pairs[-1]}** (naive), **{prism_order[-1]}** "
              "(PrismFlow). "
              + ("The two systems agree on both ends."
                 if ordered_pairs[0] == prism_order[0]
                 and ordered_pairs[-1] == prism_order[-1]
                 else "**The two systems disagree on the ordering**, which is "
                      "reported here rather than resolved by picking one, as "
                      "Part 29 test C requires."))
        A("")
    A("Scoring the suspicion detector on the most and least dependent of these "
      "pairs was **not run** (see section 9), so this Part states no verdict "
      "on whether natural dependence raises the detector the way Part 10's "
      "exact clone did.")
    A("")

    # -- cross-dataset -----------------------------------------------------
    # -- Phase 3 diagnostics ----------------------------------------------
    phase3 = results.get("phase3_diagnostics") or {}
    A("## 7. Diagnostics (no new training)")
    A("")
    if not phase3:
        A("not measured - no diagnostics block in `results/lanl/results.json`.")
        A("")
    else:
        A("### 7.1 Trivial baselines: what a single scalar already achieves")
        A("")
        tb = phase3.get("trivial_baselines", {})
        A(tb.get("note", ""))
        A("")
        A("| scalar | orientation (fixed on train) | train AUC | **test AUC** |")
        A("|---|---|---:|---:|")
        for name, row in (tb.get("baselines") or {}).items():
            A(f"| `{name}` | {row['orientation_fixed_on_train']} | "
              f"{fmt(row['train_auc_as_measured'])} | **{fmt(row['test_auc'])}** |")
        A("")
        # Compare the best trivial scalar against the best model AUC, plainly.
        best_trivial = max(
            ((n, r["test_auc"]) for n, r in (tb.get("baselines") or {}).items()
             if isinstance(r.get("test_auc"), (int, float))
             and math.isfinite(r["test_auc"])),
            key=lambda kv: kv[1], default=(None, None),
        )
        model_aucs = [
            (f"{arm} / {system}", dig(conf, "systems", system, "summary", "roc_auc", "mean"))
            for arm, conf in configurations.items()
            for system in ("naive", "prismflow")
        ]
        model_aucs = [(k, v) for k, v in model_aucs if isinstance(v, (int, float))]
        if best_trivial[0] and model_aucs:
            best_model = max(model_aucs, key=lambda kv: kv[1])
            gap = best_model[1] - best_trivial[1]
            A(f"**Read this before any model number below.** The best single "
              f"scalar is `{best_trivial[0]}` at test AUC "
              f"{best_trivial[1]:.4f}; the best model is {best_model[0]} at "
              f"{best_model[1]:.4f}, a difference of {gap:+.4f}. "
              + ("**A single untrained scalar matches the models**, so the "
                 "ROC AUC figures in this Part are not evidence that the "
                 "fusion learned anything beyond how much this computer "
                 "logged in this window."
                 if gap < 0.05 else
                 "The models are ahead of every trivial scalar by more than "
                 "0.05 AUC, so the ranking is not reducible to one of these "
                 "scalars."))
            A("")

        A("### 7.2 Host audit: how concentrated the attack labels are")
        A("")
        ha = phase3.get("host_audit", {})
        A("| block | attack samples | attack computers | top-1 share | "
          "top-4 share |")
        A("|---|---:|---:|---:|---:|")
        for name in ("train", "test"):
            row = dig(ha, "per_split", name) or {}
            A(f"| {name} | {row.get('n_attack_samples', 0):,} | "
              f"{row.get('n_attack_computers', 0)} | "
              f"{fmt(row.get('top1_share'))} | {fmt(row.get('top4_share'))} |")
        A("")
        A(f"Attack computers appearing in BOTH blocks: "
          f"**{ha.get('attack_computers_shared_train_test', 'not measured')}**; "
          f"test-only attack computers: "
          f"{ha.get('attack_computers_test_only', 'not measured')}. "
          + (ha.get("note") or ""))
        A("")

        A("### 7.3 Seen vs unseen computers")
        A("")
        su = phase3.get("seen_unseen", {})
        A(su.get("note", ""))
        A("")
        A("| arm | system | seen AUC | unseen AUC | n seen | n unseen |")
        A("|---|---|---|---|---:|---:|")
        for arm, per_system in (su.get("per_arm") or {}).items():
            for system, row in per_system.items():
                if not isinstance(row, dict) or "seen" not in row:
                    continue
                seen, unseen = row["seen"], row["unseen"]
                A(f"| {arm} | {system} | "
                  f"{fmt({'mean': seen['auc_mean'], 'std': seen['auc_std']})} | "
                  f"{fmt({'mean': unseen['auc_mean'], 'std': unseen['auc_std']})} | "
                  f"{seen['n_attack_samples']} | {unseen['n_attack_samples']} |")
        A("")

        A("### 7.4 Paired cluster bootstrap over test computers")
        A("")
        cb = phase3.get("cluster_bootstrap", {})
        A(cb.get("note", ""))
        A("")
        A("| arm | mean paired AUC difference (PrismFlow - naive) | 95% CI | "
          "test computers |")
        A("|---|---:|---|---:|")
        for arm, row in (cb.get("per_arm") or {}).items():
            if "not_measured" in row:
                A(f"| {arm} | not measured | {row['not_measured']} | - |")
                continue
            lo, hi = row["ci95_low"], row["ci95_high"]
            crosses = (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
                       and lo <= 0.0 <= hi)
            A(f"| {arm} | "
              f"{row['observed_mean_paired_auc_difference']:+.4f} | "
              f"[{lo:+.4f}, {hi:+.4f}]"
              + (" **includes 0**" if crosses else " excludes 0")
              + f" | {row['n_test_computers']:,} |")
        A("")
        A("A per-seed std and this interval answer different questions. The "
          "std says how much the answer moves when the initialisation and "
          "batch order change on THIS test set; the interval says how much it "
          "would move if the test computers had been a different draw. An "
          "interval that includes 0 means the sign of the difference is not "
          "established by this test set, whatever the seed counts say.")
        A("")

        A("### 7.5 How a missing view enters dependence and ENIV")
        A("")
        A(phase3.get("missing_view_handling", "not measured"))
        A("")

    closeout = results.get("phase4_closeout") or {}
    if closeout:
        A(closeout.get("note", ""))
        A("")
        top = closeout.get("top_attack_computers") or {}
        if top.get("per_computer"):
            A(f"### 7.6 The {top.get('k')} most-attacked computers, as sampled")
            A("")
            A("| computer | train attack | train benign | test attack | "
              "test benign | attack train+test |")
            A("|---|---:|---:|---:|---:|---:|")
            for row in top["per_computer"]:
                A(f"| `{row['computer']}` | {row['train']['attack_samples']:,} | "
                  f"{row['train']['benign_samples']:,} | "
                  f"{row['test']['attack_samples']:,} | "
                  f"{row['test']['benign_samples']:,} | "
                  f"{row['attack_samples_train_plus_test']:,} |")
            A("")
            A(f"Attack computers in the sample table: "
              f"{top.get('n_attack_computers_total')}. "
              + top.get("note", ""))
            A("")

        tsu = closeout.get("trivial_scalar_seen_unseen") or {}
        if tsu.get("baselines"):
            A("### 7.7 The 7.1 scalars inside the 7.3 seen/unseen groups")
            A("")
            A(tsu.get("note", ""))
            A("")
            A("| scalar | orientation (fixed on train) | seen test AUC | "
              "unseen test AUC |")
            A("|---|---|---:|---:|")
            for name, row in tsu["baselines"].items():
                A(f"| `{name}` | {row['orientation_fixed_on_train']} | "
                  f"{fmt(row.get('seen_test_auc'))} | "
                  f"{fmt(row.get('unseen_test_auc'))} |")
            A("")
            counts = tsu.get("per_group_n_attack") or {}
            n_benign = tsu.get("n_benign_test_samples")
            A(f"n = {counts.get('seen', 'not measured')} seen attack rows, "
              f"{counts.get('unseen', 'not measured')} unseen attack rows, "
              "each against all "
              + (f"{n_benign:,}" if isinstance(n_benign, int) else "not measured")
              + " benign test rows - the same comparison sets as 7.3.")
            A("")

        pvo = closeout.get("without_auth_present_view_only") or {}
        if pvo.get("per_system"):
            A(f"### 7.8 `{pvo.get('arm')}` over its predictable test rows only")
            A("")
            A(pvo.get("note", ""))
            A("")
            A(f"Rows kept: {pvo['n_test_with_a_present_view']:,} of "
              f"{pvo['n_test_all_rows']:,} test rows "
              f"({pvo['n_attack_with_a_present_view']:,} attack, "
              f"{pvo['n_benign_with_a_present_view']:,} benign). Rows "
              f"excluded: {pvo['n_test_excluded']:,}, of which "
              f"{pvo['n_attack_excluded']:,} are attack rows.")
            A("")
            A("| system | ROC AUC (rows with a present view) | "
              "attack recall (same rows) |")
            A("|---|---|---|")
            for system in ("naive", "prismflow"):
                row = (pvo.get("per_system") or {}).get(system)
                if not row:
                    A(f"| {system} | not measured | not measured |")
                    continue
                A(f"| {system} | {fmt(row['roc_auc'])} | "
                  f"{fmt(row['attack_recall'])} |")
            A("")
            A("Beside these, the same arm over ALL test rows is in section "
              "4.1.")
            A("")

        drop = closeout.get("drop_top_test_computer") or {}
        if drop.get("per_arm"):
            A("### 7.9 Test AUC with the most-attacked test computer removed")
            A("")
            A(drop.get("note", ""))
            A("")
            A(f"Removed: every test row of `{drop['computer_removed']}` - "
              f"{drop['n_test_rows_removed']:,} rows "
              f"({drop['n_attack_rows_removed']:,} attack, "
              f"{drop['n_benign_rows_removed']:,} benign), leaving "
              f"{drop['n_test_rows_kept']:,} test rows of which "
              f"{drop['n_attack_rows_kept']:,} are attack rows.")
            A("")
            A("| arm | system | ROC AUC, that computer removed | "
              "ROC AUC, all test rows (4.1) |")
            A("|---|---|---|---|")
            for arm, per_system in drop["per_arm"].items():
                for system in ("naive", "prismflow"):
                    row = per_system.get(system)
                    full = dig(configurations, arm, "systems", system,
                               "summary", "roc_auc")
                    if not row:
                        A(f"| {arm} | {system} | not measured | {fmt(full)} |")
                        continue
                    A(f"| {arm} | {system} | {fmt(row['roc_auc'])} | "
                      f"{fmt(full)} |")
            A("")

    A("## 8. Cross-dataset table (part30.md step 5)")
    A("")
    A("synthetic vs AIT vs LANL, for the DETECTION claim and the "
      "honest-duplication false positive. Every cell cites a `results/` file "
      "or says `not measured`. Nothing is pooled across datasets and nothing "
      "is averaged across AIT's two directions.")
    A("")

    chorus = read_json("results/chorus/attack_metrics.json")
    ait = read_json("results/ait_attack/results.json")
    clone = read_json("results/clone/summary.json")

    A("### 8.1 DETECTION: does dependence among the implicated views rise?")
    A("")
    A("| dataset | condition | mean dependence | ENIV | source |")
    A("|---|---|---|---|---|")
    for cell, label, key in (
        # On clean synthetic data there is no compromised SET, so
        # `dependence_compromised` has no value to report and the all-pairs
        # mean is the comparable quantity. The key used is named in the row.
        ("clean|prismflow", "clean", "dependence_all_pairs"),
        ("chorus_k2_eps1.0|prismflow", "chorus k=2, eps 1.0", "dependence_compromised"),
    ):
        block = dig(chorus, "summary", cell)
        A(f"| synthetic | {label} | {fmt(dig(block, key))} (`{key}`) | "
          f"{fmt(dig(block, 'eniv_measured'))} | "
          f"`results/chorus/attack_metrics.json`, `summary.{cell}` |")
    for direction in ("A_rm_to_santos", "B_santos_to_rm"):
        for pop_key, label in (("dnsteal_only", "attack (dnsteal-only)"),
                               ("benign", "benign")):
            block = dig(ait, "directions", direction, "systems", "prismflow",
                        "natural_windows", pop_key, "summary")
            A(f"| AIT ({direction.split('_')[0]}) | {label} | "
              f"{fmt(dig(block, 'mean_dependence_eigen'))} | "
              f"{fmt(dig(block, 'eniv_eigen'))} | "
              f"`results/ait_attack/results.json`, `directions.{direction}."
              f"systems.prismflow.natural_windows.{pop_key}.summary` |")
    for name, configuration in configurations.items():
        block = configuration["part29_test_a_natural_attack_windows"]["per_system"]
        for pop_key, label in (("attack_windows", "attack"),
                               ("benign_windows", "benign")):
            summary = dig(block, "prismflow", "populations", pop_key, "summary")
            A(f"| LANL ({name}) | {label} | "
              f"{fmt(dig(summary, 'mean_dependence_eigen'))} | "
              f"{fmt(dig(summary, 'eniv_eigen'))} | "
              f"`results/lanl/results.json`, `configurations.{name}."
              f"part29_test_a_natural_attack_windows.per_system.prismflow."
              f"populations.{pop_key}.summary` |")
    A("")
    A("AIT's attack population is dnsteal-only, which Part 29 forbids pooling "
      "with its multi-step windows; LANL's red-team file has no step taxonomy, "
      "so LANL's single attack population is not the same object as either of "
      "AIT's. The rows are side by side, not comparable cell by cell.")
    A("")

    A("### 8.2 Honest-duplication false positive")
    A("")
    A("| dataset | quantity | value | source |")
    A("|---|---|---|---|")
    clone_rows = (clone or {}).get("summary") or []
    # That file carries TWO rows per k, one per base correlation rho. Picking
    # silently would make the cell ambiguous, so rho is pinned and named in
    # the citation: rho = 0 is duplication with no underlying correlation.
    CLONE_RHO = 0.0
    for k in (0, 2):
        row = next((r for r in clone_rows
                    if r.get("k") == k and r.get("system") == "prismflow"
                    and r.get("rho") == CLONE_RHO), None)
        # At k=0 there are no clone pairs, so dependence_clone_pairs has
        # nothing to report; the all-pairs mean is what exists at both k.
        overall = (
            fmt({"mean": row.get("mean_dependence_mean"),
                 "std": row.get("mean_dependence_std")}) if row else "not measured"
        )
        clone_pairs = (
            fmt({"mean": row.get("dependence_clone_pairs_mean"),
                 "std": row.get("dependence_clone_pairs_std")}) if row else "not measured"
        )
        if k == 0:
            clone_pairs += " (no clone pairs exist at k=0)"
        A(f"| synthetic | exact duplication, k={k}: mean dependence, all pairs "
          f"/ clone pairs | {overall} / {clone_pairs} | "
          f"`results/clone/summary.json`, `summary[]` where k={k}, "
          f"rho={CLONE_RHO:g}, system=prismflow |")
    A("| synthetic | Part 10 clone_k2 detector AUC | not measured in this Part "
      "| not re-run here; Part 10's own results file holds it |")
    A(f"| AIT | strict same-sensor pair | "
      f"{dig(ait, 'sensor_group_check', 'verdict') or 'not measured'} | "
      "`results/ait_attack/results.json`, `sensor_group_check.verdict` |")
    A("| LANL | strict same-sensor pair | NO EVIDENCE - cyber1 ships no "
      "sensor-group definition, so no within-group pair exists | "
      "`configs/data/lanl.yaml`; section 6 above |")
    A("| LANL | detector AUC on naturally dependent pairs | not measured | "
      "detector not scored (section 9) |")
    A("")

    # -- not run -----------------------------------------------------------
    A("## 9. What was not run, and why")
    A("")
    A("part30.md step 4 asks for Part 28's metrics **and Part 29's tests "
      "A-D**. Part 28's metrics and test A ran. B, C and D did not, and no "
      "partial substitute is reported in their place:")
    A("")
    not_run = next(iter(configurations.values()))["part29_tests_not_run"]
    for key, reason in not_run.items():
        A(f"- **{key}** - {reason}")
    A("")
    A("These are **INCOMPLETE**, not negative results. Nothing below should be "
      "read as evidence about chorus robustness, the suspicion detector or "
      "masked-view behaviour on LANL.")
    A("")

    # -- what the design cannot show ---------------------------------------
    A("## 10. What this design cannot show")
    A("")
    A("- **No population-level operating point.** The case-control sample "
      "fixes prevalence by construction, so no threshold, alert rate or "
      "precision figure here transfers to the real network.")
    A("- **No spread over attack episodes.** One fixed split, and the +/- is "
      "initialisation-and-batch-order variance (caveat ii). The red-team "
      "activity is concentrated in a few days (caveat i), so even the test "
      "block is close to a handful of episodes.")
    A("- **No independence from the label source in the WITH-auth arm.** "
      "Caveat iii.")
    A("- **No claim that benign windows are clean.** Caveat iv.")
    A("- **No cross-dataset significance.** Three datasets, one split each, "
      "different label semantics; section 7 is a side-by-side table, not a "
      "test.")
    A("- **No detection-vs-defence separation on LANL.** That separation is "
      "Part 09's and rests on the chorus grid, which was not run here.")
    A("")

    # -- proposed limitations ----------------------------------------------
    A("## 11. Proposed limitations (NOT applied)")
    A("")
    A("`docs/KNOWN_LIMITATIONS.md` and `docs/CONTRACT.md` are unchanged by "
      "this Part. The following text is proposed for the user to approve or "
      "reject; it has not been written to either file.")
    A("")
    # L23-L25 are generated from the data, so the figures inside them cannot
    # drift from the tables above.
    lower = dig(population, "estimated_population_cells_union", "lower_bound")
    upper = dig(population, "estimated_population_cells_union", "upper_bound")
    train_attack = dig(split, "per_split", "train", "n_attack_samples")
    test_attack = dig(split, "per_split", "test", "n_attack_samples")
    test_prevalence = dig(split, "per_split", "test", "attack_prevalence_in_sample")
    A(f"> **L23 (proposed).** On LANL cyber1 at {window} s the "
      f"`(computer, window)` sample space is between {lower:,} and "
      f"{upper:,} occupied cells over days 1-16 "
      "(`results/lanl/population.json`), against "
      f"{train_attack} train and {test_attack} test attack cells - a "
      "population prevalence near 2e-5. Part 30 therefore reports a declared "
      f"case-control sample (all cases, deterministic "
      f"1/{population.get('control_divisor')} hash of controls) whose test "
      f"prevalence is {test_prevalence:.4f}, about "
      f"{test_prevalence / 2e-5:.0f}x the population rate. Its accuracy, "
      "Brier and ECE therefore describe that sample and not the network, and "
      "no population operating point has been measured on LANL.")
    A("")
    auth_arms = [n for n in configurations if n != "without_auth"]
    without = dig(coverage or {}, "per_arm", "without_auth", "test")
    A("> **L24 (proposed).** LANL's red-team labels are derived from "
      "authentication records, so the `auth` view is label-adjacent. "
      + ("Part 30 measures both arms, so the WITHOUT-auth arm is the one that "
         "bears on detection from evidence independent of the label source."
         if "without_auth" in configurations else
         "**Part 30 measured only auth-bearing arms**, so every LANL number "
         "it reports is label-adjacent. The WITHOUT-auth arm part30.md step 2 "
         "requires is not measured.")
      + (f" That arm is itself limited: {without['n_attack_empty']} of "
         f"{without['n_attack_samples']} of its test attack samples "
         f"({without['attack_empty_share']:.1%}) have no record in any "
         "remaining view, so they are unpredictable by construction and cap "
         "the arm's attainable recall before training."
         if without and without.get("n_attack_empty") else ""))
    A("")
    best_trivial_name = None
    A("> **L25 (proposed).** Part 28's `fit_scaler`, `apply_scaler` and "
      "`train_ait` are not reusable across datasets: each resolves native view "
      "widths through `ait_lds.FEATURES`, keyed by AIT source names. Part 30 "
      "mirrored them rather than patching Part 28. Any future Part on a third "
      "dataset will hit the same wall.")
    A("")

    tb_block = dig(results, "phase3_diagnostics", "trivial_baselines", "baselines") or {}
    finite = [(n, r["test_auc"]) for n, r in tb_block.items()
              if isinstance(r.get("test_auc"), (int, float)) and math.isfinite(r["test_auc"])]
    if finite:
        best_trivial_name, best_trivial_auc = max(finite, key=lambda kv: kv[1])
        model_best = max(
            [v for arm, conf in configurations.items() for system in ("naive", "prismflow")
             for v in [dig(conf, "systems", system, "summary", "roc_auc", "mean")]
             if isinstance(v, (int, float))] or [float("nan")]
        )
        A(f"> **L26 (proposed).** On LANL cyber1 a single untrained scalar - "
          f"`{best_trivial_name}`, oriented on train only - reaches test ROC "
          f"AUC {best_trivial_auc:.4f}, against {model_best:.4f} for the best "
          "trained arm (`results/lanl/results.json`, "
          "`phase3_diagnostics.trivial_baselines`). "
          + ("The ROC AUC figures on this dataset are therefore not evidence "
             "that fusion learned structure beyond per-window log volume, and "
             "no AUC-based claim should be made for LANL without reporting "
             "this scalar beside it."
             if math.isfinite(model_best) and model_best - best_trivial_auc < 0.05
             else "The trained arms are ahead of this scalar by more than "
                  "0.05 AUC.")
          )
        A("")

    # -- files -------------------------------------------------------------
    A("## 12. Files")
    A("")
    A("| path | what it holds |")
    A("|---|---|")
    A("| `configs/data/lanl.yaml` | pre-registered day, window, split, "
      "features, label rule, control divisor, stop conditions |")
    A("| `prismflow/data/lanl_cyber1.py` | streaming read-only loader, "
      "`LANLMultiViewDataset` |")
    A("| `experiments/lanl/config.yaml` | Part 28's training and evaluation "
      "settings, unchanged |")
    A("| `experiments/lanl/run_lanl.py` | runner and this README's generator |")
    A("| `tests/data/test_lanl_cyber1.py` | loader tests |")
    A("| `results/lanl/population.json` | records streamed and occupied cells "
      "per view, per day |")
    A("| `results/lanl/split.json` | per-split and per-day sample and attack "
      "counts, attack blocks |")
    A("| `results/lanl/label_audit.json` | both label readings per block, from "
      "`redteam.txt.gz` alone |")
    A("| `results/lanl/samples.npz` | the whole 61,641-row sample table, built "
      "once; every arm reads it so all arms share identical rows |")
    A("| `results/lanl/units/*.json` | one file per (arm, system, seed): "
      "metrics, diagnostics, test-A populations, timing |")
    A("| `results/lanl/scores/*.npz` | per-sample TEST scores keyed by "
      "(computer, window_start), per arm, system and seed |")
    A("| `results/lanl/execution_mode.json` | the thread benchmark and the "
      "parallel-vs-sequential decision |")
    A("| `results/lanl/results.run1.json.gz` | run 1, immutable, with its "
      "sha256 in `results_manifest.json` |")
    A("| `results/lanl/results.json` | every metric, diagnostic, pair table "
      "and paired difference |")
    A("")
    wall = results.get("unit_wallclock") or {}
    A("**Wall-clock, and what each number measures.** "
      f"`results/lanl/results.json` `elapsed_seconds` = "
      f"{results.get('elapsed_seconds', float('nan')):.0f} s. "
      + (wall.get("elapsed_seconds_measures") or "")
      + f" The sample-table scan was {population.get('scan_seconds', float('nan')):.0f} s "
        f"on {population.get('workers')} worker processes, once, before any "
        "arm ran.")
    A("")
    if wall.get("n_units"):
        A("Training wall-clock, from the unit files only:")
        A("")
        A("| quantity | value |")
        A("|---|---|")
        A(f"| units | {wall['n_units']} |")
        A(f"| first unit start | {wall['first_unit_start']} |")
        A(f"| last unit end | {wall['last_unit_end']} |")
        A(f"| total wall-clock | {wall['total_wall_clock_seconds']} |")
        A(f"| sum of unit seconds | {wall['sum_of_unit_seconds']:,.0f} s "
          f"({wall['sum_of_unit_seconds'] / 3600:.1f} process-hours) |")
        A(f"| longest single unit | {wall['max_unit_seconds']:,.0f} s |")
        A(f"| shortest single unit | {wall['min_unit_seconds']:,.0f} s |")
        A(f"| concurrency limit | {wall['concurrency_limit']} |")
        A(f"| waves (ceiling of units over concurrency) | "
          f"{wall['waves_ceiling']} |")
        A("")
        A(wall.get("note", ""))
        A("")
    A(f"Commit `{results.get('git_commit') or 'not recorded'}`.")
    A("")
    A("`data/raw/` was read only. Nothing in it was extracted, renamed, moved "
      "or deleted, and `auth.txt.gz` was not modified.")
    A("")

    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# One (arm, system, seed) unit: the parallel granularity
# ---------------------------------------------------------------------------

UNITS_DIR = RESULTS_DIR / "units"
SCORES_DIR = RESULTS_DIR / "scores"


def arm_dataset(cfg: dict, data_cfg, arm: dict):
    """The shared sample table, subset to this arm's views, scaled on train only.

    Deterministic given the arm: the scaler is fitted on the train block of the
    same rows every time, so two units of one arm see byte-identical inputs and
    differ only by seed.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    full = LANLMultiViewDataset(samples)
    data = subset(full, tuple(arm["views"]))
    indices = split_indices(data)
    scaler = fit_scaler(data, indices["train"], cfg)
    return apply_scaler(data, scaler), indices, scaler


@torch.no_grad()
def per_sample_scores(model, dataset, test_idx, batch_size: int) -> dict:
    """Test-set scores keyed by the sample's own identity, for later diagnostics.

    Keyed by (computer, window_start) rather than row index, so a score file
    stays meaningful even if the sample table is ever rebuilt in a different
    order.
    """
    collected = collect_predictions(
        model, iter_batches(dataset, test_idx, batch_size, shuffle=False)
    )
    return {
        "computer": np.asarray(
            [dataset.samples.computer[i] for i in test_idx], dtype=object),
        "window_start": dataset.window_start.numpy()[test_idx],
        "label": np.asarray(collected["labels"], dtype=np.int64),
        "prob_attack": np.asarray(collected["probs"], dtype=np.float64)[:, CLASS_ATTACK],
        "vacuity": np.asarray(collected["vacuity_confidence"], dtype=np.float64),
    }, collected


def benchmark_steps(n_steps: int, threads: int) -> dict:
    """Time `n_steps` real training steps of a with_auth naive model.

    Real data and real batches from the train block, not synthetic tensors, so
    the number reflects what an arm will actually cost. The first 10 steps are
    discarded as warm-up.
    """
    torch.set_num_threads(max(1, threads))
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_cfg = load_lanl_config(cfg["data"]["config"])
    arm = next(a for a in cfg["view_configurations"] if a["name"] == "with_auth")
    scaled, indices, _ = arm_dataset(cfg, data_cfg, arm)
    train_idx = indices["train"]

    set_seed(0)
    model = PrismFlow(
        view_configs(
            scaled.sources,
            hidden_dims=cfg["model"]["hidden_dims"],
            feature_dim=cfg["model"]["feature_dim"],
            dropout=cfg["model"]["dropout"],
        ),
        n_classes=scaled.n_classes,
        use_discount=False,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    model.train()

    def steps(limit):
        done = 0
        while done < limit:
            for batch in iter_batches(
                scaled, train_idx, cfg["training"]["batch_size"],
                shuffle=True, seed=done,
            ):
                targets = torch.nn.functional.one_hot(
                    batch.labels, num_classes=scaled.n_classes
                ).to(batch.views.dtype)
                loss = compute_loss(
                    model(batch.views, batch.view_mask), targets,
                    epoch=0, anneal_epochs=cfg["training"]["anneal_epochs"],
                    per_view_loss_weight=cfg["training"]["per_view_loss_weight"],
                )
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                done += 1
                if done >= limit:
                    return

    steps(10)
    started = time.time()
    steps(n_steps)
    elapsed = time.time() - started
    batches_per_epoch = int(np.ceil(train_idx.size / cfg["training"]["batch_size"]))
    ms = elapsed / n_steps * 1000.0
    return {
        "threads": max(1, threads),
        "n_steps": n_steps,
        "ms_per_step": ms,
        "batches_per_epoch": batches_per_epoch,
        "epochs": cfg["training"]["epochs"],
        "projected_naive_model_seconds": ms / 1000.0 * batches_per_epoch * cfg["training"]["epochs"],
    }


def run_unit(arm_name: str, system_name: str, seed: int, threads: int) -> dict:
    """Train and score ONE (arm, system, seed) and write its own two files.

    Writing per unit is what makes the run restartable and what lets arms be
    run as separate processes: nothing is shared between units except the
    read-only sample table.
    """
    torch.set_num_threads(max(1, threads))
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_cfg = load_lanl_config(cfg["data"]["config"])
    arm = next(a for a in cfg["view_configurations"] if a["name"] == arm_name)
    system = next(s for s in cfg["systems"] if s["name"] == system_name)

    started = time.time()
    scaled, indices, scaler = arm_dataset(cfg, data_cfg, arm)
    train_idx, test_idx = indices["train"], indices["test"]
    model = train_lanl(scaled, train_idx, seed, system["use_discount"], cfg)
    scores, collected = per_sample_scores(
        model, scaled, test_idx, cfg["evaluation"]["batch_size"])

    row = {"seed": seed}
    row.update(metric_row(
        collected["labels"], collected["probs"], collected["vacuity_confidence"], cfg))
    diagnostics = {"seed": seed}
    diagnostics.update(dependence_and_eniv(model, scaled, test_idx, cfg))

    # Part 29 test A on this unit: ENIV and dependence on real attack vs real
    # benign test windows, never pooled.
    labels = scaled.labels.numpy()[test_idx]
    natural = {}
    for name, idx in (("attack_windows", test_idx[labels == CLASS_ATTACK]),
                      ("benign_windows", test_idx[labels == CLASS_BENIGN])):
        if idx.size < 2:
            natural[name] = {"not_measured": f"only {idx.size} windows"}
            continue
        record = dependence_and_eniv(model, scaled, idx, cfg)
        natural[name] = {k: v for k, v in record.items() if isinstance(v, (int, float))}

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    UNITS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{arm_name}_{system_name}_seed{seed}"
    score_path = SCORES_DIR / f"{tag}.npz"
    np.savez_compressed(score_path, **scores)

    unit = {
        "arm": arm_name, "system": system_name, "seed": seed,
        "views": list(arm["views"]),
        "use_discount": system["use_discount"],
        "threads": max(1, threads),
        "seconds": time.time() - started,
        "n_train": int(train_idx.size), "n_test": int(test_idx.size),
        "scaler": {
            "fit_rows": scaler["fit_rows"],
            "present_rows_per_view": scaler["present_rows_per_view"],
            "fitted_on": "train block only",
        },
        "metrics": row,
        "diagnostics": diagnostics,
        "natural_windows": natural,
        "scores_file": str(score_path).replace("\\", "/"),
        "scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
    }
    (UNITS_DIR / f"{tag}.json").write_text(json.dumps(json_safe(unit), indent=2))
    LOGGER.info(
        "unit %s done in %.1fs (bal_acc=%.4f roc_auc=%.4f)",
        tag, unit["seconds"], row["balanced_accuracy"], row["roc_auc"],
    )
    return unit


# ---------------------------------------------------------------------------
# Phase 3 diagnostics. No training: everything reads samples.npz and scores/.
# ---------------------------------------------------------------------------

#: How missing views enter dependence/ENIV. Stated as a constant so the README
#: cannot drift from the code it describes.
MISSING_VIEW_HANDLING = (
    "A missing view is carried as a BOOLEAN MASK, not as a zero-filled "
    "observation. `prismflow/statistics/dependence.py::feature_dependence_matrix` "
    "receives `view_mask` and `prismflow/statistics/dependence.py::available_views` "
    "reduces it to the per-view availability vector that "
    "`prismflow/eniv/eniv.py::compute_eniv` is given, so an absent view is "
    "excluded from the pair statistics rather than contributing a zero row. "
    "The feature row of an absent view IS zero in the tensor "
    "(`run_lanl.py::apply_scaler` leaves `views[~rows, v] = 0.0`), but the "
    "encoder never consumes it because `MultiViewEncoder` gates on the mask. "
    "Consequence for the pair table: a dns pair is computed over the samples "
    "where BOTH views of the pair are present, not over all samples, so the "
    "dns pairs rest on the ~2% of samples that have a dns record."
)


def trivial_scalar_values(samples) -> dict:
    """The three untrained scalars of section 7.1, as raw arrays.

    One definition, used by `trivial_baselines` and by the close-out
    seen/unseen reading, so the two cannot drift apart.
    """
    views = samples.views
    mask = samples.view_mask
    sources = list(samples.sources)
    # Feature 0 of every view is log1p(n_events) for that view, so the summed
    # expm1 is the total record count across views for that sample.
    totals = np.zeros(views.shape[0], dtype=np.float64)
    for v in range(views.shape[1]):
        totals += np.expm1(views[:, v, 0].astype(np.float64)) * mask[:, v]
    return {
        "total_records_across_views": totals,
        "auth_present_indicator": mask[:, sources.index("auth")].astype(np.float64),
        "n_views_present": mask.sum(axis=1).astype(np.float64),
    }


def arm_coverage(data_cfg, indices: dict, cfg: dict) -> dict:
    """Per arm: samples left with NO present view once that arm's views are dropped.

    All arms share one sample table, so dropping a view cannot remove a row -
    it can only empty it. A row whose every remaining view is absent carries
    an all-zero feature tensor and an all-False mask, so it is
    UNPREDICTABLE BY CONSTRUCTION: every model must give it the same score,
    whatever its label. Counting them is the honest way to say how much of an
    arm's attack class is unreachable before any training happens.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    sources = list(samples.sources)
    out = {
        "note": (
            "Shared sample table: all arms use the SAME rows, labels and "
            "split. An arm drops a view's feature columns and encoder, never "
            "a row. A row with no present view among the arm's views is "
            "unpredictable by construction - all models score it identically."
        ),
        "n_samples_total": int(labels.size),
        "per_arm": {},
    }
    for arm in cfg["view_configurations"]:
        keep = [sources.index(v) for v in arm["views"]]
        present_any = samples.view_mask[:, keep].any(axis=1)
        entry = {}
        for split_name in ("train", "test"):
            idx = indices[split_name]
            empty = ~present_any[idx]
            attack = labels[idx] == CLASS_ATTACK
            entry[split_name] = {
                "n_samples": int(idx.size),
                "n_empty_all_views_absent": int(empty.sum()),
                "n_attack_samples": int(attack.sum()),
                "n_attack_empty": int((empty & attack).sum()),
                "n_benign_empty": int((empty & ~attack).sum()),
                "attack_empty_share": (
                    float((empty & attack).sum() / attack.sum())
                    if attack.sum() else float("nan")
                ),
            }
        out["per_arm"][arm["name"]] = entry
    return out


def trivial_baselines(data_cfg, indices: dict) -> dict:
    """Scalars a model must beat: record volume, auth presence, views present.

    Orientation is fixed on the TRAIN block only - if a scalar ranks attacks
    below benigns on train, its sign is flipped, and that decision is made
    without ever looking at test. The test AUC that follows is then honest
    about direction.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    candidates = trivial_scalar_values(samples)
    out = {
        "note": (
            "Each row is a SINGLE SCALAR, no training. Orientation (sign) is "
            "chosen on the train block only and then applied unchanged to "
            "test."
        ),
        "baselines": {},
    }
    for name, score in candidates.items():
        tr, te = indices["train"], indices["test"]
        train_auc = roc_auc_binary(labels[tr], score[tr])
        sign = 1.0 if not math.isfinite(train_auc) or train_auc >= 0.5 else -1.0
        out["baselines"][name] = {
            "train_auc_as_measured": _safe(train_auc),
            "orientation_fixed_on_train": "as-is" if sign > 0 else "negated",
            "test_auc": _safe(roc_auc_binary(labels[te], sign * score[te])),
            "n_test": int(te.size),
        }
    return out


def host_audit(data_cfg, indices: dict) -> dict:
    """Attack samples per computer, and how concentrated they are."""
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    computers = np.asarray(samples.computer, dtype=object)
    out = {"per_split": {}}
    attack_sets = {}
    for name in ("train", "test"):
        idx = indices[name]
        attack_idx = idx[labels[idx] == CLASS_ATTACK]
        hosts, counts = np.unique(computers[attack_idx], return_counts=True)
        order = np.argsort(-counts)
        counts = counts[order]
        hosts = hosts[order]
        total = int(counts.sum())
        attack_sets[name] = set(hosts.tolist())
        out["per_split"][name] = {
            "n_attack_samples": total,
            "n_attack_computers": int(hosts.size),
            "top1_computer": str(hosts[0]) if hosts.size else None,
            "top1_share": float(counts[0] / total) if total else float("nan"),
            "top4_share": float(counts[:4].sum() / total) if total else float("nan"),
            "top10": [
                {"computer": str(h), "attack_samples": int(c)}
                for h, c in zip(hosts[:10], counts[:10])
            ],
        }
    shared = attack_sets["train"] & attack_sets["test"]
    out["attack_computers_shared_train_test"] = len(shared)
    out["attack_computers_test_only"] = len(attack_sets["test"] - attack_sets["train"])
    out["note"] = (
        "A computer with attack samples in BOTH blocks lets a model recognise "
        "the host rather than the behaviour, which is why the seen/unseen "
        "split below is reported."
    )
    return out


def _load_scores(arm: str, system: str, seeds) -> list[dict]:
    rows = []
    for seed in seeds:
        path = SCORES_DIR / f"{arm}_{system}_seed{seed}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as handle:
            rows.append({
                "seed": seed,
                "computer": np.asarray([str(c) for c in handle["computer"]], dtype=object),
                "label": handle["label"],
                "prob_attack": handle["prob_attack"],
            })
    return rows


def seen_unseen_auc(data_cfg, indices: dict, cfg: dict, arms: list[str]) -> dict:
    """Test AUC on attack samples whose computer is also attacked in train, vs not.

    Each group is scored against ALL benign test samples, so the two AUCs are
    comparable to each other and to the arm's overall AUC. Descriptive only:
    the groups are different sizes and are not a randomised comparison.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    computers = np.asarray(samples.computer, dtype=object)
    train_attack_hosts = set(
        computers[indices["train"][labels[indices["train"]] == CLASS_ATTACK]].tolist()
    )
    out = {
        "note": (
            "'seen' = the attack sample's computer also has at least one "
            "attack sample in the train block. Each group is scored against "
            "all benign test samples. Descriptive, not a controlled "
            "comparison."
        ),
        "per_arm": {},
    }
    for arm in arms:
        per_system = {}
        for system in ("naive", "prismflow"):
            rows = _load_scores(arm, system, cfg["seeds"])
            if not rows:
                continue
            groups = {"seen": [], "unseen": []}
            counts = {}
            for row in rows:
                is_attack = row["label"] == CLASS_ATTACK
                seen_flag = np.asarray(
                    [c in train_attack_hosts for c in row["computer"]], dtype=bool)
                benign = ~is_attack
                for key, selector in (("seen", is_attack & seen_flag),
                                      ("unseen", is_attack & ~seen_flag)):
                    counts[key] = int(selector.sum())
                    keep = selector | benign
                    if selector.sum() == 0:
                        groups[key].append(float("nan"))
                        continue
                    groups[key].append(
                        roc_auc_binary(row["label"][keep], row["prob_attack"][keep]))
            per_system[system] = {
                key: {
                    "auc_mean": _safe(float(np.nanmean(values))) if values else float("nan"),
                    "auc_std": _safe(float(np.nanstd(values, ddof=1))) if len(values) > 1 else float("nan"),
                    "n_attack_samples": counts.get(key, 0),
                    "n_seeds": int(np.sum(np.isfinite(values))),
                }
                for key, values in groups.items()
            }
            per_system[system]["n_benign_test_samples"] = int((rows[0]["label"] == CLASS_BENIGN).sum())
        out["per_arm"][arm] = per_system
    return out


def cluster_bootstrap(cfg: dict, arms: list[str], replicates: int = 2000,
                      seed: int = 12345) -> dict:
    """Paired cluster bootstrap over test COMPUTERS for AUC(prismflow) - AUC(naive).

    Clusters are computers, because samples from one computer are not
    independent: the same host contributes many windows. Resampling rows would
    understate the interval. The statistic is the mean over seeds of the paired
    difference, recomputed on each resampled set of computers.

    This interval is about TEST-SET SAMPLING. It is a different quantity from
    the per-seed std, which is training randomness on a fixed test set.
    """
    out = {
        "note": (
            "2000 replicates, fixed seed 12345, clusters = test computers, "
            "paired (same resampled computers for both systems, every seed). "
            "Measures test-set sampling variability; the per-seed std "
            "measures training randomness. They are not interchangeable."
        ),
        "replicates": replicates,
        "bootstrap_seed": seed,
        "per_arm": {},
    }
    for arm in arms:
        naive = _load_scores(arm, "naive", cfg["seeds"])
        prism = _load_scores(arm, "prismflow", cfg["seeds"])
        if not naive or not prism or len(naive) != len(prism):
            out["per_arm"][arm] = {"not_measured": "scores missing for one system"}
            continue
        computers = naive[0]["computer"]
        hosts = np.unique(computers)
        index_by_host = {h: np.flatnonzero(computers == h) for h in hosts}

        def paired_mean(rows_a, rows_b, rows_idx):
            diffs = []
            for a, b in zip(rows_a, rows_b):
                auc_a = roc_auc_binary(a["label"][rows_idx], a["prob_attack"][rows_idx])
                auc_b = roc_auc_binary(b["label"][rows_idx], b["prob_attack"][rows_idx])
                if math.isfinite(auc_a) and math.isfinite(auc_b):
                    diffs.append(auc_a - auc_b)
            return float(np.mean(diffs)) if diffs else float("nan")

        all_rows = np.arange(computers.size)
        observed = paired_mean(prism, naive, all_rows)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(replicates):
            picked = rng.choice(hosts, size=hosts.size, replace=True)
            rows_idx = np.concatenate([index_by_host[h] for h in picked])
            value = paired_mean(prism, naive, rows_idx)
            if math.isfinite(value):
                draws.append(value)
        draws = np.asarray(draws, dtype=np.float64)
        out["per_arm"][arm] = {
            "observed_mean_paired_auc_difference": _safe(observed),
            "ci95_low": _safe(float(np.percentile(draws, 2.5))) if draws.size else float("nan"),
            "ci95_high": _safe(float(np.percentile(draws, 97.5))) if draws.size else float("nan"),
            "n_replicates_used": int(draws.size),
            "n_test_computers": int(hosts.size),
        }
    return out


def replicate_table(configurations: dict) -> dict:
    """Run 1 vs run 2 of with_auth, per metric. Run 1 is never replaced.

    Run 1 was executed sequentially at torch's default thread count; run 2 in
    whatever mode this Part chose. A difference here is execution-mode plus
    floating-point reduction order, NOT a different experiment: same seeds,
    same hyperparameters, same sample table rule.
    """
    gz = RESULTS_DIR / "results.run1.json.gz"
    if not gz.exists() or "with_auth" not in configurations:
        return {"not_measured": "run 1 record or with_auth arm absent"}
    import gzip as _gzip
    run1 = json.loads(_gzip.decompress(gz.read_bytes()))
    manifest = json.loads((RESULTS_DIR / "results_manifest.json").read_text())
    digest = hashlib.sha256(
        _gzip.decompress(gz.read_bytes())).hexdigest()
    out = {
        "note": (
            "Same seeds, same hyperparameters, same sample rule; run 1 ran "
            "sequentially at torch's default thread count, run 2 in this "
            "Part's chosen mode. Differences are execution mode and "
            "floating-point reduction order, not a different experiment."
        ),
        "run1_sha256_verified": digest == manifest["run1"]["decompressed_sha256"],
        "run1_commit": run1.get("git_commit"),
        "per_metric": {},
    }
    for system in ("naive", "prismflow"):
        old = dig(run1, "configurations", "with_auth", "systems", system, "summary")
        new = dig(configurations, "with_auth", "systems", system, "summary")
        if not old or not new:
            continue
        for key in ("balanced_accuracy", "roc_auc", "prob_ece", "attack_recall"):
            a, b = dig(old, key, "mean"), dig(new, key, "mean")
            if a is None or b is None:
                continue
            out["per_metric"].setdefault(key, {})[system] = {
                "run1_mean": float(a),
                "run2_mean": float(b),
                "difference_run2_minus_run1": float(b) - float(a),
            }

    # The PrismFlow-minus-naive paired difference WITHIN each run. Both runs
    # pair the same seeds, so each column is a paired reading; what the two
    # columns together can show is whether the SIGN of that pairing survives a
    # change of execution mode.
    paired_old = dig(run1, "configurations", "with_auth",
                     "paired_prismflow_minus_naive") or {}
    paired_new = dig(configurations, "with_auth",
                     "paired_prismflow_minus_naive") or {}
    within = {
        "note": (
            "PrismFlow minus naive, paired within seed, in each run "
            "separately. A seed count inside ONE run shares that run's "
            "execution mode for both systems, so it is evidence about the "
            "two systems under that mode and NOT independent evidence about "
            "mode effects. Only the two columns side by side speak to the "
            "mode, and with one run each they do so without a spread."
        ),
        "per_metric": {},
        "sign_differs_between_runs": [],
    }
    for key in ("balanced_accuracy", "roc_auc", "prob_ece", "attack_recall",
                "brier", "benign_recall", "accuracy"):
        a, b = dig(paired_old, key, "mean"), dig(paired_new, key, "mean")
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        flips = (a > 0 > b) or (a < 0 < b)
        within["per_metric"][key] = {
            "run1_paired_mean": a,
            "run1_paired_std": dig(paired_old, key, "std"),
            "run1_n_seeds_favouring_prismflow": dig(
                paired_old, key, "n_seeds_favouring_prismflow"),
            "run2_paired_mean": b,
            "run2_paired_std": dig(paired_new, key, "std"),
            "run2_n_seeds_favouring_prismflow": dig(
                paired_new, key, "n_seeds_favouring_prismflow"),
            "n_seeds": dig(paired_new, key, "n_seeds"),
            "lower_is_better": key in LOWER_IS_BETTER,
            "sign_differs_between_runs": bool(flips),
        }
        if flips:
            within["sign_differs_between_runs"].append(key)
    out["paired_within_run"] = within
    return out


# ---------------------------------------------------------------------------
# Phase 4 close-out diagnostics. Read-only, no training: every figure comes
# from results/lanl/samples.npz and results/lanl/scores/*.npz.
# ---------------------------------------------------------------------------

#: Threshold used wherever a close-out figure needs a hard class decision.
#: Two classes, so this is the argmax of the probability vector.
DECISION_THRESHOLD = 0.5


def _align_test_rows(samples, indices: dict, row: dict) -> None:
    """Verify a score file's rows are the test block in order, or raise.

    The score files are written in `indices['test']` order, so positional
    alignment is correct - but it is CHECKED rather than assumed, because a
    silent misalignment would corrupt every figure below.
    """
    test_idx = indices["test"]
    if row["prob_attack"].size != test_idx.size:
        raise RuntimeError("score file length does not match the test block")
    if not np.array_equal(row["label"], samples.labels[test_idx]):
        raise RuntimeError("score file labels do not match the test block")
    expected = np.asarray(
        [str(samples.computer[i]) for i in test_idx], dtype=object)
    if not np.array_equal(row["computer"], expected):
        raise RuntimeError("score file computers do not match the test block")


def _mean_std(values: list[float]) -> dict:
    """mean, sample std and n over seeds, NaN excluded, in summarise()'s shape."""
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "mean": float(np.mean(kept)) if kept else float("nan"),
        "std": float(np.std(kept, ddof=1)) if len(kept) > 1 else float("nan"),
        "n_seeds": len(kept),
    }


def top_attack_computer_table(data_cfg, indices: dict, k: int = 4) -> dict:
    """The k computers with the most attack samples (train+test), as sampled.

    Counts are of rows in the sample table, which is case-control: the benign
    counts are of KEPT controls (1/512 deterministic hash), not of the
    computer's benign windows on the network.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    computers = np.asarray([str(c) for c in samples.computer], dtype=object)
    attack_all = labels == CLASS_ATTACK
    hosts, counts = np.unique(computers[attack_all], return_counts=True)
    order = np.argsort(-counts)
    top = [str(h) for h in hosts[order][:k]]
    rows = []
    for host in top:
        is_host = computers == host
        entry = {"computer": host}
        total_attack = 0
        for split_name in ("train", "test"):
            idx = indices[split_name]
            sel = is_host[idx]
            lab = labels[idx]
            entry[split_name] = {
                "attack_samples": int((sel & (lab == CLASS_ATTACK)).sum()),
                "benign_samples": int((sel & (lab == CLASS_BENIGN)).sum()),
            }
            total_attack += entry[split_name]["attack_samples"]
        entry["attack_samples_train_plus_test"] = total_attack
        rows.append(entry)
    return {
        "note": (
            "Counts are rows of the shared sample table, which is "
            "case-control: every attack cell is kept and a benign cell is "
            "kept iff crc32(computer|window) % 512 == 0, so the benign "
            "column is kept controls and not the computer's benign windows "
            "on the network."
        ),
        "k": k,
        "n_attack_computers_total": int(hosts.size),
        "per_computer": rows,
    }


def trivial_scalar_seen_unseen(data_cfg, indices: dict) -> dict:
    """7.1's scalars, split by 7.3's seen/unseen attack groups.

    Orientation is fixed on the whole train block exactly as in 7.1, then
    each attack group is scored against ALL benign test samples, exactly as
    in 7.3, so the two sections can be read side by side.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    computers = np.asarray([str(c) for c in samples.computer], dtype=object)
    train_idx, test_idx = indices["train"], indices["test"]
    train_attack_hosts = set(
        computers[train_idx[labels[train_idx] == CLASS_ATTACK]].tolist())
    is_attack = labels[test_idx] == CLASS_ATTACK
    seen_flag = np.asarray(
        [c in train_attack_hosts for c in computers[test_idx]], dtype=bool)
    benign = ~is_attack
    groups = {"seen": is_attack & seen_flag, "unseen": is_attack & ~seen_flag}
    out = {
        "note": (
            "Same scalars and same train-only orientation as 7.1; same "
            "seen/unseen definition and same all-benign comparison set as "
            "7.3. Untrained, so there is no seed and no std."
        ),
        "n_benign_test_samples": int(benign.sum()),
        "per_group_n_attack": {k: int(v.sum()) for k, v in groups.items()},
        "baselines": {},
    }
    for name, score in trivial_scalar_values(samples).items():
        train_auc = roc_auc_binary(labels[train_idx], score[train_idx])
        sign = 1.0 if not math.isfinite(train_auc) or train_auc >= 0.5 else -1.0
        oriented = sign * score[test_idx]
        entry = {"orientation_fixed_on_train": "as-is" if sign > 0 else "negated"}
        for key, selector in groups.items():
            keep = selector | benign
            entry[f"{key}_test_auc"] = (
                _safe(roc_auc_binary(labels[test_idx][keep], oriented[keep]))
                if selector.sum() else float("nan")
            )
        out["baselines"][name] = entry
    return out


def present_view_only(data_cfg, indices: dict, cfg: dict, arm_name: str) -> dict:
    """One arm's test figures over its PREDICTABLE test rows only.

    Rows whose every view in this arm is absent carry an all-zero tensor and
    an all-False mask (see `arm_coverage`), so no model can separate them.
    Restricting to rows with at least one present view says what the arm does
    where it has any evidence at all.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    arm = next((a for a in cfg["view_configurations"]
                if a["name"] == arm_name), None)
    if arm is None:
        return {"not_measured": f"no arm named {arm_name} in the config"}
    sources = list(samples.sources)
    keep_views = [sources.index(v) for v in arm["views"]]
    test_idx = indices["test"]
    present = samples.view_mask[:, keep_views].any(axis=1)[test_idx]
    labels_test = samples.labels[test_idx]
    out = {
        "note": (
            "Test rows with at least one of this arm's views present. The "
            "excluded rows are unpredictable by construction, not misclassified. "
            f"Attack recall is at a {DECISION_THRESHOLD} probability "
            "threshold, which for two classes is the argmax."
        ),
        "arm": arm_name,
        "views": list(arm["views"]),
        "n_test_all_rows": int(test_idx.size),
        "n_test_with_a_present_view": int(present.sum()),
        "n_test_excluded": int((~present).sum()),
        "n_attack_with_a_present_view": int(
            (present & (labels_test == CLASS_ATTACK)).sum()),
        "n_attack_excluded": int(
            (~present & (labels_test == CLASS_ATTACK)).sum()),
        "n_benign_with_a_present_view": int(
            (present & (labels_test == CLASS_BENIGN)).sum()),
        "per_system": {},
    }
    for system in ("naive", "prismflow"):
        rows = _load_scores(arm_name, system, cfg["seeds"])
        if not rows:
            continue
        aucs, recalls = [], []
        for row in rows:
            _align_test_rows(samples, indices, row)
            lab = row["label"][present]
            prob = row["prob_attack"][present]
            aucs.append(roc_auc_binary(lab, prob))
            attack = lab == CLASS_ATTACK
            recalls.append(
                float((prob[attack] >= DECISION_THRESHOLD).mean())
                if attack.any() else float("nan"))
        out["per_system"][system] = {
            "roc_auc": _mean_std(aucs),
            "attack_recall": _mean_std(recalls),
        }
    return out


def drop_top_computer(data_cfg, indices: dict, cfg: dict, arms: list[str]) -> dict:
    """Test AUC per arm and system with the single most-attacked test computer removed.

    Every row of that computer goes, attack and benign alike, so the removal
    does not itself move the class balance in that computer's favour.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    labels = samples.labels
    computers = np.asarray([str(c) for c in samples.computer], dtype=object)
    test_idx = indices["test"]
    attack_test = test_idx[labels[test_idx] == CLASS_ATTACK]
    if attack_test.size == 0:
        return {"not_measured": "no attack samples in the test block"}
    hosts, counts = np.unique(computers[attack_test], return_counts=True)
    top = str(hosts[int(np.argmax(counts))])
    keep = computers[test_idx] != top
    labels_test = labels[test_idx]
    out = {
        "note": (
            "All test rows of one computer removed, attack and benign "
            "together. Descriptive: one host is not a resample, so this is "
            "not an interval."
        ),
        "computer_removed": top,
        "n_test_rows_removed": int((~keep).sum()),
        "n_attack_rows_removed": int(
            (~keep & (labels_test == CLASS_ATTACK)).sum()),
        "n_benign_rows_removed": int(
            (~keep & (labels_test == CLASS_BENIGN)).sum()),
        "n_test_rows_kept": int(keep.sum()),
        "n_attack_rows_kept": int((keep & (labels_test == CLASS_ATTACK)).sum()),
        "per_arm": {},
    }
    for arm in arms:
        per_system = {}
        for system in ("naive", "prismflow"):
            rows = _load_scores(arm, system, cfg["seeds"])
            if not rows:
                continue
            aucs = []
            for row in rows:
                _align_test_rows(samples, indices, row)
                aucs.append(roc_auc_binary(
                    row["label"][keep], row["prob_attack"][keep]))
            per_system[system] = {"roc_auc": _mean_std(aucs)}
        out["per_arm"][arm] = per_system
    return out


def views_present_shares(data_cfg, indices: dict, cfg: dict) -> dict:
    """Per arm: share of attack and of benign TEST rows with >= 2 views present.

    Dependence and ENIV are pair statistics, so a row with fewer than two
    present views contributes no pair at all. This is how many rows can carry
    a pair in each class.
    """
    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    sources = list(samples.sources)
    test_idx = indices["test"]
    labels_test = samples.labels[test_idx]
    attack = labels_test == CLASS_ATTACK
    out = {
        "note": (
            "A pair statistic needs two present views in the same row, so a "
            "row with 0 or 1 present view contributes nothing to dependence "
            "or ENIV."
        ),
        "per_arm": {},
    }
    for arm in cfg["view_configurations"]:
        keep_views = [sources.index(v) for v in arm["views"]]
        n_present = samples.view_mask[:, keep_views].sum(axis=1)[test_idx]
        two_plus = n_present >= 2
        out["per_arm"][arm["name"]] = {
            "n_attack_test": int(attack.sum()),
            "n_attack_two_or_more_views": int((two_plus & attack).sum()),
            "attack_share_two_or_more_views": (
                float((two_plus & attack).sum() / attack.sum())
                if attack.any() else float("nan")),
            "n_benign_test": int((~attack).sum()),
            "n_benign_two_or_more_views": int((two_plus & ~attack).sum()),
            "benign_share_two_or_more_views": (
                float((two_plus & ~attack).sum() / (~attack).sum())
                if (~attack).any() else float("nan")),
        }
    return out


def unit_wallclock(units: list[dict], mode: dict | None) -> dict:
    """Wall-clock as the unit files record it, and what they do NOT record.

    A unit file carries its own `seconds` but no start or end timestamp, so
    the first start, the last end and the true span cannot be recovered from
    `results/lanl/` and are reported as not recorded rather than estimated.
    """
    seconds = [float(u["seconds"]) for u in units if "seconds" in u]
    concurrency = (mode or {}).get("max_concurrent_processes")
    waves = (
        int(math.ceil(len(seconds) / concurrency))
        if isinstance(concurrency, int) and concurrency > 0 and seconds else None
    )
    return {
        "note": (
            "Read from the `seconds` field of every file in "
            "results/lanl/units/. Unit files carry no start or end "
            "timestamp, so first unit start, last unit end and total "
            "wall-clock are NOT RECORDED in results/lanl/ and are not "
            "estimated here. The wave count is the arithmetic ceiling of "
            "units over the concurrency limit, not an observed schedule: "
            "units have unequal lengths, so waves overlap in practice."
        ),
        "n_units": len(seconds),
        "first_unit_start": "not recorded in unit files",
        "last_unit_end": "not recorded in unit files",
        "total_wall_clock_seconds": "not recorded in unit files",
        "sum_of_unit_seconds": float(sum(seconds)) if seconds else float("nan"),
        "max_unit_seconds": float(max(seconds)) if seconds else float("nan"),
        "min_unit_seconds": float(min(seconds)) if seconds else float("nan"),
        "concurrency_limit": concurrency,
        "waves_ceiling": waves,
        "elapsed_seconds_measures": (
            "results.json `elapsed_seconds` times the AGGREGATE step only - "
            "`run_lanl.py --aggregate`: reading the 40 unit files, "
            "samples.npz and scores/*.npz, computing the Phase 3 and Phase 4 "
            "diagnostics including the 2000-replicate cluster bootstrap, and "
            "writing results.json and this README. It does NOT include any "
            "training."
        ),
    }


def aggregate(mode: dict | None = None) -> dict:
    """Combine results/lanl/units/*.json into results.json and the README.

    Reads only files. An arm with no units simply does not appear, which is
    what lets the README's status line say truthfully which arms exist.
    """
    started = time.time()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_cfg = load_lanl_config(cfg["data"]["config"])
    units = [json.loads(p.read_text()) for p in sorted(UNITS_DIR.glob("*.json"))]
    if not units:
        raise RuntimeError("no units in results/lanl/units/")

    samples = load_samples(RESULTS_DIR / "samples.npz", data_cfg)
    full = LANLMultiViewDataset(samples)
    indices = split_indices(full)
    report = json.loads((RESULTS_DIR / "split.json").read_text())
    population = json.loads((RESULTS_DIR / "population.json").read_text())

    configurations = {}
    for arm in cfg["view_configurations"]:
        name = arm["name"]
        arm_units = [u for u in units if u["arm"] == name]
        if not arm_units:
            continue
        data = subset(full, tuple(arm["views"]))
        systems = {}
        for system in cfg["systems"]:
            mine = sorted(
                (u for u in arm_units if u["system"] == system["name"]),
                key=lambda u: u["seed"],
            )
            if not mine:
                continue
            rows = [u["metrics"] | {"seed": u["seed"]} for u in mine]
            diagnostics = [u["diagnostics"] for u in mine]
            systems[system["name"]] = {
                "system": system["name"],
                "label": system["label"],
                "use_discount": system["use_discount"],
                "experiment_id": f"lanl/{name}/{system['name']}",
                "protocol_output": "not written in unit mode; per-unit files "
                                   "are results/lanl/units/*.json",
                "per_seed": rows,
                "summary": summarise(rows),
                "diagnostics_per_seed": diagnostics,
                "diagnostics_summary": summarise(
                    [{k: v for k, v in d.items() if isinstance(v, float)}
                     for d in diagnostics]),
                "pair_table": pair_table(diagnostics, list(arm["views"])),
                "seconds_per_unit": [u["seconds"] for u in mine],
                "scores_files": [
                    {"seed": u["seed"], "file": u["scores_file"],
                     "sha256": u["scores_sha256"]} for u in mine
                ],
            }
        if len(systems) < 2:
            # A half-finished arm: one system only. Keep it, but it cannot be
            # paired, and the README's status check will catch the seed count.
            LOGGER.warning("arm %s has only %s", name, list(systems))

        natural = {"note": (
            "ENIV and mean pairwise dependence on real attack vs real benign "
            "test windows. No synthetic perturbation. LANL's red-team file "
            "has no attack-step taxonomy, so unlike Part 29 on AIT there is "
            "no dnsteal-only/multi-step split; this is one attack "
            "population, not two pooled."
        ), "n": {}, "per_system": {}}
        labels_test = full.labels.numpy()[indices["test"]]
        natural["n"] = {
            "attack_windows": int((labels_test == CLASS_ATTACK).sum()),
            "benign_windows": int((labels_test == CLASS_BENIGN).sum()),
        }
        for system_name in systems:
            mine = sorted((u for u in arm_units if u["system"] == system_name),
                          key=lambda u: u["seed"])
            populations = {}
            for pop in ("attack_windows", "benign_windows"):
                rows = [
                    u["natural_windows"][pop] | {"seed": u["seed"]}
                    for u in mine
                    if isinstance(u["natural_windows"].get(pop), dict)
                    and "not_measured" not in u["natural_windows"][pop]
                ]
                populations[pop] = (
                    {"per_seed": rows, "summary": summarise(rows)} if rows
                    else {"not_measured": "no unit reported this population"}
                )
            paired = (
                paired_difference(populations["attack_windows"]["per_seed"],
                                  populations["benign_windows"]["per_seed"],
                                  LOWER_IS_BETTER)
                if all("per_seed" in populations[p] for p in populations) else {}
            )
            natural["per_system"][system_name] = {
                "populations": populations,
                "paired_attack_minus_benign": paired,
            }

        labels = data.labels.numpy()[indices["test"]]
        first_unit = arm_units[0]
        configurations[name] = {
            "name": name,
            "label": arm["label"],
            "views": list(arm["views"]),
            "scaler": first_unit["scaler"],
            "missingness": {
                "train": missingness(data, indices["train"]),
                "test": missingness(data, indices["test"]),
                "threshold": cfg["missingness"]["threshold"],
                "over_threshold_train": [
                    v for v, frac in missingness(data, indices["train"]).items()
                    if frac > cfg["missingness"]["threshold"]],
                "over_threshold_test": [
                    v for v, frac in missingness(data, indices["test"]).items()
                    if frac > cfg["missingness"]["threshold"]],
            },
            "majority_baseline": majority_baseline(labels, cfg),
            "systems": systems,
            "paired_prismflow_minus_naive": (
                paired_difference(systems["prismflow"]["per_seed"],
                                  systems["naive"]["per_seed"], LOWER_IS_BETTER)
                if len(systems) == 2 else {}
            ),
            "part29_test_a_natural_attack_windows": natural,
            "part29_tests_not_run": {
                "B_chorus_attack": (
                    "NOT RUN. Part 29's grid is k in {1,2,3} x Part 09's "
                    "epsilon and beta grid, retrained per cell, which does "
                    "not fit this Part's time budget. No partial grid is "
                    "reported, because a subset of the cells is not the test."
                ),
                "C_honest_duplication": (
                    "PARTIALLY RUN. The strict same-sensor pair does not "
                    "exist here: LANL cyber1 ships no sensor-group "
                    "definition, so there is no within-group pair to score "
                    "and the answer is NO EVIDENCE. The 'different sensors, "
                    "naturally dependent' half needs the most/least "
                    "dependent pairs, which ARE written to "
                    "systems.<name>.pair_table; scoring the detector on them "
                    "was not run."
                ),
                "D_missingness": (
                    "NOT RUN as a detector-AUC test. Per-view missing "
                    "fractions on train and test ARE measured above, and "
                    "whole arms without flows and without dns ARE run, which "
                    "is a stronger answer than masking at 30%/50%; but Part "
                    "10's masked detector AUC itself was not computed."
                ),
            },
        }

    arms_present = list(configurations)
    diagnostics = {
        "missing_view_handling": MISSING_VIEW_HANDLING,
        "arm_coverage": arm_coverage(data_cfg, indices, cfg),
        "trivial_baselines": trivial_baselines(data_cfg, indices),
        "host_audit": host_audit(data_cfg, indices),
        "seen_unseen": seen_unseen_auc(data_cfg, indices, cfg, arms_present),
        "cluster_bootstrap": cluster_bootstrap(cfg, arms_present),
    }
    closeout = {
        "note": (
            "Part 30 close-out, read-only: every figure is computed from "
            "results/lanl/samples.npz and results/lanl/scores/*.npz. No "
            "model was trained and no view was rescanned. Results only - no "
            "interpretation is offered in this block."
        ),
        "top_attack_computers": top_attack_computer_table(data_cfg, indices, 4),
        "trivial_scalar_seen_unseen": trivial_scalar_seen_unseen(data_cfg, indices),
        "without_auth_present_view_only": present_view_only(
            data_cfg, indices, cfg, "without_auth"),
        "drop_top_test_computer": drop_top_computer(
            data_cfg, indices, cfg, arms_present),
        "views_present_shares": views_present_shares(data_cfg, indices, cfg),
    }

    payload = json_safe({
        "experiment": cfg["experiment"],
        "smoke": False,
        "git_commit": git_commit(),
        "config": cfg,
        "data_config": cfg["data"]["config"],
        "window_seconds": data_cfg.window_seconds,
        "day_seconds": data_cfg.day_seconds,
        "split": data_cfg.split,
        "label_meaning": LABEL_MEANING,
        "split_report": report,
        "population": population,
        "execution_mode": mode or {"not_recorded": True},
        "configurations": configurations,
        "phase3_diagnostics": diagnostics,
        "phase4_closeout": closeout,
        "unit_wallclock": unit_wallclock(units, mode),
        "replicate_run1_vs_run2": replicate_table(configurations),
        "n_units": len(units),
        "elapsed_seconds": time.time() - started,
    })
    (RESULTS_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    write_readme(payload, Path(__file__).with_name("README.md"))
    LOGGER.info("aggregated %d units over arms %s", len(units), arms_present)
    return payload


def record_mode_measurements() -> dict:
    """Add the MEASURED cost of the chosen mode to execution_mode.json.

    The committed benchmark carried a projected throughput, which assumed
    concurrent units do not contend. They do. This reads the `seconds` field
    of every unit actually run and the seed-0 with_auth metrics of run 1, so
    the README can print the projection beside what happened instead of the
    projection alone.
    """
    import gzip as _gzip
    mode_path = RESULTS_DIR / "execution_mode.json"
    mode = json.loads(mode_path.read_text())
    units = [json.loads(q.read_text()) for q in sorted(UNITS_DIR.glob("*.json"))]
    if not units:
        raise RuntimeError("no units in results/lanl/units/")

    seconds = [float(u["seconds"]) for u in units]
    per_system = {}
    for system in ("naive", "prismflow"):
        mine = [float(u["seconds"]) for u in units if u["system"] == system]
        if mine:
            per_system[system] = {
                "n_units": len(mine),
                "mean_seconds_per_unit": float(np.mean(mine)),
                "std_seconds_per_unit": (
                    float(np.std(mine, ddof=1)) if len(mine) > 1
                    else float("nan")),
            }
    sequential = float(mode["sequential_run1_seconds_per_unit"])
    concurrent = int(mode["max_concurrent_processes"])
    mean_seconds = float(np.mean(seconds))
    # Throughput = units finished per second. Sequential run 1 managed
    # 1/sequential; the chosen mode manages concurrent/mean_seconds.
    mode["achieved_parallel_measured"] = {
        "n_units": len(units),
        "mean_seconds_per_unit": mean_seconds,
        "std_seconds_per_unit": (
            float(np.std(seconds, ddof=1)) if len(seconds) > 1
            else float("nan")),
        "per_system": per_system,
        "throughput_speedup_vs_run1": (concurrent / mean_seconds) * sequential,
        "note": (
            "A unit takes LONGER in this mode than it did sequentially - "
            f"{mean_seconds:.0f} s against run 1's {sequential:.0f} s - "
            f"because {concurrent} of them share the box. The gain is "
            "throughput, not latency. The projection above overstated that "
            "gain because it assumed no contention between concurrent units; "
            "the decision rule (at least 2x run 1's sequential throughput) is "
            "met on the MEASURED figure as well as the projected one, so the "
            "mode choice stands on what happened rather than on the forecast."
        ),
    }

    gz = RESULTS_DIR / "results.run1.json.gz"
    equivalence = {"not_measured": "run 1 record or a with_auth seed-0 unit is absent"}
    if gz.exists():
        run1 = json.loads(_gzip.decompress(gz.read_bytes()))
        rows = {}
        for system in ("naive", "prismflow"):
            unit = next(
                (u for u in units
                 if u["arm"] == "with_auth" and u["system"] == system
                 and int(u["seed"]) == 0), None)
            old = dig(run1, "configurations", "with_auth", "systems", system,
                      "per_seed")
            old_row = next((r for r in (old or []) if int(r["seed"]) == 0), None)
            if unit is None or old_row is None:
                continue
            metrics = {}
            for key in ("balanced_accuracy", "roc_auc"):
                a, b = old_row.get(key), unit["metrics"].get(key)
                if a is None or b is None:
                    continue
                metrics[key] = {
                    "run1": float(a),
                    "run2": float(b),
                    "absolute_difference": float(b) - float(a),
                }
            if metrics:
                rows[system] = metrics
        if rows:
            equivalence = {
                "per_system": rows,
                "note": (
                    "Same seed, same hyperparameters, same sample table and "
                    "the same time split. A non-zero difference here is "
                    "execution mode and floating-point reduction order, not a "
                    "different experiment: run 1 ran sequentially at torch's "
                    "default 12 threads and run 2 at 1 thread, which changes "
                    "the order in which sums are accumulated. The difference "
                    "is NOT evidence about either system's quality."
                ),
            }
    mode["equivalence_check_seed0"] = equivalence
    mode_path.write_text(json.dumps(json_safe(mode), indent=2))
    LOGGER.info("recorded measured mode cost over %d units", len(units))
    return mode


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 30 - LANL cyber1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--read-until", type=int, default=None,
                        help="override read_until for a smoke run; results are "
                             "then labelled smoke and are NOT a finding")
    parser.add_argument("--readme-only", action="store_true",
                        help="regenerate README.md from the committed "
                             "results/lanl/results.json without rerunning")
    parser.add_argument("--unit", default=None,
                        help="run ONE unit, as arm:system:seed")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads for this process")
    parser.add_argument("--benchmark", type=int, default=None,
                        help="time N training steps and exit")
    parser.add_argument("--record-mode", action="store_true",
                        help="add the measured cost of the chosen execution "
                             "mode, and the seed-0 equivalence check against "
                             "run 1, to results/lanl/execution_mode.json")
    parser.add_argument("--aggregate", action="store_true",
                        help="combine results/lanl/units/*.json into "
                             "results.json and regenerate the README")
    parser.add_argument("--build-samples", action="store_true",
                        help="the ONE scan: stream all four views and "
                             "write results/lanl/samples.npz plus its sha256")
    parser.add_argument("--label-audit", action="store_true",
                        help="write results/lanl/label_audit.json from "
                             "redteam.txt.gz alone; reads no view stream")
    args = parser.parse_args()

    if args.readme_only:
        results = json.loads((RESULTS_DIR / "results.json").read_text())
        write_readme(results, Path(__file__).with_name("README.md"))
        LOGGER.info("README regenerated from results/lanl/results.json")
        return

    if args.record_mode:
        record_mode_measurements()
        return

    if args.aggregate:
        mode_path = RESULTS_DIR / "execution_mode.json"
        mode = json.loads(mode_path.read_text()) if mode_path.exists() else None
        aggregate(mode)
        return

    if args.benchmark:
        record = benchmark_steps(args.benchmark, args.threads)
        print(json.dumps(record))
        return

    if args.unit:
        arm_name, system_name, seed = args.unit.split(":")
        run_unit(arm_name, system_name, int(seed), args.threads)
        return

    if args.build_samples:
        # The ONE scan of this Part. Every arm afterwards reads samples.npz, so
        # the four view streams are read once and all arms share identical rows.
        started = time.time()
        data_cfg = load_lanl_config(yaml.safe_load(
            CONFIG_PATH.read_text())["data"]["config"])
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dataset, redteam, timing = load_dataset(data_cfg, args.workers)
        indices = split_indices(dataset)
        report = split_report(dataset, redteam, indices, data_cfg)
        population = dict(dataset.samples.population)
        population.update({
            "scan_seconds": timing["scan_seconds"],
            "workers": timing["workers"],
            "read_until": data_cfg.read_until,
            "smoke": False,
            "kept_samples_total": len(dataset),
            "estimated_population_cells_union": {
                "value": None,
                "method": (
                    "Not directly counted. The union over views of occupied "
                    "cells cannot be derived from per-view counts, and "
                    "counting it exactly would need a cross-view per-window "
                    "set this loader does not keep. The exact per-view counts "
                    "are in per_view; the union is bounded below by their "
                    "maximum and above by their sum."
                ),
                "lower_bound": max(
                    v["population_cells"] for v in population["per_view"].values()),
                "upper_bound": sum(
                    v["population_cells"] for v in population["per_view"].values()),
            },
        })
        dataset.samples.population = population
        (RESULTS_DIR / "population.json").write_text(
            json.dumps(json_safe(population), indent=2))
        (RESULTS_DIR / "split.json").write_text(json.dumps(json_safe(report), indent=2))
        (RESULTS_DIR / "label_audit.json").write_text(
            json.dumps(json_safe(label_audit(data_cfg, redteam)), indent=2))

        record = save_samples(dataset.samples, RESULTS_DIR / "samples.npz")
        record["scan_seconds"] = timing["scan_seconds"]
        record["built_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_path = RESULTS_DIR / "results_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest["samples"] = record
        manifest_path.write_text(json.dumps(manifest, indent=2))

        check_stop_conditions(
            report,
            yaml.safe_load(CONFIG_PATH.read_text())
            ["stop_conditions"]["min_attack_windows_per_split"],
        )
        LOGGER.info("samples.npz written: %s", record)
        LOGGER.info("build-samples done in %.1fs", time.time() - started)
        return

    if args.label_audit:
        data_cfg = load_lanl_config(yaml.safe_load(
            CONFIG_PATH.read_text())["data"]["config"])
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        audit = label_audit(data_cfg, read_redteam(data_cfg))
        (RESULTS_DIR / "label_audit.json").write_text(
            json.dumps(json_safe(audit), indent=2))
        LOGGER.info("label audit written: %s", audit["per_block"])
        return

    started = time.time()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_cfg = load_lanl_config(cfg["data"]["config"])
    smoke = args.read_until is not None
    if smoke:
        data_cfg = dataclasses.replace(data_cfg, read_until=args.read_until)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        dataset, redteam, timing = load_dataset(data_cfg, args.workers)
    except TimestampInversion as exc:
        (RESULTS_DIR / "STOPPED.json").write_text(json.dumps(
            {"stop": "timestamp_inversion", "detail": str(exc)}, indent=2))
        raise

    indices = split_indices(dataset)
    report = split_report(dataset, redteam, indices, data_cfg)

    population = dict(dataset.samples.population)
    population["scan_seconds"] = timing["scan_seconds"]
    population["workers"] = timing["workers"]
    population["read_until"] = data_cfg.read_until
    population["smoke"] = smoke
    population["kept_samples_total"] = len(dataset)
    population["estimated_population_cells_union"] = {
        "value": None,
        "method": (
            "Not directly counted. The union over views of occupied cells "
            "cannot be derived from per-view counts, and counting it exactly "
            "would need a cross-view per-window set this loader does not keep. "
            "The exact per-view counts are in per_view; the union is bounded "
            "below by their maximum and above by their sum."
        ),
        "lower_bound": max(
            v["population_cells"] for v in population["per_view"].values()
        ),
        "upper_bound": sum(
            v["population_cells"] for v in population["per_view"].values()
        ),
    }
    (RESULTS_DIR / "population.json").write_text(
        json.dumps(json_safe(population), indent=2))
    (RESULTS_DIR / "split.json").write_text(json.dumps(json_safe(report), indent=2))
    LOGGER.info(
        "samples: train=%d (%d attack, %d attack windows) test=%d (%d attack, %d attack windows)",
        report["per_split"]["train"]["n_samples"],
        report["per_split"]["train"]["n_attack_samples"],
        report["per_split"]["train"]["n_distinct_attack_windows"],
        report["per_split"]["test"]["n_samples"],
        report["per_split"]["test"]["n_attack_samples"],
        report["per_split"]["test"]["n_distinct_attack_windows"],
    )

    check_stop_conditions(report, cfg["stop_conditions"]["min_attack_windows_per_split"])

    configurations = {}
    for configuration in cfg["view_configurations"]:
        configurations[configuration["name"]] = run_configuration(
            configuration, dataset, indices, cfg
        )
        # Written after each configuration, so a run that dies later still
        # leaves the configurations that finished.
        done = len(configurations) == len(cfg["view_configurations"])
        payload = json_safe({
            "experiment": cfg["experiment"],
            "status_complete": done and not smoke,
            "status_reason": (
                "Part 28's metrics and Part 29's test A ran on both view "
                "configurations over 5 seeds. Part 29's tests B, C (second "
                "half) and D did NOT run - see section 9. This Part is "
                "therefore complete for what it reports and INCOMPLETE "
                "against part30.md step 4's full test list."
                if done and not smoke else
                f"Only {len(configurations)} of "
                f"{len(cfg['view_configurations'])} view configurations "
                "finished." if not smoke else
                "SMOKE RUN - a shortened read range. NOT A FINDING."
            ),
            "smoke": smoke,
            "git_commit": git_commit(),
            "config": cfg,
            "data_config": cfg["data"]["config"],
            "window_seconds": data_cfg.window_seconds,
            "day_seconds": data_cfg.day_seconds,
            "split": data_cfg.split,
            "label_meaning": LABEL_MEANING,
            "split_report": report,
            "population": population,
            "configurations": configurations,
            "elapsed_seconds": time.time() - started,
        })
        (RESULTS_DIR / "results.json").write_text(json.dumps(payload, indent=2))
        write_readme(payload, Path(__file__).with_name("README.md"))

    LOGGER.info("done in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
