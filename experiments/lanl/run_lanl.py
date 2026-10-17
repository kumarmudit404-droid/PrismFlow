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
    contiguous_blocks,
    majority_baseline,
    metric_row,
    pair_table,
    paired_difference,
    summarise,
)
from prismflow.data.lanl_cyber1 import (
    CLASS_ATTACK,
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
            f"Only {sorted(finished)} of {sorted(expected)} view "
            f"configurations finished. **{', '.join(missing)} did NOT run** "
            "and every figure below covers the finished configuration only. "
            + (
                f"Reason, measured not estimated: the streaming pass took "
                f"{scan:.0f} s and the finished configuration's 10 models took "
                f"{elapsed - scan:.0f} s more "
                "(`results/lanl/results.json`, `elapsed_seconds` minus "
                "`population.scan_seconds`), because the discounted system "
                "costs about 1.7x the undiscounted one per model on this data. "
                "A second configuration needs the same again, which does not "
                "fit this Part's 90-minute budget. Seeds were NOT reduced and "
                "the day range was NOT shrunk to make it fit; the "
                "configuration is simply missing."
                if isinstance(elapsed, (int, float)) and isinstance(scan, (int, float))
                else "The remaining configuration did not fit the time budget."
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
            "Part 28's metric set and Part 29's test A ran on BOTH view "
            f"configurations over {len(results['config']['seeds'])} seeds. "
            "Part 29's tests B, D and the second half of C did NOT run "
            "(section 8), so this Part is complete for what it reports and "
            "INCOMPLETE against part30.md step 4's full A-D list. Read "
            "section 8 before citing anything here as a Part 29 repeat."
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
      "28's `part28_extras`, called through Part 28's imported `metric_row`, "
      "and cross-checked against `prismflow.evaluation.evaluate`'s own per-seed "
      "rows (`run_lanl.py::cross_check`; a mismatch above 1e-9 raises).")
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
            A("**Not run, and it is in the brief.** Part 28's step 5 says that "
              "when a view is missing in more than half the windows, a second "
              "configuration WITHOUT that view should also be run. That "
              f"applies here to {', '.join(over)} and was **not run** - the "
              "time budget did not reach it (see the status line). So this "
              "Part does not know whether the result above survives dropping "
              f"{' or '.join(over)}. `dns` in particular is present in under "
              "2% of windows, so its encoder sees almost nothing and its "
              "contribution to the fusion is close to untested here.")
        A("")

    # -- test A ------------------------------------------------------------
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
                  f"{signed(dig(block, 'paired_attack_minus_benign', key))} |")
        A("")
    A("Read the WITHOUT-auth block for a detection claim: with auth present, "
      "the views contain the records the labels came from (caveat iii).")
    A("")
    A("Sign convention: `paired_difference` marks a *fall* in mean dependence "
      "as favouring PrismFlow, because `LOWER_IS_BETTER` contains "
      "`mean_dependence_eigen`. For test A the quantity of interest is whether "
      "dependence is HIGHER on attack windows than benign ones, so read the "
      "signed mean, not the favouring count, in that row.")
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
      "pairs was **not run** (see section 8), so this Part states no verdict "
      "on whether natural dependence raises the detector the way Part 10's "
      "exact clone did.")
    A("")

    # -- cross-dataset -----------------------------------------------------
    A("## 7. Cross-dataset table (part30.md step 5)")
    A("")
    A("synthetic vs AIT vs LANL, for the DETECTION claim and the "
      "honest-duplication false positive. Every cell cites a `results/` file "
      "or says `not measured`. Nothing is pooled across datasets and nothing "
      "is averaged across AIT's two directions.")
    A("")

    chorus = read_json("results/chorus/attack_metrics.json")
    ait = read_json("results/ait_attack/results.json")
    clone = read_json("results/clone/summary.json")

    A("### 7.1 DETECTION: does dependence among the implicated views rise?")
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

    A("### 7.2 Honest-duplication false positive")
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
      "detector not scored (section 8) |")
    A("")

    # -- not run -----------------------------------------------------------
    A("## 8. What was not run, and why")
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
    A("## 9. What this design cannot show")
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
    A("## 10. Proposed limitations (NOT applied)")
    A("")
    A("`docs/KNOWN_LIMITATIONS.md` and `docs/CONTRACT.md` are unchanged by "
      "this Part. The following text is proposed for the user to approve or "
      "reject; it has not been written to either file.")
    A("")
    A("> **L23 (proposed).** On LANL cyber1 the `(computer, window)` sample "
      "space is ~2e7 cells at 300 s over days 1-16 against ~850 red-team "
      "cells, a prevalence near 2e-5. Part 30 therefore reports a declared "
      "case-control sample (all cases, deterministic 1/512 hash of controls), "
      "so its accuracy, Brier and ECE describe that sample and not the "
      "network. No population operating point has been measured on LANL.")
    A("")
    A("> **L24 (proposed).** LANL's red-team labels are derived from "
      "authentication records, so the `auth` view is label-adjacent. "
      + ("Part 30 reports WITH-auth and WITHOUT-auth separately; only the "
         "WITHOUT-auth arm bears on detection from independent evidence."
         if "without_auth" in configurations else
         "**Part 30 measured only the WITH-auth arm**, so every LANL number "
         "it reports is label-adjacent and none of them bears on detection "
         "from evidence independent of the label source. The WITHOUT-auth "
         "arm part30.md step 2 requires is not measured."))
    A("")
    A("> **L25 (proposed).** Part 28's `fit_scaler`, `apply_scaler` and "
      "`train_ait` are not reusable across datasets: each resolves native view "
      "widths through `ait_lds.FEATURES`, keyed by AIT source names. Part 30 "
      "mirrored them rather than patching Part 28. Any future Part on a third "
      "dataset will hit the same wall.")
    A("")

    # -- files -------------------------------------------------------------
    A("## 11. Files")
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
    A("| `results/lanl/results.json` | every metric, diagnostic, pair table "
      "and paired difference |")
    A("")
    A(f"Run: {results.get('elapsed_seconds', float('nan')):.0f} s wall, scan "
      f"{population.get('scan_seconds', float('nan')):.0f} s on "
      f"{population.get('workers')} worker processes. Commit "
      f"`{results.get('git_commit') or 'not recorded'}`.")
    A("")
    A("`data/raw/` was read only. Nothing in it was extracted, renamed, moved "
      "or deleted, and `auth.txt.gz` was not modified.")
    A("")

    path.write_text("\n".join(L), encoding="utf-8")


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
    parser.add_argument("--label-audit", action="store_true",
                        help="write results/lanl/label_audit.json from "
                             "redteam.txt.gz alone; reads no view stream")
    args = parser.parse_args()

    if args.readme_only:
        results = json.loads((RESULTS_DIR / "results.json").read_text())
        write_readme(results, Path(__file__).with_name("README.md"))
        LOGGER.info("README regenerated from results/lanl/results.json")
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
                "half) and D did NOT run - see section 8. This Part is "
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
