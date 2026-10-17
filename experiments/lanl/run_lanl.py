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
    return {
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
    }


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

    systems = {}
    for system in cfg["systems"]:
        systems[system["name"]] = run_system(
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
    }


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
    args = parser.parse_args()

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
        (RESULTS_DIR / "results.json").write_text(json.dumps(json_safe({
            "experiment": cfg["experiment"],
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
        }), indent=2))

    LOGGER.info("done in %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
