"""Frozen-naive control: is naive's robustness to duplicates learned during training?

In run_calibration_duplicated.py, the naive system at k copies was TRAINED on
data with those copies. This script adds a fourth condition that never sees a
duplicate during training:

  naive_frozen  the naive model trained at k = 0 (4 views), weights frozen, then
                evaluated with k copies of view 0 appended at inference only.

A model's encoder count is fixed at construction, so a 4-view model cannot take
4 + k views as-is. The frozen model is widened instead: a (4 + k)-view naive
model whose slots 0..3 hold the trained k = 0 weights unchanged, and whose k
extra slots hold exact copies of view 0's trained encoder and evidence head.
A duplicated input then yields evidence bit-identical to view 0's (checked on
every evaluation batch), so Dempster fusion counts view 0's evidence 1 + k
times with no retraining. Nothing is optimised after k = 0.

Only naive_frozen is trained and scored here, via evaluate(). The comparison
systems (naive, prismflow, naive_weights_discounted) are read from the per-seed
rows evaluate() already wrote to results/calibration_duplicated/. The k = 0
naive models are retrained from the same seeds, so naive_frozen at k = 0 must
reproduce k0_naive exactly; the script checks this and refuses to compare
otherwise.

Base data, seeds and training settings come from config_duplicated.yaml.

Usage:
    python -m experiments.calibration.run_calibration_frozen_naive
    python -m experiments.calibration.run_calibration_frozen_naive --quick --results-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import yaml

from experiments.calibration.run_calibration import build_dataset
from experiments.calibration.run_calibration_duplicated import train
from prismflow.data.corruption import duplicate_view
from prismflow.data.loaders import iter_batches
from prismflow.evaluation import evaluate
from prismflow.train import TrainConfig, build_model
from prismflow.utils.logging import get_logger

CONFIG_PATH = Path(__file__).with_name("config_duplicated.yaml")
DUPLICATED_ID = "calibration_duplicated"

SYSTEM = "naive_frozen"
COMPARED = ("naive", "prismflow", "naive_weights_discounted")
TABLE_SYSTEMS = (SYSTEM, *COMPARED)

METRICS = (
    "accuracy",
    "prob_mean_confidence",
    "prob_ece",
    "brier",
    "brier_reliability",
    "brier_resolution",
    "brier_uncertainty",
    "brier_within_bin",
    "prob_aurc",
    "vacuity_mean_confidence",
    "vacuity_ece",
)

CHANGE_METRICS = (
    "accuracy",
    "prob_mean_confidence",
    "prob_ece",
    "brier",
    "brier_reliability",
    "brier_resolution",
    "prob_aurc",
)

# Metrics the k = 0 reproduction check compares; any drift voids the comparison.
REPRODUCTION_TOLERANCE = 1e-9


def widen_frozen(model, k: int, source_view: int, data_cfg: dict):
    """A (base_views + k)-view copy of `model`; extra slots clone the source view's modules."""
    base_views = data_cfg["n_views"]
    wide = build_model(TrainConfig(n_views=base_views + k, rho=data_cfg["rho"], use_discount=False))

    state = model.state_dict()
    wide_state = {}
    for key in wide.state_dict():
        new_key = key
        for prefix in ("encoder.encoders.", "evidence_head.heads."):
            if key.startswith(prefix):
                index, rest = key[len(prefix):].split(".", 1)
                if int(index) >= base_views:
                    new_key = f"{prefix}{source_view}.{rest}"
        wide_state[key] = state[new_key].clone()
    wide.load_state_dict(wide_state, strict=True)

    for parameter in wide.parameters():
        parameter.requires_grad_(False)
    wide.eval()
    return wide


class _CheckedModel(torch.nn.Module):
    """Wraps the widened model and asserts copy slots reproduce the source view's evidence."""

    def __init__(self, model, base_views: int, source_view: int):
        super().__init__()
        self.model = model
        self.base_views = base_views
        self.source_view = source_view

    def forward(self, views, view_mask):
        output = self.model(views, view_mask)
        evidence = output.per_view_evidence
        source = evidence[:, self.source_view, :]
        for slot in range(self.base_views, evidence.shape[1]):
            if not torch.equal(evidence[:, slot, :], source):
                raise AssertionError(f"copy slot {slot} evidence differs from view {self.source_view}")
        return output


def _load_rows(results_dir: Path, k: int, system: str) -> dict[int, dict]:
    path = results_dir / DUPLICATED_ID / f"k{k}_{system}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run run_calibration_duplicated.py first")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["seed"]: row for row in report["per_seed"]}


def _stats(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def _fmt(mean, std, signed=False):
    if math.isnan(mean):
        return "n/a"
    head = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    return head if math.isnan(std) else f"{head} +/- {std:.4f}"


def render_markdown(rows, config, experiment_id):
    """rows[k][system][seed] -> per-seed metric dict written by evaluate()."""
    ks, seeds = config["duplicates"], config["seeds"]
    lines = [
        "# Frozen-naive control",
        "",
        f"Seeds: {seeds}. Base: {config['data']['n_views']} views, rho = {config['data']['rho']}; "
        f"k copies of view {config['source_view']}. `naive_frozen` from "
        f"`results/{experiment_id}/k<k>_naive_frozen/`; the other systems from "
        f"`results/{DUPLICATED_ID}/k<k>_<system>/`. Mean +/- sample std across seeds.",
        "",
    ]

    for k in ks:
        lines += [
            f"## k = {k}",
            "",
            "| metric | " + " | ".join(TABLE_SYSTEMS) + " |",
            "|---|" + "---|" * len(TABLE_SYSTEMS),
        ]
        for metric in METRICS:
            cells = [_fmt(*_stats([rows[k][s][seed][metric] for seed in seeds])) for s in TABLE_SYSTEMS]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Paired within-seed differences:",
            "",
            "| metric | naive_frozen - naive | seeds lower | naive_frozen - prismflow | seeds lower |",
            "|---|---|---|---|---|",
        ]
        for metric in CHANGE_METRICS:
            cells = []
            for other in ("naive", "prismflow"):
                diffs = [rows[k][SYSTEM][s][metric] - rows[k][other][s][metric] for s in seeds]
                cells += [_fmt(*_stats(diffs), signed=True), f"{sum(d < 0 for d in diffs)}/{len(diffs)}"]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        f"## Change from k = {ks[0]}, within seed",
        "",
        "At k = 0 all naive-family systems start from the same trained model per seed.",
        "",
        "| metric | system | " + " | ".join(f"k={k} - k={ks[0]}" for k in ks[1:]) + " |",
        "|---|---|" + "---|" * (len(ks) - 1),
    ]
    for metric in CHANGE_METRICS:
        for system in TABLE_SYSTEMS:
            cells = []
            for k in ks[1:]:
                diffs = [rows[k][system][s][metric] - rows[ks[0]][system][s][metric] for s in seeds]
                cells.append(f"{_fmt(*_stats(diffs), signed=True)} ({sum(d > 0 for d in diffs)}/{len(diffs)} up)")
            lines.append(f"| {metric} | {system} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Frozen vs trained-on-duplicates: difference in change from k = 0, within seed",
        "",
        "(naive_frozen change) - (naive change). Negative on accuracy/resolution means the "
        "frozen model degrades more; positive on ECE/reliability/Brier/AURC means it worsens more.",
        "",
        "| metric | " + " | ".join(f"k={k}" for k in ks[1:]) + " |",
        "|---|" + "---|" * (len(ks) - 1),
    ]
    for metric in CHANGE_METRICS:
        cells = []
        for k in ks[1:]:
            diffs = [
                (rows[k][SYSTEM][s][metric] - rows[ks[0]][SYSTEM][s][metric])
                - (rows[k]["naive"][s][metric] - rows[ks[0]]["naive"][s][metric])
                for s in seeds
            ]
            cells.append(f"{_fmt(*_stats(diffs), signed=True)} ({sum(d < 0 for d in diffs)}/{len(diffs)} lower)")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the frozen-naive duplication control.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default="calibration_frozen_naive")
    parser.add_argument(
        "--comparison-dir",
        default=None,
        help="results dir holding calibration_duplicated/; defaults to --results-dir",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test only: 2 epochs, skips the reproduction check -- NOT evidence, never report",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = args.experiment_id
    if args.quick:
        config["training"]["epochs"] = 2
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.calibration_frozen_naive")
    seeds = config["seeds"]
    ks = config["duplicates"]
    if ks[0] != 0:
        raise ValueError("duplicates must start at k = 0, where the frozen model is trained")
    eval_cfg = config["evaluation"]
    source_view = config["source_view"]
    base_views = config["data"]["n_views"]
    comparison_dir = Path(args.comparison_dir or args.results_dir)

    datasets, frozen = {}, {}
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits)
        frozen[seed] = train(dataset, splits, 0, source_view, seed, False, config["data"], config["training"])
        logger.info("seed %d: k=0 naive trained (%.0fs)", seed, time.perf_counter() - started)

    rows = {k: {} for k in ks}
    for k in ks:
        models = {
            seed: _CheckedModel(widen_frozen(frozen[seed], k, source_view, config["data"]), base_views, source_view)
            for seed in seeds
        }

        def loader(seed, k=k):
            dataset, splits = datasets[seed]
            for batch in iter_batches(dataset, splits[eval_cfg["split"]], eval_cfg["batch_size"]):
                yield duplicate_view(batch, source_view, k)

        report = evaluate(
            models,
            loader,
            f"{experiment_id}/k{k}_{SYSTEM}",
            seeds,
            results_dir=args.results_dir,
            n_bins=eval_cfg["n_bins"],
            coverages=eval_cfg["coverages"],
            title=f"{experiment_id}: k={k}, {SYSTEM}",
        )
        rows[k][SYSTEM] = {row["seed"]: row for row in report["per_seed"]}
        summary = report["summary"]
        logger.info(
            "k=%d naive_frozen acc=%.4f ece=%.4f reliability=%.4f resolution=%.4f aurc=%.4f",
            k, summary["accuracy"]["mean"], summary["prob_ece"]["mean"],
            summary["brier_reliability"]["mean"], summary["brier_resolution"]["mean"],
            summary["prob_aurc"]["mean"],
        )

    if args.quick:
        logger.info("quick mode: comparison and summary skipped")
        return

    for k in ks:
        for system in COMPARED:
            rows[k][system] = _load_rows(comparison_dir, k, system)

    # The retrained k = 0 naive must be the model the duplicated run trained.
    worst = max(
        abs(rows[0][SYSTEM][s][m] - rows[0]["naive"][s][m]) for s in seeds for m in CHANGE_METRICS
    )
    logger.info("k=0 reproduction check: max |naive_frozen - k0_naive| = %.3g", worst)
    if worst > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"naive_frozen at k=0 does not reproduce {DUPLICATED_ID}/k0_naive (max diff {worst:.3g}); "
            "the comparison would not be like-for-like"
        )

    out_dir = Path(args.results_dir) / experiment_id
    markdown = render_markdown(rows, config, experiment_id)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (out_dir / "config_used.json").write_text(
        json.dumps({**config, "reproduction_max_abs_diff": worst}, indent=2), encoding="utf-8"
    )
    print()
    print(markdown)


if __name__ == "__main__":
    main()
