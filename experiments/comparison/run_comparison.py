"""Part 10: evaluating the suspicion detector as a detector.

SCOPE NOTE. The Part 10 brief specifies a 5-system x 8-condition comparison
matrix. That matrix was deliberately deferred after Part 09 found that the
discount does not defend (`experiments/chorus/`): a matrix built to show a
defence that does not exist answers the wrong question, and the detector -- the
half of the system Part 09 showed to be working -- deserves the Part. So this
script measures DETECTION, with the accuracy/ECE/confidence columns kept per
condition for context. An independent-PGD condition is added to the brief's
list, because Part 09 showed it is the attack the dependence signal cannot see.

WHAT IS MEASURED, PER CONDITION, PER SEED

  detection rate at a threshold calibrated to 5% and 1% false positives on that
  seed's own CLEAN scores; ROC AUC of that condition's scores against clean;
  mean suspicion score; plus accuracy, ECE, Brier reliability, mean confidence
  and ENIV from `prismflow.evaluation.evaluate`.

ISOLATION. This script knows which condition is which -- it must, to score a
detector at all. The DETECTOR never does: `suspicion_report` receives beliefs,
the view mask, the dependence matrix and the model's own pre-fusion strata, and
nothing else (`tests/unit/test_suspicion.py` asserts it against the signatures).
Evaluation-time knowledge of the label is not detector-time knowledge of it.

Outputs under results/comparison/:
    detector.json      per-seed and summarised detector metrics
    <condition>/       metrics.json, metrics.csv, reliability_diagram.png
    progress.json      written after every condition, so an interrupted run
                       keeps what it has (Part 09 wrote nothing until the end)

Usage:
    python -m experiments.comparison.run_comparison
    python -m experiments.comparison.run_comparison --quick   # NOT evidence
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.calibration.run_calibration import build_dataset, train
from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.data.corruption import add_noise, drop_views, duplicate_view
from prismflow.data.dataset import Batch
from prismflow.data.loaders import iter_batches
from prismflow.evaluation import evaluate
from prismflow.models.defended import DefendedPrismFlow
from prismflow.statistics.suspicion import DEFAULT_PERMUTATIONS, calibrate_threshold
from prismflow.utils.logging import get_logger

PROTOCOL_METRICS = ("accuracy", "prob_ece", "brier_reliability", "vacuity_mean_confidence", "eniv")
DETECTOR_METRICS = ("auc", "detect_at_5pct", "detect_at_1pct", "mean_score", "flag_rate_shared")

# The Part 10 brief's allowed-paths list has no config.yaml, so the
# configuration lives here rather than in a file beside it.
CONFIG = {
    "experiment": {"id": "comparison"},
    "seeds": [0, 1, 2, 3, 4],
    "data": {"n_views": 4, "rho": 0.3},
    "training": {
        "epochs": 40, "batch_size": 64, "lr": 0.001,
        "anneal_epochs": 10, "per_view_loss_weight": 1.0,
    },
    "attack": {"steps": 30, "target_strategy": "least_likely", "fixed_target": 0},
    "detector": {
        # The shared threshold is reported for reference; the headline numbers
        # use per-seed thresholds calibrated to a clean false-positive rate.
        "threshold": 0.05,
        "permutations": 16,
    },
    "evaluation": {
        "split": "test", "batch_size": 64, "n_bins": 15,
        "coverages": [1.0, 0.9, 0.8, 0.5],
    },
    # `clean` MUST come first: it calibrates every threshold.
    "conditions": [
        {"name": "clean",      "kind": "clean",   "k": 0},
        {"name": "clone_k2",   "kind": "clone",   "k": 2},
        {"name": "missing_30", "kind": "missing", "k": 0, "rate": 0.3},
        {"name": "missing_50", "kind": "missing", "k": 0, "rate": 0.5},
        {"name": "noisy_1",    "kind": "noisy",   "k": 0, "sigma": 2.0},
        {"name": "chorus_k1",  "kind": "chorus",  "k": 1, "epsilon": 1.0, "beta": 1.0},
        {"name": "chorus_k2",  "kind": "chorus",  "k": 2, "epsilon": 1.0, "beta": 1.0},
        {"name": "chorus_k3",  "kind": "chorus",  "k": 3, "epsilon": 1.0, "beta": 1.0},
        # Added to the brief's list: Part 09 showed this is the attack the
        # dependence signal cannot see, so it is the detector's real test.
        {"name": "pgd_k2",     "kind": "pgd",     "k": 2, "epsilon": 1.0},
    ],
}


def roc_auc(positive, negative) -> float:
    """Rank-based AUC: P(score of an attacked sample > score of a clean one).

    0.5 is chance. Ties count as half, which matters here because a saturated
    detector produces many equal scores.
    """
    positive = np.asarray([v for v in positive if np.isfinite(v)], dtype=np.float64)
    negative = np.asarray([v for v in negative if np.isfinite(v)], dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    # average ranks within ties
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    summed = np.zeros(counts.size)
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]
    rank_sum = ranks[: positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size))


def corrupt(batch: Batch, condition: dict, seed: int, batch_index: int) -> Batch:
    """Non-adversarial corruptions. Seeded on (seed, batch, condition name) so
    every system sees the same draws."""
    kind = condition["kind"]
    # zlib.crc32, not hash(): Python randomises string hashing per process, so
    # hash() here would make the corruption draws irreproducible across runs.
    corruption_seed = zlib.crc32(f"{seed}|{batch_index}|{condition['name']}".encode()) % (2**31)
    if kind in ("clean", "chorus", "pgd"):
        # Adversarial conditions start from clean input; `attack` perturbs it.
        return batch
    if kind == "clone":
        return duplicate_view(batch, source_idx=0, k=condition["k"])
    if kind == "missing":
        return drop_views(batch, rate=condition["rate"], seed=corruption_seed)
    if kind == "noisy":
        return add_noise(batch, view_idx=0, sigma=condition["sigma"], seed=corruption_seed)
    raise ValueError(f"unknown corruption kind {kind!r}")


def attack(model, batch: Batch, condition: dict, attack_cfg: dict, seed: int) -> Batch:
    """Adversarial conditions, white-box against the model being evaluated."""
    kind = condition["kind"]
    if kind not in ("chorus", "pgd"):
        return batch
    config = ChorusConfig(
        k=condition["k"],
        epsilon=condition["epsilon"],
        beta=condition.get("beta", 1.0),
        steps=attack_cfg["steps"],
        target_strategy=attack_cfg["target_strategy"],
        fixed_target=attack_cfg["fixed_target"],
        seed=seed,
    )
    run = chorus_attack if kind == "chorus" else pgd_attack
    delta = run(model, batch.views, batch.view_mask, config).delta
    return Batch(
        views=batch.views + delta,
        view_mask=batch.view_mask.clone(),
        labels=batch.labels.clone(),
        sample_ids=batch.sample_ids.clone(),
    )


def run_condition(defended, dataset, indices, condition, attack_cfg, batch_size, seed):
    """Returns (batches actually scored, per-sample suspicion scores)."""
    batches, scores = [], []
    for index, batch in enumerate(iter_batches(dataset, indices, batch_size)):
        prepared = corrupt(batch, condition, seed, index)
        prepared = attack(defended.model, prepared, condition, attack_cfg, seed)
        batches.append(prepared)
        with torch.no_grad():
            output = defended(prepared.views, prepared.view_mask)
        if output.suspicion is None:
            scores.append(np.full(len(prepared), np.nan))
        else:
            scores.append(np.asarray(output.suspicion.score, dtype=np.float64))
    return batches, np.concatenate(scores)


def summarise(rows, keys) -> dict:
    out = {}
    for key in keys:
        values = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[key] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "n_seeds": len(values),
        }
    return out


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(float(value))
    return value


def main():
    parser = argparse.ArgumentParser(description="Part 10: suspicion detector evaluation.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="smoke test only -- NOT evidence")
    args = parser.parse_args()

    config = json.loads(json.dumps(CONFIG))  # deep copy; --quick mutates it
    experiment_id = config["experiment"]["id"]
    if args.quick:
        # Seeds are NOT reduced: `evaluate` enforces the 5-seed rule and the
        # smoke test should exercise the same path the real run takes.
        config["training"]["epochs"] = 2
        config["attack"]["steps"] = 3
        config["detector"]["permutations"] = 4
        config["conditions"] = config["conditions"][:2] + config["conditions"][-2:]
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.comparison")
    seeds, eval_cfg = config["seeds"], config["evaluation"]
    out_dir = Path(args.results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # A clone condition appends views, so it needs its own model shape.
    widths = sorted({4 + c["k"] if c["kind"] == "clone" else config["data"]["n_views"]
                     for c in config["conditions"]})
    datasets, models = {}, {}
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits)
        models[seed] = {}
        for width in widths:
            data_cfg = dict(config["data"], n_views=width)
            if width == config["data"]["n_views"]:
                source, source_splits = dataset, splits
            else:
                # Same seed, wider model: the clone condition feeds it 4 real
                # views plus k copies, so it must be built for that width.
                source, source_splits = build_dataset(seed, data_cfg)
            model = train(source, source_splits, seed, True, data_cfg, config["training"])
            model.use_discount = True
            model.eval()
            models[seed][width] = DefendedPrismFlow(
                model,
                suspicion_threshold=config["detector"]["threshold"],
                suspicion_permutations=config["detector"].get("permutations", DEFAULT_PERMUTATIONS),
                suspicion_seed=seed,
            )
        logger.info("seed %d trained widths %s (%.0fs)", seed, widths, time.perf_counter() - started)

    clean_scores = {}
    per_seed_scores = {}
    protocol_summary, detector_rows = {}, {}
    progress_path = out_dir / "progress.json"

    for condition in config["conditions"]:
        name = condition["name"]
        started = time.perf_counter()
        width = 4 + condition["k"] if condition["kind"] == "clone" else config["data"]["n_views"]

        batches_by_seed, rows = {}, []
        for seed in seeds:
            defended = models[seed][width]
            dataset, splits = datasets[seed]
            batches, scores = run_condition(
                defended, dataset, splits[eval_cfg["split"]], condition,
                config["attack"], eval_cfg["batch_size"], seed,
            )
            batches_by_seed[seed] = batches
            per_seed_scores[f"{name}|{seed}"] = scores.tolist()

            if condition["kind"] == "clean":
                clean_scores[seed] = scores
            reference = clean_scores.get(seed)
            if reference is None:
                raise RuntimeError("the clean condition must be listed first: it calibrates the threshold")

            at_5 = calibrate_threshold(reference, target_rate=0.05)
            at_1 = calibrate_threshold(reference, target_rate=0.01)
            finite = scores[np.isfinite(scores)]
            rows.append({
                "seed": seed,
                "auc": roc_auc(scores, reference),
                "detect_at_5pct": float((finite > at_5).mean()) if finite.size else float("nan"),
                "detect_at_1pct": float((finite > at_1).mean()) if finite.size else float("nan"),
                "mean_score": float(finite.mean()) if finite.size else float("nan"),
                "flag_rate_shared": float((finite > config["detector"]["threshold"]).mean())
                if finite.size else float("nan"),
                "threshold_at_5pct": float(at_5),
                "threshold_at_1pct": float(at_1),
            })

        report = evaluate(
            {seed: models[seed][width] for seed in seeds},
            lambda seed: iter(batches_by_seed[seed]),
            f"{experiment_id}/{name}",
            seeds,
            results_dir=args.results_dir,
            n_bins=eval_cfg["n_bins"],
            coverages=eval_cfg["coverages"],
            title=name,
        )
        protocol_summary[name] = report["summary"]
        detector_rows[name] = rows

        summary = summarise(rows, DETECTOR_METRICS)
        logger.info(
            "%-14s auc=%.3f det@5%%=%.3f det@1%%=%.3f score=%+.4f acc=%.3f ece=%.3f (%.0fs)",
            name, summary["auc"]["mean"], summary["detect_at_5pct"]["mean"],
            summary["detect_at_1pct"]["mean"], summary["mean_score"]["mean"],
            report["summary"]["accuracy"]["mean"], report["summary"]["prob_ece"]["mean"],
            time.perf_counter() - started,
        )
        # Checkpoint after every condition: an interrupted run keeps its work.
        progress_path.write_text(
            json.dumps(json_safe({
                "completed": list(detector_rows),
                "detector": {k: summarise(v, DETECTOR_METRICS) for k, v in detector_rows.items()},
                "protocol": protocol_summary,
            }), indent=2),
            encoding="utf-8",
        )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        commit = None

    (out_dir / "detector.json").write_text(
        json.dumps(json_safe({
            "config": config,
            "commit": commit,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "summary": {k: summarise(v, DETECTOR_METRICS) for k, v in detector_rows.items()},
            "protocol": protocol_summary,
            "per_seed": detector_rows,
            "scores": per_seed_scores,
            "note": "controlled research simulation; own models, synthetic data",
        }), indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
