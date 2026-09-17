"""The unified evaluation protocol. Every experiment reports metrics through here.

    evaluate(model, loader, experiment_id, seeds) -> dict

writes

    results/<experiment_id>/metrics.json
    results/<experiment_id>/metrics.csv
    results/<experiment_id>/reliability_diagram.png

Experiment scripts must not compute their own metrics. A metric an experiment
needs belongs in `compute_metrics`, so every result in the project is computed
by the same code.

TWO CONFIDENCE DEFINITIONS
--------------------------
The model reports two different things that can be called confidence:

  top-label probability  max_k probs[k], the expected Dirichlet probability of
                         the predicted class. A probability of being right, so
                         it is the quantity ECE/MCE is defined for. Always
                         >= 1/K.
  vacuity confidence     1 - fused uncertainty (`PrismFlowOutput.confidence`).
                         The quantity the clone experiment tracks. It is NOT a
                         probability of being right: a fully vacuous opinion
                         has vacuity confidence 0 and still predicts at 1/K.

Calibration and selective-prediction metrics are reported for both, under the
prefixes `prob_` and `vacuity_`. They answer different questions and must not be
substituted for each other in a write-up. The Brier score and its
decomposition use the full probability vector and have no prefix.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from prismflow.evaluation.calibration import (
    DEFAULT_N_BINS,
    brier_decomposition,
    expected_calibration_error,
    maximum_calibration_error,
    plot_reliability_diagram,
    reliability_bins,
)
from prismflow.evaluation.metrics import (
    DEFAULT_COVERAGES,
    accuracy,
    aurc,
    coverage_key,
    macro_precision_recall_f1,
    selective_risk,
)

MIN_SEEDS = 5
_PROB_TOLERANCE = 1e-4

CONFIDENCE_DEFINITIONS = {
    "prob": "top-label expected Dirichlet probability, max_k probs[k]",
    "vacuity": "1 - fused uncertainty (PrismFlowOutput.confidence); not a probability of being right",
}


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    vacuity_confidence: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> dict[str, float]:
    """Every protocol metric for one set of predictions. Pure function of arrays."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    vacuity_confidence = np.asarray(vacuity_confidence, dtype=np.float64)

    # float32 model outputs can sit a hair outside [0, 1]; anything beyond
    # rounding error is a real defect and must not be clipped away.
    for name, values in (("probs", probs), ("vacuity confidence", vacuity_confidence)):
        if np.any(values < -_PROB_TOLERANCE) or np.any(values > 1.0 + _PROB_TOLERANCE):
            raise ValueError(f"{name} outside [0, 1] beyond rounding tolerance")
    probs = np.clip(probs, 0.0, 1.0)
    vacuity_confidence = np.clip(vacuity_confidence, 0.0, 1.0)

    n_classes = probs.shape[1]
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    errors = 1.0 - correct

    result: dict[str, float] = {
        "n_samples": float(labels.size),
        "accuracy": accuracy(labels, predictions),
        **macro_precision_recall_f1(labels, predictions, n_classes),
    }

    decomposition = brier_decomposition(probs, labels, n_bins)
    result.update(
        brier=decomposition.brier,
        brier_reliability=decomposition.reliability,
        brier_resolution=decomposition.resolution,
        brier_uncertainty=decomposition.uncertainty,
        brier_within_bin=decomposition.within_bin,
    )

    for prefix, confidence in (("prob", probs.max(axis=1)), ("vacuity", vacuity_confidence)):
        result[f"{prefix}_mean_confidence"] = float(confidence.mean())
        result[f"{prefix}_ece"] = expected_calibration_error(confidence, correct, n_bins)
        result[f"{prefix}_mce"] = maximum_calibration_error(confidence, correct, n_bins)
        result[f"{prefix}_aurc"] = aurc(confidence, errors)
        for coverage in coverages:
            result[f"{prefix}_{coverage_key(coverage)}"] = selective_risk(confidence, errors, coverage)

    return result


def _resolve(source, seed: int, what: str):
    if isinstance(source, Mapping):
        if seed not in source:
            raise KeyError(f"no {what} for seed {seed}")
        return source[seed]
    if callable(source) and not isinstance(source, torch.nn.Module):
        return source(seed)
    raise TypeError(
        f"{what} must be a Mapping[seed, ...] or a callable seed -> ...; a single "
        f"{what} cannot represent several seeds"
    )


@torch.no_grad()
def collect_predictions(model, batches: Iterable) -> dict[str, np.ndarray | float]:
    """Run `model` over `batches` in eval mode and gather arrays for scoring.

    Also returns batch-size-weighted ENIV diagnostics when the model computes
    them (NaN otherwise). Those are reported, not scored.
    """
    was_training = model.training
    model.eval()
    labels, probs, vacuity = [], [], []
    eniv_sums = {"eniv": 0.0, "mean_dependence": 0.0, "efficiency_ratio": 0.0}
    eniv_weight = 0
    try:
        for batch in batches:
            output = model(batch.views, batch.view_mask)
            labels.append(batch.labels.cpu().numpy())
            probs.append(output.probs.detach().cpu().double().numpy())
            vacuity.append(output.confidence.detach().cpu().double().numpy())
            if output.eniv is not None:
                n = len(batch)
                eniv_sums["eniv"] += output.eniv.effective_views * n
                eniv_sums["mean_dependence"] += output.eniv.mean_dependence * n
                eniv_sums["efficiency_ratio"] += output.eniv.efficiency_ratio * n
                eniv_weight += n
    finally:
        model.train(was_training)

    if not labels:
        raise ValueError("loader produced no batches")
    diagnostics = {
        key: (value / eniv_weight if eniv_weight else float("nan")) for key, value in eniv_sums.items()
    }
    return {
        "labels": np.concatenate(labels),
        "probs": np.concatenate(probs),
        "vacuity_confidence": np.concatenate(vacuity),
        **diagnostics,
    }


def _summary(per_seed: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for key in keys:
        values = [row[key] for row in per_seed if not math.isnan(row[key])]
        summary[key] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "n_seeds": len(values),
        }
    return summary


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def evaluate(
    model: Mapping[int, torch.nn.Module] | Callable[[int], torch.nn.Module],
    loader: Mapping[int, Iterable] | Callable[[int], Iterable],
    experiment_id: str,
    seeds: Sequence[int],
    *,
    results_dir: str | Path = "results",
    n_bins: int = DEFAULT_N_BINS,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    allow_fewer_seeds: bool = False,
    title: str | None = None,
) -> dict:
    """Evaluate one system over several seeds and write the standard outputs.

    model    per seed: a Mapping {seed: model} or a callable seed -> model
             (one trained model per seed; a single model is rejected).
    loader   per seed: a Mapping {seed: iterable of Batch} or a callable
             seed -> iterable of Batch. Use a callable when the batches are a
             one-shot iterator, so each seed gets a fresh one.
    experiment_id
             output subdirectory of `results_dir`. May contain '/' to separate
             conditions, e.g. "robustness/missing_0.3/prismflow".

    Metrics are computed per seed, then summarised as mean and sample std
    (ddof=1) across seeds. Fewer than 5 seeds is refused unless
    `allow_fewer_seeds`; such outputs are flagged `not_evidence` in the JSON
    and CSV and must never be reported as findings.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    not_evidence = len(seeds) < MIN_SEEDS
    if not_evidence and not allow_fewer_seeds:
        raise ValueError(f"contract requires at least {MIN_SEEDS} seeds, got {len(seeds)}")
    coverages = list(coverages)

    per_seed, bins_by_definition = [], {"prob": [], "vacuity": []}
    for seed in seeds:
        collected = collect_predictions(_resolve(model, seed, "model"), _resolve(loader, seed, "loader"))
        row = {"seed": seed}
        row.update(
            compute_metrics(
                collected["labels"], collected["probs"], collected["vacuity_confidence"],
                n_bins=n_bins, coverages=coverages,
            )
        )
        row.update({key: collected[key] for key in ("eniv", "mean_dependence", "efficiency_ratio")})
        per_seed.append(row)

        probs = np.clip(collected["probs"], 0.0, 1.0)
        correct = (probs.argmax(axis=1) == collected["labels"]).astype(np.float64)
        bins_by_definition["prob"].append(reliability_bins(probs.max(axis=1), correct, n_bins))
        bins_by_definition["vacuity"].append(
            reliability_bins(np.clip(collected["vacuity_confidence"], 0.0, 1.0), correct, n_bins)
        )

    metric_keys = [key for key in per_seed[0] if key != "seed"]
    summary = _summary(per_seed, metric_keys)

    out_dir = Path(results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_id": experiment_id,
        "seeds": seeds,
        "not_evidence": not_evidence,
        "n_bins": n_bins,
        "coverages": coverages,
        "confidence_definitions": CONFIDENCE_DEFINITIONS,
        "summary": summary,
        "per_seed": per_seed,
        "reliability_bins": {
            definition: [{"seed": seed, **bins.to_dict()} for seed, bins in zip(seeds, bin_list)]
            for definition, bin_list in bins_by_definition.items()
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe(report), indent=2), encoding="utf-8"
    )

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "not_evidence", *metric_keys])
        for row in per_seed:
            writer.writerow([f"seed={row['seed']}", not_evidence, *(row[k] for k in metric_keys)])
        for stat in ("mean", "std"):
            writer.writerow([stat, not_evidence, *(summary[k][stat] for k in metric_keys)])

    heading = title or experiment_id
    if not_evidence:
        heading += "  [NOT EVIDENCE: fewer than 5 seeds]"
    plot_reliability_diagram(
        [
            {"label": "top-label probability", "bins": bins_by_definition["prob"]},
            {"label": "vacuity confidence (1 - u)", "bins": bins_by_definition["vacuity"]},
        ],
        out_dir / "reliability_diagram.png",
        title=f"Reliability: {heading}",
    )

    report["output_dir"] = str(out_dir)
    return report
