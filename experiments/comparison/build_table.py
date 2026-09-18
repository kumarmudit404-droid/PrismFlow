"""Render the Part 10 detector results as matrix.csv and a markdown table.

Reads `results/comparison/detector.json` and writes, beside it:

    matrix.csv     one row per condition, detector and protocol metrics
    summary.md     the same as formatted tables, plus "Where PrismFlow loses"

Nothing is computed here that `run_comparison.py` did not already measure --
this is formatting only, so the tables cannot drift from the run that produced
them.

Usage:
    python -m experiments.comparison.build_table
    python -m experiments.comparison.build_table --results-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DETECTOR_COLUMNS = ("auc", "detect_at_5pct", "detect_at_1pct", "mean_score")
PROTOCOL_COLUMNS = ("accuracy", "prob_ece", "brier_reliability", "vacuity_mean_confidence", "eniv")

HEADINGS = {
    "auc": "ROC AUC",
    "detect_at_5pct": "detection @ 5% FPR",
    "detect_at_1pct": "detection @ 1% FPR",
    "mean_score": "mean suspicion score",
    "accuracy": "accuracy",
    "prob_ece": "ECE",
    "brier_reliability": "Brier reliability",
    "vacuity_mean_confidence": "mean confidence",
    "eniv": "ENIV",
}


def cell(entry, signed=False) -> str:
    if entry is None:
        return "n/a"
    mean, std = entry.get("mean"), entry.get("std")
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    head = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return head
    return f"{head} +/- {std:.4f}"


def write_csv(path: Path, detector: dict, protocol: dict) -> None:
    columns = ["condition"]
    for key in DETECTOR_COLUMNS + PROTOCOL_COLUMNS:
        columns += [f"{key}_mean", f"{key}_std"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for name in detector:
            row = [name]
            for key in DETECTOR_COLUMNS:
                entry = detector[name].get(key, {})
                row += [entry.get("mean"), entry.get("std")]
            for key in PROTOCOL_COLUMNS:
                entry = protocol.get(name, {}).get(key, {})
                row += [entry.get("mean"), entry.get("std")]
            writer.writerow(row)


def table(conditions, values, keys, signed=()) -> list[str]:
    lines = [
        "| condition | " + " | ".join(HEADINGS[k] for k in keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for name in conditions:
        cells = [cell(values.get(name, {}).get(k), signed=k in signed) for k in keys]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def losing_cells(detector: dict, protocol: dict) -> list[str]:
    """Every condition where the detector is at or below chance, or where
    accuracy/ECE under attack is worse than on clean input.

    Required by the brief's HONESTY REQUIREMENT. It is generated from the
    numbers rather than written by hand, so it cannot quietly omit a loss.
    """
    lines = []
    clean_accuracy = protocol.get("clean", {}).get("accuracy", {}).get("mean")
    clean_ece = protocol.get("clean", {}).get("prob_ece", {}).get("mean")

    for name, metrics in detector.items():
        if name == "clean":
            continue
        auc = metrics.get("auc", {}).get("mean")
        if auc is None or math.isnan(auc):
            continue
        if auc < 0.45:
            lines.append(
                f"- **{name}**: detector INVERTED, AUC {cell(metrics['auc'])}. Scores are "
                "systematically LOWER than clean, so the flag points the wrong way: at any "
                "threshold this condition is flagged less often than clean input is."
            )
        elif auc <= 0.55:
            lines.append(
                f"- **{name}**: detector at chance, AUC {cell(metrics['auc'])}. Agreement "
                "under this condition is not distinguishable from clean."
            )
    for name in protocol:
        if name == "clean":
            continue
        accuracy = protocol[name].get("accuracy", {}).get("mean")
        ece = protocol[name].get("prob_ece", {}).get("mean")
        if clean_accuracy is not None and accuracy is not None and accuracy < clean_accuracy:
            lines.append(
                f"- **{name}**: accuracy {accuracy:.4f} against {clean_accuracy:.4f} clean "
                f"({accuracy - clean_accuracy:+.4f})."
            )
        if clean_ece is not None and ece is not None and ece > clean_ece:
            lines.append(
                f"- **{name}**: ECE {ece:.4f} against {clean_ece:.4f} clean ({ece - clean_ece:+.4f})."
            )
    return lines or ["- None found by the automatic check above."]


def main():
    parser = argparse.ArgumentParser(description="Format the Part 10 detector results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default="comparison")
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / args.experiment_id
    payload = json.loads((out_dir / "detector.json").read_text(encoding="utf-8"))
    detector, protocol = payload["summary"], payload["protocol"]
    seeds = payload["config"]["seeds"]
    conditions = list(detector)

    write_csv(out_dir / "matrix.csv", detector, protocol)

    lines = [
        "# Suspicion detector (Part 10)",
        "",
        f"Seeds: {seeds}. Mean +/- sample std across seeds. Detection thresholds are "
        "calibrated per seed on that seed's own CLEAN scores, so the `clean` row reads "
        "0.05 and 0.01 by construction and every other row is a true-positive rate at "
        "that false-positive budget. AUC is against the same seed's clean scores; 0.5 "
        "is chance.",
        "",
        "The detector never sees which condition it is scoring. This table does, "
        "because scoring a detector requires it.",
        "",
        "## Detection",
        "",
    ]
    lines += table(conditions, detector, DETECTOR_COLUMNS, signed=("mean_score",))
    lines += ["", "## Task metrics under each condition", ""]
    lines += table(conditions, protocol, PROTOCOL_COLUMNS)
    lines += ["", "## Where PrismFlow loses", "", *losing_cells(detector, protocol), ""]

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'matrix.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
