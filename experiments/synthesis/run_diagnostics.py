"""Part 14 Arm B diagnostics: is the 0.7448 ceiling real?

NOT a detector, NOT a defence, and NOT a replacement for Arm B. This file
answers three questions raised by the Arm B ablation before Arm A is allowed
to start. It imports Arm B's pipeline rather than restating it, so every
number here is directly comparable to results/synthesis/oracle.json.

D1  THE TAIL LEAK
    Adding `tail` to {dependence, detector} moved selectivity AUC 0.6724 ->
    0.7115 (+0.0391), more than clique contrast contributes. `tail` is a
    PRE-REGISTERED NEGATIVE CONTROL (Part 13: TPR@5%FPR 0.038-0.068), so
    run_oracle.py's docstring registers this as a setup alarm, not a result.

    Hypothesis: the contribution is a between-condition mean shift -- tail
    tracks WHICH CONDITION a sample came from, not the sample's content.

    Four tests, all on the same fitted rows:
      a) variance decomposition (ICC): between- vs within-condition variance
      b) condition-mean surrogate: replace each tail value with its condition
         mean. This is a pure batch-level constant with ZERO per-sample
         content. If it scores like the real tail, the real tail was acting
         as one.
      c) within-condition centring: subtract the condition mean, leaving only
         per-sample content. If the contribution vanishes, it was the shift.
      d) condition identification: tail's AUC for clone_k2 vs clone_k3 --
         two conditions that share the SAME label. Any separation there is
         condition tracking by definition.

D2  THE CLIQUE FIT TYPE
    clique_contrast has univariate AUC 0.6408 +/- 0.0551 but the linear fit
    flips its weight negative on 3/5 seeds, landing at exactly 1-univariate.
    Does a RANK-BASED fit recover the separation a linear one cancels?

    This cannot change P3's literal verdict, which is about a linear gate.
    It can change what Arm A should use.

D3  THE NaN DROP PATTERN
    clique_contrast.py raises "All-NaN slice encountered" on seeds 3 and 4,
    dropping 1 row each. A drop concentrated in one condition would bias the
    ablation; a scattered one would not. This counts them per condition.

Usage:
    python -m experiments.synthesis.run_diagnostics
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.synthesis.run_oracle import (
    CONFIG,
    SIGNALS,
    apply_logistic,
    clone_in_place,
    fit_logistic,
    fmt,
    signals_for,
    summarise,
)
from experiments.tail.run_tail import build_dataset, roc_auc, train_model, whole_split
from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger(__name__)
OUT_DIR = Path("results/synthesis")

TAIL = SIGNALS.index("tail")
CLIQUE = SIGNALS.index("clique_contrast")
DEP_DET = [SIGNALS.index("dependence"), SIGNALS.index("detector")]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks in [0, 1], ties shared. No scipy dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    # Share ranks within tied runs so a tie cannot be broken by input order.
    sorted_values = values[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks / max(len(values) - 1, 1)


def rank_transform(features: np.ndarray) -> np.ndarray:
    return np.column_stack([average_ranks(features[:, i]) for i in range(features.shape[1])])


def auc_of(features, labels, columns, ridge, iterations) -> float:
    sub = features[:, columns] if isinstance(columns, list) else features[:, [columns]]
    w, c, s = fit_logistic(sub, labels, ridge, iterations)
    scores = apply_logistic(sub, w, c, s)
    return roc_auc(scores[labels == 1], scores[labels == 0])


def signed_weight(features, labels, column, ridge, iterations) -> float:
    w, _, _ = fit_logistic(features[:, [column]], labels, ridge, iterations)
    return float(w[1])


def icc(values: np.ndarray, condition: np.ndarray) -> dict:
    """Share of total variance that is BETWEEN condition rather than within."""
    grand = values.mean()
    between = 0.0
    within = 0.0
    for name in np.unique(condition):
        group = values[condition == name]
        between += len(group) * (group.mean() - grand) ** 2
        within += ((group - group.mean()) ** 2).sum()
    total = between + within
    return {
        "between_share": float(between / total) if total > 0 else float("nan"),
        "within_share": float(within / total) if total > 0 else float("nan"),
    }


def main():
    config = json.loads(json.dumps(CONFIG))
    ridge = config["oracle"]["ridge"]
    iterations = config["oracle"]["iterations"]
    rows = []

    for seed in config["seeds"]:
        set_seed(seed)
        dataset, splits, base = build_dataset(seed, config["data"])
        model = train_model(dataset, splits, base, config["training"], seed)
        views, view_mask, _ = whole_split(
            dataset, splits["test"], config["evaluation"]["batch_size"]
        )

        attack = chorus_attack(
            model, views, view_mask,
            ChorusConfig(
                k=config["attack"]["k"], epsilon=config["attack"]["epsilon"],
                beta=config["attack"]["beta"], steps=config["attack"]["steps"], seed=seed,
            ),
        )

        # Same three conditions Arm B fits on, kept separately labelled.
        conditions = {"chorus_k3": signals_for(model, views + attack.delta, view_mask)}
        for k in config["clone"]["ks"]:
            conditions[f"clone_k{k}"] = signals_for(
                model, clone_in_place(views, config["clone"]["source"], k), view_mask
            )

        # D3: where do the non-finite rows fall, BEFORE anything is dropped?
        nan_by_condition = {
            name: {
                signal: int((~np.isfinite(values[signal])).sum()) for signal in SIGNALS
            }
            for name, values in conditions.items()
        }

        features, labels, condition = [], [], []
        for name, values in conditions.items():
            block = np.column_stack([values[signal] for signal in SIGNALS])
            features.append(block)
            labels.append(np.full(len(block), 1.0 if name == "chorus_k3" else 0.0))
            condition.append(np.full(len(block), name))
        features = np.vstack(features)
        labels = np.concatenate(labels)
        condition = np.concatenate(condition)

        keep = np.isfinite(features).all(axis=1)
        features, labels, condition = features[keep], labels[keep], condition[keep]

        # ---- D1: is tail's contribution a between-condition mean shift? ----
        condition_mean = np.zeros(len(features))
        for name in np.unique(condition):
            here = condition == name
            condition_mean[here] = features[here, TAIL].mean()

        surrogate = features.copy()
        surrogate[:, TAIL] = condition_mean          # pure batch-level constant
        centred = features.copy()
        centred[:, TAIL] = features[:, TAIL] - condition_mean   # content only

        dep_det = auc_of(features, labels, DEP_DET, ridge, iterations)
        dep_det_tail = auc_of(features, labels, DEP_DET + [TAIL], ridge, iterations)
        dep_det_surrogate = auc_of(surrogate, labels, DEP_DET + [TAIL], ridge, iterations)
        dep_det_centred = auc_of(centred, labels, DEP_DET + [TAIL], ridge, iterations)

        # Same label, different condition: any separation here is condition
        # tracking, because the gate is supposed to be blind to which clone
        # produced the row.
        clone_rows = condition != "chorus_k3"
        k2 = features[condition == "clone_k2", TAIL]
        k3 = features[condition == "clone_k3", TAIL]
        tail_condition_auc = roc_auc(k2, k3) if len(k2) and len(k3) else float("nan")

        # ---- D2: does a rank-based fit recover clique contrast? ----
        ranked = rank_transform(features)
        all_columns = list(range(len(SIGNALS)))

        row = {
            "seed": seed,
            "nan_by_condition": nan_by_condition,
            "n_dropped": int((~keep).sum()),
            "tail": {
                "dep_det": dep_det,
                "dep_det_tail": dep_det_tail,
                "dep_det_tail_condition_mean_surrogate": dep_det_surrogate,
                "dep_det_tail_within_condition_centred": dep_det_centred,
                "clone_k2_vs_k3_auc": tail_condition_auc,
                "variance": icc(features[:, TAIL], condition),
                "condition_means": {
                    name: float(features[condition == name, TAIL].mean())
                    for name in np.unique(condition)
                },
            },
            "clique": {
                "univariate": roc_auc(
                    features[labels == 1][:, CLIQUE], features[labels == 0][:, CLIQUE]
                ),
                "linear_alone": auc_of(features, labels, [CLIQUE], ridge, iterations),
                "rank_alone": auc_of(ranked, labels, [CLIQUE], ridge, iterations),
                "linear_weight_alone": signed_weight(features, labels, CLIQUE, ridge, iterations),
                "all_four_linear": auc_of(features, labels, all_columns, ridge, iterations),
                "all_four_rank": auc_of(ranked, labels, all_columns, ridge, iterations),
                "existing_three_rank": auc_of(
                    ranked, labels, [i for i in all_columns if i != CLIQUE], ridge, iterations
                ),
            },
        }
        rows.append(row)
        LOGGER.info(
            "seed=%s | tail: real=%.4f surrogate=%.4f centred=%.4f base=%.4f | "
            "clique: lin=%.4f rank=%.4f | dropped=%s",
            seed, dep_det_tail, dep_det_surrogate, dep_det_centred, dep_det,
            row["clique"]["linear_alone"], row["clique"]["rank_alone"], row["n_dropped"],
        )

    def across(path) -> dict:
        values = []
        for row in rows:
            node = row
            for key in path:
                node = node[key]
            values.append(node)
        return summarise(values)

    tail_keys = [
        "dep_det", "dep_det_tail", "dep_det_tail_condition_mean_surrogate",
        "dep_det_tail_within_condition_centred", "clone_k2_vs_k3_auc",
    ]
    clique_keys = [
        "univariate", "linear_alone", "rank_alone", "linear_weight_alone",
        "all_four_linear", "all_four_rank", "existing_three_rank",
    ]
    tail_summary = {key: across(["tail", key]) for key in tail_keys}
    tail_summary["between_condition_variance_share"] = across(
        ["tail", "variance", "between_share"]
    )
    clique_summary = {key: across(["clique", key]) for key in clique_keys}

    drops = {}
    for name in ("chorus_k3", "clone_k2", "clone_k3"):
        drops[name] = {
            signal: [row["nan_by_condition"][name][signal] for row in rows]
            for signal in SIGNALS
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "diagnostics.json").write_text(
        json.dumps({
            "config": config, "torch": torch.__version__,
            "tail": tail_summary, "clique": clique_summary,
            "nan_drops_by_condition": drops,
            "per_seed": rows,
        }, indent=2, default=float),
        encoding="utf-8",
    )

    lines = [
        "# Part 14 Arm B diagnostics",
        "",
        f"Seeds: {config['seeds']}. Same pipeline and conditions as Arm B.",
        "",
        "## D1: is the tail contribution a between-condition mean shift?",
        "",
        "| fit | tail column | selectivity AUC |",
        "|---|---|---|",
        f"| dependence + detector | (absent) | {fmt(tail_summary['dep_det'])} |",
        f"| + tail | as measured | {fmt(tail_summary['dep_det_tail'])} |",
        f"| + tail | condition MEAN only (no per-sample content) | "
        f"{fmt(tail_summary['dep_det_tail_condition_mean_surrogate'])} |",
        f"| + tail | within-condition CENTRED (content only) | "
        f"{fmt(tail_summary['dep_det_tail_within_condition_centred'])} |",
        "",
        f"Between-condition share of tail variance: "
        f"{fmt(tail_summary['between_condition_variance_share'])}",
        "",
        f"tail AUC, clone_k2 vs clone_k3 (SAME label, different condition): "
        f"{fmt(tail_summary['clone_k2_vs_k3_auc'])}",
        "",
        "## D2: does a rank-based fit recover clique contrast?",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| clique univariate AUC | {fmt(clique_summary['univariate'])} |",
        f"| clique alone, LINEAR fit | {fmt(clique_summary['linear_alone'])} |",
        f"| clique alone, RANK fit | {fmt(clique_summary['rank_alone'])} |",
        f"| clique fitted weight, alone (sign) | {fmt(clique_summary['linear_weight_alone'])} |",
        f"| all four, LINEAR | {fmt(clique_summary['all_four_linear'])} |",
        f"| all four, RANK | {fmt(clique_summary['all_four_rank'])} |",
        f"| existing three, RANK | {fmt(clique_summary['existing_three_rank'])} |",
        "",
        "## D3: where do the dropped rows fall?",
        "",
        "Non-finite counts per condition per seed (seeds in config order).",
        "",
        "| condition | " + " | ".join(SIGNALS) + " |",
        "|---|" + "---|" * len(SIGNALS),
    ]
    for name, by_signal in drops.items():
        cells = " | ".join(str(by_signal[signal]) for signal in SIGNALS)
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    (OUT_DIR / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("diagnostics written")


if __name__ == "__main__":
    main()
