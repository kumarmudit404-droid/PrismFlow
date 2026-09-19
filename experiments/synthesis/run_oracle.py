"""Part 14 Arm B: the ORACLE CEILING. Is the information there at all?

WHAT THIS IS, AND WHAT IT IS NOT

This fits a logistic gate on the four candidate signals USING ATTACK LABELS. A
detector may not do that -- docs/CONTRACT.md forbids the suspicion detector
ground truth -- so nothing here is a proposed defence, and no number from this
file may be reported as detection performance.

It answers a strictly prior question: IS THE INFORMATION PRESENT? A label-fitted
gate evaluated in-sample is an upper bound on what any legal gate could extract.
If the ceiling is low, no threshold rule, calibration scheme or ensemble built
from these four signals can do better, and building one would be wasted compute.
That is why this runs before Arm A rather than after.

THE PRIMARY METRIC: SELECTIVITY

Parts 09, 10 and 13 all measured attacked-vs-clean. That is not the question the
project keeps failing. L5's finding is that honest duplication (clone_k2, AUC
0.8743) scores HIGHER than every adversarial chorus condition -- the detector
fires hardest at the case where nothing is wrong.

So the metric here is

    selectivity AUC = ROC( score(chorus k=3, eps=2.0) vs score(clone) )

0.5 means the signals cannot tell an attack from honest redundancy; below 0.5
means they actively prefer to flag honest redundancy. The condition chosen is
the one where each signal alone is known to fail: k=3 (Part 10's detector is at
chance, AUC 0.4870) at eps 2.0 (Part 13's lambda_U has collapsed).

THE FOUR SIGNALS, all strictly per-sample

  dependence        mean off-diagonal pairwise agreement -- the quantity the
                    discount acts on
  detector          Part 10's unexplained-agreement score
  tail              Part 13's tail-agreement score. PRE-REGISTERED NEGATIVE
                    CONTROL: Part 13 measured TPR@5%FPR of 0.038-0.068, i.e. no
                    signal. It is kept in deliberately. If the oracle leans on
                    it, something is wrong with the setup, not with Part 13.
  clique_contrast   internal minus external agreement, new in Part 14, pinned
                    against constructed structures in
                    tests/unit/test_clique_contrast.py BEFORE being used here

No batch-level constant may be used as a feature: it would be constant within a
condition and would let the fit identify the condition rather than the sample.

MATCHED GEOMETRY

Part 06's `duplicate_view` APPENDS copies, so V grows to 4 + k. Chorus keeps
V = 4. Comparing signals across different view counts would confound the measure
with V -- clique contrast, agreement means and ENIV all depend on it. Clones are
therefore made IN PLACE here: clone_k2 overwrites view 1 with view 0, clone_k3
overwrites views 1 and 2. Same V, same clique size as the attack it is compared
against. This is a deliberate deviation from Part 06 and the reason is the
comparison itself.

Usage:
    python -m experiments.synthesis.run_oracle
    python -m experiments.synthesis.run_oracle --quick   # NOT evidence
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from experiments.tail.run_tail import (
    build_dataset,
    roc_auc,
    train_model,
    whole_split,
)
from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.models.prismflow import FORWARD_NULL_PERMUTATIONS
from prismflow.statistics.clique_contrast import clique_contrast
from prismflow.statistics.dependence import (
    feature_dependence_matrix,
    predicted_classes,
)
from prismflow.statistics.suspicion import pairwise_agreement, suspicion_report
from prismflow.statistics.tail_dependence import tail_agreement_score
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger(__name__)
OUT_DIR = Path("results/synthesis")

SIGNALS = ("dependence", "detector", "tail", "clique_contrast")

# The go/no-go the Part is gated on, fixed before the run.
GO_THRESHOLD = 0.6

# Pre-registered prediction 3: "neither the existing signals nor their
# combination reaches selectivity AUC 0.70 without clique contrast." The
# ablation is what isolates that -- the four-signal oracle alone cannot say
# whether clique contrast is load-bearing or merely along for the ride.
ABLATIONS = {
    "all_four": ("dependence", "detector", "tail", "clique_contrast"),
    "existing_three": ("dependence", "detector", "tail"),
    "existing_two": ("dependence", "detector"),
    "clique_only": ("clique_contrast",),
    "clique_plus_dependence": ("dependence", "clique_contrast"),
}
P3_THRESHOLD = 0.70

CONFIG = {
    "experiment": {"id": "synthesis_oracle"},
    "seeds": [0, 1, 2, 3, 4],
    "data": {"n_views": 4, "rho": 0.3, "n_samples": 2000},
    "training": {"epochs": 40, "batch_size": 64, "lr": 1e-3, "anneal_epochs": 10,
                 "per_view_loss_weight": 1.0},
    "attack": {"k": 3, "epsilon": 2.0, "beta": 1.0, "steps": 30},
    "clone": {"source": 0, "ks": [2, 3]},
    "evaluation": {"batch_size": 512},
    "oracle": {"ridge": 1e-3, "iterations": 60},
    "go_threshold": GO_THRESHOLD,
}


def clone_in_place(views: torch.Tensor, source: int, k: int) -> torch.Tensor:
    """k views made bit-identical to `source`, WITHOUT changing the view count."""
    if k < 2:
        raise ValueError(f"clone k must be >= 2, got {k}")
    cloned = views.clone()
    targets = [v for v in range(views.shape[1]) if v != source][: k - 1]
    for view in targets:
        cloned[:, view, :] = views[:, source, :]
    return cloned


@torch.no_grad()
def signals_for(model, views, view_mask) -> dict[str, np.ndarray]:
    """The four per-sample signals for one condition."""
    output = model(views, view_mask)
    evidence = output.per_view_evidence
    belief = output.per_view_belief
    strata = predicted_classes(evidence, view_mask)

    agreement = pairwise_agreement(belief, view_mask)  # [B, V, V]
    off_diagonal = ~np.eye(agreement.shape[1], dtype=bool)
    with np.errstate(invalid="ignore"):
        dependence = np.nanmean(agreement[:, off_diagonal], axis=1)

    matrix = feature_dependence_matrix(
        model.encoder(views, view_mask), evidence, view_mask,
        method=model.dependence_method, conditioning=model.dependence_conditioning,
        seed=model.dependence_seed, null_permutations=FORWARD_NULL_PERMUTATIONS,
    )
    detector = suspicion_report(
        belief, matrix, view_mask, evidence=evidence, strata=strata
    ).score

    scalar = evidence.detach().cpu().numpy()[
        np.arange(evidence.shape[0]), :, output.prediction.detach().cpu().numpy()
    ]

    return {
        "dependence": np.asarray(dependence, dtype=np.float64),
        "detector": np.asarray(detector, dtype=np.float64),
        "tail": tail_agreement_score(scalar),
        "clique_contrast": clique_contrast(agreement, view_mask),
    }


def fit_logistic(features: np.ndarray, labels: np.ndarray, ridge: float, iterations: int):
    """IRLS logistic fit. Standardised inputs, ridge for separable cases.

    Newton rather than gradient descent because the problem is tiny (4 features)
    and IRLS converges in a handful of steps; the ridge keeps the Hessian
    invertible when a signal separates the classes perfectly.
    """
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardised = (features - centre) / scale

    design = np.column_stack([np.ones(len(standardised)), standardised])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0  # never penalise the intercept

    for _ in range(iterations):
        eta = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        variance = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (labels - probability) - penalty @ weights
        hessian = (design * variance[:, None]).T @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        weights = weights + step
        if np.max(np.abs(step)) < 1e-8:
            break

    return weights, centre, scale


def apply_logistic(features, weights, centre, scale) -> np.ndarray:
    standardised = (np.asarray(features, dtype=np.float64) - centre) / scale
    design = np.column_stack([np.ones(len(standardised)), standardised])
    return design @ weights


def finite_rows(features: np.ndarray, labels: np.ndarray):
    keep = np.isfinite(features).all(axis=1)
    return features[keep], labels[keep], int((~keep).sum())


def summarise(values: list[float]) -> dict:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return {
        "mean": statistics.mean(clean) if clean else float("nan"),
        "std": statistics.stdev(clean) if len(clean) > 1 else float("nan"),
        "n_seeds": len(clean),
        "per_seed": values,
    }


def fmt(entry) -> str:
    if math.isnan(entry["mean"]):
        return "n/a"
    std = entry["std"]
    return f"{entry['mean']:.4f}" if math.isnan(std) else f"{entry['mean']:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, NOT evidence")
    args = parser.parse_args()

    config = json.loads(json.dumps(CONFIG))
    if args.quick:
        config["seeds"] = config["seeds"][:1]
        config["training"] = dict(config["training"], epochs=2)
        config["attack"] = dict(config["attack"], steps=3)
        LOGGER.warning("--quick: 1 seed, 2 epochs. NOT EVIDENCE.")

    per_seed_rows = []
    pooled = {"features": [], "labels": [], "seed": []}

    for seed in config["seeds"]:
        set_seed(seed)
        dataset, splits, base = build_dataset(seed, config["data"])
        model = train_model(dataset, splits, base, config["training"], seed)
        views, view_mask, _ = whole_split(
            dataset, splits["test"], config["evaluation"]["batch_size"]
        )

        # Positive class: the attack, at the strength where each signal fails.
        attack = chorus_attack(
            model, views, view_mask,
            ChorusConfig(
                k=config["attack"]["k"], epsilon=config["attack"]["epsilon"],
                beta=config["attack"]["beta"], steps=config["attack"]["steps"], seed=seed,
            ),
        )
        attacked = signals_for(model, views + attack.delta, view_mask)
        success = float(
            (
                model(views + attack.delta, view_mask).prediction.detach().cpu().numpy()
                == attack.target.detach().cpu().numpy()
            ).mean()
        )

        # Negative class: honest duplication at matched V and clique size.
        honest = [
            signals_for(model, clone_in_place(views, config["clone"]["source"], k), view_mask)
            for k in config["clone"]["ks"]
        ]

        def stack(source_list):
            return np.column_stack([
                np.concatenate([s[name] for s in source_list]) for name in SIGNALS
            ])

        positive = stack([attacked])
        negative = stack(honest)
        features = np.vstack([positive, negative])
        labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
        features, labels, dropped = finite_rows(features, labels)

        weights, centre, scale = fit_logistic(
            features, labels, config["oracle"]["ridge"], config["oracle"]["iterations"]
        )
        scores = apply_logistic(features, weights, centre, scale)
        oracle_auc = roc_auc(scores[labels == 1], scores[labels == 0])

        ablation_auc = {}
        for name, subset in ABLATIONS.items():
            columns = [SIGNALS.index(s) for s in subset]
            sub_features = features[:, columns]
            w, c, s = fit_logistic(
                sub_features, labels,
                config["oracle"]["ridge"], config["oracle"]["iterations"],
            )
            sub_scores = apply_logistic(sub_features, w, c, s)
            ablation_auc[name] = roc_auc(sub_scores[labels == 1], sub_scores[labels == 0])

        univariate = {
            name: roc_auc(
                features[labels == 1][:, i], features[labels == 0][:, i]
            )
            for i, name in enumerate(SIGNALS)
        }

        per_seed_rows.append({
            "seed": seed,
            "oracle_auc": oracle_auc,
            "ablation": ablation_auc,
            "univariate": univariate,
            "attack_success": success,
            "n_positive": int((labels == 1).sum()),
            "n_negative": int((labels == 0).sum()),
            "dropped_nonfinite": dropped,
            "weights": weights.tolist(),
        })
        pooled["features"].append(features)
        pooled["labels"].append(labels)
        pooled["seed"].append(np.full(len(labels), seed))

        LOGGER.info(
            "seed=%s success=%.3f | ORACLE AUC=%.4f | %s",
            seed, success, oracle_auc,
            " ".join(f"{k}={v:.3f}" for k, v in univariate.items()),
        )

    # Pooled in-sample ceiling, and leave-one-seed-out for generalisation.
    all_features = np.vstack(pooled["features"])
    all_labels = np.concatenate(pooled["labels"])
    all_seeds = np.concatenate(pooled["seed"])

    weights, centre, scale = fit_logistic(
        all_features, all_labels, config["oracle"]["ridge"], config["oracle"]["iterations"]
    )
    pooled_scores = apply_logistic(all_features, weights, centre, scale)
    pooled_auc = roc_auc(pooled_scores[all_labels == 1], pooled_scores[all_labels == 0])

    loso = []
    for seed in config["seeds"]:
        train = all_seeds != seed
        held = ~train
        if not held.any() or len(np.unique(all_labels[train])) < 2:
            continue
        w, c, s = fit_logistic(
            all_features[train], all_labels[train],
            config["oracle"]["ridge"], config["oracle"]["iterations"],
        )
        held_scores = apply_logistic(all_features[held], w, c, s)
        loso.append(roc_auc(held_scores[all_labels[held] == 1], held_scores[all_labels[held] == 0]))

    oracle = summarise([r["oracle_auc"] for r in per_seed_rows])
    univariate = {
        name: summarise([r["univariate"][name] for r in per_seed_rows]) for name in SIGNALS
    }
    ablation = {
        name: summarise([r["ablation"][name] for r in per_seed_rows]) for name in ABLATIONS
    }

    # P3 is a claim about the arms WITHOUT clique contrast: they must stay
    # below the threshold. Per-seed crossings are recorded next to the mean
    # because a mean under 0.70 with seeds on both sides is a different
    # finding from one where no seed reaches it.
    arms_without_clique = [
        name for name, subset in ABLATIONS.items() if "clique_contrast" not in subset
    ]
    per_arm = {
        name: {
            "mean": ablation[name]["mean"],
            "reaches_threshold": ablation[name]["mean"] >= P3_THRESHOLD,
            "seeds_at_or_above": sum(
                1 for v in ablation[name]["per_seed"]
                if not math.isnan(v) and v >= P3_THRESHOLD
            ),
            "n_seeds": ablation[name]["n_seeds"],
        }
        for name in ABLATIONS
    }
    p3 = {
        "threshold": P3_THRESHOLD,
        "statement": (
            "neither the existing signals nor their combination reaches "
            f"selectivity AUC {P3_THRESHOLD} without clique contrast"
        ),
        "arms_without_clique": arms_without_clique,
        "held": all(not per_arm[name]["reaches_threshold"] for name in arms_without_clique),
        "per_arm": per_arm,
    }

    loso_summary = summarise(loso)
    decision = "PROCEED to Arm A" if oracle["mean"] >= GO_THRESHOLD else "STOP: information absent"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "oracle.json").write_text(
        json.dumps({
            "config": config, "torch": torch.__version__,
            "oracle_auc": oracle, "univariate_auc": univariate,
            "ablation_auc": ablation, "p3": p3,
            "pooled_in_sample_auc": pooled_auc, "leave_one_seed_out_auc": loso_summary,
            "attack_success": summarise([r["attack_success"] for r in per_seed_rows]),
            "per_seed": per_seed_rows,
            "decision": decision,
        }, indent=2, default=float),
        encoding="utf-8",
    )

    lines = [
        "# Part 14 Arm B: oracle ceiling (NOT a detector)",
        "",
        "Fitted WITH attack labels and evaluated in-sample. This is an upper",
        "bound on what any legal gate could extract, never a detection result.",
        "",
        f"Seeds: {config['seeds']}. Primary metric: selectivity AUC, "
        f"chorus k={config['attack']['k']} eps={config['attack']['epsilon']} "
        f"vs clone k={config['clone']['ks']}.",
        "",
        f"Attack success: {fmt(summarise([r['attack_success'] for r in per_seed_rows]))}",
        "",
        "## Decision",
        "",
        f"| quantity | value |", "|---|---|",
        f"| oracle selectivity AUC (per-seed) | **{fmt(oracle)}** |",
        f"| go threshold | {GO_THRESHOLD} |",
        f"| pooled in-sample | {pooled_auc:.4f} |",
        f"| leave-one-seed-out | {fmt(loso_summary)} |",
        f"| **decision** | **{decision}** |",
        "",
        "## Each signal alone (univariate selectivity AUC)",
        "",
        "| signal | AUC | note |", "|---|---|---|",
    ]
    notes = {
        "dependence": "quantity the discount acts on",
        "detector": "Part 10, at chance on k=3 vs clean (L5)",
        "tail": "PRE-REGISTERED NEGATIVE CONTROL (Part 13: no signal)",
        "clique_contrast": "new in Part 14",
    }
    for name in SIGNALS:
        lines.append(f"| {name} | {fmt(univariate[name])} | {notes[name]} |")

    lines += [
        "",
        "AUC below 0.5 means the signal prefers to flag HONEST duplication over",
        "the attack -- the L5 failure, restated on the selectivity metric.",
        "",
        f"## P3 ablation (threshold {P3_THRESHOLD})",
        "",
        "Pre-registered prediction 3: " + p3["statement"] + ".",
        "",
        "The threshold applies to the arms WITHOUT clique contrast "
        f"({', '.join(arms_without_clique)}); P3 holds only if every one of",
        f"them stays below {P3_THRESHOLD}. Arms that include clique contrast are",
        "shown for contrast and are not what the prediction is about.",
        "",
        f"| arm | signals | AUC | vs {P3_THRESHOLD} | seeds >= {P3_THRESHOLD} |",
        "|---|---|---|---|---|",
    ]
    for name, subset in ABLATIONS.items():
        arm = per_arm[name]
        marker = "" if "clique_contrast" in subset else " (P3 arm)"
        verdict = "**REACHES**" if arm["reaches_threshold"] else "below"
        lines.append(
            f"| {name}{marker} | {', '.join(subset)} | {fmt(ablation[name])} | "
            f"{verdict} | {arm['seeds_at_or_above']}/{arm['n_seeds']} |"
        )

    delta = ablation["all_four"]["mean"] - ablation["existing_three"]["mean"]
    lines += [
        "",
        f"**P3 {'HOLDS' if p3['held'] else 'REFUTED'}.** "
        f"all_four minus existing_three = {delta:+.4f} selectivity AUC.",
        "",
        "The per-seed column is the load-bearing one: a mean below threshold",
        "with seeds on both sides is not the same result as one where no seed",
        "crosses. Read it before quoting the mean.",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("DECISION: %s (oracle AUC %.4f)", decision, oracle["mean"])


if __name__ == "__main__":
    main()
