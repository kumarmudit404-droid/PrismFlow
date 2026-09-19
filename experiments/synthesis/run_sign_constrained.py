"""Part 14: the SIGN-CONSTRAINED ceiling -- what a label-free gate can reach.

Arm B's oracle is fitted WITH attack labels. Three of its four signals owe
their usable direction to those labels, so Arm B's 0.7448 (linear) / 0.7528
(rank) is not a ceiling Arm A can inherit. This file measures what survives
when every choice that needed labels is removed.

TWO THINGS ARE REMOVED, NOT ONE

  1. SIGN. Each signal's direction is fixed from its own design rationale
     before any data is seen, never fitted.
  2. WEIGHTS. A fitted weight vector needs labels too. The combiner here is an
     equal-weight sum of signed ranks. Any cleverer weighting is exactly the
     thing Arm A cannot do.

  3. And the rank transform itself is calibrated on the HONEST rows only
     (clone conditions), then applied to everything. Ranking against the
     pooled set -- as the D2 diagnostic did -- lets the attack distribution
     shape its own scores, which a deployed gate calibrated on clean and
     honest-duplication data could not do.

THE A PRIORI SIGNS, AND WHERE THEY COME FROM

  dependence       POSITIVE. The project's established direction: more
                   inter-view agreement, more suspicion. It is the quantity
                   the discount acts on, in that direction, since Part 04.

  clique_contrast  POSITIVE. Derived mechanistically in the module docstring
                   BEFORE this data was seen: honest duplicates agree with
                   each other AND with the rest (undifferentiated, ~0), while
                   a colluding clique agrees internally and DISSENTS from the
                   rest (clearly > 0). This is the L5-derived reasoning the
                   signal was built from.

  detector         NO DEFENSIBLE A PRIORI SIGN IN THE USEFUL DIRECTION.
                   Its design rationale (suspicion.py: "what is suspicious is
                   agreement in EXCESS of both") gives POSITIVE. On this
                   metric its univariate AUC is 0.4023 -- the a priori sign is
                   empirically BACKWARDS. The only known-correct sign is
                   negative, and that is label-derived. Asserting it a priori
                   would mean claiming "less unexplained agreement means more
                   attack", which contradicts the signal's entire rationale.

  tail             SAME, and worse. Design rationale gives POSITIVE (more
                   joint extremeness, more suspicion). Univariate AUC 0.4543,
                   again backwards. It is also a PRE-REGISTERED NEGATIVE
                   CONTROL that Part 13 found carries no signal. Its useful
                   direction is a label-derived suppressor effect.

So the honest all-four variant cannot use the label-derived signs. It is run
with the DESIGN-RATIONALE signs (all positive), which is what someone
deploying these four signals without attack labels would actually write. The
point of the variant is to show what that costs, not to endorse it.

Usage:
    python -m experiments.synthesis.run_sign_constrained
"""

from __future__ import annotations

import json
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

# Fixed before the run, from each signal's own rationale. See the docstring.
A_PRIORI_SIGN = {
    "dependence": +1.0,
    "clique_contrast": +1.0,
    "detector": +1.0,      # design rationale; empirically backwards here
    "tail": +1.0,          # design rationale; empirically backwards here
}

# Signals whose a priori sign is both defensible AND not contradicted by the
# signal's own univariate behaviour. This is the deployable set.
DEPLOYABLE = ("dependence", "clique_contrast")

VARIANTS = {
    "deployable_dep_clique": DEPLOYABLE,
    "clique_alone": ("clique_contrast",),
    "dependence_alone": ("dependence",),
    "all_four_design_signs": SIGNALS,
}


def honest_calibrated_ranks(column: np.ndarray, honest: np.ndarray) -> np.ndarray:
    """Map values through the ECDF of the HONEST rows only.

    This is the transform a deployed gate could actually build: it sees clean
    and honest-duplication data at calibration time and nothing else. Values
    above everything seen in calibration saturate at 1.0, which is the correct
    behaviour for a gate -- "more extreme than anything honest" is the most it
    can say.
    """
    reference = np.sort(column[honest])
    if len(reference) == 0:
        return np.full(len(column), 0.5)
    left = np.searchsorted(reference, column, side="left")
    right = np.searchsorted(reference, column, side="right")
    return (left + right) / (2.0 * len(reference))


def main():
    config = json.loads(json.dumps(CONFIG))
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

        conditions = {"chorus_k3": signals_for(model, views + attack.delta, view_mask)}
        for k in config["clone"]["ks"]:
            conditions[f"clone_k{k}"] = signals_for(
                model, clone_in_place(views, config["clone"]["source"], k), view_mask
            )

        features, labels = [], []
        for name, values in conditions.items():
            block = np.column_stack([values[signal] for signal in SIGNALS])
            features.append(block)
            labels.append(np.full(len(block), 1.0 if name == "chorus_k3" else 0.0))
        features = np.vstack(features)
        labels = np.concatenate(labels)
        keep = np.isfinite(features).all(axis=1)
        features, labels = features[keep], labels[keep]

        # The gate calibrates on honest rows only -- it never sees an attack.
        honest = labels == 0
        ranked = np.column_stack([
            honest_calibrated_ranks(features[:, i], honest) for i in range(len(SIGNALS))
        ])

        variant_auc = {}
        for name, subset in VARIANTS.items():
            columns = [SIGNALS.index(s) for s in subset]
            signs = np.array([A_PRIORI_SIGN[s] for s in subset])
            score = (ranked[:, columns] * signs).sum(axis=1)   # equal weights
            variant_auc[name] = roc_auc(score[labels == 1], score[labels == 0])

        # What the label-fitted versions of the same two sets reach, so the
        # cost of removing label help is readable directly.
        label_fitted = {}
        for name, subset in (("deployable_dep_clique", DEPLOYABLE), ("all_four", SIGNALS)):
            columns = [SIGNALS.index(s) for s in subset]
            w, c, s = fit_logistic(
                ranked[:, columns], labels,
                config["oracle"]["ridge"], config["oracle"]["iterations"],
            )
            scores = apply_logistic(ranked[:, columns], w, c, s)
            label_fitted[name] = roc_auc(scores[labels == 1], scores[labels == 0])

        rows.append({"seed": seed, "sign_constrained": variant_auc, "label_fitted": label_fitted})
        LOGGER.info(
            "seed=%s | dep+clique=%.4f clique_alone=%.4f all_four_design=%.4f | "
            "label-fitted dep+clique=%.4f all_four=%.4f",
            seed, variant_auc["deployable_dep_clique"], variant_auc["clique_alone"],
            variant_auc["all_four_design_signs"],
            label_fitted["deployable_dep_clique"], label_fitted["all_four"],
        )

    sign_constrained = {
        name: summarise([r["sign_constrained"][name] for r in rows]) for name in VARIANTS
    }
    label_fitted = {
        name: summarise([r["label_fitted"][name] for r in rows])
        for name in ("deployable_dep_clique", "all_four")
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sign_constrained.json").write_text(
        json.dumps({
            "config": config, "torch": torch.__version__,
            "a_priori_sign": A_PRIORI_SIGN,
            "sign_constrained_auc": sign_constrained,
            "label_fitted_auc": label_fitted,
            "per_seed": rows,
        }, indent=2, default=float),
        encoding="utf-8",
    )

    lines = [
        "# Part 14: the sign-constrained (label-free) ceiling",
        "",
        f"Seeds: {config['seeds']}. No labels used for sign, weight or rank",
        "calibration. Equal-weight sum of signed ranks, ranks calibrated on the",
        "HONEST rows only. Threshold of reference: 0.70.",
        "",
        "## What a gate without attack labels can reach",
        "",
        "| variant | signals | selectivity AUC |",
        "|---|---|---|",
    ]
    for name, subset in VARIANTS.items():
        lines.append(f"| {name} | {', '.join(subset)} | {fmt(sign_constrained[name])} |")

    lines += [
        "",
        "## The cost of removing label help",
        "",
        "| set | label-fitted (rank) | sign-constrained | gap |",
        "|---|---|---|---|",
    ]
    for key, variant in (("deployable_dep_clique", "deployable_dep_clique"),
                         ("all_four", "all_four_design_signs")):
        gap = label_fitted[key]["mean"] - sign_constrained[variant]["mean"]
        lines.append(
            f"| {key} | {fmt(label_fitted[key])} | {fmt(sign_constrained[variant])} | "
            f"{gap:+.4f} |"
        )

    lines += [
        "",
        "## Signs",
        "",
        "dependence +1 (established since Part 04), clique_contrast +1",
        "(mechanistic, from the module docstring, pre-data).",
        "",
        "detector and tail have NO defensible a priori sign in the useful",
        "direction: their design rationale gives +1, their univariate AUCs are",
        "0.4023 and 0.4543 -- backwards. Their only known-correct sign is",
        "label-derived, which Arm A cannot obtain. all_four_design_signs uses",
        "the design-rationale signs to show what that costs; it is not an",
        "endorsement of deploying them.",
        "",
    ]
    (OUT_DIR / "sign_constrained.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("sign-constrained ceiling written")


if __name__ == "__main__":
    main()
