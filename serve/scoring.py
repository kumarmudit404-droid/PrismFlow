"""Scoring: the frozen library, called. Nothing here is a reimplementation.

WHAT THIS CAN AND CANNOT SAY
----------------------------
Every number below comes from `prismflow`: `feature_dependence_matrix`,
`compute_eniv`, `soft_cluster_alpha`, `evidence_discount`, `fuse_opinions`
and `suspicion_report`. This module arranges inputs and serialises outputs.
If a quantity is not in that list, it is not reported.

NO MODEL, SO NO PREDICTED CLASS. The dependence estimator normally
stratifies by the class the model predicts, so that within-class correlation
is what gets measured. A stranger's upload comes with no trained model and
no labels, so there is nothing to stratify by: dependence here is measured
in ONE stratum, unconditionally. That is a weaker quantity than the one the
experiments report, and `conditioning` in the response says so on every
call rather than leaving the reader to assume otherwise.

NO EVIDENCE, NO FUSION. The discount, the fused opinion and the suspicion
score are all functions of evidence, which features alone do not contain.
When an upload carries no `evidence` array those three fields are null with
a stated reason -- they are never synthesised from the features, because an
invented evidence head would make the fused confidence a property of this
file rather than of the user's data.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from prismflow.eniv.discount import evidence_discount
from prismflow.eniv.eniv import compute_eniv, soft_cluster_alpha
from prismflow.models.evidence import evidence_to_opinion
from prismflow.models.fusion import fuse_opinions
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.statistics.suspicion import suspicion_report

# Stated in the response so a result can be reproduced, and hashed so two
# responses that claim the same configuration provably had one.
SCORING_CONFIG = {
    "dependence_method": "pearson",
    "dependence_conditioning": "global",
    "conditioning_strata": 1,
    "null_permutations": 0,
    "eniv_methods": ["eigen", "design_effect"],
    "suspicion_permutations": 200,
}

# The three sentences that travel with every result. They are literals, not
# computed, because they are true of every call regardless of what was
# uploaded -- and a caveat that could be switched off by a good-looking
# number is not a caveat.
CALIBRATION_NOTE = "uncalibrated on your data"
DEFENCE_NOTE = "not validated - see KNOWN_LIMITATIONS L3, L4, L5, L8"
ESTIMATOR_VALIDATION_NOTE = (
    "ENIV is validated on synthetic data only - see KNOWN_LIMITATIONS L9"
)


def config_hash(seed: int) -> str:
    payload = json.dumps({"config": SCORING_CONFIG, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def jsonable(value):
    """NaN and infinity become null. Never 0.

    A NaN in this matrix means a pair had no measurable overlap. Serialising
    it as 0.0 would say the opposite -- "measured, and independent" -- which
    is the single most expensive lie this service could tell, because 0 is
    exactly the value a reader would take as good news.
    """
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def score(gated, seed: int = 0) -> dict:
    """Dependence, ENIV, per-view alpha and -- given evidence -- fusion.

    `gated` is a `serve.gate.GatedInput`; it has already been validated, and
    nothing here re-checks it or repairs it.
    """
    features = gated.features
    n_samples, n_views, _ = features.shape
    mask = gated.view_mask

    # One stratum: see the module docstring. Passing `classes` explicitly is
    # what makes it unconditional -- the estimator only derives classes when
    # none are given.
    strata = np.zeros(n_samples, dtype=np.int64)
    dependence = feature_dependence_matrix(
        features,
        features,
        view_mask=mask,
        classes=strata,
        method=SCORING_CONFIG["dependence_method"],
        conditioning=SCORING_CONFIG["dependence_conditioning"],
        seed=seed,
        null_permutations=SCORING_CONFIG["null_permutations"],
    )

    available = available_views(mask, n_views=n_views)
    eigen = compute_eniv(dependence, available=available, method="eigen")
    design = compute_eniv(dependence, available=available, method="design_effect")
    alpha = soft_cluster_alpha(dependence, available=available)

    result = {
        "shape": {"samples": n_samples, "views": n_views,
                  "features_per_view": int(features.shape[2])},
        "available_views": jsonable(available),
        "dependence_matrix": jsonable(dependence),
        "eniv": {
            "eigen": {
                "nominal_views": eigen.nominal_views,
                "effective_views": jsonable(eigen.effective_views),
                "efficiency_ratio": jsonable(eigen.efficiency_ratio),
                "mean_dependence": jsonable(eigen.mean_dependence),
            },
            "design_effect": {
                "nominal_views": design.nominal_views,
                "effective_views": jsonable(design.effective_views),
                "efficiency_ratio": jsonable(design.efficiency_ratio),
                "mean_dependence": jsonable(design.mean_dependence),
            },
        },
        "per_view_alpha": jsonable(alpha),
        "conditioning": (
            "ONE stratum, unconditional. No model and no labels were "
            "supplied, so there is no predicted class to condition on. The "
            "experiments in this project condition on the predicted class; "
            "this number does not, and is the weaker quantity."
        ),
        "seed": seed,
        "config": SCORING_CONFIG,
        "config_hash": config_hash(seed),
        "calibration": CALIBRATION_NOTE,
        "defence": DEFENCE_NOTE,
        "estimator_validation": ESTIMATOR_VALIDATION_NOTE,
    }

    if gated.evidence is None:
        absent = (
            "not measured - the upload carried no `evidence` array, and "
            "evidence is not derived from features here. Add an `evidence` "
            "[samples, views, classes] array to an .npz to get this."
        )
        result["fused"] = None
        result["fused_absent_reason"] = absent
        result["suspicion"] = None
        result["suspicion_absent_reason"] = absent
        return result

    evidence = torch.as_tensor(gated.evidence, dtype=torch.float64)
    mask_t = None if mask is None else torch.as_tensor(mask, dtype=torch.bool)

    discounted = evidence_discount(evidence, np.asarray(alpha, dtype=np.float64))
    belief, uncertainty = evidence_to_opinion(discounted)
    fused_belief, fused_uncertainty = fuse_opinions(belief, uncertainty, mask_t)

    report = suspicion_report(
        belief,
        dependence,
        view_mask=None if mask is None else mask,
        evidence=gated.evidence,
        seed=seed,
        n_permutations=SCORING_CONFIG["suspicion_permutations"],
    )

    result["fused"] = {
        "mean_belief_per_class": jsonable(
            fused_belief.mean(dim=0).detach().cpu().numpy()),
        "mean_uncertainty": jsonable(
            float(fused_uncertainty.mean().detach().cpu())),
        "n_classes": int(fused_belief.shape[-1]),
    }
    result["suspicion"] = {
        "mean_score": jsonable(float(np.nanmean(report.score))),
        "max_score": jsonable(float(np.nanmax(report.score))),
        "flagged_fraction": jsonable(float(np.mean(report.flag))),
        "threshold": jsonable(float(report.threshold))
        if hasattr(report, "threshold") else None,
        "note": (
            "Unexplained agreement, in cosine units. This detector fires on "
            "honest duplication as readily as on collusion - see "
            "KNOWN_LIMITATIONS L5."
        ),
    }
    return result
