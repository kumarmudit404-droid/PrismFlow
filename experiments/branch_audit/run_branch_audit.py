"""Part 11 follow-up: WHERE DID THE SHARED INFORMATION GO?

Part 11 found that disentanglement pushes ENIV UP (3.04 -> 3.43 at rho 0.6) on
data that genuinely is dependent, and named one leading explanation:

    "Nothing in the objective requires the shared branches to stay aligned
    ACROSS views ... a projection is free to satisfy that constraint by
    rotating each view's shared subspace independently, which destroys
    measurable cross-view structure."

THAT EXPLANATION IS TESTABLE AND THIS FILE TESTS IT -- along with the
alternative it overlooked.

H1 (the README's): per-view rotation of the shared subspace destroys the
    measured dependence.

    H1 has a problem the README did not notice. Dependence is measured with
    `method="cca"`, and `prismflow/statistics/dependence.py` states plainly
    that CCA is "invariant to any invertible linear remapping of either side".
    A rotation IS an invertible linear remapping. If the estimator cannot see a
    rotation, rotation cannot be the cause. The ROTATION PROBE below applies an
    independent random orthogonal matrix to each view's shared branch and
    re-measures. If ENIV barely moves, H1 is dead.

H2 (the alternative): the branches SWAPPED ROLES.

    The penalty only requires shared_v to be statistically independent of
    private_v. Nothing whatsoever requires the branch NAMED "shared" to be the
    one carrying the across-view information. A model can satisfy the penalty
    perfectly by routing the common latent into `private` and the view-specific
    noise into `shared` -- the two are still independent within the view, the
    loss is happy, and dependence measured on `shared` collapses. ENIV then
    rises for a measurement reason, not because redundancy was removed.

    H2 predicts: dependence(private) > dependence(shared), and dependence on
    concat(shared, private) stays close to V1's, because no information was
    destroyed -- only moved to the branch nobody is reading.

WHAT IS MEASURED. For every trained model, on the test split, the SAME
estimator (cca / pairwise_holdout / 4 null permutations -- the model's own
forward settings) is run over four representations:

    shared          what Part 11 measured, and what its ENIV came from
    private         the branch nobody looked at
    both            concat(shared, private): the full representation
    shared_rotated  shared, with an independent random orthogonal rotation per
                    view -- the H1 probe

V1 is measured on its encoder features, as its own forward path does, and is
the reference point for "what the dependence actually is".

NOTHING HERE IS A FIX. No frozen module is modified. Part 11's modules are
imported and used exactly as committed; this file only reads representations
that were always there and applies an estimator to them.

Usage:
    python -m experiments.branch_audit.run_branch_audit
    python -m experiments.branch_audit.run_branch_audit --quick   # NOT evidence
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.disentangle.run_disentangle import (
    CONFIG as DISENTANGLE_CONFIG,
    build_dataset,
    fmt,
    json_safe,
    make_model,
    summarise,
    train_system,
)
from prismflow.data.loaders import iter_batches
from prismflow.eniv.eniv import compute_eniv
from prismflow.models.prismflow import FORWARD_NULL_PERMUTATIONS
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger(__name__)

OUT_DIR = Path("results/branch_audit")

# The degenerate lambda_2 >= 10 runs are excluded on purpose: Part 11 showed
# they collapse to chance accuracy and near-zero confidence, so "where their
# information went" is not a meaningful question -- there is no information.
SYSTEMS = ("v1", "v2_lambda0", "v2_hsic_l1", "v2_orthogonality")
REPRESENTATIONS = ("shared", "private", "both", "shared_rotated")

CONFIG = {
    "experiment": {"id": "branch_audit"},
    "seeds": DISENTANGLE_CONFIG["seeds"],
    "rhos": DISENTANGLE_CONFIG["rhos"],
    "data": DISENTANGLE_CONFIG["data"],
    "training": DISENTANGLE_CONFIG["training"],
    "v2": DISENTANGLE_CONFIG["v2"],
    "evaluation": {"split": "test", "batch_size": 256},
    # The model's own forward-path settings, so the audit reproduces the number
    # Part 11 reported rather than measuring a different quantity.
    "dependence": {
        "method": "cca",
        "conditioning": "pairwise_holdout",
        "null_permutations": FORWARD_NULL_PERMUTATIONS,
        "seed": 0,
    },
}


def random_rotation(dim: int, generator: torch.Generator, dtype) -> torch.Tensor:
    """A Haar-ish random orthogonal matrix via QR of a Gaussian.

    The sign fix on R's diagonal is what makes the distribution proper rather
    than QR-implementation-dependent; without it this is not a uniform draw.
    """
    gaussian = torch.randn(dim, dim, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


@torch.no_grad()
def collect_representations(model, dataset, indices, batch_size):
    """Run the model over the split and keep the branches, evidence and mask.

    Returns None for `shared`/`private` on V1, which has no split -- V1's
    reference representation is its encoder features.
    """
    shared, private, features, evidence, masks = [], [], [], [], []
    for batch in iter_batches(dataset, indices, batch_size):
        output = model(batch.views, batch.view_mask)
        evidence.append(output.per_view_evidence)
        masks.append(output.view_mask)
        if getattr(output, "shared", None) is not None:
            shared.append(output.shared)
            private.append(output.private)
        else:
            features.append(model.encoder(batch.views, batch.view_mask))

    out = {
        "evidence": torch.cat(evidence),
        "view_mask": torch.cat(masks),
        "shared": torch.cat(shared) if shared else None,
        "private": torch.cat(private) if private else None,
        "features": torch.cat(features) if features else None,
    }
    return out


def measure(representation, evidence, view_mask, dep_cfg, n_views):
    """Dependence matrix + ENIV on one representation, V1's estimator settings."""
    matrix = feature_dependence_matrix(
        representation,
        evidence,
        view_mask,
        method=dep_cfg["method"],
        conditioning=dep_cfg["conditioning"],
        seed=dep_cfg["seed"],
        null_permutations=dep_cfg["null_permutations"],
    )
    present = available_views(view_mask, n_views=n_views)
    eniv = compute_eniv(matrix, present)

    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    return {
        # effective_views is the ENIV the model reports; mean_dependence comes
        # from ENIVResult so it matches Part 11's diagnostic exactly.
        "eniv": float(eniv.effective_views),
        "mean_dependence": float(eniv.mean_dependence),
        "max_dependence": float(finite.max()) if finite.size else float("nan"),
    }


def audit_model(system, model, dataset, splits, dep_cfg, batch_size, seed):
    """Measure every representation for one trained model."""
    reps = collect_representations(model, dataset, splits["test"], batch_size)
    evidence, view_mask = reps["evidence"], reps["view_mask"]
    n_views = view_mask.shape[1]

    if reps["shared"] is None:
        # V1: no branches. Its features are the reference.
        row = measure(reps["features"], evidence, view_mask, dep_cfg, n_views)
        return {"features": row}

    shared, private = reps["shared"], reps["private"]

    generator = torch.Generator().manual_seed(10_000 + seed)
    rotated = torch.stack(
        [
            shared[:, v, :] @ random_rotation(shared.shape[-1], generator, shared.dtype)
            for v in range(n_views)
        ],
        dim=1,
    )

    sources = {
        "shared": shared,
        "private": private,
        "both": torch.cat([shared, private], dim=-1),
        "shared_rotated": rotated,
    }
    return {
        name: measure(tensor, evidence, view_mask, dep_cfg, n_views)
        for name, tensor in sources.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, NOT evidence")
    args = parser.parse_args()

    config = json.loads(json.dumps(CONFIG))
    if args.quick:
        config["seeds"] = config["seeds"][:1]
        config["rhos"] = config["rhos"][:1]
        config["training"] = dict(config["training"], epochs=2)
        LOGGER.warning("--quick: 1 seed, 2 epochs. NOT EVIDENCE.")

    dep_cfg = config["dependence"]
    batch_size = config["evaluation"]["batch_size"]

    per_seed: dict[str, list[dict]] = {}
    for rho in config["rhos"]:
        for system in SYSTEMS:
            for seed in config["seeds"]:
                set_seed(seed)
                dataset, splits, base = build_dataset(seed, rho, config["data"])
                model = make_model(system, base, config["v2"])
                train_system(system, model, dataset, splits, base, config["training"], seed)

                rows = audit_model(
                    system, model, dataset, splits, dep_cfg, batch_size, seed
                )
                key = f"rho{rho}_{system}"
                for rep, values in rows.items():
                    per_seed.setdefault(f"{key}|{rep}", []).append({"seed": seed, **values})
                LOGGER.info(
                    "rho=%s %s seed=%s -> %s",
                    rho, system, seed,
                    {r: round(v["eniv"], 3) for r, v in rows.items()},
                )

    metrics = ("eniv", "mean_dependence", "max_dependence")
    protocol = {k: summarise(v, metrics) for k, v in per_seed.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "torch": torch.__version__,
        "protocol": json_safe(protocol),
        "per_seed": json_safe(per_seed),
    }
    (OUT_DIR / "branch_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Where did the shared information go? (Part 11 follow-up)",
        "",
        f"Seeds: {config['seeds']}. Mean +/- sample std across seeds.",
        "",
        "Estimator is the model's own: "
        f"{dep_cfg['method']} / {dep_cfg['conditioning']} / "
        f"{dep_cfg['null_permutations']} null permutations.",
        "",
    ]
    for metric in metrics:
        lines += [
            f"## {metric}",
            "",
            "| rho | system | " + " | ".join(REPRESENTATIONS) + " |",
            "|---|---|" + "---|" * len(REPRESENTATIONS),
        ]
        for rho in config["rhos"]:
            for system in SYSTEMS:
                key = f"rho{rho}_{system}"
                if system == "v1":
                    entry = protocol.get(f"{key}|features")
                    cells = [f"{fmt(entry[metric])} (features)"] + ["n/a"] * (
                        len(REPRESENTATIONS) - 1
                    )
                else:
                    cells = [
                        fmt(protocol[f"{key}|{rep}"][metric])
                        if f"{key}|{rep}" in protocol else "n/a"
                        for rep in REPRESENTATIONS
                    ]
                lines.append(f"| {rho} | {system} | " + " | ".join(cells) + " |")
        lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
