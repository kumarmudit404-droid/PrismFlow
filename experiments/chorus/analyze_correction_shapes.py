"""L4 follow-up: would a different CORRECTION SHAPE bind where the proportional
one does not?

READ-ONLY ANALYSIS. Nothing here is wired into the model. `discount.py`,
`eniv.py`, `fusion.py` and `prismflow.py` are called, never modified. The
alternative corrections are applied post hoc to the raw evidence a saved,
already-executed attack produced.

METHOD. The Part 09 models are reconstructed by deterministic retraining and the
SAVED perturbation tensors in `results/chorus/perturbations/` are replayed --
the same method validated in `experiments/chorus/README.md` section 6. The
replay's per-seed success rates are checked against
`results/chorus/attack_metrics.json` before any new number is reported, and the
baseline belief figures are checked against section 6's table. If either check
fails the run aborts rather than print unvalidated numbers.

THE SHAPES

  current        alpha from `soft_cluster_alpha`, applied by `evidence_discount`.
                 What the model does today.
  capped         current, then every per-view belief clamped to <= CAP, with the
                 removed mass returned to vacuity. A BLANKET rule: it is applied
                 to all views, because a correction cannot know which views are
                 compromised. Honest views sit below the cap and are untouched,
                 which is why it can bind asymmetrically.
  floor_alpha    alpha lower-bounded at 0.25, per the brief's wording.
  alpha_ceiling  alpha UPPER-bounded at 0.25. See the note below: on this data
                 the floor reading is a no-op, and this is the reading that
                 tests what L4 actually proposed.
  per_sample     alpha recomputed per sample from a per-sample redundancy proxy
                 instead of the batch-level dependence matrix.

NOTE ON "min alpha = 0.25". The measured alphas on the compromised views are
0.5021 (eps 1.0) and 0.4331 (eps 2.0), both already above 0.25, so a LOWER bound
at 0.25 cannot change any number on this data -- it is a no-op by construction
and is reported as such. L4's sentence "alpha near 0.25 with a floor" meant
driving alpha DOWN to about 0.25, which is an upper bound on alpha. Both
readings are computed so the ambiguity is visible rather than silently resolved.

NOTE ON "per-sample dependence". Dependence is a distributional quantity: it is
a statistic ACROSS samples, and there is no such thing as the correlation of a
single sample. So a genuine per-sample dependence estimate does not exist. What
is used here is a per-sample REDUNDANCY PROXY -- the cosine between two views'
belief vectors on that sample, clipped to [0, 1] -- fed to the same
`soft_cluster_alpha`. This is the statistic Part 10's detector uses. Treat the
per_sample row as "what a per-sample agreement-driven correction would do", not
as "per-sample dependence", which is not measurable.

Usage:
    python -m experiments.chorus.analyze_correction_shapes
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.calibration.run_calibration import build_dataset, train
from prismflow.attacks.chorus import ChorusConfig, choose_targets, select_views
from prismflow.data.dataset import Batch
from prismflow.data.loaders import iter_batches
from prismflow.eniv.discount import evidence_discount
from prismflow.eniv.eniv import soft_cluster_alpha
from prismflow.models.evidence import evidence_to_opinion, opinion_to_probs
from prismflow.models.fusion import fuse_opinions
from prismflow.utils.logging import get_logger

CONFIG_PATH = Path(__file__).with_name("config.yaml")
RESULTS = Path("results/chorus")
CELLS = ("chorus_k2_eps1.0", "chorus_k2_eps2.0")

CAP = 0.3
ALPHA_BOUND = 0.25

# From experiments/chorus/README.md section 6, used as a regression check on the
# reconstruction: (compromised -> target, honest -> true) on successful attacks.
SECTION_6 = {
    "chorus_k2_eps1.0": (0.5553, 0.1668),
    "chorus_k2_eps2.0": (0.7301, 0.2130),
}

SHAPES = ("current", "capped", "floor_alpha", "alpha_ceiling", "per_sample")

# ORACLE SWEEP. Each of these is an alpha applied ONLY to the compromised views,
# using ground-truth knowledge of which views those are. That is not a defence --
# no deployable mechanism knows this -- it is an upper bound, to separate two
# questions: is the correction's SHAPE the binding constraint, or its inability
# to tell compromised views from honest ones? Nothing selective is achievable
# without detection, and Part 10 measured detection at chance for k=3.
ORACLE_ALPHAS = (0.25, 0.1, 0.05, 0.01)


def per_sample_alpha(belief: torch.Tensor) -> torch.Tensor:
    """[B, V] alpha from a per-sample redundancy proxy (see module docstring)."""
    normed = belief / belief.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = torch.einsum("bik,bjk->bij", normed, normed).clamp(0.0, 1.0).cpu().numpy()
    return torch.tensor(
        np.stack([soft_cluster_alpha(matrix) for matrix in cosine]),
        dtype=belief.dtype,
        device=belief.device,
    )


def apply_shape(shape: str, evidence: torch.Tensor, alpha_view: torch.Tensor, raw_belief):
    """Return (belief [B, V, K], uncertainty [B, V]) under one correction shape."""
    if shape == "current":
        return evidence_to_opinion(evidence_discount(evidence, alpha_view))
    if shape == "capped":
        belief, _ = evidence_to_opinion(evidence_discount(evidence, alpha_view))
        capped = belief.clamp(max=CAP)
        # Mass removed from belief returns to vacuity, keeping the simplex exact.
        return capped, 1.0 - capped.sum(dim=-1)
    if shape == "floor_alpha":
        return evidence_to_opinion(evidence_discount(evidence, alpha_view.clamp_min(ALPHA_BOUND)))
    if shape == "alpha_ceiling":
        return evidence_to_opinion(evidence_discount(evidence, alpha_view.clamp_max(ALPHA_BOUND)))
    if shape == "per_sample":
        return evidence_to_opinion(evidence_discount(evidence, per_sample_alpha(raw_belief)))
    raise ValueError(f"unknown shape {shape!r}")


def mean_over(belief: torch.Tensor, views, index: torch.Tensor) -> torch.Tensor:
    """Mean over `views` of each sample's belief in class `index[sample]`."""
    if not views:
        return torch.full((belief.shape[0],), float("nan"))
    chosen = belief[:, list(views), :]
    return chosen.gather(-1, index.view(-1, 1, 1).expand(-1, chosen.shape[1], 1)).squeeze(-1).mean(dim=1)


def summarise(values) -> tuple[float, float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan"), float("nan")
    return statistics.mean(finite), (statistics.stdev(finite) if len(finite) > 1 else float("nan"))


def fmt(pair) -> str:
    mean, std = pair
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.4f}" if math.isnan(std) else f"{mean:.4f} +/- {std:.4f}"


def main():
    logger = get_logger("prismflow.correction_shapes")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    saved = json.loads((RESULTS / "attack_metrics.json").read_text())
    seeds = config["seeds"]
    cells = {cell["name"]: cell for cell in config["cells"]}

    tables = {}
    for cell_name in CELLS:
        cell = cells[cell_name]
        shapes = list(SHAPES) + [f"oracle_a{a}" for a in ORACLE_ALPHAS]
        rows = {shape: {"succ_target": [], "succ_honest": [], "fail_target": [],
                        "fail_honest": [], "success": []} for shape in shapes}
        replayed = []

        for seed in seeds:
            dataset, splits = build_dataset(seed, config["data"])
            model = train(dataset, splits, seed, True, config["data"], config["training"])
            model.use_discount = True
            model.eval()

            attack_config = ChorusConfig(
                k=cell["k"], epsilon=cell["epsilon"], beta=cell["beta"],
                steps=config["attack"]["steps"],
                target_strategy=config["attack"]["target_strategy"],
                fixed_target=config["attack"]["fixed_target"], seed=seed,
            )
            compromised = list(select_views(model.n_views, attack_config))
            honest = [v for v in range(model.n_views) if v not in compromised]

            delta_all = torch.as_tensor(
                np.load(RESULTS / "perturbations" / f"{cell_name}_prismflow_seed{seed}.npz")["delta"]
            )

            totals = {shape: {"succ_t": 0.0, "succ_h": 0.0, "fail_t": 0.0, "fail_h": 0.0,
                              "success": 0} for shape in shapes}
            n_total = n_success = 0
            offset = 0

            with torch.no_grad():
                for batch in iter_batches(dataset, splits["test"], config["evaluation"]["batch_size"]):
                    n = len(batch)
                    delta = delta_all[offset:offset + n]
                    offset += n
                    target = choose_targets(model, batch.views, batch.view_mask, attack_config)
                    perturbed = Batch(
                        views=batch.views + delta, view_mask=batch.view_mask.clone(),
                        labels=batch.labels.clone(), sample_ids=batch.sample_ids.clone(),
                    )

                    output = model(perturbed.views, perturbed.view_mask)
                    evidence = output.per_view_evidence
                    alpha_view = torch.tensor(output.eniv.per_view_alpha, dtype=evidence.dtype)
                    raw_belief, _ = evidence_to_opinion(evidence)

                    # The sample set is FIXED by the model as it stands today, so
                    # every shape is scored on the same samples section 6 used.
                    success = output.prediction == target
                    n_total += n
                    n_success += int(success.sum())

                    for shape in shapes:
                        if shape.startswith("oracle_a"):
                            # Ground-truth selectivity: only the compromised
                            # views are pushed down. Upper bound, not a defence.
                            oracle = alpha_view.clone()
                            oracle[compromised] = float(shape.removeprefix("oracle_a"))
                            belief, uncertainty = evidence_to_opinion(
                                evidence_discount(evidence, oracle)
                            )
                        else:
                            belief, uncertainty = apply_shape(shape, evidence, alpha_view, raw_belief)
                        on_target = mean_over(belief, compromised, target)
                        on_true = mean_over(belief, honest, batch.labels)
                        totals[shape]["succ_t"] += float(on_target[success].sum())
                        totals[shape]["succ_h"] += float(on_true[success].sum())
                        totals[shape]["fail_t"] += float(on_target[~success].sum())
                        totals[shape]["fail_h"] += float(on_true[~success].sum())

                        # Bonus: would the attack still land under this shape?
                        fused_belief, fused_uncertainty = fuse_opinions(
                            belief, uncertainty, perturbed.view_mask
                        )
                        prediction = opinion_to_probs(fused_belief, fused_uncertainty).argmax(dim=-1)
                        totals[shape]["success"] += int((prediction == target).sum())

            replayed.append(n_success / n_total)
            for shape in shapes:
                block = totals[shape]
                rows[shape]["succ_target"].append(block["succ_t"] / n_success if n_success else float("nan"))
                rows[shape]["succ_honest"].append(block["succ_h"] / n_success if n_success else float("nan"))
                failures = n_total - n_success
                rows[shape]["fail_target"].append(block["fail_t"] / failures if failures else float("nan"))
                rows[shape]["fail_honest"].append(block["fail_h"] / failures if failures else float("nan"))
                rows[shape]["success"].append(block["success"] / n_total)

        # --- validation gates -------------------------------------------------
        reference = [row["success_rate"] for row in saved["per_seed"][f"{cell_name}|prismflow"]]
        if not all(abs(a - b) < 1e-9 for a, b in zip(replayed, reference)):
            raise RuntimeError(
                f"{cell_name}: replay does not match saved success rates "
                f"({replayed} vs {reference}); refusing to report derived numbers"
            )
        expected_target, expected_honest = SECTION_6[cell_name]
        got_target = summarise(rows["current"]["succ_target"])[0]
        got_honest = summarise(rows["current"]["succ_honest"])[0]
        if abs(got_target - expected_target) > 5e-4 or abs(got_honest - expected_honest) > 5e-4:
            raise RuntimeError(
                f"{cell_name}: baseline does not reproduce README section 6 "
                f"({got_target:.4f}/{got_honest:.4f} vs {expected_target}/{expected_honest})"
            )
        logger.info("%s: replay and section-6 baseline both reproduce exactly", cell_name)
        tables[cell_name] = rows

    # --- report ---------------------------------------------------------------
    lines = []
    for cell_name in CELLS:
        rows = tables[cell_name]
        lines += [
            "",
            f"### {cell_name}",
            "",
            "| shape | compromised -> target | honest -> true | ratio | margin inverted? | attack success |",
            "|---|---|---|---|---|---|",
        ]
        for shape in rows:
            target = summarise(rows[shape]["succ_target"])
            honest = summarise(rows[shape]["succ_honest"])
            ratio = target[0] / honest[0] if honest[0] else float("nan")
            inverted = "**YES**" if target[0] < honest[0] else "no"
            lines.append(
                f"| {shape} | {fmt(target)} | {fmt(honest)} | {ratio:.2f}x | {inverted} | "
                f"{fmt(summarise(rows[shape]['success']))} |"
            )
        lines += [
            "",
            "Failed-attack samples, same shapes:",
            "",
            "| shape | compromised -> target | honest -> true |",
            "|---|---|---|",
        ]
        for shape in rows:
            lines.append(
                f"| {shape} | {fmt(summarise(rows[shape]['fail_target']))} | "
                f"{fmt(summarise(rows[shape]['fail_honest']))} |"
            )

    print("\n".join(lines))
    (RESULTS / "correction_shapes.json").write_text(
        json.dumps(
            {
                "cap": CAP,
                "alpha_bound": ALPHA_BOUND,
                "seeds": seeds,
                "cells": {c: tables[c] for c in CELLS},
                "note": "read-only post-hoc analysis; nothing wired into the model",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
