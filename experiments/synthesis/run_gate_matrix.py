"""Part 14: the deployable gate across attack x stress conditions.

A NEW matrix, designed here. It is NOT the Part 10 comparison matrix (5
systems x 8 conditions, deferred, and metric'd on accuracy/ECE/Brier/ENIV),
and it is not "the 30-cell matrix" -- no such design exists in this project.
This one evaluates ONE object: the deployable two-signal gate.

THE GATE UNDER TEST, fixed and never refitted per condition:

    score = rank(dependence) + rank(clique_contrast)

signs (+1, +1) fixed a priori, equal weights, ranks calibrated on the HONEST
rows of each cell only. Measured at 0.6743 +/- 0.0492 on chorus k=3 / eps 2.0
(results/synthesis/sign_constrained.json). Nothing here is allowed to refit
it: a gate that is retuned per condition is not a gate, it is a curve fit.

16 cells: 4 attacks x 4 stress levels.

    attacks  chorus k=1 | chorus k=2 | chorus k=3 | PGD k=2
    stress   none | missing 30% | missing 50% | noisy 1 view

PGD is included because Part 09 showed it is the attack the dependence
signal cannot see, and the Part 10 scope memory requires it in any matrix
this project builds. It is the cell most likely to embarrass the gate, which
is why it is in rather than out.

Negatives in every cell are clean + clone_k2 + clone_k3 UNDER THE SAME
STRESS, so a cell compares like with like: the question is never "can the
gate tell an attack from a clean sample", it is "can it tell an attack from
HONEST redundancy", which is the failure L5 recorded.

Clones are made IN PLACE (run_oracle.clone_in_place), not by the Part 06
`duplicate_view` which appends and grows V. Matched view count is required
for clique contrast to be comparable across conditions.

THREE NUMBERS PER CELL, and the third is not optional:

    gate AUC        selectivity, on DEFINED rows only
    undefined rate  its OWN column, never folded into "dropped rows"
    oracle bound    four signals, label-fitted, rank -- the upper bound the
                    gate is NOT expected to reach, carried so the
                    oracle-vs-deployable gap is visible in every cell

The 30%-missing cell has a pre-registered verdict attached before this ran:
see experiments/synthesis/PREREGISTRATION_missing_views.md (commit 1fa074a).

Usage:
    python -m experiments.synthesis.run_gate_matrix
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.synthesis.run_oracle import (
    CONFIG,
    SIGNALS,
    apply_logistic,
    clone_in_place,
    fit_logistic,
    signals_for,
    summarise,
)
from experiments.synthesis.run_sign_constrained import (
    A_PRIORI_SIGN,
    DEPLOYABLE,
    honest_calibrated_ranks,
)
from experiments.tail.run_tail import build_dataset, roc_auc, train_model, whole_split
from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.data.corruption import add_noise, drop_views
from prismflow.data.dataset import Batch
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger(__name__)
OUT_DIR = Path("results/synthesis")

STRESS = [
    {"name": "none", "kind": "none"},
    {"name": "missing_30", "kind": "missing", "rate": 0.3},
    {"name": "missing_50", "kind": "missing", "rate": 0.5},
    {"name": "noisy_1", "kind": "noisy", "sigma": 2.0},
]

ATTACKS = [
    {"name": "chorus_k1", "kind": "chorus", "k": 1, "epsilon": 2.0},
    {"name": "chorus_k2", "kind": "chorus", "k": 2, "epsilon": 2.0},
    {"name": "chorus_k3", "kind": "chorus", "k": 3, "epsilon": 2.0},
    {"name": "pgd_k2", "kind": "pgd", "k": 2, "epsilon": 2.0},
]

PREREG = {"severe_undefined": 0.15, "material_undefined": 0.05, "severe_auc": 0.60}


def apply_stress(views, view_mask, labels, stress, seed, name):
    if stress["kind"] == "none":
        return views, view_mask
    corruption_seed = zlib.crc32(f"{seed}|{name}".encode()) % (2**31)
    batch = Batch(
        views=views, view_mask=view_mask, labels=labels,
        sample_ids=torch.arange(len(views)),
    )
    if stress["kind"] == "missing":
        out = drop_views(batch, rate=stress["rate"], seed=corruption_seed)
    elif stress["kind"] == "noisy":
        out = add_noise(batch, view_idx=0, sigma=stress["sigma"], seed=corruption_seed)
    else:
        raise ValueError(f"unknown stress {stress['kind']!r}")
    return out.views, out.view_mask


def run_attack(model, views, view_mask, attack, seed):
    config = ChorusConfig(
        k=attack["k"], epsilon=attack["epsilon"], beta=1.0,
        steps=CONFIG["attack"]["steps"], seed=seed,
    )
    run = chorus_attack if attack["kind"] == "chorus" else pgd_attack
    return run(model, views, view_mask, config).delta


def stack_signals(values) -> np.ndarray:
    return np.column_stack([values[name] for name in SIGNALS])


def main():
    config = json.loads(json.dumps(CONFIG))
    rows = []

    for seed in config["seeds"]:
        set_seed(seed)
        dataset, splits, base = build_dataset(seed, config["data"])
        model = train_model(dataset, splits, base, config["training"], seed)
        views, view_mask, labels = whole_split(
            dataset, splits["test"], config["evaluation"]["batch_size"]
        )

        for stress in STRESS:
            s_views, s_mask = apply_stress(
                views, view_mask, labels, stress, seed, stress["name"]
            )

            # Honest negatives, under the same stress as the attack they face.
            negative_blocks = [stack_signals(signals_for(model, s_views, s_mask))]
            for k in config["clone"]["ks"]:
                negative_blocks.append(stack_signals(signals_for(
                    model, clone_in_place(s_views, config["clone"]["source"], k), s_mask
                )))
            negative = np.vstack(negative_blocks)

            for attack in ATTACKS:
                delta = run_attack(model, s_views, s_mask, attack, seed)
                positive = stack_signals(signals_for(model, s_views + delta, s_mask))

                features = np.vstack([positive, negative])
                cell_labels = np.concatenate([
                    np.ones(len(positive)), np.zeros(len(negative))
                ])

                # UNDEFINED RATE, measured before anything is dropped.
                defined = np.isfinite(features).all(axis=1)
                undefined_rate = float((~defined).mean())
                clique_undefined = float(
                    (~np.isfinite(features[:, SIGNALS.index("clique_contrast")])).mean()
                )

                kept, kept_labels = features[defined], cell_labels[defined]
                if len(np.unique(kept_labels)) < 2:
                    LOGGER.warning(
                        "seed=%s %s/%s: one class entirely undefined, cell unscorable",
                        seed, stress["name"], attack["name"],
                    )
                    gate_auc = float("nan")
                    oracle_auc = float("nan")
                else:
                    honest = kept_labels == 0
                    ranked = np.column_stack([
                        honest_calibrated_ranks(kept[:, i], honest)
                        for i in range(len(SIGNALS))
                    ])

                    columns = [SIGNALS.index(s) for s in DEPLOYABLE]
                    signs = np.array([A_PRIORI_SIGN[s] for s in DEPLOYABLE])
                    score = (ranked[:, columns] * signs).sum(axis=1)
                    gate_auc = roc_auc(score[kept_labels == 1], score[kept_labels == 0])

                    w, c, s = fit_logistic(
                        ranked, kept_labels,
                        config["oracle"]["ridge"], config["oracle"]["iterations"],
                    )
                    oracle_scores = apply_logistic(ranked, w, c, s)
                    oracle_auc = roc_auc(
                        oracle_scores[kept_labels == 1], oracle_scores[kept_labels == 0]
                    )

                rows.append({
                    "seed": seed,
                    "stress": stress["name"],
                    "attack": attack["name"],
                    "gate_auc": gate_auc,
                    "oracle_auc": oracle_auc,
                    "undefined_rate": undefined_rate,
                    "clique_undefined_rate": clique_undefined,
                    "n_positive": int(len(positive)),
                    "n_negative": int(len(negative)),
                })

            LOGGER.info(
                "seed=%s stress=%s done (undefined %.3f)",
                seed, stress["name"], rows[-1]["undefined_rate"],
            )

    # ---- aggregate to cells ----
    cells = {}
    for stress in STRESS:
        for attack in ATTACKS:
            matching = [
                r for r in rows
                if r["stress"] == stress["name"] and r["attack"] == attack["name"]
            ]
            cells[f"{stress['name']}|{attack['name']}"] = {
                "stress": stress["name"],
                "attack": attack["name"],
                "gate_auc": summarise([r["gate_auc"] for r in matching]),
                "oracle_auc": summarise([r["oracle_auc"] for r in matching]),
                "undefined_rate": summarise([r["undefined_rate"] for r in matching]),
                "clique_undefined_rate": summarise(
                    [r["clique_undefined_rate"] for r in matching]
                ),
            }

    # ---- the pre-registered verdict, applied mechanically ----
    missing30 = [c for key, c in cells.items() if c["stress"] == "missing_30"]
    worst_undefined = max(c["undefined_rate"]["mean"] for c in missing30)
    worst_auc = min(
        c["gate_auc"]["mean"] for c in missing30
        if not np.isnan(c["gate_auc"]["mean"])
    )
    if worst_undefined > PREREG["severe_undefined"] or worst_auc < PREREG["severe_auc"]:
        band = "SEVERE"
    elif worst_undefined > PREREG["material_undefined"]:
        band = "MATERIAL"
    else:
        band = "CLEAN"

    verdict = {
        "band": band,
        "criteria": PREREG,
        "missing_30_worst_undefined_rate": worst_undefined,
        "missing_30_worst_gate_auc": worst_auc,
        "preregistration": "experiments/synthesis/PREREGISTRATION_missing_views.md",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate_matrix.json").write_text(
        json.dumps({
            "config": config, "torch": torch.__version__,
            "gate": {"signals": list(DEPLOYABLE), "signs": A_PRIORI_SIGN,
                     "refitted_per_condition": False},
            "stress": STRESS, "attacks": ATTACKS,
            "cells": cells, "verdict": verdict, "per_seed": rows,
        }, indent=2, default=float),
        encoding="utf-8",
    )

    def cell(stress_name, attack_name, key):
        entry = cells[f"{stress_name}|{attack_name}"][key]
        if np.isnan(entry["mean"]):
            return "n/a"
        return f"{entry['mean']:.3f}"

    lines = [
        "# Part 14: deployable gate across attack x stress (16 cells)",
        "",
        "Gate: rank(dependence) + rank(clique_contrast), signs fixed a priori,",
        "equal weights, NOT refitted per condition. Negatives are clean +",
        "clone_k2 + clone_k3 under the same stress. 5 seeds, mean shown.",
        "",
        "## Gate selectivity AUC (defined rows)",
        "",
        "| stress | " + " | ".join(a["name"] for a in ATTACKS) + " |",
        "|---|" + "---|" * len(ATTACKS),
    ]
    for stress in STRESS:
        cells_text = " | ".join(cell(stress["name"], a["name"], "gate_auc") for a in ATTACKS)
        lines.append(f"| {stress['name']} | {cells_text} |")

    lines += [
        "",
        "## Undefined rate (own column, not folded into dropped rows)",
        "",
        "| stress | " + " | ".join(a["name"] for a in ATTACKS) + " |",
        "|---|" + "---|" * len(ATTACKS),
    ]
    for stress in STRESS:
        cells_text = " | ".join(
            cell(stress["name"], a["name"], "undefined_rate") for a in ATTACKS
        )
        lines.append(f"| {stress['name']} | {cells_text} |")

    lines += [
        "",
        "## Four-signal oracle upper bound (label-fitted, NOT deployable)",
        "",
        "| stress | " + " | ".join(a["name"] for a in ATTACKS) + " |",
        "|---|" + "---|" * len(ATTACKS),
    ]
    for stress in STRESS:
        cells_text = " | ".join(cell(stress["name"], a["name"], "oracle_auc") for a in ATTACKS)
        lines.append(f"| {stress['name']} | {cells_text} |")

    lines += [
        "",
        "## Pre-registered missing-views verdict",
        "",
        f"Band: **{band}**. Worst 30%-missing cell: undefined rate "
        f"{worst_undefined:.4f}, gate AUC {worst_auc:.4f}.",
        f"Criteria fixed before the run in {verdict['preregistration']}: "
        f"SEVERE above {PREREG['severe_undefined']:.0%} undefined or AUC below "
        f"{PREREG['severe_auc']:.2f}, MATERIAL above "
        f"{PREREG['material_undefined']:.0%}.",
        "",
    ]
    (OUT_DIR / "gate_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("gate matrix written | pre-registered band: %s", band)


if __name__ == "__main__":
    main()
