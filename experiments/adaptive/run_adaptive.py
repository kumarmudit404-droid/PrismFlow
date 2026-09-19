"""Part 14: the adaptive adversary. Does knowing the discount help the attacker?

THE QUESTION, AND WHAT IT IS NOT

This is NOT the gate question. `experiments/synthesis/` is closed as a
negative result: a label-free gate over these signals is at chance in 15 of
16 conditions, so there is nothing there worth adapting against. That thread
is closed formally here by one gradient-free arm and then dropped.

The open question is the DISCOUNT. PrismFlow's claim is that the ENIV
discount corrects fused belief when views are dependent. Part 09 attacked it
with an adversary that did not know it existed. An adversary that does know
would try to make its views agree on a wrong class while APPEARING
independent to the estimator -- paying a penalty on measured dependence:

    - gamma * max(0, rho_hat(C) - tau)

gamma = 0 recovers the Part 09 Chorus attack exactly (pinned in
tests/unit/test_adaptive.py), so everything the sweep shows is attributable
to the evasion term.

FOUR ARMS

  sweep          attack success and MEASURED dependence vs gamma. Reporting
                 success alone would be unreadable: if success is flat, it
                 matters enormously whether rho moved. Flat success with
                 flat rho means the attacker never evaded; flat success with
                 falling rho means it evaded and gained nothing.

  transfer       perturbations crafted on the UNDEFENDED model (same
                 weights, discount off) applied to the defended one. If
                 transfer succeeds where the white-box attack fails, the
                 white-box failure was gradient masking.

  random_search  gradient-FREE, same budget. If this beats the gradient
                 attack, our gradient is broken and no robustness claim from
                 this file stands.

  gate_check     the gradient-free search pointed at the deployable gate,
                 to close the synthesis thread with a measurement rather
                 than an assertion.

Usage:
    python -m experiments.adaptive.run_adaptive
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.synthesis.run_oracle import CONFIG, SIGNALS, clone_in_place, signals_for
from experiments.synthesis.run_sign_constrained import (
    A_PRIORI_SIGN,
    DEPLOYABLE,
    honest_calibrated_ranks,
)
from experiments.tail.run_tail import build_dataset, roc_auc, train_model, whole_split
from prismflow.attacks.adaptive import (
    AdaptiveConfig,
    adaptive_attack,
    measured_dependence,
    random_search_attack,
)
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger(__name__)
OUT_DIR = Path("results/adaptive")

GAMMAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
ATTACK = {"k": 3, "epsilon": 2.0, "beta": 1.0, "steps": 30}
RANDOM_RESTARTS = 64


def attack_success(model, views, view_mask, delta, target) -> float:
    with torch.no_grad():
        prediction = model(views + delta, view_mask).prediction
    return float((prediction.cpu() == target.cpu()).float().mean())


def gate_auc(model, attacked_views, view_mask, clone_views, view_masks) -> float:
    """Deployable gate selectivity: attacked vs honest duplication.

    Gate fixed exactly as in experiments/synthesis: a priori signs, equal
    weights, ranks calibrated on the honest rows. Never refitted.
    """
    positive = signals_for(model, attacked_views, view_mask)
    blocks = [np.column_stack([positive[s] for s in SIGNALS])]
    for cloned, mask in zip(clone_views, view_masks):
        honest = signals_for(model, cloned, mask)
        blocks.append(np.column_stack([honest[s] for s in SIGNALS]))

    features = np.vstack(blocks)
    labels = np.concatenate(
        [np.ones(len(blocks[0]))] + [np.zeros(len(b)) for b in blocks[1:]]
    )
    defined = np.isfinite(features).all(axis=1)
    features, labels = features[defined], labels[defined]
    if len(np.unique(labels)) < 2:
        return float("nan")

    honest_rows = labels == 0
    ranked = np.column_stack([
        honest_calibrated_ranks(features[:, i], honest_rows) for i in range(len(SIGNALS))
    ])
    columns = [SIGNALS.index(s) for s in DEPLOYABLE]
    signs = np.array([A_PRIORI_SIGN[s] for s in DEPLOYABLE])
    score = (ranked[:, columns] * signs).sum(axis=1)
    return roc_auc(score[labels == 1], score[labels == 0])


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
        clones = [
            clone_in_place(views, config["clone"]["source"], k) for k in config["clone"]["ks"]
        ]
        clone_masks = [view_mask] * len(clones)

        def make_config(gamma, tau):
            return AdaptiveConfig(
                k=ATTACK["k"], epsilon=ATTACK["epsilon"], beta=ATTACK["beta"],
                steps=ATTACK["steps"], gamma=gamma, tau=tau, seed=seed,
            )

        # tau is the CLEAN measured dependence among the views the attacker
        # controls: the attacker is asking to look no more dependent than the
        # honest baseline. Fixed per seed, before the sweep.
        compromised = tuple(range(ATTACK["k"]))
        tau = float(measured_dependence(model, views, view_mask, compromised))

        for gamma in GAMMAS:
            result = adaptive_attack(model, views, view_mask, make_config(gamma, tau))
            rows.append({
                "seed": seed, "arm": "sweep", "gamma": gamma, "tau": tau,
                "success": attack_success(
                    model, views, view_mask, result.delta, result.target
                ),
                "rho_initial": result.rho_initial,
                "rho_final": result.rho_final,
                "gate_auc": gate_auc(
                    model, views + result.delta, view_mask, clones, clone_masks
                ),
                "steps_accepted": result.steps_accepted,
            })

        # ---- transfer: craft with the discount OFF, apply with it ON ----
        was = model.use_discount
        model.use_discount = False
        try:
            transfer = adaptive_attack(model, views, view_mask, make_config(0.0, tau))
        finally:
            model.use_discount = was
        rows.append({
            "seed": seed, "arm": "transfer", "gamma": 0.0, "tau": tau,
            "success": attack_success(
                model, views, view_mask, transfer.delta, transfer.target
            ),
            "rho_initial": transfer.rho_initial,
            "rho_final": float(
                measured_dependence(
                    model, views + transfer.delta, view_mask, compromised
                )
            ),
            "gate_auc": gate_auc(
                model, views + transfer.delta, view_mask, clones, clone_masks
            ),
            "steps_accepted": transfer.steps_accepted,
        })

        # ---- gradient-free control, same epsilon ball ----
        control = random_search_attack(
            model, views, view_mask, make_config(0.0, tau), restarts=RANDOM_RESTARTS
        )
        rows.append({
            "seed": seed, "arm": "random_search", "gamma": 0.0, "tau": tau,
            "success": attack_success(
                model, views, view_mask, control.delta, control.target
            ),
            "rho_initial": control.rho_initial,
            "rho_final": control.rho_final,
            "gate_auc": gate_auc(
                model, views + control.delta, view_mask, clones, clone_masks
            ),
            "steps_accepted": control.steps_accepted,
        })

        # ---- closing the gate thread: gradient-free, aimed at evasion ----
        gate_probe = random_search_attack(
            model, views, view_mask, make_config(20.0, tau), restarts=RANDOM_RESTARTS
        )
        rows.append({
            "seed": seed, "arm": "gate_check", "gamma": 20.0, "tau": tau,
            "success": attack_success(
                model, views, view_mask, gate_probe.delta, gate_probe.target
            ),
            "rho_initial": gate_probe.rho_initial,
            "rho_final": gate_probe.rho_final,
            "gate_auc": gate_auc(
                model, views + gate_probe.delta, view_mask, clones, clone_masks
            ),
            "steps_accepted": gate_probe.steps_accepted,
        })

        swept = [r for r in rows if r["seed"] == seed and r["arm"] == "sweep"]
        LOGGER.info(
            "seed=%s tau=%.4f | gamma=0 success=%.3f rho=%.4f | gamma=%g success=%.3f "
            "rho=%.4f | transfer=%.3f random=%.3f",
            seed, tau,
            swept[0]["success"], swept[0]["rho_final"],
            GAMMAS[-1], swept[-1]["success"], swept[-1]["rho_final"],
            [r for r in rows if r["seed"] == seed and r["arm"] == "transfer"][0]["success"],
            [r for r in rows if r["seed"] == seed and r["arm"] == "random_search"][0]["success"],
        )

    def summarise_rows(selected, key):
        values = [r[key] for r in selected]
        clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        return {
            "mean": float(np.mean(clean)) if clean else float("nan"),
            "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan"),
            "n_seeds": len(clean),
            "per_seed": values,
        }

    sweep = {}
    for gamma in GAMMAS:
        selected = [r for r in rows if r["arm"] == "sweep" and r["gamma"] == gamma]
        sweep[str(gamma)] = {
            key: summarise_rows(selected, key)
            for key in ("success", "rho_final", "gate_auc")
        }

    arms = {}
    for arm in ("transfer", "random_search", "gate_check"):
        selected = [r for r in rows if r["arm"] == arm]
        arms[arm] = {
            key: summarise_rows(selected, key)
            for key in ("success", "rho_final", "gate_auc")
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adaptive.json").write_text(
        json.dumps({
            "config": config, "torch": torch.__version__,
            "attack": ATTACK, "gammas": GAMMAS, "random_restarts": RANDOM_RESTARTS,
            "sweep": sweep, "arms": arms, "per_seed": rows,
        }, indent=2, default=float),
        encoding="utf-8",
    )

    def fmt(entry):
        if np.isnan(entry["mean"]):
            return "n/a"
        if np.isnan(entry["std"]):
            return f"{entry['mean']:.4f}"
        return f"{entry['mean']:.4f} +/- {entry['std']:.4f}"

    lines = [
        "# Part 14: adaptive adversary",
        "",
        f"Seeds: {config['seeds']}. Attack: k={ATTACK['k']}, eps={ATTACK['epsilon']}, "
        f"beta={ATTACK['beta']}, {ATTACK['steps']} steps.",
        "tau per seed = clean measured dependence among the compromised views:",
        "the attacker asks to look no more dependent than an honest baseline.",
        "",
        "## Sweep: does knowing the discount help?",
        "",
        "| gamma | attack success | measured dependence (final) | deployable gate AUC |",
        "|---|---|---|---|",
    ]
    for gamma in GAMMAS:
        entry = sweep[str(gamma)]
        lines.append(
            f"| {gamma:g} | {fmt(entry['success'])} | {fmt(entry['rho_final'])} | "
            f"{fmt(entry['gate_auc'])} |"
        )

    lines += [
        "",
        "gamma = 0 IS the Part 09 Chorus attack (pinned in tests/unit/test_adaptive.py).",
        "",
        "## Controls",
        "",
        "| arm | attack success | measured dependence | gate AUC |",
        "|---|---|---|---|",
    ]
    for arm in ("transfer", "random_search", "gate_check"):
        entry = arms[arm]
        lines.append(
            f"| {arm} | {fmt(entry['success'])} | {fmt(entry['rho_final'])} | "
            f"{fmt(entry['gate_auc'])} |"
        )
    lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("adaptive results written")


if __name__ == "__main__":
    main()
