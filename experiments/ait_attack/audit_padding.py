"""Part 29 step 4(a): does the frozen chorus attack write into the
structurally-zero padding columns?

AIT views are 8-10 features wide and the dataset tensor is padded to the widest
view (10), so every view but `suricata` carries columns that are structurally
zero in the data. The Part 29 post-run brief asked this README to state that the
frozen chorus attack ALSO writes +/- epsilon into those padding columns, with
per-view counts.

It does not, and this script is the measurement that says so. `chorus_attack`
builds its step as `step_size * gradient.sign() * mask` where `mask` covers all
10 columns of a compromised view (`prismflow/attacks/chorus.py:148-166`), so the
padding columns ARE inside the attack's mask. They are nonetheless never moved,
because `view_configs` gives each view's encoder its own native input width
(`experiments/ait_clean/run_ait_clean.py:300-301`), the encoder never reads the
padded columns, and the gradient with respect to them is therefore identically
zero. `sign(0) == 0`, so the masked step writes nothing.

Writes `results/ait_attack/padding_audit.json`. Measures, per direction and per
view: the padding width, that the DATA is zero there, the attack delta on those
columns for chorus and pgd at two epsilons, and the objective's gradient with
respect to them. Nothing here is inferred from reading the attack code alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from experiments.ait_attack.run_ait_attack import CONFIG_PATH, prepare_direction, train_ait
from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import (ChorusConfig, choose_targets, chorus_attack,
                                      chorus_objective)
from prismflow.data.ait_lds import FEATURES

OUT = Path("results/ait_attack/padding_audit.json")
EPSILONS = (1.0, 2.0)
#: One seed. This is a STRUCTURAL property of the encoder's input width, not a
#: statistic over initialisations: the gradient with respect to a column no
#: encoder reads is zero for every seed by construction. Reported as a single
#: deterministic audit, NOT as a 5-seed finding.
SEED = 0


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    native = {view: len(features) for view, features in FEATURES.items()}
    width = max(native.values())

    report = {
        "question": ("Does the frozen chorus attack write +/- epsilon into the "
                     "structurally-zero padding columns of a compromised view?"),
        "answer": None,
        "seed": SEED,
        "seed_note": ("One seed. The gradient with respect to a column the encoder "
                      "never reads is zero by construction, for every seed, so this "
                      "is a structural audit and NOT a 5-seed finding."),
        "padded_view_width": width,
        "per_view_native_and_padding": {
            view: {"native_dims": n, "padding_columns": width - n}
            for view, n in native.items()
        },
        "mechanism": {
            "attack_mask_covers_padding": True,
            "attack_mask_site": "prismflow/attacks/chorus.py:148-149",
            "step_site": "prismflow/attacks/chorus.py:165-166",
            "encoder_native_width_site": "experiments/ait_clean/run_ait_clean.py:300-301",
            "why_zero": ("view_configs gives each view's encoder its native input "
                         "width, so the padded columns are never read; the gradient "
                         "with respect to them is identically zero and sign(0) == 0."),
        },
        "directions": {},
    }

    any_padding_written = False
    for direction in cfg["directions"]:
        dname = direction["name"]
        prepared = prepare_direction(direction, cfg, {})
        dataset, index = prepared["test"], prepared["test_index"]
        sources = list(dataset.sources)
        views, mask = dataset.views[index], dataset.view_mask[index]
        model = train_ait(prepared["train"], prepared["train_index"], SEED, True, cfg)

        per_view = {}
        for position, view in enumerate(sources):
            n_native = native[view]
            n_pad = width - n_native
            data_pad = views[:, position, n_native:]
            entry = {
                "native_dims": n_native,
                "padding_columns": n_pad,
                "n_windows": int(views.shape[0]),
                "padding_cells": int(data_pad.numel()),
                "data_max_abs_in_padding": float(data_pad.abs().max()) if data_pad.numel() else 0.0,
                "attacks": {},
            }
            for kind in ("chorus", "pgd"):
                for epsilon in EPSILONS:
                    config = ChorusConfig(
                        k=1, epsilon=epsilon, beta=cfg["attack"]["beta_primary"],
                        steps=cfg["attack"]["steps"],
                        target_strategy=cfg["attack"]["target_strategy"],
                        fixed_target=cfg["attack"]["fixed_target"],
                        seed=SEED, views=(position,),
                    )
                    target = choose_targets(model, views, mask, config)
                    runner = chorus_attack if kind == "chorus" else pgd_attack
                    result = runner(model, views, mask, config, target=target)
                    if tuple(result.compromised) != (position,):
                        raise RuntimeError(
                            f"asked to compromise {position} but attacked "
                            f"{tuple(result.compromised)}; the subset was not honoured"
                        )
                    delta = result.delta
                    native_delta = delta[:, position, :n_native]
                    pad_delta = delta[:, position, n_native:]
                    n_pad_nonzero = int((pad_delta.abs() > 0).sum()) if n_pad else 0
                    any_padding_written = any_padding_written or n_pad_nonzero > 0
                    entry["attacks"][f"{kind}_eps{epsilon}"] = {
                        "native_cells": int(native_delta.numel()),
                        "native_cells_perturbed": int((native_delta.abs() > 0).sum()),
                        "native_max_abs_delta": float(native_delta.abs().max()),
                        "padding_cells": int(pad_delta.numel()),
                        "padding_cells_perturbed": n_pad_nonzero,
                        "padding_max_abs_delta": (
                            float(pad_delta.abs().max()) if pad_delta.numel() else 0.0
                        ),
                    }
            # Why: the gradient the attack takes the sign of.
            if n_pad:
                config = ChorusConfig(
                    k=1, epsilon=EPSILONS[0], beta=cfg["attack"]["beta_primary"],
                    steps=cfg["attack"]["steps"],
                    target_strategy=cfg["attack"]["target_strategy"],
                    fixed_target=cfg["attack"]["fixed_target"],
                    seed=SEED, views=(position,),
                )
                target = choose_targets(model, views, mask, config)
                probe = torch.zeros_like(views).requires_grad_(True)
                objective = chorus_objective(
                    model, views + probe, mask, (position,), target, config.beta
                )
                (gradient,) = torch.autograd.grad(objective, probe)
                g_pad = gradient[:, position, n_native:]
                g_native = gradient[:, position, :n_native]
                entry["objective_gradient"] = {
                    "padding_max_abs": float(g_pad.abs().max()),
                    "padding_nonzero_cells": int((g_pad != 0).sum()),
                    "padding_cells": int(g_pad.numel()),
                    "native_max_abs": float(g_native.abs().max()),
                }
            per_view[view] = entry
        report["directions"][dname] = {
            "n_test_windows": int(index.size),
            "sources_in_tensor_order": sources,
            "per_view": per_view,
        }

    report["answer"] = (
        "NO. Across both directions, all five views, chorus and pgd, at epsilon "
        "1.0 and 2.0, every padding cell's delta is exactly zero and the "
        "objective's gradient with respect to every padding column is exactly "
        "zero. The padding columns lie inside the attack's mask but are "
        "unreachable through the encoder, so the brief's premise that the attack "
        "writes +/- epsilon into them is NOT what this pipeline does."
        if not any_padding_written else
        "YES -- at least one padding cell was perturbed; see per-view counts."
    )
    report["padding_ever_written"] = any_padding_written

    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")
    print(report["answer"])


if __name__ == "__main__":
    main()
