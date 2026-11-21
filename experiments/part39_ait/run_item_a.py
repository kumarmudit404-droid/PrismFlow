"""Part 39 item A: a decision threshold chosen on the TRAINING scenario (L16).

NOT YET EXECUTED. Written 2026-11-21 against the committed
`experiments/part39_ait/PRE_REGISTRATION.md` and `config.yaml`, which are the
authority for every choice here. No run has produced a number from this file.

WHAT IT DOES. For each (direction, system, seed) in the config grid: retrains
the model with Part 28's recipe, selects the threshold that maximises Youden's
J on the TRAINING scenario, and reports balanced accuracy at that threshold on
the TEST scenario beside Part 28's committed argmax-at-0.5 figure.

NOTHING IS REIMPLEMENTED. `load_scenario`, `labelled_indices`, `fit_scaler`,
`apply_scaler`, `train_ait`, `part28_extras` and `roc_auc_binary` are imported
from `experiments/ait_clean/run_ait_clean.py` and used unchanged, so the
retrained models are Part 28's models and the argmax-at-0.5 figures come from
the same function that produced the committed ones. `prismflow/**` is
import-only. The dataset/scaler/train/predict wiring below mirrors
`run_ait_clean.run_direction` and `run_system` line for line:

    train_indices = labelled_indices(train_dataset)
    scaler       = fit_scaler(train_dataset, train_indices, cfg)
    train_scaled = apply_scaler(train_dataset, scaler)
    test_scaled  = apply_scaler(test_dataset, scaler)
    model        = train_ait(train_scaled, train_indices, seed, use_discount, cfg)
    collected    = collect_predictions(model, iter_batches(test_scaled, ...))

WHY THERE IS NO VALIDATION SPLIT. Part 28's own `train_ait` docstring says it:
"There is no validation split and no early stopping: a split inside the
training scenario would be a within-scenario split, which this Part forbids."
`docs/part27-ait-datacard.md` section 8 gives the reason -- a random window
split leaks across neighbouring windows of one attack run, and a chronological
split on `santos` is close to a label split. So the threshold is selected on
the WHOLE training scenario. See PRE_REGISTRATION section 1.

THE L29 CONFIDENCE KEY. `collect_predictions` returns the column under the key
`vacuity_confidence`; the column holds a CONFIDENCE, so `0.0` on an
evidence-free row is the right value and the *name* is the defect. That file is
frozen and is not edited. This runner writes the quantity under the key
`confidence`, records a `confidence_convention` string, and asserts before
every write that no key named `vacuity` is present.

    PYTHONPATH=. ./.venv/Scripts/python.exe experiments/part39_ait/run_item_a.py --dry-run
    PYTHONPATH=. ./.venv/Scripts/python.exe experiments/part39_ait/run_item_a.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ait_clean.run_ait_clean import (                 # noqa: E402
    CLASS_ATTACK,
    CLASS_BENIGN,
    apply_scaler,
    fit_scaler,
    labelled_indices,
    load_scenario,
    part28_extras,
    train_ait,
)
from prismflow.data.loaders import iter_batches                   # noqa: E402
from prismflow.evaluation.protocol import collect_predictions      # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
#: Part 28's committed figures. The reproduction gate reads
#: directions.<direction>.systems.<system>.per_seed[*].balanced_accuracy.
PART28_RESULTS = ROOT / "results" / "ait_clean" / "results.json"

CONFIDENCE_CONVENTION = (
    "higher means more evidence. This is the column run_ait_clean.py collects "
    "as `vacuity_confidence`; the name there is inverted (L29) and that file is "
    "frozen. No key written by this runner is named `vacuity`."
)


# ---------------------------------------------------------------------------
# the threshold rule (PRE_REGISTRATION section 1)
# ---------------------------------------------------------------------------

def balanced_accuracy_at(labels: np.ndarray, prob_attack: np.ndarray,
                         threshold: float) -> tuple[float, float, float]:
    """(balanced accuracy, attack recall, benign recall) at `prob_attack >= t`.

    Matches `part28_extras`'s definition -- the mean of the per-class recalls --
    with the argmax replaced by a threshold. Both classes must be present; the
    caller asserts that, because a mean over one present recall would be a
    different statistic silently.
    """
    predicted_attack = prob_attack >= threshold
    attack = labels == CLASS_ATTACK
    benign = labels == CLASS_BENIGN
    attack_recall = float(predicted_attack[attack].mean())
    benign_recall = float((~predicted_attack[benign]).mean())
    return (attack_recall + benign_recall) / 2.0, attack_recall, benign_recall


def youdens_j(balanced_accuracy: float) -> float:
    """J = sensitivity + specificity - 1 = 2 * balanced_accuracy - 1.

    A strictly increasing transform of balanced accuracy, so the two have the
    SAME argmax. PRE_REGISTRATION section 1 records this: the brief named
    Youden's J, L16's closing condition names "maximise balanced accuracy on
    train", and the selected threshold is reported under both names.
    """
    return 2.0 * balanced_accuracy - 1.0


def select_threshold(labels: np.ndarray, prob_attack: np.ndarray) -> dict:
    """Exhaustive scan over the observed scores. No new algorithm.

    Grid: every distinct `prob_attack` on the rows given, plus 0.0 and 1.0
    (PRE_REGISTRATION section 1). Exhaustive, so there is no step size to tune
    and no grid resolution chosen after seeing results.

    TIE-BREAK is "smaller t", and it comes for free: the grid is sorted
    ascending and `numpy.argmax` returns the FIRST occurrence of the maximum,
    so the smallest maximising threshold wins without a second pass.
    """
    grid = np.unique(np.concatenate([
        np.asarray(prob_attack, dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]))
    scores = np.array([balanced_accuracy_at(labels, prob_attack, t)[0]
                       for t in grid], dtype=np.float64)
    best = int(np.argmax(scores))          # first maximum == smallest t
    t = float(grid[best])
    ba, attack_recall, benign_recall = balanced_accuracy_at(labels, prob_attack, t)
    return {
        "threshold": t,
        "grid_size": int(grid.size),
        "train_balanced_accuracy_at_threshold": ba,
        "train_youdens_j_at_threshold": youdens_j(ba),
        "train_attack_recall_at_threshold": attack_recall,
        "train_benign_recall_at_threshold": benign_recall,
        "degenerate_at_selection": t in (0.0, 1.0),
        "rule": (
            "argmax of Youden's J over every distinct train prob_attack plus "
            "0.0 and 1.0; ties to the smaller threshold. J = 2*BA - 1, so this "
            "is also argmax balanced accuracy on train."
        ),
    }


# ---------------------------------------------------------------------------
# the reproduction gate (PRE_REGISTRATION section 5.1)
# ---------------------------------------------------------------------------

def part28_committed() -> dict[tuple[str, str, int], float]:
    """Part 28's per-seed TEST-scenario argmax-at-0.5 balanced accuracy.

    WHICH FIGURE THIS IS, because the review brief and the committed data
    differ. The brief asks the gate to compare "training balanced accuracy"
    against "Part 28's committed value". **Part 28 never reports a
    training-scenario balanced accuracy** -- every committed figure is on the
    TEST scenario -- so no such committed value exists to compare against. The
    gate therefore runs on the TEST-scenario argmax-at-0.5 figure, which is
    what PRE_REGISTRATION section 5.1 specifies and what exists on disk. The
    train-scenario argmax-at-0.5 value is still computed and reported, labelled
    as having no committed counterpart.
    """
    if not PART28_RESULTS.exists():
        raise SystemExit(
            f"{PART28_RESULTS} is missing. The reproduction gate cannot run "
            "without Part 28's committed per-seed figures, and without the "
            "gate this item compares two training runs rather than two "
            "thresholds. Stopping."
        )
    data = json.loads(PART28_RESULTS.read_text())
    out: dict[tuple[str, str, int], float] = {}
    for direction, dv in data["directions"].items():
        for system, sv in dv["systems"].items():
            for row in sv["per_seed"]:
                out[(direction, system, int(row["seed"]))] = float(
                    row["balanced_accuracy"])
    return out


def check_gate(key: tuple[str, str, int], measured: float,
               committed: dict, tolerance: float) -> dict:
    """Fatal on mismatch. Threshold selection never runs on a failed gate."""
    if key not in committed:
        raise SystemExit(
            f"{key}: Part 28 has no committed per-seed balanced accuracy for "
            "this cell. Refusing to proceed: there is nothing to reproduce "
            "against."
        )
    expected = committed[key]
    delta = measured - expected
    if abs(delta) > tolerance:
        raise SystemExit(
            f"REPRODUCTION GATE FAILED for {key}.\n"
            f"  committed argmax-at-0.5 balanced accuracy: {expected!r}\n"
            f"  retrained:                                 {measured!r}\n"
            f"  difference:                                {delta!r}\n"
            f"  tolerance:                                 {tolerance!r}\n"
            "The retrained model is not Part 28's model, so comparing its "
            "thresholds would compare two training runs rather than two "
            "thresholds. Stopping before any threshold is selected.\n"
            "SUSPECT THREAD COUNT FIRST: Part 34 measured a committed "
            "eniv_eigen of 3.4396320614332856 missing its own value by 0.0107 "
            "under OMP_NUM_THREADS=1 (3.4289614712229826). Re-run at the "
            "machine default before concluding anything. The observed count "
            f"here was {torch.get_num_threads()}."
        )
    return {"committed": expected, "retrained": measured, "difference": delta,
            "tolerance": tolerance, "passed": True}


# ---------------------------------------------------------------------------
# aggregation (PRE_REGISTRATION section 4)
# ---------------------------------------------------------------------------

def agg(values: list[float]) -> dict:
    """mean +/- sample std (ddof=1). A cell below 5 seeds is not a finding."""
    return {
        "mean": float(st.fmean(values)),
        "std": float(st.stdev(values)) if len(values) > 1 else None,
        "n": len(values),
        "per_seed": [float(v) for v in values],
    }


def signs(diffs: list[float]) -> dict:
    """Per-seed signs, reported BEFORE any aggregate mean (section 4.2)."""
    return {
        "seeds_positive": sum(d > 0 for d in diffs),
        "seeds_negative": sum(d < 0 for d in diffs),
        "seeds_exactly_zero": sum(d == 0 for d in diffs),
        "per_seed": [float(d) for d in diffs],
    }


def assert_no_vacuity(obj, where: str) -> None:
    """L29: no key written by this runner may be named `vacuity`."""
    if isinstance(obj, dict):
        if "vacuity" in obj:
            raise AssertionError(
                f"refusing to write a key named 'vacuity' at {where} (L29). "
                "The column holds a CONFIDENCE; this runner writes "
                "'confidence'."
            )
        for k, v in obj.items():
            assert_no_vacuity(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_vacuity(v, f"{where}[{i}]")


# ---------------------------------------------------------------------------
# one unit
# ---------------------------------------------------------------------------

def run_unit(direction: dict, system: dict, seed: int, train_scaled, train_indices,
             test_scaled, test_indices, cfg: dict, committed: dict,
             tolerance: float) -> dict:
    """Train, gate, select on train, report on test. In that order."""
    model = train_ait(train_scaled, train_indices, seed,
                      system["use_discount"], cfg)
    batch_size = cfg["evaluation"]["batch_size"]

    test_collected = collect_predictions(
        model, iter_batches(test_scaled, test_indices, batch_size, shuffle=False))
    test_labels = np.asarray(test_collected["labels"], dtype=np.int64)
    test_probs = np.asarray(test_collected["probs"], dtype=np.float64)
    if set(np.unique(test_labels)) != {CLASS_BENIGN, CLASS_ATTACK}:
        raise SystemExit(
            f"{direction['name']}/{system['name']}/seed{seed}: the test block "
            "is not two-class, so balanced accuracy is a different statistic. "
            "Stopping.")

    # GATE FIRST. part28_extras is Part 28's own function, so the figure being
    # compared is produced by the same code that produced the committed one.
    test_at_argmax = part28_extras(test_labels, test_probs)
    gate = check_gate((direction["name"], system["name"], seed),
                      test_at_argmax["balanced_accuracy"], committed, tolerance)

    # Only now may a threshold be selected.
    train_collected = collect_predictions(
        model, iter_batches(train_scaled, train_indices, batch_size, shuffle=False))
    train_labels = np.asarray(train_collected["labels"], dtype=np.int64)
    train_probs = np.asarray(train_collected["probs"], dtype=np.float64)
    if set(np.unique(train_labels)) != {CLASS_BENIGN, CLASS_ATTACK}:
        raise SystemExit(
            f"{direction['name']}/{system['name']}/seed{seed}: the train block "
            "is not two-class. Stopping.")

    train_prob_attack = train_probs[:, CLASS_ATTACK]
    test_prob_attack = test_probs[:, CLASS_ATTACK]

    selection = select_threshold(train_labels, train_prob_attack)
    t = selection["threshold"]

    test_ba, test_attack_recall, test_benign_recall = balanced_accuracy_at(
        test_labels, test_prob_attack, t)

    # The train-scenario argmax-at-0.5 figure. Reported, but there is NO
    # committed Part 28 counterpart -- Part 28 only ever reports test-scenario
    # metrics -- so it is labelled rather than gated.
    train_at_argmax = part28_extras(train_labels, train_probs)

    # `argmax` picks class 1 iff prob_attack > 0.5 (numpy breaks ties toward
    # index 0 = benign), while the threshold rule uses >=. The two differ only
    # on rows with prob_attack exactly 0.5. Both are reported rather than
    # assumed equal.
    test_ba_at_half, _, _ = balanced_accuracy_at(test_labels, test_prob_attack, 0.5)

    unit = {
        "direction": direction["name"],
        "system": system["name"],
        "seed": seed,
        "torch_threads_observed": int(torch.get_num_threads()),
        "reproduction_gate": gate,
        "selection_on_train": selection,
        "train_argmax_0_5": {
            "balanced_accuracy": train_at_argmax["balanced_accuracy"],
            "attack_recall": train_at_argmax["attack_recall"],
            "benign_recall": train_at_argmax["benign_recall"],
            "no_committed_counterpart": (
                "Part 28 reports no training-scenario metric, so this figure "
                "is reported for context and is NOT gated against anything."),
        },
        "test_argmax_0_5": {
            "balanced_accuracy": test_at_argmax["balanced_accuracy"],
            "attack_recall": test_at_argmax["attack_recall"],
            "benign_recall": test_at_argmax["benign_recall"],
            "roc_auc": test_at_argmax["roc_auc"],
            "n_test": test_at_argmax["n_test"],
            "attack_prevalence": test_at_argmax["attack_prevalence"],
            "balanced_accuracy_at_ge_0_5": test_ba_at_half,
            "argmax_vs_ge_half_note": (
                "argmax selects attack iff prob_attack > 0.5; the threshold "
                "rule uses >=. They differ only on rows with prob_attack "
                "exactly 0.5."),
        },
        "test_at_train_chosen_threshold": {
            "threshold": t,
            "balanced_accuracy": test_ba,
            "attack_recall": test_attack_recall,
            "benign_recall": test_benign_recall,
        },
        "delta_balanced_accuracy_test": (
            test_ba - test_at_argmax["balanced_accuracy"]),
        "confidence_convention": CONFIDENCE_CONVENTION,
        "test_confidence_mean": float(
            np.asarray(test_collected["vacuity_confidence"],
                       dtype=np.float64).mean()),
    }
    assert_no_vacuity(unit, "unit")
    return unit


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def describe_grid(cfg: dict) -> list[str]:
    lines = []
    for direction in cfg["directions"]:
        for system in cfg["systems"]:
            for seed in cfg["seeds"]:
                lines.append("%-16s %-10s seed%d   train=%-16s test=%s"
                             % (direction["name"], system["name"], seed,
                                direction["train"], direction["test"]))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the config grid and stop before any training")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    grid = describe_grid(cfg)

    if args.dry_run:
        print("Part 39 item A -- DRY RUN. No model trained, no threshold "
              "selected, no metric computed.")
        print("config: %s" % args.config)
        print("threshold rule: %s on %s"
              % (cfg["threshold"]["method"], cfg["threshold"]["select_on"]))
        print("  equivalent to: %s" % cfg["threshold"]["equivalent_to"])
        print("H1 bar: |delta balanced accuracy| < %s"
              % cfg["hypothesis"]["bar_abs_balanced_accuracy_delta"])
        print("reproduction gate: %s, tolerance %s"
              % (cfg["stopping_rules"]["reproduction_gate"]["requirement"].strip(),
                 cfg["stopping_rules"]["reproduction_gate"]["tolerance_abs"]))
        print("\n%d units:" % len(grid))
        for line in grid:
            print("  " + line)
        print("\nObserved torch threads: %d. Every unit records this; see the "
              "gate's note on Part 34." % torch.get_num_threads())
        print("Stopping before training, as --dry-run requires.")
        return 0

    committed = part28_committed()
    tolerance = float(cfg["stopping_rules"]["reproduction_gate"]["tolerance_abs"])
    bar = float(cfg["hypothesis"]["bar_abs_balanced_accuracy_delta"])

    out_root = ROOT / cfg["output"]["results_root"]
    units_dir = out_root / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted({d[k] for d in cfg["directions"] for k in ("train", "test")})
    datasets = {name: load_scenario(name, cfg) for name in scenarios}

    units: list[dict] = []
    per_cell: dict[str, dict] = {}

    for direction in cfg["directions"]:
        if direction["train"] == direction["test"]:
            raise SystemExit(
                f"direction {direction['name']} has train == test; the two "
                "must be DIFFERENT scenarios.")
        train_dataset, _ = datasets[direction["train"]]
        test_dataset, _ = datasets[direction["test"]]
        train_indices = labelled_indices(train_dataset)
        test_indices = labelled_indices(test_dataset)
        # Fitted on the TRAIN scenario's label_known rows only, refitted per
        # direction, never shared between directions. run_direction's order.
        scaler = fit_scaler(train_dataset, train_indices, cfg)
        train_scaled = apply_scaler(train_dataset, scaler)
        test_scaled = apply_scaler(test_dataset, scaler)

        for system in cfg["systems"]:
            cell_units = []
            for seed in cfg["seeds"]:
                unit = run_unit(direction, system, seed, train_scaled,
                                train_indices, test_scaled, test_indices,
                                cfg, committed, tolerance)
                path = units_dir / ("%s_%s_seed%d.json"
                                    % (direction["name"], system["name"], seed))
                path.write_text(json.dumps(unit, indent=1))
                units.append(unit)
                cell_units.append(unit)
                print("done %-16s %-10s seed%d  t=%.6f  test BA %.10f -> %.10f "
                      "(delta %+.10f)"
                      % (direction["name"], system["name"], seed,
                         unit["selection_on_train"]["threshold"],
                         unit["test_argmax_0_5"]["balanced_accuracy"],
                         unit["test_at_train_chosen_threshold"]["balanced_accuracy"],
                         unit["delta_balanced_accuracy_test"]))

            key = "%s/%s" % (direction["name"], system["name"])
            diffs = [u["delta_balanced_accuracy_test"] for u in cell_units]
            at_argmax = agg([u["test_argmax_0_5"]["balanced_accuracy"]
                             for u in cell_units])
            at_threshold = agg([
                u["test_at_train_chosen_threshold"]["balanced_accuracy"]
                for u in cell_units])
            cell = {
                "n_seeds": len(cell_units),
                "signs_of_delta_before_mean": signs(diffs),
                "delta_balanced_accuracy": agg(diffs),
                "test_balanced_accuracy_argmax_0_5": at_argmax,
                "test_balanced_accuracy_at_train_chosen_threshold": at_threshold,
                "selected_threshold": agg(
                    [u["selection_on_train"]["threshold"] for u in cell_units]),
                "train_balanced_accuracy_at_threshold": agg(
                    [u["selection_on_train"]["train_balanced_accuracy_at_threshold"]
                     for u in cell_units]),
                "degenerate_at_selection_per_seed": [
                    u["selection_on_train"]["degenerate_at_selection"]
                    for u in cell_units],
                "h1_cell_within_bar": abs(
                    at_threshold["mean"] - at_argmax["mean"]) < bar,
                "h1_signed_difference_of_means": (
                    at_threshold["mean"] - at_argmax["mean"]),
            }
            if len(cell_units) < len(cfg["seeds"]):
                cell["NOT_A_FINDING"] = (
                    "n = %d of %d seeds. A mean over a partial cell is not a "
                    "finding." % (len(cell_units), len(cfg["seeds"])))
            per_cell[key] = cell

    all_within = all(c["h1_cell_within_bar"] for c in per_cell.values())
    verdict = "CONFIRMED" if all_within else "DISCONFIRMED"
    failing = {k: c["h1_signed_difference_of_means"]
               for k, c in per_cell.items() if not c["h1_cell_within_bar"]}

    results = {
        "part": 39,
        "item": "A",
        "closes": "L16, first clause only",
        "authority": "experiments/part39_ait/PRE_REGISTRATION.md",
        "what_changed_from_part28": (
            "the decision threshold, and nothing else. Loader, window length, "
            "sample mode, model, training recipe, scaler and label filter are "
            "imported from experiments/ait_clean/run_ait_clean.py and used "
            "unchanged."),
        "threshold_rule": cfg["threshold"],
        "no_validation_split_because": (
            "docs/part27-ait-datacard.md section 8: a random window split "
            "leaks across neighbouring windows of one attack run, and a "
            "chronological split on santos is close to a label split. Part "
            "28's own train_ait docstring records the same prohibition."),
        "seed_meaning": (
            "A seed selects parameter initialisation and minibatch order only. "
            "Train and test are whole scenarios, so a seed cannot redraw or "
            "re-partition the data. Every +/- is "
            "initialisation-and-batch-order variance on a fixed pair of "
            "scenarios, NOT a spread over independent attacks: the test sets "
            "hold 3 (santos) and 1 (russellmitchell) contiguous attack "
            "episodes."),
        "pooling_policy": (
            "Directions are never pooled, averaged or concatenated, and "
            "systems are separate cells."),
        "torch_threads_observed": int(torch.get_num_threads()),
        "reproduction_gate": {
            "status": "PASSED",
            "what": ("every unit's TEST-scenario argmax-at-0.5 balanced "
                     "accuracy reproduced Part 28's committed per-seed value"),
            "source": ("results/ait_clean/results.json, directions.<direction>"
                       ".systems.<system>.per_seed[*].balanced_accuracy"),
            "tolerance_abs": tolerance,
            "note": ("the brief asked for a gate on TRAINING balanced "
                     "accuracy; Part 28 reports no training-scenario metric, "
                     "so there is no committed value to compare against. The "
                     "gate runs on the test-scenario figure, as "
                     "PRE_REGISTRATION section 5.1 specifies."),
        },
        "h1": {
            "statement": cfg["hypothesis"]["h1"].strip(),
            "bar_abs_balanced_accuracy_delta": bar,
            "verdict": verdict,
            "cells_outside_bar": failing,
            "read_the_sign": (
                "H1 is a NULL hypothesis about threshold choice. A "
                "train-chosen threshold that IMPROVES balanced accuracy by "
                "more than the bar DISCONFIRMS H1 while being a better "
                "outcome. Disconfirmed upward: 0.5 was the wrong operating "
                "point and the ranking was always there. Disconfirmed "
                "downward: 0.5 was accidentally favourable and Part 28's "
                "argmax-at-0.5 figures are optimistic as deployment numbers. "
                "Neither may be written as passed or failed without the sign."),
            "direction_b_caveat": (
                "direction B's committed stds are 0.0685 (naive) and 0.0455 "
                "(prismflow), so a 0.02 bar sits well inside one std there: a "
                "cell can satisfy H1 while the per-seed values move a great "
                "deal. Read signs_of_delta_before_mean."),
        },
        "aggregation_rule": (
            "mean +/- sample std (ddof=1) over the 5 seeds, per cell. A cell "
            "with fewer than 5 seeds is not a finding."),
        "per_cell": per_cell,
        "units": units,
        "cannot_show": cfg["cannot_show"],
        "no_status_change": (
            "docs/KNOWN_LIMITATIONS.md is not edited. Proposals go to "
            "docs/proposed_limitations_update.md for the review gate."),
    }
    assert_no_vacuity(results, "results")

    out_path = ROOT / cfg["output"]["aggregate"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))

    print("\n%-28s %6s %16s %16s %12s %s"
          % ("direction/system", "n", "BA at 0.5", "BA at t", "delta", "within bar"))
    for key, cell in per_cell.items():
        print("%-28s %6d %16.10f %16.10f %+12.10f %s"
              % (key, cell["n_seeds"],
                 cell["test_balanced_accuracy_argmax_0_5"]["mean"],
                 cell["test_balanced_accuracy_at_train_chosen_threshold"]["mean"],
                 cell["h1_signed_difference_of_means"],
                 cell["h1_cell_within_bar"]))
    print("\nH1 (|delta| < %s in all four cells): %s" % (bar, verdict))
    if failing:
        for key, delta in failing.items():
            print("  outside bar: %-28s %+0.10f" % (key, delta))
    print("wrote %s" % out_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
