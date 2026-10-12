"""Part 28 addendum: non-fusion learned baselines on the AIT windows.

WHY THIS EXISTS
---------------
`run_ait_clean.py` compares two systems that share everything except the
discount -- the same encoders, the same evidence heads, the same training loop
-- plus a majority-class baseline that fits nothing. So every comparison there
is fusion against fusion, or fusion against a constant. The gap between a
fused multi-view model and a FLAT classifier on the same features was not
measured, and in direction B both systems reach ROC AUC 0.9999, where the
discriminative work may be almost entirely in the one view that sees the
exfiltration directly.

This script adds the two baselines that bound that question:

  flat_all_views        logistic regression on every view's features
                        concatenated, plus the five view_mask indicators
  single_view_<name>    one logistic regression per view, on that view's
                        features and its own mask indicator

It answers "does the architecture earn anything over a flat model on the same
features" and "which single view carries the signal". It makes no other claim,
and it changes nothing in `run_ait_clean.py` or its outputs: this script reads
`results/ait_clean/results.json` to quote the naive and PrismFlow rows beside
its own and never writes to it.

WHAT THE LABEL MEANS
--------------------
Unchanged from `run_ait_clean.py`, and it governs these numbers exactly as it
governs those. A window is attack=1 if ANY officially labelled line falls in
it, and on both scenarios that label is dominated by one continuously-running
DNS exfiltration tool, so the task is "was the exfiltration tool running in
this window", NOT "detect an APT" and not "detect a multi-step intrusion". A
single view scoring well here says that view sees the exfiltration tool; it
does not say that view detects intrusions.

SAME DATA, SAME SPLIT, SAME SCALER, SAME METRIC CODE
----------------------------------------------------
Everything that could make this an unfair comparison is imported from
`run_ait_clean.py` rather than reimplemented:

  `load_scenario`, `labelled_indices`   the same windows, label_known only
  `fit_scaler`, `apply_scaler`          fitted on the TRAINING scenario's
                                        label_known rows only, refitted per
                                        direction, never shared
  `metric_row`                          the same `compute_metrics` +
                                        `part28_extras` pair every row in
                                        Part 28 goes through

Both directions are run separately and are never pooled, averaged or
concatenated, for the same reason as in `run_ait_clean.py`: two directions are
two measurements on two test sets with different prevalences (0.602 and 0.900
attack among labelled windows).

DETERMINISM, AND WHY THERE IS NO +/- HERE
-----------------------------------------
Logistic regression with the lbfgs solver is a deterministic function of its
inputs: there is no initialisation draw and no minibatch order, so a seed has
nothing to select and five runs would return five identical numbers. Reporting
`x +/- 0.0000` over five such runs would fake the 5-SEED RULE rather than
satisfy it -- the rule exists to report variation, and here there is none to
report. So every baseline row below is ONE value with no `+/-`, and
`verify_determinism` refits each model a second time and asserts the
coefficients and probabilities are bit-identical, so the claim is checked
rather than asserted.

The naive and PrismFlow reference rows DO carry a `+/-`, because those are
5-seed means over model initialisation and batch order. The two kinds of row
must not be read as the same kind of estimate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from experiments.ait_clean.run_ait_clean import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    CONFIG_PATH,
    LABEL_MEANING,
    apply_scaler,
    fit_scaler,
    labelled_indices,
    load_scenario,
    metric_row,
)
from prismflow.data.ait_lds import FEATURES
from prismflow.utils.logging import get_logger

LOGGER = get_logger("prismflow.ait_baselines")

#: Fixed and stated, not tuned. lbfgs is deterministic, so these three values
#: plus the data fully determine every number this script writes.
SOLVER = "lbfgs"
MAX_ITER = 5000
C_VALUE = 1.0
#: Left at scikit-learn's default. Direction A trains at 90.0% attack
#: prevalence and direction B at 60.2%, and a plain unweighted baseline is
#: what "flat classifier on the same features" means. Reweighting would be a
#: different baseline and is not run here.
CLASS_WEIGHT = None

#: The metrics the addendum reports, in the order the tables use.
REPORT_METRICS = [
    ("roc_auc", "ROC AUC", 4),
    ("balanced_accuracy", "balanced accuracy", 4),
    ("macro_f1", "macro-F1", 4),
    ("attack_recall", "attack recall", 4),
    ("benign_recall", "benign recall", 4),
    ("brier", "Brier", 4),
    ("prob_ece", "ECE (15 bins)", 4),
]

RESULTS_JSON = Path("results/ait_clean/results.json")
OUT_JSON = Path("results/ait_clean/baselines.json")
OUT_MD = Path("results/ait_clean/baselines.md")

DETERMINISM_NOTE = (
    "Logistic regression with the lbfgs solver is a deterministic function of "
    "its inputs: no initialisation draw, no minibatch order, so a seed has "
    "nothing to select and five runs return five identical numbers. Each "
    "baseline row is therefore ONE value with no +/-. Reporting +/- 0.0000 "
    "over five identical runs would fake the 5-SEED RULE rather than satisfy "
    "it. Determinism is checked, not assumed: every model is refitted and its "
    "coefficients and predicted probabilities asserted bit-identical. The "
    "naive and PrismFlow reference rows DO carry a +/-, because those are "
    "5-seed means over model initialisation and batch order; the two kinds of "
    "row are not the same kind of estimate."
)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def native_dims(sources) -> list[int]:
    return [len(FEATURES[s]) for s in sources]


def flat_features(dataset, indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Every view's native columns concatenated, then the five mask indicators.

    Missing views are zero-filled, which is what `apply_scaler` already leaves
    behind: an absent view's row is the zero vector, and the mask indicator is
    what tells the model the zeros mean "absent" rather than "observed zero".
    Without the indicators a zero-filled row is indistinguishable from a real
    measurement at the scaled mean, so the indicators are part of the baseline
    and not an extra.

    The padded columns beyond each view's native width are dropped: they are
    structurally zero for every row and carry nothing.
    """
    views = dataset.views.numpy()[indices]
    mask = dataset.view_mask.numpy()[indices]
    dims = native_dims(dataset.sources)
    blocks, names = [], []
    for v, source in enumerate(dataset.sources):
        blocks.append(views[:, v, : dims[v]])
        names += [f"{source}:{f.name}" for f in FEATURES[source]]
    blocks.append(mask.astype(np.float64))
    names += [f"mask:{s}" for s in dataset.sources]
    return np.concatenate(blocks, axis=1).astype(np.float64), names


def single_view_features(dataset, indices: np.ndarray, view: int) -> tuple[np.ndarray, list[str]]:
    """One view's native columns plus its own mask indicator."""
    views = dataset.views.numpy()[indices]
    mask = dataset.view_mask.numpy()[indices]
    source = dataset.sources[view]
    dim = native_dims(dataset.sources)[view]
    features = np.concatenate(
        [views[:, view, :dim], mask[:, view : view + 1].astype(np.float64)], axis=1
    ).astype(np.float64)
    names = [f"{source}:{f.name}" for f in FEATURES[source]] + [f"mask:{source}"]
    return features, names


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def require_sklearn():
    """Import scikit-learn or stop. A dependency is never added silently."""
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "scikit-learn is not installed, and this addendum must not add a "
            "dependency. Install it deliberately (pip install scikit-learn) and "
            f"re-run, or treat the baselines as not measured. Import said: {error}"
        ) from error
    return LogisticRegression, sklearn.__version__


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray):
    """One logistic regression with the fixed, stated solver settings."""
    LogisticRegression, _ = require_sklearn()
    model = LogisticRegression(
        solver=SOLVER, max_iter=MAX_ITER, C=C_VALUE, class_weight=CLASS_WEIGHT
    )
    model.fit(x_train, y_train)
    # sklearn sorts classes, so predict_proba column 1 is the attack class.
    # Asserted rather than assumed: a silent reordering would invert every
    # ROC AUC in this file.
    if not np.array_equal(model.classes_, np.array([CLASS_BENIGN, CLASS_ATTACK])):
        raise AssertionError(f"unexpected class order {model.classes_}")
    return model


def probabilities(model, x: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probs.shape[1] != 2:
        raise AssertionError(f"expected 2 columns, got {probs.shape}")
    return probs


def verify_determinism(x_train, y_train, x_test, model, probs) -> dict:
    """Refit and demand bit-identical coefficients and probabilities.

    This is what licenses reporting a single value with no `+/-`. If a future
    scikit-learn makes lbfgs non-deterministic, this raises instead of quietly
    turning a stated fact into a false one.
    """
    again = fit_logistic(x_train, y_train)
    coef_identical = bool(
        np.array_equal(model.coef_, again.coef_)
        and np.array_equal(model.intercept_, again.intercept_)
    )
    probs_identical = bool(np.array_equal(probs, probabilities(again, x_test)))
    if not (coef_identical and probs_identical):
        raise AssertionError(
            "refit differed, so the no-+/- reporting in this file would be false: "
            f"coefficients identical={coef_identical}, probabilities "
            f"identical={probs_identical}"
        )
    return {"refit_coefficients_identical": True, "refit_probabilities_identical": True}


# ---------------------------------------------------------------------------
# One baseline, one direction
# ---------------------------------------------------------------------------


def run_baseline(
    name: str,
    builder,
    train_dataset,
    train_indices,
    test_dataset,
    test_indices,
    cfg: dict,
) -> dict:
    """Fit on the training scenario, score on the test scenario, one row out."""
    x_train, feature_names = builder(train_dataset, train_indices)
    y_train = train_dataset.labels.numpy()[train_indices]
    x_test, _ = builder(test_dataset, test_indices)
    y_test = test_dataset.labels.numpy()[test_indices]

    model = fit_logistic(x_train, y_train)
    probs = probabilities(model, x_test)
    # vacuity=None: a logistic regression has no Dirichlet vacuity, so the
    # `vacuity_` keys are dropped rather than filled with a stand-in. Same
    # treatment as the majority-class baseline in run_ait_clean.py.
    row = metric_row(y_test, probs, None, cfg)
    row.update(
        {
            "baseline": name,
            "n_features": int(x_train.shape[1]),
            "feature_names": feature_names,
            "n_train": int(x_train.shape[0]),
            "solver": SOLVER,
            "max_iter": MAX_ITER,
            "C": C_VALUE,
            "class_weight": CLASS_WEIGHT,
            "deterministic": True,
            **verify_determinism(x_train, y_train, x_test, model, probs),
        }
    )
    return row


def reference_rows(direction_record: dict) -> dict:
    """The naive and PrismFlow rows from results.json, quoted, not recomputed.

    Read-only. These are 5-seed means with a sample std, produced by
    `run_ait_clean.py`; this script does not retrain them and does not write
    to that file.
    """
    out = {}
    for system, record in direction_record["systems"].items():
        summary = record["summary"]
        out[system] = {
            "label": record["label"],
            "n_seeds": summary["balanced_accuracy"]["n_seeds"],
            "source": "results/ait_clean/results.json",
            "metrics": {
                key: {"mean": summary[key]["mean"], "std": summary[key]["std"]}
                for key, _, _ in REPORT_METRICS
                if key in summary
            },
        }
    out["majority_class_baseline"] = {
        "label": "majority-class baseline",
        "n_seeds": None,
        "source": "results/ait_clean/results.json",
        "metrics": {
            key: {"mean": direction_record["majority_class_baseline"][key], "std": None}
            for key, _, _ in REPORT_METRICS
            if key in direction_record["majority_class_baseline"]
        },
    }
    return out


def run_direction(direction: dict, datasets: dict, clean: dict, cfg: dict) -> dict:
    train_dataset, _ = datasets[direction["train"]]
    test_dataset, _ = datasets[direction["test"]]
    if direction["train"] == direction["test"]:
        raise SystemExit(f"{direction['name']}: train and test must differ")

    train_indices = labelled_indices(train_dataset)
    test_indices = labelled_indices(test_dataset)
    scaler = fit_scaler(train_dataset, train_indices, cfg)
    train_scaled = apply_scaler(train_dataset, scaler)
    test_scaled = apply_scaler(test_dataset, scaler)

    baselines = {}
    LOGGER.info("%s: flat_all_views", direction["name"])
    baselines["flat_all_views"] = run_baseline(
        "flat_all_views", flat_features,
        train_scaled, train_indices, test_scaled, test_indices, cfg,
    )
    for view, source in enumerate(test_scaled.sources):
        name = f"single_view_{source}"
        LOGGER.info("%s: %s", direction["name"], name)
        baselines[name] = run_baseline(
            name,
            lambda d, i, v=view: single_view_features(d, i, v),
            train_scaled, train_indices, test_scaled, test_indices, cfg,
        )

    clean_record = clean["directions"][direction["name"]]
    structure = clean_record["test_structure"]
    return {
        "direction": direction["name"],
        "label": direction["label"],
        "train_scenario": direction["train"],
        "test_scenario": direction["test"],
        "label_meaning": LABEL_MEANING,
        "n_train": int(len(train_indices)),
        "train_attack_prevalence": float(
            (train_scaled.labels.numpy()[train_indices] == CLASS_ATTACK).mean()
        ),
        "prevalence": {
            "n_test": structure["n_label_known"],
            "n_attack": structure["n_attack"],
            "n_benign": structure["n_benign"],
            "attack_prevalence": structure["attack_prevalence_labelled"],
            "n_attack_blocks": structure["n_attack_blocks"],
        },
        "scaler": {
            "fitted_on": direction["train"],
            "fit_rows": scaler["fit_rows"],
            "note": (
                "the same policy as run_ait_clean.py: fitted on the training "
                "scenario's label_known rows only, refitted per direction, never "
                "shared between directions"
            ),
        },
        "baselines": baselines,
        "reference": reference_rows(clean_record),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_one(value, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not measured"
    return f"{value:.{places}f}"


def fmt_ref(entry, places: int = 4) -> str:
    if entry is None or entry.get("mean") is None:
        return "not measured"
    if entry.get("std") is None:
        return f"{entry['mean']:.{places}f}"
    return f"{entry['mean']:.{places}f} +/- {entry['std']:.{places}f}"


def prevalence_cell(record: dict) -> str:
    p = record["prevalence"]
    return f"{p['attack_prevalence']:.3f} (n={p['n_test']}, {p['n_attack']}a/{p['n_benign']}b)"


def direction_table(record: dict) -> list[str]:
    p = record["prevalence"]
    lines = [
        f"### Direction {record['label']}",
        "",
        f"**Train** `{record['train_scenario']}`, {record['n_train']} label_known "
        f"windows, prevalence {record['train_attack_prevalence']:.3f}. "
        f"**Test** `{record['test_scenario']}`, n = {p['n_test']} label_known "
        f"windows, {p['n_attack']} attack, {p['n_benign']} benign, **attack "
        f"prevalence {p['attack_prevalence']:.3f}**, "
        f"{p['n_attack_blocks']} independent contiguous attack block(s). The "
        f"scaler is fitted on `{record['train_scenario']}` only.",
        "",
        "| model | test prevalence | "
        + " | ".join(name for _, name, _ in REPORT_METRICS)
        + " |",
        "|---|---|" + "---|" * len(REPORT_METRICS),
    ]
    prevalence = prevalence_cell(record)
    for name, row in record["baselines"].items():
        lines.append(
            f"| `{name}` | {prevalence} | "
            + " | ".join(fmt_one(row.get(key), places) for key, _, places in REPORT_METRICS)
            + " |"
        )
    for name, ref in record["reference"].items():
        suffix = f" ({ref['n_seeds']} seeds)" if ref["n_seeds"] else ""
        lines.append(
            f"| *{ref['label']}*{suffix} | {prevalence} | "
            + " | ".join(
                fmt_ref(ref["metrics"].get(key), places) for key, _, places in REPORT_METRICS
            )
            + " |"
        )
    lines += [
        "",
        "Rows in `code font` are this addendum's deterministic baselines: one "
        "value, no `+/-`. Rows in *italics* are quoted from "
        "`results/ait_clean/results.json` for reference and are 5-seed means "
        "with a sample std, except the majority-class baseline, which is "
        "deterministic on a fixed test set.",
        "",
    ]
    return lines


def write_report(results: dict, path: Path) -> None:
    lines = [
        "# Part 28 addendum - non-fusion learned baselines",
        "",
        f"Produced by `experiments/ait_clean/run_ait_baselines.py`"
        + (f", commit `{results['commit']}`" if results.get("commit") else "")
        + f", scikit-learn {results['sklearn_version']}.",
        "",
        "## What the label means",
        "",
        LABEL_MEANING,
        "",
        "This governs these numbers exactly as it governs those in "
        "`results/ait_clean/summary.md`. A single view scoring well here says "
        "that view sees the exfiltration tool; it does not say that view "
        "detects intrusions.",
        "",
        "## What was fitted",
        "",
        f"- `flat_all_views`: one logistic regression on every view's native "
        f"features concatenated, plus the five `view_mask` indicators. Missing "
        f"views are zero-filled, and the indicators are what distinguish a "
        f"zero-filled row from an observed zero.",
        "- `single_view_<name>`: one logistic regression per view, on that view's "
        "native features and its own mask indicator.",
        "",
        f"Solver `{SOLVER}`, `max_iter={MAX_ITER}`, `C={C_VALUE}`, "
        f"`class_weight={CLASS_WEIGHT}` - fixed and stated, not tuned. The same "
        "windows (`label_known` only), the same scaler policy (fitted on the "
        "training scenario alone, refitted per direction) and the same metric "
        "functions as `run_ait_clean.py`.",
        "",
        "## Determinism",
        "",
        DETERMINISM_NOTE,
        "",
        "## Two directions, reported separately and never pooled",
        "",
        "Two directions are two measurements on two test sets with different "
        "prevalences, so a mean over them would be meaningless. There is no "
        "pooled table here, no averaged number across A and B, and no "
        "concatenated test set.",
        "",
    ]
    for record in results["directions"].values():
        lines += direction_table(record)
    lines += [
        "## Files",
        "",
        "- `results/ait_clean/baselines.json` - every number above, both "
        "directions, with each direction's prevalence block and the per-model "
        "feature list.",
        "- `results/ait_clean/baselines.md` - this file.",
        "",
        "`results/ait_clean/results.json` and `results/ait_clean/summary.md` were "
        "not modified by this addendum; the reference rows are read from the "
        "former.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def json_safe(value):
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    _, sklearn_version = require_sklearn()
    cfg = yaml.safe_load(args.config.read_text())

    if not RESULTS_JSON.is_file():
        raise SystemExit(
            f"{RESULTS_JSON} not found. Run experiments/ait_clean/run_ait_clean.py "
            f"first: this addendum quotes its naive and PrismFlow rows and will not "
            f"invent them."
        )
    clean = json.loads(RESULTS_JSON.read_text())

    scenarios = sorted({d[k] for d in cfg["directions"] for k in ("train", "test")})
    if len(scenarios) < 2:
        raise SystemExit("two processed scenarios are required; see /part26")
    datasets = {s: load_scenario(s, cfg) for s in scenarios}

    results = {
        "part": 28,
        "addendum": "non-fusion learned baselines",
        "commit": git_commit(),
        "sklearn_version": sklearn_version,
        "label_meaning": LABEL_MEANING,
        "determinism": DETERMINISM_NOTE,
        "pooling_policy": (
            "Direction A and direction B are reported separately and are never "
            "pooled, averaged or concatenated."
        ),
        "solver": {
            "solver": SOLVER, "max_iter": MAX_ITER, "C": C_VALUE,
            "class_weight": CLASS_WEIGHT,
        },
        "reference_source": str(RESULTS_JSON).replace("\\", "/"),
        "directions": {},
    }
    for direction in cfg["directions"]:
        results["directions"][direction["name"]] = run_direction(
            direction, datasets, clean, cfg
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(json_safe(results), indent=2) + "\n", encoding="utf-8")
    write_report(results, OUT_MD)
    LOGGER.info("wrote %s and %s", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()
