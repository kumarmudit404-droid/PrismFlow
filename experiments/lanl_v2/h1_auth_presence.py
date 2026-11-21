"""H1's trivial baselines on Part 38's sample table. No training, no model.

WHY THIS FILE EXISTS. `run_lanl_v2.py` never computes H1. The statistic it
needs is defined at `experiments/lanl_v2/metrics.py:80`
(`auth_presence_auc`), but the runner's only metric calls are `m.evaluate`
and `m.constant_baseline_brier`, so no committed file carried the baseline and
H1 was `not measured` until this pass. The sweep does not need to finish for
this: the baseline is seed-free and is a property of the SAMPLE, not of a
trained unit.

WHAT H1 IS. Pre-registration section 1:

    H1 (L24 -- selection vs view drop). With control cells drawn without
    reference to the `auth` stream, the auth-presence-only trivial baseline on
    the test block falls materially below Part 30's committed test AUC of
    0.6637 [...]. "Materially" is pre-set at a drop of at least 0.05 AUC. That
    baseline is seed-free, so the bar applies to the single recomputed value.

WHY IT IMPORTS THE FROZEN PART 30 RUNNER. The scalars and the orientation rule
are not re-implemented here. `trivial_scalar_values` and `roc_auc_binary` are
imported from the frozen `experiments/lanl/run_lanl.py`, so Part 38's value is
produced by the same code that produced the 0.6637 it is compared against. A
re-implementation could differ in the sign rule or in the auth column index
and the comparison would be meaningless. Frozen modules are import-only and
nothing here edits one.

ORIENTATION, as Part 30 fixed it: the sign is chosen on the TRAIN block only
and applied unchanged to test, so a scalar that ranks attacks below benigns on
train is negated without test ever being consulted.

ALL THREE SCALARS ARE REPORTED, not only auth presence. Part 30 committed the
three together and only `auth_present_indicator` carries H1's bar; the other
two are what make a disconfirmation readable, because L24's open clause asks
which of the remaining occupancy signals carries the separation. They are
labelled `barred: false`.

    PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_auth_presence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lanl.run_lanl import (                           # noqa: E402
    roc_auc_binary,
    trivial_scalar_values,
)
from experiments.lanl_v2 import metrics as m                      # noqa: E402
from prismflow.data.lanl_cyber1 import (                          # noqa: E402
    load_lanl_config,
    load_samples,
)

OUT = ROOT / "results" / "lanl_v2"

#: `results/lanl/results.json`,
#: phase3_diagnostics.trivial_baselines.baselines.<name>.test_auc
PART30_TEST_AUC = {
    "auth_present_indicator": 0.6636552213548774,
    "total_records_across_views": 0.6842788175411642,
    "n_views_present": 0.5624544155807845,
}
#: Pre-registration section 1: "a drop of at least 0.05 AUC".
H1_MIN_DROP = 0.05
H1_SCALAR = "auth_present_indicator"


def main(tag: str = "v2_days16") -> int:
    cfg = load_lanl_config()
    samples = load_samples(OUT / f"samples_{tag}.npz", cfg)
    labels = samples.labels
    block = np.asarray(samples.block)
    # Same row selection as run_lanl_v2.unit: the whole block, no
    # label_known filter. LANLSamples documents label_known as all True for
    # LANL, and the runner does not filter on it either, so an extra filter
    # here would score different rows than the sweep does.
    tr = np.flatnonzero(block == "train")
    te = np.flatnonzero(block == "test")
    if not np.array_equal(np.unique(samples.label_known), np.array([True])):
        raise AssertionError(
            "label_known is not all True on this table; the row selection "
            "above assumes it is (LANLSamples docstring). Stopping rather "
            "than silently scoring a different population."
        )

    scalars = trivial_scalar_values(samples)
    auth_col = list(samples.sources).index("auth")

    out = {
        "part": 38,
        "what": (
            "H1's trivial baselines on Part 38's sample table. Seed-free, no "
            "training, no model, no inference. One scalar per row."
        ),
        "not_a_sweep_result": (
            "Computed from results/lanl_v2/samples_%s.npz alone. Independent "
            "of the 40-unit sweep and of its execution mode: no thread count "
            "applies and no ENIV is involved." % tag
        ),
        "tag": tag,
        "orientation_rule": (
            "Sign chosen on the TRAIN block only and applied unchanged to "
            "test. Imported from the frozen experiments/lanl/run_lanl.py so "
            "Part 38's value and Part 30's 0.6637 come from the same code."
        ),
        "auth_column_index": int(auth_col),
        "sources": list(samples.sources),
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "test_prevalence": float((labels[te] == 1).mean()),
        "baselines": {},
    }

    for name, score in scalars.items():
        train_auc = roc_auc_binary(labels[tr], score[tr])
        sign = 1.0 if not np.isfinite(train_auc) or train_auc >= 0.5 else -1.0
        test_auc = m.auth_presence_auc(labels[te], sign * score[te])
        row = {
            "train_auc_as_measured": float(train_auc),
            "orientation_fixed_on_train": "as-is" if sign > 0 else "negated",
            "test_auc": test_auc,
            "n_test": int(te.size),
            "part30_test_auc": PART30_TEST_AUC[name],
            "barred": name == H1_SCALAR,
        }
        if test_auc is None:
            row["test_auc_status"] = (
                "not measured: the test block is not two-class, so no AUC "
                "is defined"
            )
        else:
            row["drop_from_part30"] = PART30_TEST_AUC[name] - test_auc
        out["baselines"][name] = row

    h1 = out["baselines"][H1_SCALAR]
    out["h1"] = {
        "scalar": H1_SCALAR,
        "part30_test_auc": PART30_TEST_AUC[H1_SCALAR],
        "min_drop_required": H1_MIN_DROP,
        "bar_test_auc_at_or_below": PART30_TEST_AUC[H1_SCALAR] - H1_MIN_DROP,
        "part38_test_auc": h1["test_auc"],
        "drop_achieved": h1.get("drop_from_part30"),
        "verdict": (
            "NOT MEASURED" if h1["test_auc"] is None
            else "CONFIRMED" if h1["drop_from_part30"] >= H1_MIN_DROP
            else "DISCONFIRMED"
        ),
        "verdict_means": (
            "CONFIRMED: the auth-presence baseline fell by at least 0.05, so "
            "selection on non-auth occupancy removed the separation auth "
            "presence alone used to carry. DISCONFIRMED: it did not fall that "
            "far, and per pre-registration section 1 the selection signal has "
            "another source, which is a reportable finding in its own right. "
            "Neither verdict changes any limitation's status: that needs the "
            "full sweep and the review gate (pre-registration section 7.6)."
        ),
        "not_a_status_change": (
            "docs/KNOWN_LIMITATIONS.md is not edited by this pass. L24's "
            "closing condition is a condition on the DESIGN, met by selection "
            "rule (a); H1 is a separate falsifiable prediction about what that "
            "design produces."
        ),
    }

    path = OUT / "h1_auth_presence.json"
    path.write_text(json.dumps(out, indent=1))

    print("H1 trivial baselines, tag %s" % tag)
    print("  train rows %d, test rows %d, test prevalence %.10f"
          % (tr.size, te.size, out["test_prevalence"]))
    print("  auth column index %d of %s" % (auth_col, list(samples.sources)))
    print("%-28s %-10s %12s %12s %12s %s"
          % ("scalar", "orient", "train AUC", "test AUC", "Part 30", "drop"))
    for name, row in out["baselines"].items():
        print("%-28s %-10s %12.10f %12s %12.10f %s%s"
              % (name, row["orientation_fixed_on_train"],
                 row["train_auc_as_measured"],
                 "not measured" if row["test_auc"] is None
                 else "%.10f" % row["test_auc"],
                 row["part30_test_auc"],
                 "n/a" if row["test_auc"] is None
                 else "%+.10f" % -row["drop_from_part30"],
                 "   <- H1" if row["barred"] else ""))
    print("\nH1: bar is test AUC <= %.10f (0.6637 - 0.05). measured %s -> %s"
          % (out["h1"]["bar_test_auc_at_or_below"],
             "not measured" if h1["test_auc"] is None
             else "%.10f" % h1["test_auc"],
             out["h1"]["verdict"]))
    print("wrote results/lanl_v2/h1_auth_presence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
