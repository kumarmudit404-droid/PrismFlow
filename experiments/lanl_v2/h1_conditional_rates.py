"""Part 30's 410/410 against 12,323/18,319 statistic, on both sample tables.

NO TRAINING, NO INFERENCE, NO MODEL. Counts and ratios only, seed-free,
computed from two committed sample tables. Independent of the sweep and of its
execution mode.

WHAT THIS ANSWERS. H1 is disconfirmed and the occupancy decomposition
(`h1_occupancy_decomposition.py`) says auth presence still reaches test AUC
0.6612 on a sample whose selection never consulted the auth stream. An AUC is
a ranking; it does not say whether the underlying CONTINGENCY survived. L24's
claim is a contingency:

    Every attack sample has an `auth` record while about a third of benign
    samples do not (410 of 410 attack against 12,323 of 18,319 benign in the
    test block), so auth presence alone reaches test AUC 0.6637.

So the question is whether 410/410 still holds. This pass recomputes the
contingency on BOTH tables, for ALL FOUR views, and reports them side by side.

WHERE THE COMPARISON NUMBERS COME FROM, and a deviation from the brief. The
brief asks for "Part 30's committed figures from results/lanl/results.json".
**12,323 is not in that file.** It is prose in `docs/KNOWN_LIMITATIONS.md`
L24, which cites `phase3_diagnostics.arm_coverage`,
`phase3_diagnostics.trivial_baselines` and `results/lanl/samples.npz`; the
first two carry neither count. Rather than quote prose, this pass RECOMPUTES
Part 30's side from `results/lanl/samples.npz`, which is committed and
tracked, and asserts that the recomputation reproduces L24's two figures. A
quoted number that cannot be resolved to a path is not evidence; a
recomputation that reproduces it is.

THE TWO TABLES ARE NOT ROW-COMPARABLE. Pre-registration section 2: Part 38's
sample "is not a subset or a superset of Part 30's, and the two sample tables
are not comparable row-for-row. Every cross-Part comparison in this programme
is a comparison of designs, not of runs." So a RATE may be compared between
them; a count difference may not be read as a change in the network.

    PYTHONPATH=. ./.venv/Scripts/python.exe \
        experiments/lanl_v2/h1_conditional_rates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from prismflow.data.lanl_cyber1 import (                          # noqa: E402
    load_lanl_config,
    load_samples,
)

OUT = ROOT / "results" / "lanl_v2"
PART30_SAMPLES = ROOT / "results" / "lanl" / "samples.npz"

#: docs/KNOWN_LIMITATIONS.md L24, the two figures this pass must reproduce
#: from results/lanl/samples.npz before any comparison is made.
L24_AUTH_ATTACK_WITH = 410
L24_AUTH_ATTACK_TOTAL = 410
L24_AUTH_BENIGN_WITH = 12323
L24_AUTH_BENIGN_TOTAL = 18319


def contingency(samples, idx) -> dict:
    """Per view: how many attack and benign rows hold a record of that view."""
    labels = samples.labels[idx]
    mask = samples.view_mask[idx]
    attack = labels == 1
    benign = labels == 0
    out = {
        "n_rows": int(idx.size),
        "n_attack": int(attack.sum()),
        "n_benign": int(benign.sum()),
        "prevalence": float(attack.mean()) if idx.size else None,
        "per_view": {},
    }
    for i, name in enumerate(samples.sources):
        aw = int(mask[attack, i].sum())
        bw = int(mask[benign, i].sum())
        row = {
            "attack_with_record": aw,
            "attack_total": int(attack.sum()),
            "benign_with_record": bw,
            "benign_total": int(benign.sum()),
            "attack_rate": float(aw / attack.sum()) if attack.sum() else None,
            "benign_rate": float(bw / benign.sum()) if benign.sum() else None,
        }
        row["rate_gap_attack_minus_benign"] = (
            None if row["attack_rate"] is None or row["benign_rate"] is None
            else row["attack_rate"] - row["benign_rate"])
        row["attack_rate_is_exactly_one"] = aw == int(attack.sum())
        out["per_view"][name] = row
    return out


def test_index(samples) -> np.ndarray:
    return np.flatnonzero(np.asarray(samples.block) == "test")


def main(tag: str = "v2_days16") -> int:
    cfg = load_lanl_config()

    p30 = load_samples(PART30_SAMPLES, cfg)
    p38 = load_samples(OUT / f"samples_{tag}.npz", cfg)
    if tuple(p30.sources) != tuple(p38.sources):
        raise AssertionError(
            f"view order differs: Part 30 {p30.sources}, Part 38 "
            f"{p38.sources}; the per-view comparison below would misalign")

    c30 = contingency(p30, test_index(p30))
    c38 = contingency(p38, test_index(p38))

    # Gate: reproduce L24's two figures before comparing anything to them.
    a30 = c30["per_view"]["auth"]
    checks = {
        "auth_attack_with_record": (a30["attack_with_record"],
                                    L24_AUTH_ATTACK_WITH),
        "auth_attack_total": (a30["attack_total"], L24_AUTH_ATTACK_TOTAL),
        "auth_benign_with_record": (a30["benign_with_record"],
                                    L24_AUTH_BENIGN_WITH),
        "auth_benign_total": (a30["benign_total"], L24_AUTH_BENIGN_TOTAL),
    }
    mismatched = {k: {"recomputed": got, "l24_prose": want}
                  for k, (got, want) in checks.items() if got != want}
    if mismatched:
        raise AssertionError(
            "recomputation of Part 30's table does not reproduce L24's "
            f"figures: {mismatched}. Stopping rather than comparing Part 38 "
            "against a number that cannot be reproduced.")

    out = {
        "part": 38,
        "what": (
            "Per-view record-presence contingency on the TEST block of both "
            "sample tables. Counts and rates only, seed-free, no training, no "
            "inference, no model."
        ),
        "not_a_sweep_result": (
            "Computed from results/lanl/samples.npz and "
            "results/lanl_v2/samples_%s.npz alone. Independent of the 40-unit "
            "sweep and of its execution mode." % tag
        ),
        "l24_reproduction_gate": {
            "status": "PASSED",
            "what_was_checked": (
                "L24's 410/410 attack and 12,323/18,319 benign auth figures, "
                "recomputed from results/lanl/samples.npz. 12,323 is prose in "
                "docs/KNOWN_LIMITATIONS.md and is not a field in "
                "results/lanl/results.json, so it is reproduced from the "
                "committed table rather than quoted."
            ),
            "checks": {k: {"recomputed": got, "l24_prose": want}
                       for k, (got, want) in checks.items()},
        },
        "not_row_comparable": (
            "Pre-registration section 2: Part 38's sample is not a subset or "
            "a superset of Part 30's and the two tables are not comparable "
            "row-for-row. Compare RATES, never counts, and read any "
            "difference as a difference of designs."
        ),
        "sources": list(p38.sources),
        "part30": c30,
        "part38": c38,
        "per_view_rate_comparison": {},
    }

    for name in p38.sources:
        r30 = c30["per_view"][name]
        r38 = c38["per_view"][name]
        out["per_view_rate_comparison"][name] = {
            "part30_attack_rate": r30["attack_rate"],
            "part38_attack_rate": r38["attack_rate"],
            "attack_rate_change": r38["attack_rate"] - r30["attack_rate"],
            "part30_benign_rate": r30["benign_rate"],
            "part38_benign_rate": r38["benign_rate"],
            "benign_rate_change": r38["benign_rate"] - r30["benign_rate"],
            "part30_rate_gap": r30["rate_gap_attack_minus_benign"],
            "part38_rate_gap": r38["rate_gap_attack_minus_benign"],
            "rate_gap_change": (r38["rate_gap_attack_minus_benign"]
                                - r30["rate_gap_attack_minus_benign"]),
            "part30_attack_rate_is_exactly_one":
                r30["attack_rate_is_exactly_one"],
            "part38_attack_rate_is_exactly_one":
                r38["attack_rate_is_exactly_one"],
        }

    auth = out["per_view_rate_comparison"]["auth"]
    out["auth_verdict"] = {
        "question": (
            "Did L24's contingency -- every attack cell holds an auth record, "
            "a third of benign cells do not -- survive a selection that never "
            "consults the auth stream?"
        ),
        "part30_attack_rate": auth["part30_attack_rate"],
        "part38_attack_rate": auth["part38_attack_rate"],
        "still_exactly_one_on_part38": auth["part38_attack_rate_is_exactly_one"],
        "rate_gap_change": auth["rate_gap_change"],
        "reading": (
            "SURVIVED: the attack-side rate is still 1.0, so every attack row "
            "holds an auth record even though auth was never consulted in "
            "selection -- the contingency is a property of the data."
            if auth["part38_attack_rate_is_exactly_one"] else
            "DID NOT SURVIVE INTACT: the attack-side rate is no longer "
            "exactly 1.0, so some attack rows now hold no auth record. The "
            "rate gap, not the 410/410 framing, is what carries over."
        ),
        "what_this_does_not_say": (
            "Nothing here is about the model. These are sample-table counts; "
            "no arm, no seed and no trained unit is involved, and no "
            "limitation's status changes on this pass (pre-registration "
            "section 7.6)."
        ),
    }

    path = OUT / "h1_conditional_rates.json"
    path.write_text(json.dumps(out, indent=1))

    print("Conditional rates on the TEST block, both tables")
    print("  L24 reproduction gate: PASSED "
          "(410/410 attack, 12,323/18,319 benign, recomputed)")
    print("  Part 30 test: %d rows, %d attack, %d benign, prevalence %.10f"
          % (c30["n_rows"], c30["n_attack"], c30["n_benign"],
             c30["prevalence"]))
    print("  Part 38 test: %d rows, %d attack, %d benign, prevalence %.10f"
          % (c38["n_rows"], c38["n_attack"], c38["n_benign"],
             c38["prevalence"]))
    print("\n%-6s %-24s %-24s %10s" % ("view", "Part 30 attack/benign",
                                       "Part 38 attack/benign", "gap chg"))
    for name in p38.sources:
        r30 = c30["per_view"][name]
        r38 = c38["per_view"][name]
        cmp = out["per_view_rate_comparison"][name]
        print("%-6s %-24s %-24s %+10.6f"
              % (name,
                 "%d/%d  %d/%d" % (r30["attack_with_record"], r30["attack_total"],
                                   r30["benign_with_record"], r30["benign_total"]),
                 "%d/%d  %d/%d" % (r38["attack_with_record"], r38["attack_total"],
                                   r38["benign_with_record"], r38["benign_total"]),
                 cmp["rate_gap_change"]))
    print("\n%-6s %10s %10s %10s %10s" % ("view", "P30 atk", "P38 atk",
                                          "P30 ben", "P38 ben"))
    for name in p38.sources:
        c = out["per_view_rate_comparison"][name]
        print("%-6s %10.6f %10.6f %10.6f %10.6f"
              % (name, c["part30_attack_rate"], c["part38_attack_rate"],
                 c["part30_benign_rate"], c["part38_benign_rate"]))
    print("\nauth: attack rate %.6f -> %.6f, exactly 1.0 on Part 38: %s"
          % (auth["part30_attack_rate"], auth["part38_attack_rate"],
             auth["part38_attack_rate_is_exactly_one"]))
    print("  %s" % out["auth_verdict"]["reading"])
    print("\nwrote results/lanl_v2/h1_conditional_rates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
