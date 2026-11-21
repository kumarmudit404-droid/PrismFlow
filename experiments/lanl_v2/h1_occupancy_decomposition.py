"""What carries the separation H1's disconfirmation left unexplained.

NO TRAINING, NO INFERENCE, NO MODEL. One scalar per row, seed-free, computed
from the committed sample table plus the 4.8 kB redteam file. Independent of
the sweep and of its execution mode.

WHY THIS EXISTS. H1 is disconfirmed: selecting controls without reference to
`auth` left the auth-presence baseline essentially where Part 30 found it
(0.6611838192 against 0.6636552213548774, a drop of 0.0025 against the 0.05
the pre-registration required). Pre-registration section 1 pre-declared that
outcome as meaning "the selection signal has another source". This pass asks
which one, by scoring every selection input as its own trivial baseline on the
same test block.

WHAT IS SCORED, and why each one is here

Part 30's three, imported from the frozen `experiments/lanl/run_lanl.py` so
they cannot drift from the committed values they are compared against:
`total_records_across_views`, `auth_present_indicator`, `n_views_present`.

Per-view occupancy, one indicator per view. `auth_present_indicator` is the
same column as Part 30's and is recomputed here only as a cross-check that
this pass and `h1_auth_presence.py` agree.

`n_non_auth_views_present` -- the quantity selection rule (a) actually gates
on: a cell enters the sample iff it is occupied in at least one of `proc`,
`flows`, `dns`. The nearest thing to "the selection signal" there is.

The two rule-(b) branches, as predicates, not as paraphrases:
`attack_host_indicator` is `cell.computer in attack_hosts`, where the host set
is rebuilt exactly as the sampler builds it -- `read_redteam` then
`cells_in_read_range` then `attack_hosts`, so a host whose only attack lands
outside the read range is not counted. `identity_hash_kept` is
`crc32(computer|window) % 512 == 0`, the control divisor unchanged from Part
30. `fallback_survivor_k2` is `crc32(...) % 4 == 0`, the predicate that decided
which attack-host benign cells survived the k = 2 size fallback.

THE TAUTOLOGY, STATED BEFORE ITS NUMBER IS READ

`attack_host_indicator` is 1 on EVERY attack row by construction: a cell with
`attack = 1` makes its own computer an attack host. Its AUC is therefore a
property of the sampling rule and not a discovered signal, and it is marked
`tautological: true` in the output. Pre-registration section 2 already
declares this -- "Rule (b) uses host-level label knowledge for selection: whether
a computer has any attack cell. Per-host prevalence therefore differs between
attack and non-attack hosts by construction" -- and says what follows: "the
figure that matters is the WITHIN-ATTACK-HOST contrast rather than the overall
prevalence". So every scalar is reported twice: over the whole test block, and
restricted to attack-host rows, where the tautology is held constant.

A near-constant scalar has an uninformative AUC, so `positive_rate` is
reported beside every indicator. Read the rate first.

    PYTHONPATH=. ./.venv/Scripts/python.exe \
        experiments/lanl_v2/h1_occupancy_decomposition.py
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lanl.run_lanl import (                           # noqa: E402
    roc_auc_binary,
    trivial_scalar_values,
)
from experiments.lanl_v2 import sampler as sp                     # noqa: E402
from prismflow.data.lanl_cyber1 import (                          # noqa: E402
    load_lanl_config,
    load_samples,
    read_redteam,
)

OUT = ROOT / "results" / "lanl_v2"

#: results/lanl_v2/h1_auth_presence.json, baselines.<name>.test_auc
H1_TEST_AUC = 0.6611838191552647
H1_SCALAR = "auth_present_indicator"
#: Pre-registration section 2: unchanged from Part 30.
CONTROL_DIVISOR = 512
#: results/lanl_v2/build_v2_days16.json, size_fallback.attack_host_benign_share_k
FALLBACK_K = 2

#: Scalars whose AUC is a property of the sampling rule, not a signal.
TAUTOLOGICAL = {"attack_host_indicator"}


def cell_keys(samples, cfg):
    """(computer_bytes, window_index) per row, as the sampler's predicates see it.

    `build_lanl_samples` stores `computer.decode()` and
    `window * window_seconds + 1`, so both are invertible. The round-trip is
    asserted rather than assumed: a silent off-by-one here would compute the
    crc of a different cell and the hash scalars would be noise that looks
    like a measurement.
    """
    starts = np.asarray(samples.window_start, dtype=np.float64)
    windows = np.rint((starts - 1.0) / cfg.window_seconds).astype(np.int64)
    if not np.array_equal(windows * cfg.window_seconds + 1, starts):
        raise AssertionError(
            "window_start does not invert to a window index under "
            f"window_seconds={cfg.window_seconds}; refusing to hash the "
            "wrong cell"
        )
    return [(c.encode(), int(w)) for c, w in zip(samples.computer, windows)]


def build_scalars(samples, cfg, hosts) -> dict[str, np.ndarray]:
    scalars = dict(trivial_scalar_values(samples))
    mask = samples.view_mask
    sources = list(samples.sources)

    for i, name in enumerate(sources):
        key = f"occupancy_{name}"
        scalars[key] = mask[:, i].astype(np.float64)

    non_auth = [i for i, s in enumerate(sources) if s != "auth"]
    scalars["n_non_auth_views_present"] = (
        mask[:, non_auth].sum(axis=1).astype(np.float64))

    keys = cell_keys(samples, cfg)
    crc = np.asarray([zlib.crc32(c + b"|" + str(w).encode())
                      for c, w in keys], dtype=np.int64)
    scalars["attack_host_indicator"] = np.asarray(
        [c.encode() in hosts for c in samples.computer], dtype=np.float64)
    scalars["identity_hash_kept"] = (crc % CONTROL_DIVISOR == 0).astype(np.float64)
    scalars["fallback_survivor_k2"] = (
        crc % (2 ** FALLBACK_K) == 0).astype(np.float64)
    return scalars


def score_block(labels, scalars, tr, te, label: str) -> dict:
    """Train-only orientation, then test AUC. Part 30's rule, unchanged."""
    out = {}
    for name, score in scalars.items():
        n_classes_train = len(np.unique(labels[tr]))
        n_classes_test = len(np.unique(labels[te]))
        row = {
            "positive_rate_test": float(np.mean(score[te] > 0)),
            "distinct_values_test": int(np.unique(score[te]).size),
            "tautological": name in TAUTOLOGICAL,
        }
        if n_classes_train != 2 or n_classes_test != 2:
            row["test_auc"] = None
            row["test_auc_status"] = (
                f"not measured: {label} is not two-class "
                f"(train {n_classes_train}, test {n_classes_test})")
            out[name] = row
            continue
        train_auc = roc_auc_binary(labels[tr], score[tr])
        sign = 1.0 if not np.isfinite(train_auc) or train_auc >= 0.5 else -1.0
        test_auc = roc_auc_binary(labels[te], sign * score[te])
        row["train_auc_as_measured"] = float(train_auc)
        row["orientation_fixed_on_train"] = "as-is" if sign > 0 else "negated"
        if not np.isfinite(test_auc):
            row["test_auc"] = None
            row["test_auc_status"] = (
                "not measured: the scalar is constant on this block, so no "
                "AUC is defined")
        else:
            row["test_auc"] = float(test_auc)
            row["vs_h1_auth_presence"] = float(test_auc) - H1_TEST_AUC
        out[name] = row
    return out


def main(tag: str = "v2_days16") -> int:
    cfg = load_lanl_config()
    samples = load_samples(OUT / f"samples_{tag}.npz", cfg)
    labels = samples.labels
    block = np.asarray(samples.block)
    tr = np.flatnonzero(block == "train")
    te = np.flatnonzero(block == "test")

    redteam = read_redteam(cfg)
    in_range = sp.cells_in_read_range(cfg, redteam["cells"])
    hosts = sp.attack_hosts(in_range)

    scalars = build_scalars(samples, cfg, hosts)
    on_attack_host = scalars["attack_host_indicator"] > 0

    full = score_block(labels, scalars, tr, te, "the test block")
    tr_h = tr[on_attack_host[tr]]
    te_h = te[on_attack_host[te]]
    within = score_block(labels, scalars, tr_h, te_h,
                         "the attack-host test rows")

    ranked = sorted(
        ((n, r["test_auc"]) for n, r in full.items()
         if r["test_auc"] is not None and not r["tautological"]),
        key=lambda kv: abs(kv[1] - 0.5), reverse=True)

    out = {
        "part": 38,
        "what": (
            "Every selection input scored as its own trivial baseline, to ask "
            "which signal carries the separation H1's disconfirmation left "
            "unexplained. Seed-free scalars, no training, no inference, no "
            "model."
        ),
        "not_a_sweep_result": (
            "Computed from results/lanl_v2/samples_%s.npz plus redteam.txt.gz "
            "alone. Independent of the 40-unit sweep and of its execution "
            "mode: no thread count applies and no ENIV is involved." % tag
        ),
        "tag": tag,
        "h1_reference": {
            "scalar": H1_SCALAR,
            "test_auc": H1_TEST_AUC,
            "source": "results/lanl_v2/h1_auth_presence.json",
            "verdict": "DISCONFIRMED",
        },
        "orientation_rule": (
            "Sign chosen on the TRAIN rows of the block being scored, applied "
            "unchanged to its test rows. Part 30's rule, imported from the "
            "frozen experiments/lanl/run_lanl.py."
        ),
        "tautology_warning": (
            "attack_host_indicator is 1 on every attack row BY CONSTRUCTION: "
            "an attack cell makes its own computer an attack host. Its AUC "
            "measures the sampling rule, not a discovered signal, and it is "
            "excluded from the ranking. Pre-registration section 2 declares "
            "this and says the figure that matters is the within-attack-host "
            "contrast, which is the second block below."
        ),
        "read_positive_rate_first": (
            "A near-constant indicator has an uninformative AUC. "
            "positive_rate_test and distinct_values_test are reported beside "
            "every scalar for that reason."
        ),
        "sources": list(samples.sources),
        "control_divisor": CONTROL_DIVISOR,
        "fallback_k": FALLBACK_K,
        "attack_hosts_in_read_range": len(hosts),
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "n_train_on_attack_hosts": int(tr_h.size),
        "n_test_on_attack_hosts": int(te_h.size),
        "test_prevalence": float((labels[te] == 1).mean()),
        "test_prevalence_on_attack_hosts": (
            float((labels[te_h] == 1).mean()) if te_h.size else None),
        "whole_test_block": full,
        "within_attack_hosts": within,
        "ranking_excluding_tautological": [
            {"scalar": n, "test_auc": a, "distance_from_chance": abs(a - 0.5)}
            for n, a in ranked
        ],
    }

    path = OUT / "h1_occupancy_decomposition.json"
    path.write_text(json.dumps(out, indent=1))

    def table(title, rows, n):
        print("\n%s  (n_test = %d)" % (title, n))
        print("%-30s %7s %8s %12s %12s %10s"
              % ("scalar", "pos", "distinct", "train AUC", "test AUC", "vs H1"))
        for name, r in rows.items():
            auc = r.get("test_auc")
            print("%-30s %7.4f %8d %12s %12s %10s%s"
                  % (name, r["positive_rate_test"], r["distinct_values_test"],
                     ("%.10f" % r["train_auc_as_measured"])
                     if "train_auc_as_measured" in r else "n/a",
                     "not measured" if auc is None else "%.10f" % auc,
                     "n/a" if auc is None
                     else "%+.6f" % r["vs_h1_auth_presence"],
                     "   <- TAUTOLOGICAL" if r["tautological"] else ""))

    print("H1 occupancy decomposition, tag %s" % tag)
    print("  attack hosts in read range: %d" % len(hosts))
    print("  test rows %d (prevalence %.10f), of which on attack hosts %d"
          % (te.size, out["test_prevalence"], te_h.size))
    table("WHOLE TEST BLOCK", full, te.size)
    table("WITHIN ATTACK HOSTS", within, te_h.size)
    print("\nRanked by distance from chance, tautological excluded:")
    for row in out["ranking_excluding_tautological"]:
        print("  %-30s %.10f  (%+.6f from chance)"
              % (row["scalar"], row["test_auc"],
                 row["test_auc"] - 0.5))
    print("\nwrote results/lanl_v2/h1_occupancy_decomposition.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
