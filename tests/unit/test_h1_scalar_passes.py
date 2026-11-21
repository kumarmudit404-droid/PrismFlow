"""Part 38 tests for the three H1 scalar passes.

WHAT CAN ACTUALLY BE WRONG HERE. The passes are read-only arithmetic over
committed tables, and the statistics themselves come from frozen code by
import. So the failure modes are not "is roc_auc right" -- they are:

- a pass stops being deterministic, or starts depending on the thread count,
  which would make a committed number irreproducible (the same failure
  Part 34 measured on `eniv_eigen`, where `OMP_NUM_THREADS=1` misses Part
  28's committed value by 0.0107);
- the two passes stop agreeing, which would mean one of them is computing
  something other than what its name says. For a binary indicator
  `AUC = 0.5 + |attack_rate - benign_rate| / 2` exactly, so the occupancy
  AUCs and the conditional rates are the same measurement twice and must
  agree to machine precision;
- L24's figures stop being reproducible from `results/lanl/samples.npz`,
  which would invalidate every Part 30 comparison the passes make;
- `attack_host_indicator` stops being the tautology it is documented as,
  which is what justifies excluding it from the ranking;
- a statistic gets quietly reimplemented instead of imported from the frozen
  `experiments/lanl/run_lanl.py`, which is what keeps Part 38's values
  comparable to Part 30's committed ones.

Each of those is a test below. Structural tests skip when their committed
file is absent, so a clean checkout still passes.

THE PASSES WRITE INTO `results/`, so `preserve_outputs` snapshots the three
JSON files and restores them afterwards. A test must never be able to
overwrite a committed result with a worse one, including by passing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.lanl_v2 import h1_conditional_rates as cond
from experiments.lanl_v2 import h1_occupancy_decomposition as occ
from prismflow.data.lanl_cyber1 import load_lanl_config, load_samples

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "lanl_v2"
PART30_SAMPLES = ROOT / "results" / "lanl" / "samples.npz"

AUTH_JSON = OUT / "h1_auth_presence.json"
OCC_JSON = OUT / "h1_occupancy_decomposition.json"
COND_JSON = OUT / "h1_conditional_rates.json"

PASSES = {
    "h1_auth_presence": AUTH_JSON,
    "h1_occupancy_decomposition": OCC_JSON,
    "h1_conditional_rates": COND_JSON,
}

#: docs/KNOWN_LIMITATIONS.md L24, reproduced from results/lanl/samples.npz.
L24_FIGURES = {"attack_with": 410, "attack_total": 410,
               "benign_with": 12323, "benign_total": 18319}

#: The identity below is exact in real arithmetic; this is the floating-point
#: slack allowed on a value of order 1. Measured difference is 0.0.
MACHINE_PRECISION = 1e-12


@pytest.fixture
def preserve_outputs():
    """Snapshot the three committed JSONs and put them back afterwards."""
    saved = {p: p.read_bytes() for p in PASSES.values() if p.exists()}
    try:
        yield
    finally:
        for path, data in saved.items():
            path.write_bytes(data)


def run_pass(name: str, env_extra: dict | None = None) -> bytes:
    """Run one pass in a subprocess and return the bytes it wrote."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.update(env_extra or {})
    done = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "lanl_v2" / f"{name}.py")],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert done.returncode == 0, (
        f"{name} exited {done.returncode}\nstdout:\n{done.stdout}\n"
        f"stderr:\n{done.stderr}")
    return PASSES[name].read_bytes()


def require(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not committed yet")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_h1_auth_presence_bit_exact(preserve_outputs):
    """Re-running the pass reproduces the committed bytes exactly.

    Not "close to": H1's bar is a 0.05 drop and the measured drop is 0.0025,
    so a pass whose output wandered in the fourth decimal could flip the
    verdict without looking wrong.
    """
    committed = require(AUTH_JSON)
    produced = json.loads(run_pass("h1_auth_presence").decode())
    assert produced == committed
    assert produced["h1"]["verdict"] == "DISCONFIRMED"
    assert produced["baselines"]["auth_present_indicator"]["test_auc"] == (
        committed["baselines"]["auth_present_indicator"]["test_auc"])


@pytest.mark.parametrize("name", sorted(PASSES))
def test_h1_passes_unchanged_under_single_thread(name, preserve_outputs):
    """Each pass gives byte-identical output at OMP_NUM_THREADS=1.

    This is the check Part 34 showed is not automatic: a committed
    `eniv_eigen` fails a 1e-6 assertion when re-run at one thread. These
    passes must not inherit that, because their values are quoted without a
    thread count beside them.
    """
    if not PASSES[name].exists():
        pytest.skip(f"{PASSES[name].name} not committed yet")
    default = run_pass(name)
    single = run_pass(name, {"OMP_NUM_THREADS": "1"})
    assert single == default


# ---------------------------------------------------------------------------
# the two passes must be the same measurement twice
# ---------------------------------------------------------------------------

def test_h1_occupancy_decomposition_rates_cross_validate():
    """For a binary indicator, AUC == 0.5 + |attack_rate - benign_rate| / 2.

    The occupancy pass computes the left side with `roc_auc_binary`; the
    conditional-rate pass computes the right side from counts. They are
    independent implementations of one quantity, so disagreement means one of
    them is not measuring what its name says.
    """
    o = require(OCC_JSON)
    c = require(COND_JSON)
    checked = 0
    for view in c["sources"]:
        row = o["whole_test_block"][f"occupancy_{view}"]
        assert row["distinct_values_test"] == 2, (
            f"occupancy_{view} is not a binary indicator; the identity "
            "below does not apply to it")
        gap = c["per_view_rate_comparison"][view]["part38_rate_gap"]
        implied = 0.5 + abs(gap) / 2.0
        assert row["test_auc"] == pytest.approx(implied, abs=MACHINE_PRECISION), (
            f"{view}: decomposition says {row['test_auc']!r}, rates imply "
            f"{implied!r}")
        checked += 1
    assert checked == 4

    # The same algebra on Part 30's rates must return L24's committed 0.6637,
    # which is what makes the cross-Part comparison meaningful at all.
    p30_gap = c["per_view_rate_comparison"]["auth"]["part30_rate_gap"]
    assert 0.5 + abs(p30_gap) / 2.0 == pytest.approx(
        0.6636552213548774, abs=MACHINE_PRECISION)


def test_h1_auth_presence_agrees_with_decomposition():
    """`auth_present_indicator` and `occupancy_auth` are the same column."""
    a = require(AUTH_JSON)
    o = require(OCC_JSON)
    assert (a["baselines"]["auth_present_indicator"]["test_auc"]
            == o["whole_test_block"]["occupancy_auth"]["test_auc"])
    assert a["auth_column_index"] == o["sources"].index("auth") == 0


# ---------------------------------------------------------------------------
# L24's figures must stay reproducible from Part 30's committed table
# ---------------------------------------------------------------------------

def test_h1_conditional_rates_gate_passes():
    """The committed pass recorded the L24 gate as passed, with its checks."""
    c = require(COND_JSON)
    assert c["l24_reproduction_gate"]["status"] == "PASSED"
    for name, check in c["l24_reproduction_gate"]["checks"].items():
        assert check["recomputed"] == check["l24_prose"], name


def test_l24_figures_recomputed_from_part30_table():
    """410/410 and 12,323/18,319, recomputed here, not read from the pass.

    12,323 is prose in docs/KNOWN_LIMITATIONS.md and is not a field in
    results/lanl/results.json, so this is the only place the number is
    actually checked against data rather than against another document.
    """
    if not PART30_SAMPLES.exists():
        pytest.skip("results/lanl/samples.npz not present")
    samples = load_samples(PART30_SAMPLES, load_lanl_config())
    test = np.flatnonzero(np.asarray(samples.block) == "test")
    labels = samples.labels[test]
    auth = samples.view_mask[test, list(samples.sources).index("auth")]
    attack = labels == 1
    benign = labels == 0
    assert int(attack.sum()) == L24_FIGURES["attack_total"]
    assert int(benign.sum()) == L24_FIGURES["benign_total"]
    assert int(auth[attack].sum()) == L24_FIGURES["attack_with"]
    assert int(auth[benign].sum()) == L24_FIGURES["benign_with"]
    # L24's claim is the ratio, not the counts: every attack cell has one.
    assert auth[attack].all()


def test_attack_rows_identical_on_both_tables():
    """Both designs keep every attack cell, so the attack rows must match.

    If this ever fails, a cross-Part per-view comparison is comparing two
    different attack populations and the rate table is meaningless.
    """
    c = require(COND_JSON)
    for view in c["sources"]:
        cmp = c["per_view_rate_comparison"][view]
        assert cmp["part30_attack_rate"] == cmp["part38_attack_rate"], view
    assert c["part30"]["n_attack"] == c["part38"]["n_attack"] == 410


# ---------------------------------------------------------------------------
# the documented tautology, and the frozen imports
# ---------------------------------------------------------------------------

def test_attack_host_indicator_is_tautological():
    """The excluded scalar is 1.0 on every attack row, by construction.

    This is what justifies dropping it from the ranking rather than reading
    its AUC as a finding. Recomputed from the sample table, not read back
    from the pass's own output.
    """
    tag = "v2_days16"
    table = OUT / f"samples_{tag}.npz"
    if not table.exists():
        pytest.skip(f"{table.name} not present")
    cfg = load_lanl_config()
    samples = load_samples(table, cfg)
    hosts = occ.sp.attack_hosts(occ.sp.cells_in_read_range(
        cfg, occ.read_redteam(cfg)["cells"]))
    scalars = occ.build_scalars(samples, cfg, hosts)
    indicator = scalars["attack_host_indicator"]
    attack = samples.labels == 1
    assert attack.any()
    assert (indicator[attack] == 1.0).all()
    # Not vacuous: it is NOT 1.0 everywhere, so it does carry information
    # about which rows are on attack hosts.
    assert indicator.min() == 0.0
    assert "attack_host_indicator" in occ.TAUTOLOGICAL


def test_frozen_imports_intact():
    """The statistics are the frozen objects themselves, not copies.

    A reimplementation could differ in the train-only sign rule or the auth
    column index, and every comparison against Part 30's committed 0.6637
    would silently stop meaning anything.

    Identity (`is`), not `__module__`: `roc_auc_binary` is DEFINED in Part
    28's `experiments/ait_clean/run_ait_clean.py` and re-exported through
    `experiments/lanl/run_lanl.py`, so its `__module__` is the former. What
    matters is that all four Part 38 call sites hold the same function object
    the frozen Part 30 runner holds.
    """
    import experiments.lanl.run_lanl as run_lanl
    from experiments.ait_clean.run_ait_clean import (
        roc_auc_binary as defined_in_part28,
    )
    import experiments.lanl_v2.fill_analysis as fill
    import experiments.lanl_v2.h1_auth_presence as auth

    assert run_lanl.roc_auc_binary is defined_in_part28
    for module in (occ, auth, fill):
        assert module.roc_auc_binary is run_lanl.roc_auc_binary, module.__name__
    for module in (occ, auth):
        assert module.trivial_scalar_values is run_lanl.trivial_scalar_values
    # `trivial_scalar_values` really is defined by the frozen Part 30 runner.
    assert run_lanl.trivial_scalar_values.__module__ == "experiments.lanl.run_lanl"

    # The two rule-(b) predicates must come from the Part 38 sampler, which
    # is the module the synthetic audit covers cell by cell.
    assert occ.sp.attack_hosts.__module__ == "experiments.lanl_v2.sampler"
    assert occ.sp.cells_in_read_range.__module__ == "experiments.lanl_v2.sampler"


def test_no_pass_writes_a_vacuity_key():
    """L29: the column is a confidence. No Part 38 output may name it `vacuity`."""
    for path in PASSES.values():
        if path.exists():
            assert "vacuity" not in json.loads(path.read_text()).keys()
            # nor nested anywhere
            assert '"vacuity"' not in path.read_text()
