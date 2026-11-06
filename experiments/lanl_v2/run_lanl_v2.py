r"""Part 38 step 1 runner. New code; experiments/lanl/run_lanl.py is FROZEN.

    .venv\Scripts\python.exe experiments/lanl_v2/run_lanl_v2.py reproduce --days 1
    .venv\Scripts\python.exe experiments/lanl_v2/run_lanl_v2.py build --rule v2 --days 6
    .venv\Scripts\python.exe experiments/lanl_v2/run_lanl_v2.py smoke --seeds 0,1

WHAT THIS IS NOT
----------------
Not a copy of `run_lanl.py`. The scaler and the trainer come from
`prismflow.training`, the dataset-agnostic package that resolved L25, so the
arithmetic is the one already pinned by
`tests/unit/test_training_agnostic.py` rather than a third mirror of it.

THE CONFIDENCE KEY
------------------
`run_lanl.py` writes `collected["vacuity_confidence"]` under the npz key
`"vacuity"`. The column holds a CONFIDENCE -- `prismflow.evaluation.protocol`
collects `output.confidence` -- so 0.0 on an evidence-free row is the right
value and the NAME is the defect (L29). That file is frozen and is not edited.
This runner writes the key `confidence` and asserts, before every write, that
no key named `vacuity` is present.

MODES (L28)
-----------
`--mode threads1` and `--mode default` set the thread count and RECORD what
`torch.get_num_threads()` actually reports afterwards, plus this machine's
measured default. A figure is conditional on its thread count; ENIV
reproduces only at the count it was computed at.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lanl_v2 import metrics as m                      # noqa: E402
from experiments.lanl_v2 import sampler as sp                     # noqa: E402
from prismflow.data.lanl_cyber1 import (                          # noqa: E402
    LANLMultiViewDataset,
    load_lanl_config,
    load_samples,
    read_redteam,
    save_samples,
    subset,
)
from prismflow.data.loaders import iter_batches                   # noqa: E402
from prismflow.evaluation.protocol import collect_predictions      # noqa: E402
from prismflow.training import apply_scaler, fit_scaler, train     # noqa: E402

OUT = ROOT / "results" / "lanl_v2"
DAY = 86400
WINDOW = 300

# The committed Part 30 figures the reproduction gate compares against.
# results/lanl/results.json, split_report.per_split.<block>.per_day
PART30_RESULTS = ROOT / "results" / "lanl" / "results.json"

MODEL_CFG = {"hidden_dims": [64, 64], "feature_dim": 32, "dropout": 0.0}
TRAINING_CFG = {"epochs": 40, "batch_size": 64, "lr": 0.001,
                "anneal_epochs": 10, "per_view_loss_weight": 1.0}
SCALING_CFG = {"method": "zscore", "present_rows_only": True, "epsilon": 1.0e-8}

# Pre-registration section 7.
MIN_ATTACK_WINDOWS_PER_SPLIT = 10
MAX_TOTAL_SAMPLES = 400_000


class StopCondition(RuntimeError):
    """A pre-registered stopping rule fired. Report, do not work around."""


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def peak_rss_mb():
    """This process's PEAK working set, from the OS high-water mark.

    Not a sample: Windows tracks the maximum itself, so a peak reached between
    two calls is still reported. The `gzip -dc` children are separate
    processes of a few MB each and are not counted. Returns None off Windows
    rather than a wrong number.
    """
    import ctypes
    import ctypes.wintypes as wt

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
            (n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    try:
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        # Without these, the pseudo-handle comes back as a 32-bit int and the
        # call fails: the default restype is c_int, not HANDLE.
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        ok = psapi.GetProcessMemoryInfo(
            k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    except (OSError, AttributeError):
        return None
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1) if ok else None


# ---------------------------------------------------------------------------
# modes (L28)
# ---------------------------------------------------------------------------

def machine_default_threads() -> int:
    """Measured, never assumed: L28 records 8 here against run 1's claim of 12."""
    return int(torch.get_num_threads())


def apply_mode(name: str) -> dict:
    default = machine_default_threads()
    if name == "threads1":
        torch.set_num_threads(1)
        concurrency = 11
    elif name == "default":
        torch.set_num_threads(default)
        concurrency = 1
    else:
        raise SystemExit("--mode must be threads1 or default")
    return {
        "mode": name,
        "torch_threads_observed": int(torch.get_num_threads()),
        "concurrency": concurrency,
        "machine_default_threads_measured": default,
    }


# ---------------------------------------------------------------------------
# ranges
# ---------------------------------------------------------------------------

def prefix_config(cfg, days: int):
    """Read only the first `days` days, with a split that fits inside them.

    Used by the reproduction gate and the smoke. The real sweep uses the
    committed configs/data/lanl.yaml split untouched.

    Every boundary lands on a 300 s window edge, which is what keeps a window
    from belonging to two blocks. Asserted, not assumed.
    """
    if days >= 16:
        return cfg
    if days <= 2:                       # single-block prefix: train only
        split = {"train": {"start": 1, "end": days * DAY, "days": [1, days]},
                 "gap": {"start": days * DAY + 1, "end": days * DAY, "days": []},
                 "test": {"start": days * DAY + 1, "end": days * DAY, "days": []}}
    else:
        # train | one-day gap | test, in thirds of the prefix.
        train_days = max(1, days - 3)
        gap_day = train_days + 1
        split = {
            "train": {"start": 1, "end": train_days * DAY,
                      "days": [1, train_days]},
            "gap": {"start": train_days * DAY + 1, "end": gap_day * DAY,
                    "days": [gap_day, gap_day]},
            "test": {"start": gap_day * DAY + 1, "end": days * DAY,
                     "days": [gap_day + 1, days]},
        }
    for name, b in split.items():
        if b["end"] >= b["start"] and ((b["start"] - 1) % WINDOW or b["end"] % WINDOW):
            raise AssertionError(f"{name} does not align to the {WINDOW}s grid")
    return dataclasses.replace(cfg, read_until=days * DAY, split=split)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def sample_table(cfg, rule: str) -> tuple[object, dict]:
    """Select cells under `rule`, then extract features for exactly those."""
    t0 = time.time()
    redteam = read_redteam(cfg)
    cells = redteam["cells"]                      # whole file, every day
    in_range = sp.cells_in_read_range(cfg, cells)
    hosts = sp.attack_hosts(in_range)             # L27 hosts: read range only

    if rule == "part30":
        views, keep = cfg.views, sp.part30_keep
    elif rule == "v2":
        views, keep = sp.NON_AUTH_VIEWS, sp.v2_keep
    else:
        raise SystemExit("--rule must be part30 or v2")

    selection = sp.select_cells(cfg, cells, hosts, keep, views)
    t_select = time.time() - t0

    # Only train/test cells can become rows, so carrying the gap days and the
    # out-of-range cases into the feature pass buys nothing but accumulators.
    selected = sp.restrict_to_blocks(cfg, selection["selected"])
    fallback = None
    if rule == "v2":
        fallback = sp.size_fallback(cfg, selected, cells, hosts, MAX_TOTAL_SAMPLES)
        selected = fallback.pop("selected")

    t1 = time.time()
    samples = sp.build(cfg, selected, redteam)
    t_features = time.time() - t1

    report = {
        "rule": rule,
        "attack_hosts": len(hosts),
        "attack_cells_in_redteam_file": len(cells),
        "attack_cells_in_read_range": len(in_range),
        "selection": {k: v for k, v in selection.items() if k != "selected"},
        "cells_in_train_or_test_blocks": len(selected),
        "size_fallback": fallback,
        "n_samples": int(len(samples.labels)),
        "seconds": {"selection_pass": round(t_select, 1),
                    "feature_pass": round(t_features, 1)},
        "peak_rss_mb": peak_rss_mb(),
        "per_day": sp.per_day_counts(samples),
    }
    return samples, report


def split_summary(samples) -> dict:
    out = {}
    block = np.asarray(samples.block)
    for name in ("train", "test"):
        sel = block == name
        labels = samples.labels[sel]
        windows = {int(w) // WINDOW for w in samples.window_start[sel][labels == 1]}
        out[name] = {
            "n_samples": int(sel.sum()),
            "n_attack_samples": int((labels == 1).sum()),
            "n_benign_samples": int((labels == 0).sum()),
            "attack_prevalence_in_sample": (float((labels == 1).mean())
                                            if sel.sum() else None),
            "n_distinct_attack_windows": len(windows),
        }
    return out


def _print_fallback(report: dict) -> None:
    f = report.get("size_fallback")
    if not f:
        return
    print("size fallback: attack-host benign share %s (k=%d), %d of %d such "
          "cells dropped, %d rows -> %d against a bar of %d"
          % (f["attack_host_benign_share"], f["attack_host_benign_share_k"],
             f["attack_host_benign_cells_dropped"],
             f["attack_host_benign_cells_before"],
             f["rows_before_fallback"], f["rows_after_fallback"],
             f["max_total_bar"]))


def check_stop_conditions(summary: dict, total: int) -> None:
    for name, block in summary.items():
        if block["n_samples"] == 0:
            continue
        got = block["n_distinct_attack_windows"]
        if got < MIN_ATTACK_WINDOWS_PER_SPLIT:
            raise StopCondition(
                f"{name}: {got} attack windows, floor is "
                f"{MIN_ATTACK_WINDOWS_PER_SPLIT} (pre-registration 7.2)")
    if total > MAX_TOTAL_SAMPLES:
        raise StopCondition(
            f"{total} samples exceeds the {MAX_TOTAL_SAMPLES} bar "
            f"(pre-registration 2, 7.3)")


# ---------------------------------------------------------------------------
# the reproduction gate (pre-registration 7.1)
# ---------------------------------------------------------------------------

def committed_part30_per_day() -> dict:
    r = json.loads(PART30_RESULTS.read_text())
    per_split = r["split_report"]["per_split"]
    return {b: {int(d): v for d, v in per_split[b]["per_day"].items()}
            for b in ("train", "test")}


def reproduce(days: int) -> int:
    """Part 30's rule on a prefix, against Part 30's committed per-day counts.

    A prefix is compared rather than the full 16 days because the per-day
    counts are committed PER DAY: day 1 under this rule must hold exactly the
    3,491 samples Part 30 recorded for it, whatever the read range, since the
    rule is a function of cell identity alone.
    """
    cfg = prefix_config(load_lanl_config(), days)
    samples, report = sample_table(cfg, "part30")
    committed = committed_part30_per_day()

    print("\n=== REPRODUCTION GATE: Part 30's rule, days 1-%d ===" % days)
    print("%-5s %-6s %9s %9s   %9s %9s   %s"
          % ("day", "block", "samples", "committed", "attack", "committed", "verdict"))
    rows, bad = 0, 0
    for block, days_map in sorted(report["per_day"].items()):
        for day, got in sorted(days_map.items()):
            want = None
            for cblock in ("train", "test"):
                if day in committed[cblock]:
                    want = committed[cblock][day]
                    break
            if want is None:
                print("%-5d %-6s %9d %9s   %9d %9s   no committed row"
                      % (day, block, got["samples"], "-", got["attack_samples"], "-"))
                continue
            rows += 1
            ok = (got["samples"] == want["samples"]
                  and got["attack_samples"] == want["attack_samples"]
                  and got["attack_windows"] == want["attack_windows"])
            bad += 0 if ok else 1
            print("%-5d %-6s %9d %9d   %9d %9d   %s"
                  % (day, block, got["samples"], want["samples"],
                     got["attack_samples"], want["attack_samples"],
                     "match" if ok else "MISMATCH"))
    print("\nselection pass %.1fs, feature pass %.1fs, %d cells selected"
          % (report["seconds"]["selection_pass"],
             report["seconds"]["feature_pass"], report["selection"]["n_selected"]))
    print("records scanned:", report["selection"]["records_scanned"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"reproduction_days{days}.json").write_text(
        json.dumps(report, indent=1, default=str))
    if bad or rows == 0:
        print("\nGATE FAILED: %d of %d compared days mismatch "
              "(pre-registration 7.1 -- stop and report)" % (bad, rows))
        return 1
    print("\nGATE PASSED: %d of %d compared days match exactly" % (rows, rows))
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(rule: str, days: int) -> int:
    cfg = prefix_config(load_lanl_config(), days)
    samples, report = sample_table(cfg, rule)
    summary = split_summary(samples)
    report["split_summary"] = summary
    total = sum(b["n_samples"] for b in summary.values())

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{rule}_days{days}"
    print("\n=== BUILD %s ===" % tag)
    for name, b in summary.items():
        print("%-6s n=%-7d attack=%-6d benign=%-7d prevalence=%s windows=%d"
              % (name, b["n_samples"], b["n_attack_samples"], b["n_benign_samples"],
                 ("%.8f" % b["attack_prevalence_in_sample"]
                  if b["attack_prevalence_in_sample"] is not None else "n/a"),
                 b["n_distinct_attack_windows"]))
    print("total %d samples, %d attack hosts, %d attack cells in read range "
          "(%d in the whole redteam file)"
          % (total, report["attack_hosts"], report["attack_cells_in_read_range"],
             report["attack_cells_in_redteam_file"]))
    _print_fallback(report)
    print("selection %.1fs, feature pass %.1fs, peak RSS %s MB"
          % (report["seconds"]["selection_pass"],
             report["seconds"]["feature_pass"], report["peak_rss_mb"]))

    try:
        check_stop_conditions(summary, total)
    except StopCondition as exc:
        report["stop_condition"] = str(exc)
        (OUT / f"build_{tag}.json").write_text(json.dumps(report, indent=1, default=str))
        print("\nSTOP CONDITION: %s" % exc)
        return 1

    t2 = time.time()
    sha = save_samples(samples, OUT / f"samples_{tag}.npz")
    report["seconds"]["save_npz"] = round(time.time() - t2, 1)
    report["peak_rss_mb"] = peak_rss_mb()
    report["samples_sha256"] = sha
    print("save %.1fs, peak RSS after save %s MB"
          % (report["seconds"]["save_npz"], report["peak_rss_mb"]))
    (OUT / f"build_{tag}.json").write_text(json.dumps(report, indent=1, default=str))
    print("wrote results/lanl_v2/samples_%s.npz" % tag)
    return 0


# ---------------------------------------------------------------------------
# train / score one unit
# ---------------------------------------------------------------------------

def unit(samples, arm_views, system: str, seed: int, mode: dict) -> dict:
    """One (arm, system, seed). Returns metrics and the per-row score arrays.

    Nothing here mutates `samples`: the scaler is applied into a NEW array and
    wrapped in a fresh dataset, so running a second seed cannot z-score
    already-scaled columns a second time.
    """
    block = np.asarray(samples.block)
    train_idx = np.flatnonzero(block == "train")
    test_idx = np.flatnonzero(block == "test")

    dataset = LANLMultiViewDataset(samples)
    if tuple(arm_views) != tuple(samples.sources):
        dataset = subset(dataset, tuple(arm_views))
    arm = dataset.samples
    native_dims = list(dataset.view_dims)      # the loader's own native widths

    # The caller selects the train rows: fitting on the whole table would fit
    # on the test block.
    scaler = fit_scaler(arm.views[train_idx], arm.view_mask[train_idx],
                        native_dims, SCALING_CFG["epsilon"])
    scaled = apply_scaler(arm.views, arm.view_mask, scaler)
    dataset = LANLMultiViewDataset(dataclasses.replace(arm, views=scaled))

    t0 = time.time()
    model = train(dataset, train_idx, native_dims, 2, seed,
                  system == "prismflow", MODEL_CFG, TRAINING_CFG)
    seconds = time.time() - t0

    collected = collect_predictions(
        model, iter_batches(dataset, test_idx, 256, shuffle=False))
    prob = np.asarray(collected["probs"], dtype=np.float64)[:, 1]
    labels = np.asarray(collected["labels"], dtype=np.int64)
    scores = {
        "computer": np.asarray([samples.computer[i] for i in test_idx], dtype=object),
        "window_start": samples.window_start[test_idx],
        "label": labels,
        "prob_attack": prob,
        # L29: this column is a CONFIDENCE. run_lanl.py names it "vacuity".
        "confidence": np.asarray(collected["vacuity_confidence"], dtype=np.float64),
        "confidence_convention": np.asarray(
            "higher = more evidence; an all-False view-mask row reads 0.0; "
            "this is NOT vacuity", dtype=object),
        "mode": np.asarray(json.dumps(mode), dtype=object),
    }
    if "vacuity" in scores:
        raise AssertionError("refusing to write a key named 'vacuity' (L29)")

    present = arm.view_mask[test_idx].any(axis=1)
    result = m.evaluate(labels, prob)
    result["constant_baseline_brier"] = m.constant_baseline_brier(labels)
    result["eniv_eigen"] = _safe(collected.get("eniv"))
    result["mean_dependence"] = _safe(collected.get("mean_dependence"))
    result["train_seconds"] = round(seconds, 1)
    result["all_false_mask_rows"] = int((~present).sum())
    result["arm_views"] = list(arm_views)
    result["system"] = system
    result["seed"] = seed
    return {"metrics": result, "scores": scores}


def _safe(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


# ---------------------------------------------------------------------------
# probe: the step rate and the per-process memory the sweep estimate needs
# ---------------------------------------------------------------------------

#: Part 30's four arms, unchanged (results/lanl/results.json
#: `config.view_configurations`). Named here so a probe or a sweep selects an
#: arm by name rather than by a hand-typed view list.
ARMS = {
    "without_auth": ("proc", "flows", "dns"),
    "without_flows": ("auth", "proc", "dns"),
    "without_dns": ("auth", "proc", "flows"),
    "with_auth": ("auth", "proc", "flows", "dns"),
}


def probe(tag: str, arm: str, system: str, epochs: int, mode_name: str) -> int:
    """Time real epochs on a BUILT table, to get ms/step and peak RSS per unit.

    Runs `train` itself with the epoch count overridden -- not a
    reimplementation of the training step -- so the rate is the sweep's own
    code path on the sweep's own data. Part 30's `execution_mode` benchmark is
    the same measurement at 200 steps on Part 30's table.

    Peak RSS here is ONE unit process. Mode A runs 11 of them at once, so the
    sweep's memory requirement is this figure times the concurrency, not this
    figure.
    """
    mode = apply_mode(mode_name)
    cfg = load_lanl_config()
    path = OUT / f"samples_{tag}.npz"
    t0 = time.time()
    samples = load_samples(path, cfg)
    t_load = time.time() - t0

    block = np.asarray(samples.block)
    train_idx = np.flatnonzero(block == "train")
    dataset = LANLMultiViewDataset(samples)
    if tuple(ARMS[arm]) != tuple(samples.sources):
        dataset = subset(dataset, tuple(ARMS[arm]))
    a = dataset.samples
    native_dims = list(dataset.view_dims)

    t1 = time.time()
    scaler = fit_scaler(a.views[train_idx], a.view_mask[train_idx],
                        native_dims, SCALING_CFG["epsilon"])
    dataset = LANLMultiViewDataset(dataclasses.replace(
        a, views=apply_scaler(a.views, a.view_mask, scaler)))
    t_scale = time.time() - t1

    batch = TRAINING_CFG["batch_size"]
    steps_per_epoch = -(-len(train_idx) // batch)      # iter_batches keeps the tail
    t2 = time.time()
    train(dataset, train_idx, native_dims, 2, 0, system == "prismflow",
          MODEL_CFG, dict(TRAINING_CFG, epochs=epochs))
    t_train = time.time() - t2

    ms_per_step = 1000.0 * t_train / (epochs * steps_per_epoch)
    out = {
        "probe": True, "tag": tag, "arm": arm, "views": list(ARMS[arm]),
        "system": system, "mode": mode,
        "n_train_rows": int(len(train_idx)), "n_rows": int(len(samples.labels)),
        "batch_size": batch, "steps_per_epoch": steps_per_epoch,
        "epochs_timed": epochs,
        "seconds": {"load_npz": round(t_load, 1), "fit_apply_scaler": round(t_scale, 1),
                    "train_timed_epochs": round(t_train, 1)},
        "ms_per_step": round(ms_per_step, 3),
        "projected_unit_train_seconds_at_40_epochs":
            round(ms_per_step * steps_per_epoch * TRAINING_CFG["epochs"] / 1000.0, 1),
        "peak_rss_mb_one_unit_process": peak_rss_mb(),
    }
    print(json.dumps(out, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"probe_{tag}_{arm}_{system}_{mode['mode']}.json").write_text(
        json.dumps(out, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------
# smoke (pre-registration 8)
# ---------------------------------------------------------------------------

def smoke(seeds: list[int], days: int, mode_name: str) -> int:
    mode = apply_mode(mode_name)
    cfg = prefix_config(load_lanl_config(), days)
    samples, report = sample_table(cfg, "v2")
    summary = split_summary(samples)
    total = sum(b["n_samples"] for b in summary.values())
    print("\n=== SMOKE: with_auth only, %d seeds, days 1-%d ==="
          % (len(seeds), days))
    print("PIPELINE EVIDENCE, NOT A FINDING: %d seeds not 5, one arm not four, "
          "a reduced range and a reduced split." % len(seeds))
    print("mode:", json.dumps(mode))
    for name, b in summary.items():
        print("%-6s n=%-7d attack=%-6d windows=%-4d prevalence=%.8f"
              % (name, b["n_samples"], b["n_attack_samples"],
                 b["n_distinct_attack_windows"],
                 b["attack_prevalence_in_sample"] or 0.0))
    _print_fallback(report)
    print("build peak RSS %s MB" % report["peak_rss_mb"])
    check_stop_conditions(summary, total)

    OUT.mkdir(parents=True, exist_ok=True)
    out = {"smoke": True, "not_a_finding": True, "mode": mode,
           "days": days, "build": report, "split_summary": summary, "units": {}}
    for seed in seeds:
        res = unit(samples, list(samples.sources), "naive", seed, mode)
        np.savez_compressed(OUT / f"smoke_scores_seed{seed}.npz", **res["scores"])
        res["metrics"]["peak_rss_mb"] = peak_rss_mb()
        out["units"][str(seed)] = res["metrics"]
        print("\nseed %d  %s" % (seed, json.dumps(res["metrics"])))
    out["peak_rss_mb"] = peak_rss_mb()
    (OUT / "smoke.json").write_text(json.dumps(out, indent=1, default=str))
    print("smoke peak RSS %s MB" % out["peak_rss_mb"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce"); r.add_argument("--days", type=int, default=1)
    b = sub.add_parser("build")
    b.add_argument("--rule", default="v2"); b.add_argument("--days", type=int, default=16)
    pr = sub.add_parser("probe")
    pr.add_argument("--tag", default="v2_days16")
    pr.add_argument("--arm", default="with_auth", choices=sorted(ARMS))
    pr.add_argument("--system", default="naive", choices=("naive", "prismflow"))
    pr.add_argument("--epochs", type=int, default=1)
    pr.add_argument("--mode", default="default")
    s = sub.add_parser("smoke")
    s.add_argument("--seeds", default="0,1"); s.add_argument("--days", type=int, default=6)
    s.add_argument("--mode", default="default")
    a = ap.parse_args()
    if a.cmd == "reproduce":
        return reproduce(a.days)
    if a.cmd == "build":
        return build(a.rule, a.days)
    if a.cmd == "probe":
        return probe(a.tag, a.arm, a.system, a.epochs, a.mode)
    return smoke([int(x) for x in a.seeds.split(",")], a.days, a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
