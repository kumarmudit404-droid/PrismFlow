"""Part 35 - LANL re-analysis from the saved test-block scores. NO TRAINING.

WHAT THIS READS AND WHAT IT NEVER WRITES
----------------------------------------
Every input lives in `results/lanl/` and is opened read-only. `samples.npz`
and all 40 `scores/*.npz` are verified against the committed sha256 in
`results/lanl/results_manifest.json` before a single byte is used, and a
mismatch raises. `results.run1.json.gz` is verified against its DECOMPRESSED
sha256, the form the manifest records. Nothing in `results/lanl/` is created,
edited or deleted by this module.

No model is built, no scaler is fitted, torch is never imported. The only
quantity this Part fits is the two-parameter Platt calibrator of step 2c, and
that one is cross-fitted over test computers and labelled DIAGNOSTIC: there
are no train-block scores in `results/lanl/`, so it cannot be fitted out of
sample and is never a deployable recalibration.

REUSED, NOT RE-IMPLEMENTED
--------------------------
`roc_auc_binary` and the class constants come from Part 28
(`experiments/ait_clean/run_ait_clean.py`), and the Brier decomposition comes
from `prismflow.evaluation.protocol.compute_metrics`. Both are frozen and
unedited. `ClusterAUC` below is an exact reformulation of the Mann-Whitney AUC
under integer row multiplicities, not a second definition of AUC: the unit
test asserts it against `roc_auc_binary` on the explicitly repeated index set.

Every analysis, threshold and decision rule is read from `config.yaml`, which
was committed before any number here was computed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

from experiments.ait_clean.run_ait_clean import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    roc_auc_binary,
)
from prismflow.evaluation.protocol import compute_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "config.yaml"
LANL_DIR = ROOT / "results" / "lanl"
OUT_DIR = ROOT / "results" / "lanl_reanalysis"
PART30_CONFIG = ROOT / "experiments" / "lanl" / "config.yaml"
PROPOSALS = ROOT / "docs" / "proposed_limitations_update.md"

# Part 30's committed figure that step 4's scalar must reproduce before any
# interval is computed (results.json, the JSON path below).
SCALAR_GATE_PATH = (
    "phase3_diagnostics.trivial_baselines.baselines."
    "total_records_across_views.test_auc"
)
SCALAR_GATE_VALUE = 0.6843  # to 4 dp, as pre-registered

PART35_BEGIN = "<!-- PART35:BEGIN -->"
PART35_END = "<!-- PART35:END -->"

NOT_MEASURED = "not_measured"


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> dict:
    """Check every manifest sha256 before reading. A mismatch raises.

    The manifest covers samples.npz, all 40 scores files and run 1's
    decompressed bytes. results.json, population.json and split.json carry no
    sha256 in it; they are recorded here by the sha256 computed now, so a
    later reader can tell which bytes produced these numbers.
    """
    manifest = json.loads((LANL_DIR / "results_manifest.json").read_text(encoding="utf-8"))
    checked: dict[str, dict] = {}
    mismatches: list[str] = []

    samples = manifest["samples"]
    got = sha256_file(LANL_DIR / "samples.npz")
    checked["results/lanl/samples.npz"] = {
        "expected_sha256": samples["sha256"], "got_sha256": got,
        "match": got == samples["sha256"],
    }
    if got != samples["sha256"]:
        mismatches.append("results/lanl/samples.npz")

    for name, entry in sorted(manifest["scores"]["files"].items()):
        got = sha256_file(LANL_DIR / "scores" / name)
        checked[f"results/lanl/scores/{name}"] = {
            "expected_sha256": entry["sha256"], "got_sha256": got,
            "match": got == entry["sha256"],
        }
        if got != entry["sha256"]:
            mismatches.append(f"results/lanl/scores/{name}")

    run1 = manifest["run1"]
    got = sha256_gzip_decompressed(LANL_DIR / "results.run1.json.gz")
    checked["results/lanl/results.run1.json.gz (decompressed)"] = {
        "expected_sha256": run1["decompressed_sha256"], "got_sha256": got,
        "match": got == run1["decompressed_sha256"],
    }
    if got != run1["decompressed_sha256"]:
        mismatches.append("results/lanl/results.run1.json.gz")

    unmanifested = {}
    for name in ("results.json", "population.json", "split.json"):
        unmanifested[f"results/lanl/{name}"] = {
            "sha256_now": sha256_file(LANL_DIR / name),
            "note": "no sha256 in results_manifest.json; recorded, not verified",
        }

    if mismatches:
        raise SystemExit(
            "PART 35 STOPPED - sha256 mismatch against "
            "results/lanl/results_manifest.json for:\n  "
            + "\n  ".join(mismatches)
            + "\nresults/lanl/ is read-only and these numbers are not computed."
        )
    return {
        "manifest_note": manifest["note"],
        "n_verified": len(checked),
        "all_match": True,
        "verified": checked,
        "recorded_not_in_manifest": unmanifested,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Samples:
    """The committed sample table, with the test block indexed by join key."""

    def __init__(self, path: Path):
        npz = np.load(path, allow_pickle=True)
        self.views = npz["views"]
        self.view_mask = npz["view_mask"]
        self.labels = npz["labels"].astype(np.int64)
        self.window_start = npz["window_start"]
        self.computer = npz["computer"].astype(str)
        self.block = npz["block"].astype(str)
        self.meta = json.loads(str(npz["meta"][0]))
        self.sources = list(self.meta["sources"])
        self.is_test = self.block == "test"
        self.is_train = self.block == "train"
        keys = self._keys(np.arange(self.labels.size))
        if len(set(keys)) != len(keys):
            raise SystemExit("PART 35 STOPPED - duplicate (computer, window_start) in samples.npz")
        self._row_of_key = {k: i for i, k in enumerate(keys)}

    def _keys(self, rows) -> list[tuple[str, float]]:
        return list(zip(self.computer[rows].tolist(), self.window_start[rows].tolist()))

    def rows_for(self, computer: np.ndarray, window_start: np.ndarray) -> np.ndarray:
        """Map score rows to sample rows on the exact join key. Missing key raises."""
        keys = list(zip(np.asarray(computer).astype(str).tolist(),
                        np.asarray(window_start).tolist()))
        if len(set(keys)) != len(keys):
            raise SystemExit("PART 35 STOPPED - duplicate join key inside a scores file")
        try:
            return np.array([self._row_of_key[k] for k in keys], dtype=np.int64)
        except KeyError as exc:
            raise SystemExit(
                f"PART 35 STOPPED - scores row {exc.args[0]!r} has no row in samples.npz"
            ) from exc


def load_scores(arm: str, system: str, seed: int) -> dict:
    npz = np.load(LANL_DIR / "scores" / f"{arm}_{system}_seed{seed}.npz", allow_pickle=True)
    return {
        "computer": npz["computer"].astype(str),
        "window_start": npz["window_start"],
        "label": npz["label"].astype(np.int64),
        "prob_attack": npz["prob_attack"].astype(np.float64),
        "vacuity": npz["vacuity"].astype(np.float64),
    }


def two_class(p: np.ndarray) -> np.ndarray:
    """(N, 2) probabilities from prob_attack, benign first, matching Part 30."""
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack([1.0 - p, p])


def brier_block(labels: np.ndarray, p: np.ndarray, n_bins: int) -> dict:
    """Brier and its decomposition, from the frozen protocol. Never re-derived."""
    if labels.size == 0:
        return {k: NOT_MEASURED for k in
                ("brier", "brier_reliability", "brier_resolution",
                 "brier_uncertainty", "brier_within_bin")}
    probs = two_class(p)
    row = compute_metrics(labels, probs, probs.max(axis=1), n_bins=n_bins, coverages=[1.0])
    return {k: float(row[k]) for k in
            ("brier", "brier_reliability", "brier_resolution",
             "brier_uncertainty", "brier_within_bin")}


# ---------------------------------------------------------------------------
# Helpers the unit tests cover
# ---------------------------------------------------------------------------
def ipw_weights(labels: np.ndarray, attack_weight: float, benign_weight: float) -> np.ndarray:
    """Inverse-probability weight per row: 1 for a case, 512 for a kept control.

    All cases were kept, so a case stands for itself. A control was kept iff
    crc32(computer|window) % 512 == 0, so a kept control stands for 512
    occupied cells - exactly as far as that hash is uniform over cells.
    """
    labels = np.asarray(labels)
    w = np.full(labels.shape, float(benign_weight), dtype=np.float64)
    w[labels == CLASS_ATTACK] = float(attack_weight)
    return w


def weighted_confusion(labels: np.ndarray, p: np.ndarray, w: np.ndarray,
                       threshold: float) -> dict[str, float]:
    """Weighted TP/FP/FN/TN. An alarm is prob_attack >= threshold."""
    labels = np.asarray(labels)
    alarm = np.asarray(p, dtype=np.float64) >= threshold
    attack = labels == CLASS_ATTACK
    return {
        "tp": float(w[alarm & attack].sum()),
        "fp": float(w[alarm & ~attack].sum()),
        "fn": float(w[~alarm & attack].sum()),
        "tn": float(w[~alarm & ~attack].sum()),
    }


def weighted_rates(conf: dict[str, float]) -> dict[str, float]:
    alarms = conf["tp"] + conf["fp"]
    positives = conf["tp"] + conf["fn"]
    negatives = conf["fp"] + conf["tn"]
    return {
        "precision": conf["tp"] / alarms if alarms > 0 else float("nan"),
        "recall": conf["tp"] / positives if positives > 0 else float("nan"),
        "false_positive_rate": conf["fp"] / negatives if negatives > 0 else float("nan"),
    }


def weighted_brier(labels: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """The two-class Brier under row weights: sum_i w_i sum_k (p_ik - y_ik)^2 / sum w."""
    labels = np.asarray(labels)
    p = np.asarray(p, dtype=np.float64)
    y = (labels == CLASS_ATTACK).astype(np.float64)
    per_row = 2.0 * (p - y) ** 2  # (p-y)^2 + ((1-p)-(1-y))^2
    return float((w * per_row).sum() / w.sum())


def threshold_for_false_alarm_budget(labels: np.ndarray, p: np.ndarray, w: np.ndarray,
                                     budget: float) -> float:
    """Smallest candidate threshold whose weighted false positives are <= budget.

    Candidates are the unique scores plus 1 + max, so the zero-alarm threshold
    is always reachable. Smallest = most permissive threshold inside the
    budget, which is the operating point an analyst would actually pick.
    """
    p = np.asarray(p, dtype=np.float64)
    benign = np.asarray(labels) == CLASS_BENIGN
    candidates = np.concatenate([np.unique(p), [p.max() + 1.0]])
    # weighted FP is non-increasing in the threshold, so the first candidate
    # inside the budget is the answer.
    bs = p[benign]
    bw = w[benign]
    order = np.argsort(bs, kind="stable")
    cum = np.cumsum(bw[order][::-1])[::-1]  # weight at or above each sorted score
    pos = np.searchsorted(bs[order], candidates, side="left")
    fp = np.where(pos < cum.size, cum[np.minimum(pos, cum.size - 1)], 0.0)
    fp = np.where(pos >= cum.size, 0.0, fp)
    ok = np.flatnonzero(fp <= budget)
    return float(candidates[ok[0]]) if ok.size else float(candidates[-1])


def computer_fold(computers, n_folds: int = 2) -> np.ndarray:
    """Fold index from a fixed hash of the computer NAME: crc32 % n_folds.

    The name alone, not (computer|window): every window of a host lands in one
    fold, so a calibrator never sees the host it is applied to.
    """
    return np.array(
        [zlib.crc32(str(c).encode("utf-8")) % n_folds for c in np.asarray(computers)],
        dtype=np.int64,
    )


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


def platt_fit(p: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(a, b) of sigmoid(a * logit(p) + b) by minimising mean log-loss, BFGS."""
    x = _logit(p)
    y = (np.asarray(labels) == CLASS_ATTACK).astype(np.float64)
    n = max(y.size, 1)

    def objective(theta):
        z = theta[0] * x + theta[1]
        # log-sum-exp stable cross-entropy
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        s = 1.0 / (1.0 + np.exp(-z))
        g = s - y
        return loss, np.array([float(g @ x) / n, float(g.sum()) / n])

    res = minimize(objective, np.array([1.0, 0.0]), jac=True, method="BFGS",
                   options={"maxiter": 500})
    return float(res.x[0]), float(res.x[1])


def platt_apply(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))


def cross_fitted_platt(p: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, dict]:
    """Out-of-fold Platt probabilities: fit on one fold, apply to the other."""
    out = np.array(p, dtype=np.float64, copy=True)
    params = {}
    for f in np.unique(folds):
        fit_rows = folds != f
        apply_rows = folds == f
        a, b = platt_fit(p[fit_rows], labels[fit_rows])
        out[apply_rows] = platt_apply(p[apply_rows], a, b)
        params[f"fold_{int(f)}"] = {
            "a": a, "b": b,
            "n_fit_rows": int(fit_rows.sum()),
            "n_applied_rows": int(apply_rows.sum()),
            "n_fit_attack": int((labels[fit_rows] == CLASS_ATTACK).sum()),
        }
    return out, params


def prior_shift(p: np.ndarray, pi_train: float, pi_test: float) -> np.ndarray:
    """ORACLE prior-shift correction from the train prior to the TEST prior."""
    r = (pi_test / pi_train) / ((1.0 - pi_test) / (1.0 - pi_train))
    p = np.asarray(p, dtype=np.float64)
    return r * p / (r * p + (1.0 - p))


class ClusterAUC:
    """Mann-Whitney AUC under integer cluster multiplicities, precomputed once.

    The positive and negative row sets are fixed; a bootstrap replicate only
    changes how many times each cluster appears. Repeating a row m times
    multiplies its contribution to every attack-benign pair by m, so the AUC
    is a weighted rank sum - no index set has to be materialised and no
    re-sort is needed per replicate. Ties are credited 0.5, as in
    `roc_auc_binary`, which the unit test checks this against.
    """

    def __init__(self, scores: np.ndarray, pos_rows: np.ndarray, neg_rows: np.ndarray,
                 cluster_of_row: np.ndarray):
        scores = np.asarray(scores, dtype=np.float64)
        neg = scores[neg_rows]
        order = np.argsort(neg, kind="stable")
        self._neg_cluster_sorted = np.asarray(cluster_of_row)[neg_rows][order]
        neg_sorted = neg[order]
        pos = scores[pos_rows]
        self._lo = np.searchsorted(neg_sorted, pos, side="left")
        self._hi = np.searchsorted(neg_sorted, pos, side="right")
        self._pos_cluster = np.asarray(cluster_of_row)[pos_rows]

    def auc(self, multiplicity: np.ndarray) -> float:
        w_neg = np.asarray(multiplicity, dtype=np.float64)[self._neg_cluster_sorted]
        cum = np.concatenate(([0.0], np.cumsum(w_neg)))
        w_pos = np.asarray(multiplicity, dtype=np.float64)[self._pos_cluster]
        n_pos = w_pos.sum()
        n_neg = cum[-1]
        if n_pos <= 0 or n_neg <= 0:
            return float("nan")
        below = cum[self._lo]
        ties = cum[self._hi] - cum[self._lo]
        return float((w_pos * (below + 0.5 * ties)).sum() / (n_pos * n_neg))


def summarise(values) -> dict:
    """mean, SAMPLE std (ddof=1, Part 30's convention) and the per-seed values.

    ddof=1 because `run_ait_clean.summarise` uses `statistics.stdev`, so every
    +/- in this Part is the same quantity as every +/- in Part 30's tables.
    """
    arr = np.asarray([v for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": NOT_MEASURED, "std": NOT_MEASURED, "n_seeds": 0, "per_seed": list(arr)}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else NOT_MEASURED,
        "n_seeds": int(finite.size),
        "per_seed": [float(v) for v in arr],
    }


# ---------------------------------------------------------------------------
# Step 2 - L29
# ---------------------------------------------------------------------------
def step2(cfg: dict, samples: Samples, part30: dict, arm_views: dict) -> dict:
    shared = cfg["shared"]
    spec = cfg["step2_l29"]
    n_bins = int(shared["n_bins"])
    baseline = float(shared["constant_baseline_brier"])
    split = json.loads((LANL_DIR / "split.json").read_text(encoding="utf-8"))
    pi_train = float(split["per_split"]["train"]["attack_prevalence_in_sample"])
    pi_test = float(split["per_split"]["test"]["attack_prevalence_in_sample"])

    out = {
        "question": (
            "L29: why is the Brier score of both systems above the "
            "constant-prevalence baseline in without_auth and without_flows?"
        ),
        "constant_baseline_brier": baseline,
        "baseline_definition": "2 * pi * (1 - pi) at test prevalence, two-class form",
        "train_prevalence": pi_train,
        "test_prevalence": pi_test,
        "decision_rule": spec["decision_rule"],
        "per_arm": {},
        "reproduction_check": {
            "note": (
                "The uncorrected Brier recomputed from scores/*.npz against the "
                "committed per-seed value in results/lanl/results.json. This is "
                "a check that the saved scores are the scores the committed "
                "metrics were computed from."
            ),
            "max_abs_difference": None,
            "tolerance": 1e-9,
            "per_unit": {},
        },
    }

    worst = 0.0
    for arm in shared["arms"]:
        role = "failing (L29 names it)" if arm in shared["l29_failing_arms"] else "reference"
        keep = [samples.sources.index(v) for v in arm_views[arm]]
        arm_entry = {
            "role": role,
            "views": arm_views[arm],
            "per_system": {},
        }
        for system in shared["systems"]:
            per_seed = {
                "uncorrected": [], "a_present": [], "a_absent": [],
                "b_prior_shift": [], "c_platt": [],
            }
            rel_res = {k: {"reliability": [], "resolution": []} for k in
                       ("uncorrected", "a_present", "a_absent", "b_prior_shift", "c_platt")}
            counts = {}
            platt_params = {}
            mechanism: list[dict] = []
            for seed in shared["seeds"]:
                s = load_scores(arm, system, seed)
                rows = samples.rows_for(s["computer"], s["window_start"])
                if not np.array_equal(samples.labels[rows], s["label"]):
                    raise SystemExit(
                        f"PART 35 STOPPED - label mismatch between scores and samples "
                        f"for {arm}/{system}/seed{seed}"
                    )
                labels = s["label"]
                p = s["prob_attack"]
                present = samples.view_mask[rows][:, keep].any(axis=1)

                blocks = {
                    "uncorrected": (labels, p),
                    "a_present": (labels[present], p[present]),
                    "a_absent": (labels[~present], p[~present]),
                    "b_prior_shift": (labels, prior_shift(p, pi_train, pi_test)),
                }
                folds = computer_fold(s["computer"])
                p_platt, params = cross_fitted_platt(p, labels, folds)
                blocks["c_platt"] = (labels, p_platt)
                platt_params[f"seed{seed}"] = params

                for name, (lab, prob) in blocks.items():
                    b = brier_block(lab, prob, n_bins)
                    per_seed[name].append(
                        b["brier"] if b["brier"] != NOT_MEASURED else float("nan"))
                    for key, field in (("reliability", "brier_reliability"),
                                       ("resolution", "brier_resolution")):
                        v = b[field]
                        rel_res[name][key].append(
                            v if v != NOT_MEASURED else float("nan"))

                absent_prob = np.unique(p[~present])
                absent_vac = np.unique(s["vacuity"][~present])
                mechanism.append({
                    "seed": seed,
                    "absent_row_prob_attack_values": absent_prob[:4].tolist(),
                    "absent_row_prob_attack_is_constant": bool(absent_prob.size <= 1),
                    "absent_row_vacuity_confidence_values": absent_vac[:4].tolist(),
                    "present_row_mean_vacuity_confidence": (
                        float(s["vacuity"][present].mean()) if present.any() else NOT_MEASURED),
                })
                if not counts:
                    counts = {
                        "n_test_rows": int(labels.size),
                        "a_present_rows": int(present.sum()),
                        "a_present_attack": int(((labels == CLASS_ATTACK) & present).sum()),
                        "a_absent_rows": int((~present).sum()),
                        "a_absent_attack": int(((labels == CLASS_ATTACK) & ~present).sum()),
                        "fold_sizes": {f"fold_{f}": int((folds == f).sum())
                                       for f in np.unique(folds)},
                    }

                committed = next(
                    r for r in part30["configurations"][arm]["systems"][system]["per_seed"]
                    if int(r["seed"]) == seed)
                diff = abs(float(committed["brier"]) - per_seed["uncorrected"][-1])
                worst = max(worst, diff)
                out["reproduction_check"]["per_unit"][f"{arm}/{system}/seed{seed}"] = {
                    "committed_brier": float(committed["brier"]),
                    "recomputed_brier": per_seed["uncorrected"][-1],
                    "abs_difference": diff,
                }

            def cause(name: str) -> dict:
                brier = summarise(per_seed[name])
                vals = np.asarray(per_seed[name], dtype=np.float64)
                below = int(np.sum(np.isfinite(vals) & (vals < baseline)))
                return {
                    "brier": brier,
                    "brier_reliability": summarise(rel_res[name]["reliability"]),
                    "brier_resolution": summarise(rel_res[name]["resolution"]),
                    "seeds_below_baseline": below,
                    "n_seeds": int(np.sum(np.isfinite(vals))),
                }

            sub_pi = (counts["a_present_attack"] / counts["a_present_rows"]
                      if counts["a_present_rows"] else float("nan"))
            entry = {
                "counts": counts,
                "uncorrected": cause("uncorrected"),
                "a_all_false_view_mask": {
                    "present_rows": cause("a_present"),
                    "absent_rows": cause("a_absent"),
                    "present_subset_prevalence": sub_pi,
                    "present_subset_own_constant_baseline": (
                        2.0 * sub_pi * (1.0 - sub_pi) if np.isfinite(sub_pi) else NOT_MEASURED),
                    "absent_row_mechanism": {
                        "note": (
                            "What the saved scores actually are on a row whose every "
                            "arm view is absent. The field is the protocol's "
                            "vacuity_confidence, so 0.0 is the LEAST confident value "
                            "it can take - the selective-prediction machinery already "
                            "ranks these rows last. Observed, not changed: these are "
                            "Part 30's committed scores and this Part fixes nothing."
                        ),
                        "per_seed": mechanism,
                    },
                    "oracle": False,
                    "diagnostic_only": False,
                },
                "b_prior_shift": {
                    **cause("b_prior_shift"),
                    "oracle": True,
                    "oracle_note": spec["causes"]["b_prior_shift"]["oracle_note"].strip(),
                },
                "c_cross_fitted_platt": {
                    **cause("c_platt"),
                    "oracle": False,
                    "diagnostic_only": True,
                    "diagnostic_note": spec["causes"]["c_cross_fitted_platt"]["diagnostic_note"].strip(),
                    "fitted_parameters": platt_params,
                },
            }
            arm_entry["per_system"][system] = entry
        out["per_arm"][arm] = arm_entry

    out["reproduction_check"]["max_abs_difference"] = worst
    out["reproduction_check"]["passes"] = bool(worst <= 1e-9)
    if worst > 1e-9:
        raise SystemExit(
            "PART 35 STOPPED - the Brier recomputed from results/lanl/scores/*.npz "
            f"differs from the committed per-seed value by up to {worst:.3e}. The "
            "saved scores are not the scores the committed metrics came from; "
            "that is a defect in a frozen Part and is reported, not patched."
        )

    # Verdicts, by the pre-registered rule.
    verdicts = {}
    for arm in shared["arms"]:
        per_system = out["per_arm"][arm]["per_system"]
        arm_v = {}
        for cause_name, getter in (
            ("a_all_false_view_mask", lambda e: e["a_all_false_view_mask"]["present_rows"]),
            ("b_prior_shift", lambda e: e["b_prior_shift"]),
            ("c_cross_fitted_platt", lambda e: e["c_cross_fitted_platt"]),
        ):
            counts_by_system = {sys_: getter(per_system[sys_])["seeds_below_baseline"]
                                for sys_ in shared["systems"]}
            arm_v[cause_name] = {
                "seeds_below_baseline_per_system": counts_by_system,
                "accounts_for_the_failure": all(v == 5 for v in counts_by_system.values()),
            }
        uncorrected_below = {sys_: per_system[sys_]["uncorrected"]["seeds_below_baseline"]
                             for sys_ in shared["systems"]}
        arm_v["uncorrected_seeds_below_baseline"] = uncorrected_below
        arm_v["arm_fails_l29"] = all(v == 0 for v in uncorrected_below.values())
        arm_v["any_cause_accounts_for_it"] = any(
            arm_v[c]["accounts_for_the_failure"] for c in
            ("a_all_false_view_mask", "b_prior_shift", "c_cross_fitted_platt"))
        verdicts[arm] = arm_v
    out["verdicts"] = verdicts
    return out


# ---------------------------------------------------------------------------
# Step 3 - L23
# ---------------------------------------------------------------------------
def step3(cfg: dict, samples: Samples) -> dict:
    shared = cfg["shared"]
    spec = cfg["step3_l23"]
    wspec = spec["weights"]
    split = json.loads((LANL_DIR / "split.json").read_text(encoding="utf-8"))
    population = json.loads((LANL_DIR / "population.json").read_text(encoding="utf-8"))
    days = split["per_split"]["test"]["days"]
    n_days = int(days[1] - days[0] + 1)
    per_view = population["per_view"]
    test_day_keys = [str(d) for d in range(days[0], days[1] + 1)]
    test_cells = {v: int(sum(per_view[v]["population_cells_per_day"][k] for k in test_day_keys))
                  for v in per_view}

    out = {
        "question": "L23: what do the metrics look like at the population rate?",
        "weights": {"attack": wspec["attack"], "benign": wspec["benign"],
                    "source": wspec["source"].strip()},
        "caveat": spec["caveat_to_state"].strip(),
        "n_test_days": n_days,
        "test_days": days,
        "population_cells": {
            "all_days": {
                "per_view": {v: per_view[v]["population_cells"] for v in per_view},
                "union_lower_bound": population["estimated_population_cells_union"]["lower_bound"],
                "union_upper_bound": population["estimated_population_cells_union"]["upper_bound"],
                "union_method": population["estimated_population_cells_union"]["method"],
            },
            "test_days_only": {
                "per_view": test_cells,
                "union_lower_bound": max(test_cells.values()),
                "union_upper_bound": sum(test_cells.values()),
                "note": (
                    "Summed from population.json population_cells_per_day over "
                    "days 13-16. The union over views is bounded below by the "
                    "largest single view and above by the sum, for the same "
                    "reason population.json gives for the all-day figure."
                ),
            },
        },
        "threshold_selection": spec["threshold_selection"].strip(),
        "per_arm": {},
    }

    targets = [(t["name"], t.get("false_alarms_per_test_day")) for t in spec["thresholds"]]
    for arm in shared["arms"]:
        arm_entry = {}
        for system in shared["systems"]:
            acc = {name: {"threshold": [], "precision": [], "recall": [],
                          "false_positive_rate": [], "weighted_fp_per_day": [],
                          "weighted_tp": [], "n_sampled_alarms": []}
                   for name, _ in targets}
            wbrier = []
            unweighted_prevalence = None
            weighted_prevalence = None
            for seed in shared["seeds"]:
                s = load_scores(arm, system, seed)
                labels, p = s["label"], s["prob_attack"]
                w = ipw_weights(labels, wspec["attack"], wspec["benign"])
                wbrier.append(weighted_brier(labels, p, w))
                if unweighted_prevalence is None:
                    unweighted_prevalence = float((labels == CLASS_ATTACK).mean())
                    weighted_prevalence = float(
                        w[labels == CLASS_ATTACK].sum() / w.sum())
                for name, rate in targets:
                    if rate is None:
                        t = float(shared["default_threshold"])
                    else:
                        t = threshold_for_false_alarm_budget(
                            labels, p, w, float(rate) * n_days)
                    conf = weighted_confusion(labels, p, w, t)
                    rates = weighted_rates(conf)
                    acc[name]["threshold"].append(t)
                    acc[name]["precision"].append(rates["precision"])
                    acc[name]["recall"].append(rates["recall"])
                    acc[name]["false_positive_rate"].append(rates["false_positive_rate"])
                    acc[name]["weighted_fp_per_day"].append(conf["fp"] / n_days)
                    acc[name]["weighted_tp"].append(conf["tp"])
                    acc[name]["n_sampled_alarms"].append(float((p >= t).sum()))
            arm_entry[system] = {
                "weighted_brier": summarise(wbrier),
                "sample_prevalence": unweighted_prevalence,
                "weighted_prevalence": weighted_prevalence,
                "at_threshold": {
                    name: {
                        "target_false_alarms_per_test_day": rate,
                        **{k: summarise(v) for k, v in acc[name].items()},
                    }
                    for name, rate in targets
                },
            }
        out["per_arm"][arm] = arm_entry
    return out


# ---------------------------------------------------------------------------
# Step 4 - L26
# ---------------------------------------------------------------------------
def trivial_scalar(samples: Samples) -> np.ndarray:
    """total_records_across_views, Part 30's definition, recomputed here.

    Feature 0 of every view is log1p(n_events) for that view, so the summed
    expm1 over PRESENT views is the total record count of that sample.
    """
    views = samples.views
    total = np.zeros(views.shape[0], dtype=np.float64)
    for v in range(views.shape[1]):
        total += np.expm1(views[:, v, 0].astype(np.float64)) * samples.view_mask[:, v]
    return total


def step4(cfg: dict, samples: Samples, part30: dict) -> dict:
    shared = cfg["shared"]
    spec = cfg["step4_l26"]
    boot = spec["bootstrap"]
    replicates = int(boot["replicates"])
    seed = int(boot["seed"])

    # --- the scalar and its gate -------------------------------------------
    scalar_all = trivial_scalar(samples)
    train_auc = roc_auc_binary(samples.labels[samples.is_train], scalar_all[samples.is_train])
    orientation = "as-is" if train_auc >= 0.5 else "negated"
    oriented = scalar_all if train_auc >= 0.5 else -scalar_all
    test_auc = roc_auc_binary(samples.labels[samples.is_test], oriented[samples.is_test])
    committed = part30
    for part in SCALAR_GATE_PATH.split("."):
        committed = committed[part]
    gate = {
        "scalar": spec["scalar"]["name"],
        "train_auc_recomputed": float(train_auc),
        "orientation_fixed_on_train": orientation,
        "test_auc_recomputed": float(test_auc),
        "part30_committed_test_auc": float(committed),
        "part30_json_path": f"results/lanl/results.json :: {SCALAR_GATE_PATH}",
        "required_to_4dp": SCALAR_GATE_VALUE,
        "match_to_4dp": bool(round(float(test_auc), 4) == SCALAR_GATE_VALUE),
    }
    if not gate["match_to_4dp"]:
        raise SystemExit(
            "PART 35 STOPPED - the recomputed test AUC of "
            f"total_records_across_views is {test_auc:.6f}, which is not "
            f"{SCALAR_GATE_VALUE} to 4 dp. No interval is computed."
        )

    # --- row geometry, shared by every vector ------------------------------
    reference = load_scores(shared["arms"][0], shared["systems"][0], shared["seeds"][0])
    rows = samples.rows_for(reference["computer"], reference["window_start"])
    labels = reference["label"]
    computers = reference["computer"]
    hosts, cluster_of_row = np.unique(computers, return_inverse=True)

    train_attack_hosts = set(
        samples.computer[samples.is_train & (samples.labels == CLASS_ATTACK)].tolist())
    attack = labels == CLASS_ATTACK
    benign_rows = np.flatnonzero(~attack)
    seen_rows = np.flatnonzero(
        attack & np.array([c in train_attack_hosts for c in computers]))
    unseen_rows = np.flatnonzero(
        attack & np.array([c not in train_attack_hosts for c in computers]))

    scalar_test = oriented[rows]

    # --- build every AUC machine once --------------------------------------
    groups = {"seen": seen_rows, "unseen": unseen_rows}
    machines: dict[str, dict[str, list[ClusterAUC]]] = {}
    for group, pos_rows in groups.items():
        machines[group] = {}
        for arm in shared["arms"]:
            for system in shared["systems"]:
                vec = []
                for s_ in shared["seeds"]:
                    sc = load_scores(arm, system, s_)
                    if not np.array_equal(sc["computer"], computers) or not np.array_equal(
                            sc["window_start"], reference["window_start"]):
                        raise SystemExit(
                            f"PART 35 STOPPED - {arm}/{system}/seed{s_} score rows are not "
                            "in the same order as the reference unit; the paired bootstrap "
                            "would compare different rows."
                        )
                    vec.append(ClusterAUC(sc["prob_attack"], pos_rows, benign_rows,
                                          cluster_of_row))
                machines[group][f"{arm}/{system}"] = vec
        machines[group]["scalar"] = [ClusterAUC(scalar_test, pos_rows, benign_rows,
                                                cluster_of_row)]

    unit_mult = np.ones(hosts.size, dtype=np.float64)
    observed = {g: {k: float(np.mean([m.auc(unit_mult) for m in v]))
                    for k, v in machines[g].items()} for g in groups}

    rng = np.random.default_rng(seed)
    draws = {g: {k: [] for k in machines[g]} for g in groups}
    paired = {k: [] for k in machines["unseen"] if k != "scalar"}
    for _ in range(replicates):
        picked = rng.integers(0, hosts.size, size=hosts.size)
        mult = np.bincount(picked, minlength=hosts.size).astype(np.float64)
        rep = {}
        for g in groups:
            rep[g] = {}
            for key, vec in machines[g].items():
                vals = [m.auc(mult) for m in vec]
                value = float(np.mean(vals)) if all(np.isfinite(vals)) else float("nan")
                rep[g][key] = value
                if np.isfinite(value):
                    draws[g][key].append(value)
        scalar_unseen = rep["unseen"]["scalar"]
        for key in paired:
            v = rep["unseen"][key]
            if np.isfinite(v) and np.isfinite(scalar_unseen):
                paired[key].append(v - scalar_unseen)

    def interval(values) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {"ci95_low": NOT_MEASURED, "ci95_high": NOT_MEASURED,
                    "n_replicates_used": 0}
        return {
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "n_replicates_used": int(arr.size),
            "bootstrap_mean": float(arr.mean()),
        }

    out = {
        "question": "L26: what interval sits on the seen and unseen AUCs, and on the model-minus-scalar gap?",
        "scalar_gate": gate,
        "bootstrap": {
            "kind": boot["kind"], "replicates": replicates, "seed": seed,
            "n_test_computers": int(hosts.size),
            "statistic": boot["statistic"].strip(),
            "interval": boot["interval"],
        },
        "group_sizes": {"n_benign_test_rows": int(benign_rows.size),
                        "n_seen_attack_rows": int(seen_rows.size),
                        "n_unseen_attack_rows": int(unseen_rows.size)},
        "per_group": {
            g: {key: {"observed": observed[g][key], **interval(draws[g][key])}
                for key in machines[g]}
            for g in groups
        },
        "paired_model_minus_scalar_unseen": {
            key: {
                "observed": observed["unseen"][key] - observed["unseen"]["scalar"],
                **interval(paired[key]),
                "excludes_zero": bool(
                    interval(paired[key])["ci95_low"] != NOT_MEASURED
                    and (interval(paired[key])["ci95_low"] > 0.0
                         or interval(paired[key])["ci95_high"] < 0.0)),
            }
            for key in paired
        },
    }
    return out


# ---------------------------------------------------------------------------
# Step 5 - L27
# ---------------------------------------------------------------------------
def step5(cfg: dict, samples: Samples) -> dict:
    shared = cfg["shared"]
    spec = cfg["step5_l27"]
    reference = load_scores(shared["arms"][0], shared["systems"][0], shared["seeds"][0])
    computers = reference["computer"]
    attack = reference["label"] == CLASS_ATTACK
    names, counts = np.unique(computers[attack], return_counts=True)
    order = sorted(range(names.size), key=lambda i: (-int(counts[i]), str(names[i])))
    top = [str(names[i]) for i in order[:5]]

    out = {
        "question": "L27: how much of each AUC rests on one host?",
        "host_rule": spec["hosts"].strip(),
        "interval": spec["interval"].strip(),
        "hosts": [],
        "per_arm": {},
    }
    for host in top:
        rows = computers == host
        out["hosts"].append({
            "computer": host,
            "test_rows_removed": int(rows.sum()),
            "test_attack_rows_removed": int((rows & attack).sum()),
            "test_benign_rows_removed": int((rows & ~attack).sum()),
        })

    for arm in shared["arms"]:
        arm_entry = {}
        for system in shared["systems"]:
            scores = [load_scores(arm, system, s_) for s_ in shared["seeds"]]
            all_test = summarise([roc_auc_binary(s["label"], s["prob_attack"]) for s in scores])
            per_host = {}
            for host in top:
                keep = scores[0]["computer"] != host
                aucs = [roc_auc_binary(s["label"][keep], s["prob_attack"][keep]) for s in scores]
                kept = summarise(aucs)
                per_host[host] = {
                    "roc_auc_host_removed": kept,
                    "change_from_all_test": (
                        kept["mean"] - all_test["mean"]
                        if kept["mean"] != NOT_MEASURED else NOT_MEASURED),
                    "n_rows_scored": int(keep.sum()),
                    "n_attack_rows_scored": int((keep & attack).sum()),
                }
            arm_entry[system] = {"roc_auc_all_test": all_test, "per_host": per_host}
        out["per_arm"][arm] = arm_entry
    return out


# ---------------------------------------------------------------------------
# Step 6 - L28
# ---------------------------------------------------------------------------
def step6(cfg: dict, part30: dict) -> dict:
    spec = cfg["step6_l28"]
    arm = spec["arm"]
    run1 = json.loads(gzip.open(LANL_DIR / "results.run1.json.gz", "rt", encoding="utf-8").read())
    modes = {"run1": run1, "run2": part30}

    out = {
        "question": "L28: how large is the with_auth spread across execution modes, against the per-seed std?",
        "arm": arm,
        "modes": {name: dict(spec["modes"][name]) for name in modes},
        "arms_left_open": spec["arms_left_open"],
        "open_note": spec["open_note"].strip(),
        "per_system": {},
    }
    for system in cfg["shared"]["systems"]:
        per_metric = {}
        for metric in spec["metrics"]:
            by_mode = {}
            for mode, blob in modes.items():
                rows = blob["configurations"][arm]["systems"][system]["per_seed"]
                values = [float(r[metric]) for r in sorted(rows, key=lambda r: r["seed"])]
                by_mode[mode] = values
            v1 = np.asarray(by_mode["run1"], dtype=np.float64)
            v2 = np.asarray(by_mode["run2"], dtype=np.float64)
            pooled = np.concatenate([v1, v2])
            per_metric[metric] = {
                "run1": {"per_seed": v1.tolist(), "mean": float(v1.mean()),
                         "within_mode_std": float(v1.std(ddof=1))},
                "run2": {"per_seed": v2.tolist(), "mean": float(v2.mean()),
                         "within_mode_std": float(v2.std(ddof=1))},
                "across_mode_mean_difference_run2_minus_run1": float(v2.mean() - v1.mean()),
                "max_abs_same_seed_difference": float(np.max(np.abs(v2 - v1))),
                "pooled_std_10_values": float(pooled.std(ddof=1)),
                "pooled_over_largest_within_mode_std": (
                    float(pooled.std(ddof=1) / max(v1.std(ddof=1), v2.std(ddof=1)))
                    if max(v1.std(ddof=1), v2.std(ddof=1)) > 0 else NOT_MEASURED),
            }
        out["per_system"][system] = per_metric
    return out


# ---------------------------------------------------------------------------
# README and proposals
# ---------------------------------------------------------------------------
def _f(value, places: int = 4) -> str:
    if value == NOT_MEASURED or value is None:
        return "not measured"
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "not measured"
    return f"{value:.{places}f}"


def _ms(block: dict, places: int = 4) -> str:
    if block.get("mean") == NOT_MEASURED:
        return "not measured"
    return f"{_f(block['mean'], places)} +/- {_f(block['std'], places)}"


def render_readme(res: dict) -> str:
    A: list[str] = []
    add = A.append
    shared = res["config"]["shared"]
    s2, s3, s4, s5, s6 = (res["step2_l29"], res["step3_l23"], res["step4_l26"],
                          res["step5_l27"], res["step6_l28"])

    add("# Part 35 - LANL re-analysis from the saved scores")
    add("")
    add("No training. Every number below is computed from Part 30's committed "
        "test-block scores and sample table, both verified against "
        "`results/lanl/results_manifest.json` before being read "
        f"({res['integrity']['n_verified']} sha256 checks, all matched). "
        "`results/lanl/` is read-only and unchanged.")
    add("")
    add("Pre-registered in `experiments/lanl_reanalysis/config.yaml`, committed "
        f"before any number here existed (commit `{res['preregistration_commit']}`). "
        "This file is generated from `results/lanl_reanalysis/results.json` by "
        "`experiments/lanl_reanalysis/run.py`, so no figure in it is typed by hand.")
    add("")
    add("**There are no train-block scores in `results/lanl/`.** Anything needing "
        "a fit is cross-fitted over test computers and labelled DIAGNOSTIC. None "
        "of it is a deployable recalibration.")
    add("")

    # ---- step 2 ----
    add("## 1. L29 - why two arms score worse than a constant")
    add("")
    add(f"Constant-prevalence baseline: **{s2['constant_baseline_brier']}** "
        f"= 2*pi*(1-pi) at test prevalence {s2['test_prevalence']:.4f}, "
        "two-class Brier (`docs/EVALUATION_PROTOCOL.md`: "
        "mean_i sum_k (p_ik - y_ik)^2, twice the binary form).")
    add("")
    add("**Pre-registered decision rule.** " + " ".join(s2["decision_rule"].split()))
    add("")
    add(f"Saved scores reproduce the committed per-seed Brier of all 40 units to "
        f"{s2['reproduction_check']['max_abs_difference']:.2e} "
        f"(tolerance {s2['reproduction_check']['tolerance']:.0e}), so these are the "
        "scores Part 30's metrics came from.")
    add("")
    add("### 1.1 Brier per cause, 5 seeds, mean +/- std")
    add("")
    add("`present` = rows with at least one view present for that arm (cause a). "
        "`prior shift` is an ORACLE: it uses the test prevalence. `Platt` is a "
        "DIAGNOSTIC: fitted on test rows, cross-fitted by computer fold.")
    add("")
    add("| arm | role | system | uncorrected | (a) present rows | (a) absent rows | "
        "(b) prior shift ORACLE | (c) cross-fit Platt DIAGNOSTIC |")
    add("|---|---|---|---|---|---|---|---|")
    for arm in shared["arms"]:
        entry = s2["per_arm"][arm]
        for system in shared["systems"]:
            e = entry["per_system"][system]
            add(f"| `{arm}` | {entry['role']} | {system} | {_ms(e['uncorrected']['brier'])} | "
                f"{_ms(e['a_all_false_view_mask']['present_rows']['brier'])} | "
                f"{_ms(e['a_all_false_view_mask']['absent_rows']['brier'])} | "
                f"{_ms(e['b_prior_shift']['brier'])} | {_ms(e['c_cross_fitted_platt']['brier'])} |")
    add("")
    add("### 1.2 Reliability and resolution of the same five columns")
    add("")
    add("| arm | system | quantity | uncorrected | (a) present | (a) absent | "
        "(b) prior shift | (c) Platt |")
    add("|---|---|---|---|---|---|---|---|")
    for arm in shared["arms"]:
        entry = s2["per_arm"][arm]
        for system in shared["systems"]:
            e = entry["per_system"][system]
            for field, label in (("brier_reliability", "reliability"),
                                 ("brier_resolution", "resolution")):
                add(f"| `{arm}` | {system} | {label} | {_ms(e['uncorrected'][field])} | "
                    f"{_ms(e['a_all_false_view_mask']['present_rows'][field])} | "
                    f"{_ms(e['a_all_false_view_mask']['absent_rows'][field])} | "
                    f"{_ms(e['b_prior_shift'][field])} | {_ms(e['c_cross_fitted_platt'][field])} |")
    add("")
    add("### 1.3 Row counts and the subset's own baseline")
    add("")
    add("| arm | test rows | present rows (attack) | absent rows (attack) | "
        "present-subset prevalence | its own 2*pi*(1-pi) |")
    add("|---|---:|---|---|---:|---:|")
    for arm in shared["arms"]:
        e = s2["per_arm"][arm]["per_system"][shared["systems"][0]]
        c = e["counts"]
        a = e["a_all_false_view_mask"]
        add(f"| `{arm}` | {c['n_test_rows']} | {c['a_present_rows']} "
            f"({c['a_present_attack']}) | {c['a_absent_rows']} ({c['a_absent_attack']}) | "
            f"{_f(a['present_subset_prevalence'])} | "
            f"{_f(a['present_subset_own_constant_baseline'])} |")
    add("")
    add("The rule is read against the fixed 0.0428. The subset's own baseline is "
        "context, not the rule: on the present-row subset the prevalence changes, "
        "so a constant predictor there is a different number.")
    add("")
    add("**What the absent rows actually carry.** On every row whose arm views are "
        "all absent, in both systems and all 5 seeds, `prob_attack` is exactly 0.5 "
        "and the saved `vacuity_confidence` is exactly 0.0 - the least confident "
        "value it can take. A constant 0.5 scores the maximum two-class Brier of "
        "0.5 on every such row, whatever its label, which is the 0.5000 +/- 0.0000 "
        "column above. Observed in Part 30's committed scores; nothing here changes "
        "them.")
    add("")
    add("| arm | absent rows | their share of the test block | their contribution to Brier |")
    add("|---|---:|---:|---:|")
    for arm in shared["arms"]:
        c = s2["per_arm"][arm]["per_system"][shared["systems"][0]]["counts"]
        share = c["a_absent_rows"] / c["n_test_rows"]
        add(f"| `{arm}` | {c['a_absent_rows']} | {share:.4f} | {share * 0.5:.4f} |")
    add("")
    add("Against a whole budget of 0.0428, that column is the arithmetic of L29.")
    add("")
    add("### 1.4 Verdict per cause")
    add("")
    add("| arm | cause | seeds below 0.0428 (naive / prismflow) | accounts for the failure? |")
    add("|---|---|---|---|")
    labels = {"a_all_false_view_mask": "(a) all-False view mask",
              "b_prior_shift": "(b) prior shift (ORACLE)",
              "c_cross_fitted_platt": "(c) cross-fitted Platt (DIAGNOSTIC)"}
    for arm in shared["arms"]:
        v = s2["verdicts"][arm]
        for cause, label in labels.items():
            counts = v[cause]["seeds_below_baseline_per_system"]
            verdict = "**YES**" if v[cause]["accounts_for_the_failure"] else "no"
            add(f"| `{arm}` | {label} | {counts['naive']}/5 / {counts['prismflow']}/5 | "
                f"{verdict} |")
    add("")
    for arm in shared["l29_failing_arms"]:
        v = s2["verdicts"][arm]
        if not v["any_cause_accounts_for_it"]:
            add(f"- `{arm}`: **none does.** No single cause, corrected alone, brings "
                "Brier below 0.0428 in 5 of 5 seeds for both systems.")
        else:
            won = [labels[c] for c in labels if v[c]["accounts_for_the_failure"]]
            add(f"- `{arm}`: accounted for by {', '.join(won)}.")
    add("")
    add("**\"Accounts for\" is not \"fixed\".** The rows cause (a) names are "
        "unpredictable by construction (Part 30 section 4.2): every model scores "
        "them identically whatever the label. Restricting the metric to rows with a "
        "present view changes the evaluated population, it does not change a "
        "probability - a deployed detector still meets those cells and still emits "
        "0.5. The prior shift makes both failing arms WORSE, and it is an ORACLE. "
        "The Platt diagnostic does not meet the rule either. So the operative "
        "sentence of L29 is unchanged: the probability outputs of those arms should "
        "not be used.")
    add("")

    # ---- step 3 ----
    add("## 2. L23 - metrics at the population rate")
    add("")
    add(f"Inverse-probability weights: attack **{s3['weights']['attack']}**, benign "
        f"**{s3['weights']['benign']}**. " + " ".join(s3["weights"]["source"].split()))
    add("")
    add("**" + " ".join(s3["caveat"].split()) + "**")
    add("")
    pc = s3["population_cells"]
    add(f"Population cells, days 1-16: between **{pc['all_days']['union_lower_bound']:,}** "
        f"and **{pc['all_days']['union_upper_bound']:,}** occupied "
        "`(computer, window)` cells (`results/lanl/population.json`). Over the test "
        f"days ({s3['test_days'][0]}-{s3['test_days'][1]}, {s3['n_test_days']} days) "
        f"alone: between **{pc['test_days_only']['union_lower_bound']:,}** and "
        f"**{pc['test_days_only']['union_upper_bound']:,}**. "
        + " ".join(pc["test_days_only"]["note"].split()))
    add("")
    add("| view | occupied cells, days 1-16 | occupied cells, test days |")
    add("|---|---:|---:|")
    for view, cells in pc["all_days"]["per_view"].items():
        add(f"| `{view}` | {cells:,} | {pc['test_days_only']['per_view'][view]:,} |")
    add("")
    add("### 2.1 Weighted Brier, and what the sample prevalence becomes")
    add("")
    add("| arm | system | sample prevalence | weighted prevalence | weighted Brier |")
    add("|---|---|---:|---:|---|")
    for arm in shared["arms"]:
        for system in shared["systems"]:
            e = s3["per_arm"][arm][system]
            add(f"| `{arm}` | {system} | {_f(e['sample_prevalence'], 6)} | "
                f"{_f(e['weighted_prevalence'], 8)} | {_ms(e['weighted_brier'], 6)} |")
    add("")
    add("### 2.2 Population-weighted precision, recall and FPR")
    add("")
    add("Targets are per TEST DAY. `achieved` is what the threshold actually "
        "delivers: a weighted false-positive count moves in steps of 512 cells, so "
        f"over {s3['n_test_days']} test days no rate between 0 and 128 per day is "
        "representable. All three pre-registered targets - 1, 10 and 100 false "
        "alarms per day - are below that step, so all three collapse onto the same "
        "zero-alarm threshold. **A weighted precision of 1.0 there means no SAMPLED "
        "control fired**: it estimates the population false-alarm count as 0 with a "
        "resolution of 512 cells, and is not a detector with no false alarms.")
    add("")
    add("| arm | system | operating point | target FA/day | achieved FA/day | "
        "threshold | precision | recall | FPR | sampled alarms |")
    add("|---|---|---|---:|---|---|---|---|---|---|")
    for arm in shared["arms"]:
        for system in shared["systems"]:
            for name, e in s3["per_arm"][arm][system]["at_threshold"].items():
                target = e["target_false_alarms_per_test_day"]
                add(f"| `{arm}` | {system} | {name} | "
                    f"{'0.5 prob' if target is None else target} | "
                    f"{_ms(e['weighted_fp_per_day'], 1)} | {_ms(e['threshold'], 6)} | "
                    f"{_ms(e['precision'], 6)} | {_ms(e['recall'], 4)} | "
                    f"{_ms(e['false_positive_rate'], 8)} | {_ms(e['n_sampled_alarms'], 1)} |")
    add("")

    # ---- step 4 ----
    add("## 3. L26 - intervals on the seen and unseen AUCs")
    add("")
    g = s4["scalar_gate"]
    add(f"**Gate passed before any interval.** `{g['scalar']}`, recomputed from "
        f"`results/lanl/samples.npz` with Part 30's train-fixed orientation "
        f"({g['orientation_fixed_on_train']}, train AUC {_f(g['train_auc_recomputed'])}), "
        f"gives test AUC **{_f(g['test_auc_recomputed'])}** against Part 30's committed "
        f"{_f(g['part30_committed_test_auc'])} ({g['part30_json_path']}).")
    add("")
    b = s4["bootstrap"]
    add(f"Computer-cluster bootstrap: **{b['replicates']}** replicates, fixed seed "
        f"**{b['seed']}**, clusters = the **{b['n_test_computers']:,}** test computers, "
        "drawn with replacement. " + " ".join(b["statistic"].split()) + " "
        f"Interval: {b['interval']}.")
    add("")
    gs = s4["group_sizes"]
    add(f"Groups: **{gs['n_seen_attack_rows']}** seen and "
        f"**{gs['n_unseen_attack_rows']}** unseen attack rows, each against all "
        f"**{gs['n_benign_test_rows']:,}** benign test rows.")
    add("")
    add("| arm / system | seen AUC | seen 95% CI | unseen AUC | unseen 95% CI |")
    add("|---|---|---|---|---|")
    keys = [f"{arm}/{system}" for arm in shared["arms"] for system in shared["systems"]]
    for key in keys + ["scalar"]:
        seen = s4["per_group"]["seen"][key]
        unseen = s4["per_group"]["unseen"][key]
        name = "`total_records_across_views` (untrained)" if key == "scalar" else f"`{key}`"
        add(f"| {name} | {_f(seen['observed'])} | "
            f"[{_f(seen['ci95_low'])}, {_f(seen['ci95_high'])}] | "
            f"{_f(unseen['observed'])} | "
            f"[{_f(unseen['ci95_low'])}, {_f(unseen['ci95_high'])}] |")
    add("")
    add("### 3.1 Paired model-minus-scalar on unseen computers")
    add("")
    add("Paired: both AUCs are computed on the same resampled computers in every "
        "replicate, so the draw cancels.")
    add("")
    add("| arm / system | observed difference | 95% CI | excludes 0? |")
    add("|---|---|---|---|")
    for key in keys:
        e = s4["paired_model_minus_scalar_unseen"][key]
        add(f"| `{key}` | {_f(e['observed'])} | [{_f(e['ci95_low'])}, "
            f"{_f(e['ci95_high'])}] | {'**yes**' if e['excludes_zero'] else 'no, includes 0'} |")
    add("")

    # ---- step 5 ----
    add("## 4. L27 - leave one host out")
    add("")
    add(" ".join(s5["host_rule"].split()))
    add("")
    add("**Interval:** " + " ".join(s5["interval"].split()))
    add("")
    add("Rank is by ATTACK rows, so a host lower in the table can still have more "
        "total test rows than the one above it.")
    add("")
    add("| rank | computer | test rows removed | attack | benign |")
    add("|---:|---|---:|---:|---:|")
    for i, h in enumerate(s5["hosts"], start=1):
        add(f"| {i} | `{h['computer']}` | {h['test_rows_removed']} | "
            f"{h['test_attack_rows_removed']} | {h['test_benign_rows_removed']} |")
    add("")
    host_names = [h["computer"] for h in s5["hosts"]]
    add("| arm / system | all-test AUC | " + " | ".join(f"minus `{h}`" for h in host_names) + " |")
    add("|---|---|" + "---|" * len(host_names))
    for arm in shared["arms"]:
        for system in shared["systems"]:
            e = s5["per_arm"][arm][system]
            cells = []
            for h in host_names:
                ph = e["per_host"][h]
                cells.append(f"{_ms(ph['roc_auc_host_removed'])} ({_f(ph['change_from_all_test'])})")
            add(f"| `{arm}/{system}` | {_ms(e['roc_auc_all_test'])} | " + " | ".join(cells) + " |")
    add("")
    add("The bracketed figure is the change from that unit's own all-test AUC.")
    add("")

    # ---- step 6 ----
    add("## 5. L28 - with_auth across two execution modes")
    add("")
    m = s6["modes"]
    add(f"`{s6['arm']}` only. run 1: {m['run1']['threads']} threads, "
        f"{m['run1']['concurrency']} (`{m['run1']['file']}`). run 2: "
        f"{m['run2']['threads']} thread, concurrency {m['run2']['concurrency']} "
        f"(`{m['run2']['file']}`).")
    add("")
    add("**" + " ".join(s6["open_note"].split()) + "** Arms still OPEN: "
        + ", ".join(f"`{a}`" for a in s6["arms_left_open"]) + ".")
    add("")
    add("| system | metric | run 1 mean | run 1 std | run 2 mean | run 2 std | "
        "run2 - run1 | max same-seed gap | pooled std (10) | pooled / larger within-mode std |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for system in shared["systems"]:
        for metric, e in s6["per_system"][system].items():
            add(f"| {system} | `{metric}` | {_f(e['run1']['mean'])} | "
                f"{_f(e['run1']['within_mode_std'])} | {_f(e['run2']['mean'])} | "
                f"{_f(e['run2']['within_mode_std'])} | "
                f"{_f(e['across_mode_mean_difference_run2_minus_run1'])} | "
                f"{_f(e['max_abs_same_seed_difference'])} | "
                f"{_f(e['pooled_std_10_values'])} | "
                f"{_f(e['pooled_over_largest_within_mode_std'], 2)} |")
    add("")

    # ---- status ----
    add("## 6. Proposed status of each limitation")
    add("")
    add("Proposals only. `docs/KNOWN_LIMITATIONS.md` is NOT edited by this Part; "
        "the text is in `docs/proposed_limitations_update.md` for the user to "
        "accept or reject.")
    add("")
    add("| limitation | proposed status | why |")
    add("|---|---|---|")
    for item in res["proposed_status"]:
        add(f"| {item['limitation']} | **{item['status']}** | {item['why']} |")
    add("")
    add("## 7. What this Part cannot do")
    add("")
    for line in res["cannot"]:
        add(f"- {line}")
    add("")
    add("## 8. Files")
    add("")
    add("| path | what it holds |")
    add("|---|---|")
    add("| `experiments/lanl_reanalysis/config.yaml` | the pre-registration, committed first |")
    add("| `experiments/lanl_reanalysis/run.py` | this analysis and this README's generator |")
    add("| `results/lanl_reanalysis/results.json` | every number above, with its provenance |")
    add("| `results/lanl_reanalysis/integrity.json` | the sha256 of every input, as checked |")
    add("| `tests/unit/test_lanl_reanalysis.py` | weighting, fold assignment, bootstrap helper |")
    add("| `docs/proposed_limitations_update.md` | the proposed limitation text |")
    add("")
    add(f"Elapsed: {res['elapsed_seconds']:.1f} s, analysis only - no training, "
        "nothing rescanned. git HEAD at the time of the run: "
        f"`{res['git_commit']}` - `run.py` is committed in the step that follows "
        "the run, so HEAD here is the step before it.")
    add("")
    return "\n".join(A) + "\n"


def render_proposals(res: dict) -> str:
    s2, s4, s5, s6 = (res["step2_l29"], res["step4_l26"], res["step5_l27"], res["step6_l28"])
    shared = res["config"]["shared"]
    A: list[str] = []
    add = A.append
    add(PART35_BEGIN)
    add("")
    add("## Part 35 proposals (L23, L26, L27, L28 `with_auth`, L29)")
    add("")
    add("Written by `experiments/lanl_reanalysis/run.py` from "
        "`results/lanl_reanalysis/results.json`. Every figure is generated, not "
        "transcribed. `docs/KNOWN_LIMITATIONS.md` is unchanged.")
    add("")
    for item in res["proposed_status"]:
        add(f"### {item['limitation']} - proposed **{item['status']}**")
        add("")
        add(item["why"])
        add("")
        add("> " + item["text"].replace("\n", "\n> "))
        add("")
    add(PART35_END)
    return "\n".join(A) + "\n"


def write_proposals(res: dict) -> None:
    block = render_proposals(res)
    text = PROPOSALS.read_text(encoding="utf-8")
    if PART35_BEGIN in text and PART35_END in text:
        head, rest = text.split(PART35_BEGIN, 1)
        _, tail = rest.split(PART35_END, 1)
        text = head + block.rstrip("\n") + tail
    else:
        text = text.rstrip("\n") + "\n\n---\n\n" + block
    PROPOSALS.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Status proposals (prose assembled from the computed verdicts)
# ---------------------------------------------------------------------------
def proposed_status(res: dict) -> list[dict]:
    s2, s3, s4, s5, s6 = (res["step2_l29"], res["step3_l23"], res["step4_l26"],
                          res["step5_l27"], res["step6_l28"])
    shared = res["config"]["shared"]
    items = []

    # L29
    failing = shared["l29_failing_arms"]
    cause_names = ("a_all_false_view_mask", "b_prior_shift", "c_cross_fitted_platt")
    cause_labels = {"a_all_false_view_mask": "(a) the all-False view mask",
                    "b_prior_shift": "(b) the prior shift (ORACLE)",
                    "c_cross_fitted_platt": "(c) cross-fitted Platt (DIAGNOSTIC)"}
    l29_lines = []
    for arm in failing:
        e0 = s2["per_arm"][arm]["per_system"][shared["systems"][0]]
        c = e0["counts"]
        l29_lines.append(
            f"`{arm}`: {c['a_absent_rows']} of {c['n_test_rows']} test rows have an "
            f"all-False view mask ({c['a_absent_attack']} of them attack rows). On "
            "every one of them, in both systems and all 5 seeds, `prob_attack` is "
            "exactly 0.5 and the saved `vacuity_confidence` is exactly 0.0 - the "
            "least confident value it can take. A constant 0.5 scores the maximum "
            "two-class Brier of 0.5 per row, so those rows alone contribute "
            f"{c['a_absent_rows'] / c['n_test_rows'] * 0.5:.4f} to the arm's Brier, "
            "against a whole budget of 0.0428.")
        for system in shared["systems"]:
            e = s2["per_arm"][arm]["per_system"][system]
            a = e["a_all_false_view_mask"]
            l29_lines.append(
                f"`{arm}` {system}: uncorrected {_ms(e['uncorrected']['brier'])}; "
                f"present rows only {_ms(a['present_rows']['brier'])} against that "
                f"subset's own constant baseline "
                f"{_f(a['present_subset_own_constant_baseline'])}; absent rows "
                f"{_ms(a['absent_rows']['brier'])}; prior-shift ORACLE "
                f"{_ms(e['b_prior_shift']['brier'])}; cross-fitted Platt DIAGNOSTIC "
                f"{_ms(e['c_cross_fitted_platt']['brier'])}")
    verdicts_text = []
    for arm in failing:
        v = s2["verdicts"][arm]
        won = [cause_labels[c] for c in cause_names if v[c]["accounts_for_the_failure"]]
        lost = [
            f"{cause_labels[c]} "
            f"({v[c]['seeds_below_baseline_per_system']['naive']}/5 naive, "
            f"{v[c]['seeds_below_baseline_per_system']['prismflow']}/5 prismflow)"
            for c in cause_names if not v[c]["accounts_for_the_failure"]]
        verdicts_text.append(
            f"`{arm}`: "
            + (f"accounted for by {', '.join(won)}" if won
               else "**none of the three causes accounts for it**")
            + (f"; not accounted for by {', '.join(lost)}" if lost else ""))
    items.append({
        "limitation": "L29",
        "status": "CLOSED",
        "why": (
            "L29's own \"what would change our understanding\" is \"a test of each "
            "candidate cause, and a Brier below the constant baseline once the "
            "responsible one is fixed\". All three named candidates were tested under "
            "a rule fixed before any number existed, one of them is responsible, and "
            "removing its rows does put Brier below the baseline. The responsible "
            "cause is NOT fixed, because the rows it names are unpredictable by "
            "construction - excluding them changes the evaluated population rather "
            "than the model - so the operative warning is unchanged. Proposed CLOSED "
            "as diagnosed; the user may prefer OPEN on the strength of the unfixed "
            "half, and that is their call."
        ),
        "text": (
            "**L29 (update, Part 35).** Each candidate cause was corrected alone on "
            "the saved test-block scores, 5 seeds, both systems, against the fixed "
            "two-class constant baseline 0.0428. Pre-registered rule: a cause "
            "accounts for the failure only if correcting it alone puts Brier below "
            "0.0428 in 5 of 5 seeds for both systems.\n"
            + "\n".join(f"- {line}" for line in l29_lines)
            + "\n- Verdicts: " + "; ".join(verdicts_text) + ".\n"
            "- **\"Accounts for\" is not \"fixed\".** The rows cause (a) names are "
            "unpredictable by construction (Part 30 section 4.2): every model must "
            "score them identically whatever their label. Restricting the metric to "
            "the rows with a present view changes the evaluated population; it does "
            "not change a probability. A deployed detector still meets those cells "
            "and still emits 0.5 on them.\n"
            "- The prior-shift correction makes it WORSE in both failing arms, and it "
            "is an ORACLE (it uses the test prevalence). The Platt recalibration is a "
            "DIAGNOSTIC fitted on test rows and cross-fitted by computer fold, "
            "because `results/lanl/` holds no train-block scores; it does not meet "
            "the rule either. Neither is a deployable fix, so the operative sentence "
            "stands: **probability outputs of those arms should not be used.**\n"
            "- A route for Part 38, not run here: the protocol's own "
            "`vacuity_confidence` is already exactly 0.0 on precisely these rows, so "
            "abstaining on them is a selective-prediction decision the existing "
            "machinery can express, rather than a new model.\n"
            "- Evidence: `results/lanl_reanalysis/results.json`, `step2_l29`; "
            "`experiments/lanl_reanalysis/README.md` section 1."
        ),
    })

    # L23
    rows = []
    for arm in shared["arms"]:
        for system in shared["systems"]:
            e = s3["per_arm"][arm][system]
            d = e["at_threshold"]["default"]
            rows.append(
                f"`{arm}` {system}: weighted Brier {_ms(e['weighted_brier'], 6)}, "
                f"weighted prevalence {_f(e['weighted_prevalence'], 8)}, and at the "
                f"0.5 threshold weighted precision {_ms(d['precision'], 6)}, recall "
                f"{_ms(d['recall'], 4)}, FPR {_ms(d['false_positive_rate'], 8)} at "
                f"{_ms(d['weighted_fp_per_day'], 0)} weighted false alarms per test day")
    items.append({
        "limitation": "L23",
        "status": "CLOSED",
        "why": (
            "L23 asks for \"a measurement at, or reweighted to, the population "
            "prevalence\". That measurement now exists, and it confirms the "
            "limitation rather than lifting it: the reweighting is only as good as "
            "the uniformity of the control hash, and its false-alarm resolution is "
            "512 cells, so it still gives no operating point that transfers."
        ),
        "text": (
            "**L23 (update, Part 35).** Metrics were recomputed under "
            "inverse-probability weights (attack 1, benign 512, the declared 1/512 "
            "control hash) on the saved test-block scores, 5 seeds per arm and "
            "system.\n"
            + "\n".join(f"- {r}" for r in rows)
            + "\n- The weights reconstruct the occupied-cell population only as far "
            "as the crc32 control hash is uniform over cells; uniformity is assumed, "
            "not measured. A weighted false-positive count moves in steps of 512 "
            f"cells over {s3['n_test_days']} test days, so **no alarm rate between 0 "
            "and 128 per day is representable** and all three pre-registered targets "
            "(1, 10 and 100 false alarms per test day) collapse onto the same "
            "zero-alarm threshold. The weighted precision of 1.0 there means no "
            "SAMPLED control fired, which estimates the population false-alarm count "
            "as 0 with a resolution of 512 cells; it does not mean a detector with "
            "no false alarms. A reweighted estimate is therefore not an operating "
            "point.\n"
            f"- Population cells over the test days alone: between "
            f"{s3['population_cells']['test_days_only']['union_lower_bound']:,} and "
            f"{s3['population_cells']['test_days_only']['union_upper_bound']:,} "
            "(`results/lanl/population.json`, summed per day over days 13-16).\n"
            "- Evidence: `results/lanl_reanalysis/results.json`, `step3_l23`; "
            "`experiments/lanl_reanalysis/README.md` section 2."
        ),
    })

    # L26
    paired = s4["paired_model_minus_scalar_unseen"]
    excl = [k for k, v in paired.items() if v["excludes_zero"] and v["observed"] > 0]
    incl = [k for k, v in paired.items() if not v["excludes_zero"]]
    below = [k for k, v in paired.items() if v["excludes_zero"] and v["observed"] < 0]
    # L26's bar: clear the scalar by more than 0.01 AND have the interval exclude 0.
    clears = [k for k, v in paired.items()
              if v["observed"] > 0.01 and v["excludes_zero"] and v["observed"] > 0]
    items.append({
        "limitation": "L26",
        "status": "CLOSED",
        "why": (
            "L26 asks for \"an unseen-computer result that clears the scalar by more "
            "than 0.01 with an interval attached\". The interval is now attached, by "
            "the pre-registered 2000-replicate computer-cluster bootstrap, and the "
            "result confirms the limitation: "
            + (f"{len(clears)} arm-and-system pair(s) clear that bar"
               if clears else
               "NO arm-and-system pair clears that bar, and every paired "
               "model-minus-scalar interval on unseen computers includes 0")
            + ". Closed as confirmed, with the numbers whichever way they fell."
        ),
        "text": (
            "**L26 (update, Part 35).** A computer-cluster bootstrap (2000 "
            f"replicates, fixed seed {s4['bootstrap']['seed']}, "
            f"{s4['bootstrap']['n_test_computers']:,} test computers drawn with "
            "replacement) was run on the saved scores, paired against the untrained "
            "scalar inside each replicate. The scalar was recomputed from "
            "`samples.npz` with Part 30's train-fixed orientation and reproduces the "
            f"committed test AUC {_f(s4['scalar_gate']['test_auc_recomputed'])} before "
            "any interval was computed.\n"
            f"- Scalar `total_records_across_views`: seen AUC "
            f"{_f(s4['per_group']['seen']['scalar']['observed'])} "
            f"[{_f(s4['per_group']['seen']['scalar']['ci95_low'])}, "
            f"{_f(s4['per_group']['seen']['scalar']['ci95_high'])}], unseen AUC "
            f"{_f(s4['per_group']['unseen']['scalar']['observed'])} "
            f"[{_f(s4['per_group']['unseen']['scalar']['ci95_low'])}, "
            f"{_f(s4['per_group']['unseen']['scalar']['ci95_high'])}].\n"
            + "\n".join(
                f"- Paired model-minus-scalar on unseen computers, `{k}`: "
                f"{_f(paired[k]['observed'])} [{_f(paired[k]['ci95_low'])}, "
                f"{_f(paired[k]['ci95_high'])}], "
                + ("excludes 0" if paired[k]["excludes_zero"] else "**includes 0**")
                for k in paired)
            + "\n- "
            + (f"Above the scalar with an interval excluding 0: {', '.join('`'+k+'`' for k in excl) if excl else 'none'}. "
               f"Below it with an interval excluding 0: {', '.join('`'+k+'`' for k in below) if below else 'none'}. "
               f"Interval includes 0: {', '.join('`'+k+'`' for k in incl) if incl else 'none'}.")
            + " L26's own clearance bar is +0.01 with an interval attached; the "
            "arm-and-system pairs that clear it are "
            + (", ".join("`" + k + "`" for k in clears) if clears else "**none**")
            + ".\n"
            "- Evidence: `results/lanl_reanalysis/results.json`, `step4_l26`; "
            "`experiments/lanl_reanalysis/README.md` section 3."
        ),
    })

    # L27
    worst = []
    for arm in shared["arms"]:
        for system in shared["systems"]:
            e = s5["per_arm"][arm][system]
            for host, ph in e["per_host"].items():
                if ph["change_from_all_test"] != NOT_MEASURED:
                    worst.append((abs(ph["change_from_all_test"]), arm, system, host,
                                  ph["change_from_all_test"]))
    worst.sort(reverse=True)
    items.append({
        "limitation": "L27",
        "status": "OPEN",
        "why": (
            "L27's \"what would change our understanding\" is a DESIGN change - a "
            "sample that keeps all or a stated share of the benign cells of attack "
            "hosts. That needs a re-sample and a retrain, which is Part 38. Part 35 "
            "only widens the sensitivity measurement from one host to five, so L27 "
            "stays OPEN with new evidence attached."
        ),
        "text": (
            "**L27 (new evidence, Part 35; still OPEN).** The leave-one-host-out "
            "measurement of section 7.9 was extended from the single top host to "
            "the 5 computers with the most test attack rows, for all four arms and "
            "both systems, on the saved scores (5 seeds, mean +/- std, no "
            "interval - one host is not a resample).\n"
            + "\n".join(
                f"- `{h['computer']}`: {h['test_attack_rows_removed']} attack and "
                f"{h['test_benign_rows_removed']} benign test rows"
                for h in s5["hosts"])
            + "\n- Largest AUC change from removing one host: "
            + "; ".join(f"`{a}/{s}` minus `{h}` {d:+.4f}"
                        for _, a, s, h, d in worst[:3])
            + ".\n- The design change L27 names has not been made, so the "
            "limitation is unchanged. Evidence: "
            "`results/lanl_reanalysis/results.json`, `step5_l27`; "
            "`experiments/lanl_reanalysis/README.md` section 4."
        ),
    })

    # L28
    ratios = []
    for system in shared["systems"]:
        for metric, e in s6["per_system"][system].items():
            if e["pooled_over_largest_within_mode_std"] != NOT_MEASURED:
                ratios.append((e["pooled_over_largest_within_mode_std"], system, metric, e))
    ratios.sort(reverse=True)
    items.append({
        "limitation": "L28",
        "status": "OPEN",
        "why": (
            "L28 asks for \"seeds run across at least two execution modes, so the "
            "reported std covers mode as well as seed\". Only `with_auth` has two "
            "modes in `results/lanl/`; the quantity is now reported per metric for "
            "that arm, and the other three arms have no second mode to compare, so "
            "they stay OPEN until Part 38 runs them."
        ),
        "text": (
            "**L28 (new evidence, Part 35; still OPEN).** For `with_auth` only, the "
            "per-seed std within each execution mode, the spread across modes and "
            "the pooled spread are now reported per metric, both systems "
            "(`results/lanl/results.run1.json.gz` at 12 threads sequential against "
            "`results/lanl/results.json` at 1 thread, concurrency 11).\n"
            + "\n".join(
                f"- {s} `{m}`: within-mode std {_f(e['run1']['within_mode_std'])} "
                f"(run 1) and {_f(e['run2']['within_mode_std'])} (run 2); across-mode "
                f"mean difference {_f(e['across_mode_mean_difference_run2_minus_run1'])}; "
                f"largest same-seed gap {_f(e['max_abs_same_seed_difference'])}; pooled "
                f"std over the 10 values {_f(e['pooled_std_10_values'])}, which is "
                f"{_f(r, 2)}x the larger within-mode std"
                for r, s, m, e in ratios)
            + "\n- **The reading, which sharpens L28 rather than lifting it.** The "
            "two modes' 5-seed MEANS are close: the largest across-mode mean "
            f"difference over these metrics is {_f(max(abs(e['across_mode_mean_difference_run2_minus_run1']) for _, _, _, e in ratios))}, "
            "and pooling the 10 values raises the std to at most "
            f"{_f(max(r for r, _, _, _ in ratios), 2)}x the larger within-mode std. "
            "What moves is the SAME SEED: the largest same-seed gap is "
            f"{_f(max(e['max_abs_same_seed_difference'] for _, _, _, e in ratios))}, "
            "which is several within-mode stds. So the mode selects between "
            "deterministic answers per seed, as L28 already says, and the 5-seed "
            "mean absorbs most of it; a SINGLE-seed figure is the one that must not "
            "be read as mode-independent.\n"
            "- The three remaining arms (`without_auth`, `without_flows`, "
            "`without_dns`) were run in one mode only, so no cross-mode spread "
            "exists for them and L28 is not closed. Evidence: "
            "`results/lanl_reanalysis/results.json`, `step6_l28`; "
            "`experiments/lanl_reanalysis/README.md` section 5."
        ),
    })
    return items


CANNOT = [
    "**It cannot recalibrate anything deployably.** `results/lanl/` holds "
    "test-block scores only. There are no train-block scores, so every fit here "
    "is cross-fitted inside the test block and labelled DIAGNOSTIC.",
    "**It cannot retrain.** No arm is re-fitted, no view is rescanned and no "
    "hyperparameter is changed, so nothing here can fix a cause it diagnoses.",
    "**It cannot give a population operating point.** The weighted estimates of "
    "section 2 inherit the control hash's uniformity assumption and resolve "
    "false-alarm counts only in steps of 512 cells.",
    "**It cannot separate host from behaviour.** Section 4 measures how much each "
    "AUC rests on one host; the sample that would separate them does not exist.",
    "**It cannot speak for the three single-mode arms.** Section 5 covers "
    "`with_auth` alone, because only that arm was run twice.",
    "**The bootstrap interval is test-set sampling only.** It is not the per-seed "
    "std and not a spread over attack episodes: the split is fixed and the "
    "red-team activity is concentrated in a few days.",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover - a detached checkout is not a failure
        return "unknown"


def preregistration_commit() -> str:
    try:
        return subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "experiments/lanl_reanalysis/config.yaml"],
            cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 35 - LANL re-analysis, no training")
    parser.add_argument("--steps", default="2,3,4,5,6",
                        help="comma-separated step numbers to run (default all)")
    args = parser.parse_args()
    wanted = {int(x) for x in args.steps.split(",") if x.strip()}

    started = time.time()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    integrity = verify_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "integrity.json").write_text(
        json.dumps(integrity, indent=1) + "\n", encoding="utf-8")

    samples = Samples(LANL_DIR / "samples.npz")
    part30 = json.loads((LANL_DIR / "results.json").read_text(encoding="utf-8"))
    part30_cfg = yaml.safe_load(PART30_CONFIG.read_text(encoding="utf-8"))
    arm_views = {a["name"]: list(a["views"]) for a in part30_cfg["view_configurations"]}

    res = {
        "experiment": cfg["experiment"]["id"],
        "description": " ".join(cfg["experiment"]["description"].split()),
        "no_training": True,
        "git_commit": git_commit(),
        "preregistration_commit": preregistration_commit(),
        "config": cfg,
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "integrity": {k: v for k, v in integrity.items() if k != "verified"},
        "integrity_detail_file": "results/lanl_reanalysis/integrity.json",
        "arm_views": arm_views,
    }
    if 2 in wanted:
        res["step2_l29"] = step2(cfg, samples, part30, arm_views)
    if 3 in wanted:
        res["step3_l23"] = step3(cfg, samples)
    if 4 in wanted:
        res["step4_l26"] = step4(cfg, samples, part30)
    if 5 in wanted:
        res["step5_l27"] = step5(cfg, samples)
    if 6 in wanted:
        res["step6_l28"] = step6(cfg, part30)
    res["elapsed_seconds"] = time.time() - started

    if wanted == {2, 3, 4, 5, 6}:
        res["proposed_status"] = proposed_status(res)
        res["cannot"] = CANNOT
        (OUT_DIR / "results.json").write_text(
            json.dumps(res, indent=1) + "\n", encoding="utf-8")
        (HERE / "README.md").write_text(render_readme(res), encoding="utf-8")
        write_proposals(res)
        print(f"wrote {OUT_DIR / 'results.json'}, {HERE / 'README.md'} and "
              f"the Part 35 block of {PROPOSALS}")
    else:
        path = OUT_DIR / f"partial_steps_{'_'.join(str(s) for s in sorted(wanted))}.json"
        path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"PARTIAL run ({sorted(wanted)}): wrote {path}. No README, no proposals.")


if __name__ == "__main__":
    main()
