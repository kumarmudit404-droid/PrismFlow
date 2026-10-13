"""Part 29: the chorus attack, the suspicion detector, natural-dependence
false alarms and extra missingness, on real AIT log data.

WHAT THE LABEL MEANS -- read this before any number below
---------------------------------------------------------
A window is attack=1 if ANY officially labelled line falls in it, and on both
scenarios that label is dominated by one continuously-running DNS exfiltration
tool: 1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on
santos carry only dnsteal-family steps (`docs/part26-ait-loader.md` sections 0
and 12). So the task is "WAS THE EXFILTRATION TOOL RUNNING IN THIS WINDOW",
NOT "detect an APT" and not "detect a multi-step intrusion". A result from
this script described as intrusion, APT or campaign detection is a misreport
even when every number in it is correct.

ONE VIEW SOLVES THAT LABEL, AND IT DECIDES WHAT THIS PART MEASURES
------------------------------------------------------------------
`docs/KNOWN_LIMITATIONS.md` L17: `single_view_dns` scores balanced accuracy
0.9898 on santos and 1.0000 on russellmitchell, and no other single view
clears 0.7061. So "k views compromised" is not one experiment. Compromising
`dns` attacks the only view carrying the label; compromising any subset of
`apache`, `auth`, `audit` and `suricata` perturbs views that score 0.3733 to
0.7061 balanced accuracy on their own.

Every test-B result is therefore reported in two blocks -- `dns` compromised
and `dns` not compromised -- and nothing is averaged across that split.
`assert_no_pooling` refuses to summarise a mixed list, so the rule is enforced
by the code rather than left to whoever reads the table.

THE MULTI-STEP WINDOW DEFINITION USED HERE
-------------------------------------------
STRICT: attack windows carrying at least one step outside
`EXFILTRATION_TOOL_STEPS` (the dnsteal family plus the exfiltration tool's own
service record). That is 3 windows on santos and 4 on russellmitchell.
`experiments/ait_clean/config.yaml` and the `run_ait_clean.py` module docstring
quote a LOOSER count for russellmitchell (5, beyond the dnsteal family alone);
this script does not use that definition anywhere and does not mix the two.
3 and 4 windows is a handful-of-windows result: descriptive only, no ROC AUC
over it, and the count is printed beside every number.

METRICS ARE IMPORTED, NOT REIMPLEMENTED
----------------------------------------
Balanced accuracy, ROC AUC and per-class recall come from `part28_extras` in
`experiments/ait_clean/run_ait_clean.py`, together with `compute_metrics`,
exactly as Part 28 does. `docs/EVALUATION_PROTOCOL.md` records that as a named
exception to the "no experiment computes its own metrics" rule and states that
Parts 29 and 30 must share the one function. The detector's AUC goes through
the same `roc_auc_binary` that `part28_extras` itself calls.

SCOPE AND ETHICS
----------------
A controlled research simulation. Every number comes from this project's own
models on a public research dataset, inside this repository. No live systems,
no network traffic, no intrusion tooling.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.ait_clean.run_ait_clean import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    DNSTEAL_FAMILY,
    EXFILTRATION_TOOL_STEPS,
    LABEL_MEANING,
    LOWER_IS_BETTER,
    apply_scaler,
    fit_scaler,
    labelled_indices,
    load_scenario,
    metric_row,
    paired_difference,
    part28_extras,
    roc_auc_binary,
    train_ait,
)
from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import ChorusConfig, choose_targets, chorus_attack
from prismflow.data.ait_lds import FEATURES, sensor_groups
from prismflow.data.corruption import drop_views
from prismflow.data.dataset import Batch
from prismflow.data.loaders import iter_batches
from prismflow.eniv.eniv import compute_eniv
from prismflow.models.defended import DefendedPrismFlow
from prismflow.models.prismflow import REPORTING_NULL_PERMUTATIONS
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.utils.logging import get_logger

LOGGER = get_logger("prismflow.ait_attack")
CONFIG_PATH = Path(__file__).with_name("config.yaml")
PART28_RESULTS = Path("results/ait_clean/results.json")

#: Printed beside every multi-step number, every time, not once in a caption.
HANDFUL_CAVEAT = (
    "3 windows on santos and 4 on russellmitchell. Any claim about the "
    "multi-step intrusion is a handful-of-windows result: descriptive only, "
    "no ROC AUC, no significance claim, and no mean +/- std over seeds "
    "presented as a finding."
)

#: The label Part 10's `clone_k2` comparison must NOT be read as. Test C's
#: substitute block is a different construction and says so wherever it
#: appears.
NATURAL_DEPENDENCE_LABEL = "different sensors, naturally dependent"


# ---------------------------------------------------------------------------
# Honesty guards: the dns split, and "not measured"
# ---------------------------------------------------------------------------

#: Written wherever a quantity was not computed. Never 0, never blank, never
#: interpolated from a neighbouring cell.
NOT_MEASURED = "not measured"


def dns_index(sources) -> int:
    """Position of the `dns` view. Named, not hard-coded to 2."""
    return list(sources).index("dns")


def has_dns(subset, sources) -> bool:
    return dns_index(sources) in tuple(subset)


def assert_no_pooling(rows: list[dict], where: str) -> None:
    """Refuse to summarise a list of subsets that mixes the two dns sides.

    L17 says the result depends almost entirely on whether the compromised set
    includes `dns`, so a mean over both sides is not a weaker statement than
    the split one -- it is a different and misleading quantity. This is called
    before every aggregation in test B.
    """
    sides = {bool(row["dns_compromised"]) for row in rows}
    if len(sides) > 1:
        raise AssertionError(
            f"{where}: refusing to aggregate {len(rows)} subsets spanning BOTH "
            f"dns sides. docs/KNOWN_LIMITATIONS.md L17 forbids pooling across "
            f"the dns split; aggregate within a side."
        )


# ---------------------------------------------------------------------------
# Setup: Part 28's data path, scaler and training, unchanged
# ---------------------------------------------------------------------------


def check_part28(cfg: dict) -> dict:
    """Stop unless Part 28 exists and ran at this window length and mode."""
    if not PART28_RESULTS.is_file():
        raise SystemExit(
            f"missing {PART28_RESULTS}. Part 29 reuses Part 28's trained models "
            f"and its committed pair table -- run /part28 first."
        )
    part28 = json.loads(PART28_RESULTS.read_text())
    data = cfg["data"]
    for name, record in part28["directions"].items():
        structure = record["test_structure"]
        if (
            structure["window_seconds"] != data["window_seconds"]
            or structure["sample_mode"] != data["sample_mode"]
        ):
            raise SystemExit(
                f"Part 28 direction {name} ran at "
                f"{structure['window_seconds']}s/{structure['sample_mode']}, "
                f"this config asks for {data['window_seconds']}s/{data['sample_mode']}. "
                f"The brief requires the same window and mode as Part 28."
            )
    return part28


def prepare_direction(direction: dict, cfg: dict, cache: dict) -> dict:
    """Load, scale and index one direction exactly as Part 28 does."""
    for scenario in (direction["train"], direction["test"]):
        if scenario not in cache:
            cache[scenario] = load_scenario(scenario, cfg)[0]
    train_raw, test_raw = cache[direction["train"]], cache[direction["test"]]
    train_index = labelled_indices(train_raw)
    test_index = labelled_indices(test_raw)
    scaler = fit_scaler(train_raw, train_index, cfg)
    return {
        "train": apply_scaler(train_raw, scaler),
        "test": apply_scaler(test_raw, scaler),
        "train_index": train_index,
        "test_index": test_index,
        "scaler": scaler,
        "raw_test": test_raw,
    }


def feature_scale(prepared: dict) -> dict:
    """Per-view feature std on the TEST set, the measurement that justifies
    carrying Part 09's epsilon grid over unrescaled.

    Measured over present native columns only, because an absent view's row is
    all zeros and carries no observation.
    """
    dataset = prepared["test"]
    index = prepared["test_index"]
    views = dataset.views.numpy()[index]
    mask = dataset.view_mask.numpy()[index]
    native = prepared["scaler"]["native_dims"]
    per_view = {}
    stds = []
    for position, source in enumerate(dataset.sources):
        rows = mask[:, position]
        if rows.sum() < 2:
            per_view[source] = {"std": NOT_MEASURED, "reason": "fewer than 2 present rows"}
            continue
        block = views[rows, position, : native[position]].astype(np.float64)
        q75, q25 = np.percentile(block, [75, 25])
        stds.append(float(block.std()))
        per_view[source] = {
            "std": float(block.std()),
            "iqr": float(q75 - q25),
            "median_abs": float(np.median(np.abs(block))),
            "max_abs": float(np.abs(block).max()),
            "n_present": int(rows.sum()),
        }
    return {
        "per_view": per_view,
        "min_per_view_std": min(stds) if stds else float("nan"),
        "median_per_view_std": float(np.median(stds)) if stds else float("nan"),
        "max_per_view_std": max(stds) if stds else float("nan"),
        "synthetic_reference_std": 1.3,
        "note": (
            "Part 09 chose epsilon 0.2-2.0 against synthetic features with std "
            "~1.3. The MEDIAN per-view std here brackets that, so the grid is "
            "carried over unrescaled. Where one view's std is far from the "
            "median, a fixed epsilon is a different relative perturbation for "
            "that view; see the README limitation."
        ),
    }


# ---------------------------------------------------------------------------
# Window populations: benign, dnsteal-only, multi-step. Never pooled.
# ---------------------------------------------------------------------------


def window_populations(prepared: dict, scenario: str) -> dict:
    """Split one direction's test rows into the three populations of test A.

    `multi_step` uses the STRICT definition (beyond EXFILTRATION_TOOL_STEPS),
    the same `beyond_set` logic as Part 28's `test_structure`. The looser
    count, beyond DNSTEAL_FAMILY alone, is reported beside it under its own
    name so neither source document's figure is dropped and the two are never
    conflated.
    """
    dataset = prepared["raw_test"]
    index = prepared["test_index"]
    labels = dataset.labels.numpy()
    steps = dataset.samples.attack_steps

    def beyond(allowed: tuple[str, ...]) -> set[int]:
        allowed_set = set(allowed)
        return {
            row
            for row in index
            if labels[row] == CLASS_ATTACK
            and steps[row]
            and not set(steps[row]) <= allowed_set
        }

    beyond_family = beyond(DNSTEAL_FAMILY)
    multi_step = beyond(EXFILTRATION_TOOL_STEPS)

    attack_rows = [row for row in index if labels[row] == CLASS_ATTACK]
    benign_rows = [row for row in index if labels[row] == CLASS_BENIGN]
    dnsteal_only = [row for row in attack_rows if row not in beyond_family]

    # The window that is beyond the dnsteal family but is NOT a multi-step
    # step: it carries the exfiltration tool's own SERVICE_START /
    # SERVICE_STOP record, which is exactly what the strict and loose
    # definitions differ by. One window per scenario. It belongs to neither
    # named population, so it is named here rather than silently dropped --
    # without it the populations would not add up to the test set and a reader
    # could not tell which window went missing or why.
    service_only = sorted(beyond_family - multi_step)

    return {
        "scenario": scenario,
        "definition": "strict: attack windows with a step outside EXFILTRATION_TOOL_STEPS",
        "benign": np.asarray(sorted(benign_rows), dtype=np.int64),
        "dnsteal_only": np.asarray(sorted(dnsteal_only), dtype=np.int64),
        "multi_step": np.asarray(sorted(multi_step), dtype=np.int64),
        "exfiltration_service_only": np.asarray(service_only, dtype=np.int64),
        "n_benign": len(benign_rows),
        "n_dnsteal_only": len(dnsteal_only),
        "n_multi_step": len(multi_step),
        "n_exfiltration_service_only": len(service_only),
        "n_attack_beyond_dnsteal_family_looser_count": len(beyond_family),
        "partition_note": (
            "benign + dnsteal_only + multi_step + exfiltration_service_only "
            "accounts for every label_known row. The last population is the "
            "single window the strict and loose multi-step definitions differ "
            "by; it is reported separately and pooled with neither."
        ),
        "handful_caveat": HANDFUL_CAVEAT,
    }


# ---------------------------------------------------------------------------
# Dependence and ENIV on an arbitrary views tensor
# ---------------------------------------------------------------------------


def dependence_of(model, views: torch.Tensor, view_mask: torch.Tensor, cfg: dict) -> dict:
    """The reporting-grade [V, V] matrix and ENIV for given views.

    Part 28's `dependence_and_eniv` takes a dataset and an index array, so it
    cannot score a PERTURBED tensor. This calls the identical frozen
    estimators -- `feature_dependence_matrix` and `compute_eniv` -- with the
    model's own method, conditioning and seed, and with
    `REPORTING_NULL_PERMUTATIONS` (12), the same reporting-grade value Part 28
    and Part 15 use. `test_clean_dependence_matches_part28` checks this
    reproduces Part 28's committed matrix on clean input, so the wrapper is
    pinned to that Part's numbers rather than merely resembling them.
    """
    with torch.no_grad():
        output = model(views, view_mask)
        matrix = feature_dependence_matrix(
            model.encoder(views, view_mask),
            output.per_view_evidence,
            view_mask,
            method=model.dependence_method,
            conditioning=model.dependence_conditioning,
            seed=model.dependence_seed,
            null_permutations=REPORTING_NULL_PERMUTATIONS,
        )
    available = available_views(view_mask)
    matrix = np.asarray(matrix, dtype=np.float64)
    record = {"matrix": [[_safe(x) for x in row] for row in matrix.tolist()]}

    # DEGENERACY CHECK. On a handful of samples the permutation null subtracts
    # everything the estimator measured and the off-diagonal comes back
    # exactly 0, which makes ENIV pin at the view count and the efficiency
    # ratio at 1.0. Those are NOT measurements of perfect independence, they
    # are the estimator reporting that it had nothing to work with. The flag
    # travels with the numbers so no table can quote 5.0 as a finding.
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    degenerate = bool(finite.size == 0 or np.all(finite == 0.0))
    record["n_samples"] = int(views.shape[0])
    record["dependence_degenerate"] = degenerate
    record["degenerate_reason"] = (
        (
            f"every off-diagonal dependence entry is exactly 0 at n = "
            f"{views.shape[0]}: the estimator could not measure dependence at "
            f"this sample size, so ENIV pins at the view count and the "
            f"efficiency ratio at 1.0. Read these as {NOT_MEASURED}, NOT as "
            f"perfect independence."
        )
        if degenerate
        else None
    )

    for method in cfg["diagnostics"]["eniv_methods"]:
        result = compute_eniv(matrix, available=available, method=method)
        record[f"eniv_{method}"] = _safe(result.effective_views)
        record[f"mean_dependence_{method}"] = _safe(result.mean_dependence)
        record[f"efficiency_ratio_{method}"] = _safe(result.efficiency_ratio)
    return record


def mean_within_subset(matrix: np.ndarray, subset: tuple[int, ...]) -> float:
    """Mean dependence among the compromised views.

    NaN at k = 1 by construction: one view has no within-pair to average. Part
    09 recorded the same gap -- the single-view attack is invisible to this
    diagnostic -- and it is reported as `not measured`, never as 0.
    """
    members = tuple(subset)
    if len(members) < 2:
        return float("nan")
    values = [
        matrix[a, b]
        for a, b in itertools.combinations(members, 2)
        if math.isfinite(matrix[a, b])
    ]
    return float(np.mean(values)) if values else float("nan")


def _safe(value) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


# ---------------------------------------------------------------------------
# One attacked (or corrupted) pass over a test set
# ---------------------------------------------------------------------------


#: Why the detector columns are empty for naive fusion. `use_discount=False`
#: leaves `output.dependence_matrix` None, so `DefendedPrismFlow` builds no
#: `SuspicionReport` and there is no score to report -- verified, not assumed.
#: Every such cell reads `not measured` with this reason. It is NOT 0: a 0
#: would claim the detector ran and never fired.
NO_DETECTOR_REASON = (
    "naive fusion runs with use_discount=False, which leaves "
    "output.dependence_matrix None, so DefendedPrismFlow produces no "
    "SuspicionReport. The detector is undefined for this system, not silent."
)


def _forward_rows(defended, views: torch.Tensor, view_mask: torch.Tensor) -> dict:
    """One scored pass. `score`/`excess` are NaN/None when the system has no
    dependence matrix to detect against (naive fusion); see
    NO_DETECTOR_REASON."""
    with torch.no_grad():
        out = defended(views, view_mask)
    return {
        "probs": out.probs.numpy(),
        "vacuity": out.confidence.numpy(),
        "score": (
            np.full(views.shape[0], np.nan)
            if out.suspicion is None
            else np.asarray(out.suspicion.score, dtype=np.float64)
        ),
        "excess": None if out.suspicion is None else np.asarray(out.suspicion.excess),
    }


def run_cell(
    defended,
    dataset,
    index: np.ndarray,
    cfg: dict,
    *,
    kind: str,
    subset: tuple[int, ...] = (),
    epsilon: float = 0.0,
    beta: float = 1.0,
    seed: int = 0,
    missing_rate: float = 0.0,
) -> dict:
    """Score one condition over a whole test set, batch by batch.

    Batches are `attack.batch_size` (1200), which is ONE batch for both test
    sets (n = 1,108 and 1,145). This differs from Part 09 and Part 10, which
    batched at 64, and the deviation is deliberate:

    * The dependence estimator's permutation null runs once per batch, so
      fewer batches is much cheaper -- measured 6.0 s against 14.3 s per
      PrismFlow attack on direction A. That is what makes the full 25-subset
      enumeration affordable at all.
    * It matches Part 28's measurement convention, which computes dependence
      and ENIV over the whole test set at once.

    What it changes: the chorus objective's backtracking accepts or rejects on
    the BATCH MEAN, so with one batch that decision is global. Per-sample
    outcomes can move either way against finer batching -- this is a
    different attack, not a strictly weaker or stronger one. Measured on
    direction A, seed 0, PrismFlow, the success rates agree closely across
    batch sizes (views (2,4): 0.5000 at 64, 0.5153 at 256, 0.5171 at 1200;
    views (0,1,3): 0.0578 / 0.0542 / 0.0614), i.e. well inside the seed
    spread, so the choice does not carry the result.
    """
    model = defended.model
    attack_cfg = cfg["attack"]
    batch_size = attack_cfg["batch_size"]

    probs, vacuity, scores, labels = [], [], [], []
    targets, perturbed_views, perturbed_masks = [], [], []
    excess_blocks = []

    for position, batch in enumerate(iter_batches(dataset, index, batch_size)):
        views, view_mask = batch.views, batch.view_mask

        if kind == "missing":
            # Seeded on (seed, batch, rate) so every system sees the same draws.
            corrupted = drop_views(
                Batch(
                    views=views.clone(),
                    view_mask=view_mask.clone(),
                    labels=batch.labels.clone(),
                    sample_ids=batch.sample_ids.clone(),
                ),
                rate=missing_rate,
                # zlib.crc32, not hash(): Python randomises string hashing per
                # process, so hash() would make the draws irreproducible
                # across runs. Part 10's run_comparison.py does the same.
                seed=zlib.crc32(f"{seed}|{position}|{missing_rate}".encode()) % (2**31),
            )
            views, view_mask = corrupted.views, corrupted.view_mask
            target = torch.full((views.shape[0],), -1, dtype=torch.long)
        elif kind in ("chorus", "pgd"):
            config = ChorusConfig(
                epsilon=epsilon,
                beta=beta if kind == "chorus" else 0.0,
                steps=attack_cfg["steps"],
                target_strategy=attack_cfg["target_strategy"],
                fixed_target=attack_cfg["fixed_target"],
                seed=seed,
                views=tuple(subset),
            )
            target = choose_targets(model, views, view_mask, config)
            runner = chorus_attack if kind == "chorus" else pgd_attack
            result = runner(model, views, view_mask, config, target=target)
            views = views + result.delta
        else:  # clean
            target = torch.full((views.shape[0],), -1, dtype=torch.long)

        rows = _forward_rows(defended, views, view_mask)
        probs.append(rows["probs"])
        vacuity.append(rows["vacuity"])
        scores.append(rows["score"])
        if rows["excess"] is not None:
            excess_blocks.append(rows["excess"])
        labels.append(batch.labels.numpy())
        targets.append(target.numpy())
        perturbed_views.append(views)
        perturbed_masks.append(view_mask)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    targets = np.concatenate(targets)
    scores = np.concatenate(scores)

    row = metric_row(labels, probs, np.concatenate(vacuity), cfg)
    predictions = probs.argmax(axis=1)

    # Attack success: the target is the least-likely class on CLEAN input, so
    # it is never the clean prediction and a clean cell scores 0 by
    # construction -- Part 09's definition, unchanged.
    if kind in ("chorus", "pgd"):
        row["attack_success"] = float((predictions == targets).mean())
    else:
        row["attack_success"] = float("nan")

    full_views = torch.cat(perturbed_views, dim=0)
    full_mask = torch.cat(perturbed_masks, dim=0)
    dependence = dependence_of(model, full_views, full_mask, cfg)
    matrix = np.asarray(dependence["matrix"], dtype=np.float64)
    for key, value in dependence.items():
        if key != "matrix":
            row[key] = value
    row["dependence_compromised"] = mean_within_subset(matrix, subset)
    row["suspicion_mean_score"] = (
        float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
    )
    row["_scores"] = scores
    row["_matrix"] = matrix
    row["_excess"] = np.concatenate(excess_blocks) if excess_blocks else None
    return row


# ---------------------------------------------------------------------------
# Detector scoring, through the imported AUC
# ---------------------------------------------------------------------------


def detector_auc(attacked_scores: np.ndarray, clean_scores: np.ndarray) -> float:
    """P(attacked score > clean score), ties at 0.5.

    Goes through `roc_auc_binary`, the function `part28_extras` itself calls,
    rather than a second implementation -- `docs/EVALUATION_PROTOCOL.md`
    requires Parts 29 and 30 to share Part 28's metric code. The two-sample
    form is obtained by labelling attacked rows CLASS_ATTACK and clean rows
    CLASS_BENIGN; NaN scores are dropped first, which is input preparation and
    not a change to the statistic.
    """
    attacked = np.asarray(attacked_scores, dtype=np.float64)
    clean = np.asarray(clean_scores, dtype=np.float64)
    attacked = attacked[np.isfinite(attacked)]
    clean = clean[np.isfinite(clean)]
    if attacked.size == 0 or clean.size == 0:
        return float("nan")
    labels = np.concatenate(
        [np.full(attacked.size, CLASS_ATTACK), np.full(clean.size, CLASS_BENIGN)]
    )
    return roc_auc_binary(labels, np.concatenate([attacked, clean]))


def detection_at_budgets(attacked: np.ndarray, clean: np.ndarray, budgets) -> dict:
    """True-positive rate at thresholds calibrated on this seed's clean scores."""
    from prismflow.statistics.suspicion import calibrate_threshold, detection_rates

    out = {}
    for budget in budgets:
        threshold = calibrate_threshold(clean, target_rate=budget)
        out[f"detect_at_{int(round(budget * 100))}pct"] = detection_rates(attacked, threshold)
    return out


# ---------------------------------------------------------------------------
# Subset enumeration, split by dns membership
# ---------------------------------------------------------------------------


def enumerate_subsets(sources, k_grid) -> list[dict]:
    """All subsets of every k in `k_grid`, each tagged with its dns side.

    With 5 views and k in {1, 2, 3} this is 25 subsets: k=1 gives 1 with dns
    and 4 without, k=2 gives 4 and 6, k=3 gives 6 and 4. The runner enumerates
    exhaustively -- no sampling -- and `subset_sampling` in the results JSON
    records that.
    """
    names = list(sources)
    out = []
    for k in k_grid:
        for subset in itertools.combinations(range(len(names)), k):
            out.append(
                {
                    "subset": subset,
                    "names": [names[i] for i in subset],
                    "k": k,
                    "dns_compromised": has_dns(subset, names),
                }
            )
    return out


def attack_grid(k: int, cfg: dict) -> list[tuple[str, float, float]]:
    """The (kind, epsilon, beta) points run for a subset of size k.

    Part 09's grid, carried over in full:

      chorus, beta = 1, epsilon 0.2 / 0.5 / 1.0 / 2.0      every k
      chorus, epsilon 1.0, beta 0 / 5                      k >= 2 only
      pgd,    epsilon 1.0 / 2.0                             k >= 2
      pgd,    epsilon 1.0                                   k = 1

    Two exclusions, both because the excluded cell is provably the same
    number as one already run, not because of cost:

    * beta is INERT at k = 1. `chorus_objective` adds the agreement penalty
      only when the compromised set has more than one member, so beta 0 and
      beta 5 at k = 1 are bit-identical to beta 1. Recording them would put
      the same figure in three rows and inflate the apparent grid.
    * chorus IS pgd at k = 1, for the same reason: with no within-pairs the
      objective separates per view. One pgd cell is kept at k = 1 so the
      identity is visible in the results rather than only asserted in a test
      (`test_chorus_equals_pgd_at_k1`), and `pgd_epsilon_grid`'s second point
      is skipped there as a duplicate of the chorus cell at that epsilon.
    """
    attack = cfg["attack"]
    grid: list[tuple[str, float, float]] = [
        ("chorus", epsilon, attack["beta_primary"]) for epsilon in attack["epsilon_grid"]
    ]
    if k >= 2:
        grid += [
            ("chorus", attack["beta_sweep_epsilon"], beta)
            for beta in attack["beta_grid"]
            if beta != attack["beta_primary"]
        ]
        grid += [("pgd", epsilon, 0.0) for epsilon in attack["pgd_epsilon_grid"]]
    else:
        grid.append(("pgd", attack["beta_sweep_epsilon"], 0.0))
    return grid


def summarise_rows(rows: list[dict], keys) -> dict:
    """mean, sample std (ddof=1), n_seeds per key. NaN excluded, never zeroed."""
    out = {}
    for key in keys:
        values = [
            float(r[key])
            for r in rows
            # `bool` is a subclass of `int`, so a flag like
            # `dependence_degenerate` would otherwise be averaged into a
            # meaningless 0.0-1.0 "metric". Excluded explicitly.
            if key in r
            and isinstance(r[key], (int, float))
            and not isinstance(r[key], bool)
            and math.isfinite(float(r[key]))
        ]
        out[key] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "n_seeds": len(values),
        }
        if not values:
            out[key]["status"] = NOT_MEASURED
    return out


# ---------------------------------------------------------------------------
# Test C: the pairs, read from Part 28's committed table
# ---------------------------------------------------------------------------


def read_pair_table(part28: dict, direction: str, system: str) -> list[dict]:
    return part28["directions"][direction]["systems"][system]["pair_table"]


def ranked_pairs(part28: dict, direction: str, cfg: dict) -> dict:
    """The two most and two least dependent pairs, from the committed table.

    Read, never re-derived. Both systems' orderings are checked: if `naive`
    and `prismflow` disagree on which pairs those are, that disagreement is
    reported and no single ordering is picked, as the brief requires.
    """
    wanted = cfg["natural_dependence"]
    orderings = {}
    for system in ("naive", "prismflow"):
        rows = sorted(read_pair_table(part28, direction, system), key=lambda r: -r["mean"])
        orderings[system] = {
            "most": [r["pair"] for r in rows[: wanted["n_most_dependent"]]],
            "least": [r["pair"] for r in rows[-wanted["n_least_dependent"] :]][::-1],
            "values": {r["pair"]: {"mean": r["mean"], "std": r["std"]} for r in rows},
        }

    agree = (
        orderings["naive"]["most"] == orderings["prismflow"]["most"]
        and orderings["naive"]["least"] == orderings["prismflow"]["least"]
    )
    expected_most = wanted["expected_most_dependent"][direction]
    expected_least = wanted["expected_least_dependent"][direction]
    matches_config = (
        orderings["naive"]["most"] == list(expected_most)
        and orderings["naive"]["least"] == list(expected_least)
    )
    if not matches_config:
        raise SystemExit(
            f"{direction}: Part 28's pair table no longer gives the pairs this "
            f"config names. Committed most/least (naive): "
            f"{orderings['naive']['most']} / {orderings['naive']['least']}; "
            f"config expects {list(expected_most)} / {list(expected_least)}. "
            f"Part 28 is frozen -- REPORT this, do not edit either file."
        )
    return {
        "label": NATURAL_DEPENDENCE_LABEL,
        "source": cfg["natural_dependence"]["pair_table_source"],
        "systems_agree_on_ordering": agree,
        "most_dependent": orderings["naive"]["most"],
        "least_dependent": orderings["naive"]["least"],
        "per_system": orderings,
        "caveat": (
            "These views sit on DIFFERENT physical sensors. The dependence is a "
            "property of the scenario, not of shared hardware. This is NOT a "
            "same-sensor result and must not be read as one."
        ),
    }


def pair_excess_scores(excess: np.ndarray, pair: str, sources) -> np.ndarray:
    """Per-sample unexplained agreement for ONE named pair.

    The detector's headline score is a max over all pairs, so it cannot say
    which pair fired. `SuspicionReport.excess` is [B, V, V] per-pair, and this
    reads the one entry. NaN where the pair is not measurable in that sample
    (either view absent, or a wholly vacuous belief vector).
    """
    names = list(sources)
    left, right = pair.split("-")
    i, j = names.index(left), names.index(right)
    return np.asarray(excess[:, i, j], dtype=np.float64)


def natural_dependence_block(
    excess: np.ndarray, pairs: dict, sources, cfg: dict
) -> dict:
    """Test C's substitute quantity: does natural dependence raise the detector?

    Constructed as pair-against-pair on CLEAN data -- the AUC that the
    most-dependent pairs' excess beats the least-dependent pairs' excess.

    THIS IS NOT PART 10's `clone_k2` AUC. That was condition-against-clean on
    an exactly duplicated view. Here there is no corrupted condition to
    compare against, because the dependence is already present in the clean
    data, so the contrast has to be between pairs. The two numbers answer
    related questions and are not on the same scale; the README says so
    wherever they appear together.
    """
    if excess is None:
        return {"status": NOT_MEASURED, "reason": "the detector produced no per-pair excess"}

    per_pair = {}
    for pair in pairs["most_dependent"] + pairs["least_dependent"]:
        values = pair_excess_scores(excess, pair, sources)
        finite = values[np.isfinite(values)]
        per_pair[pair] = {
            "mean_excess": float(finite.mean()) if finite.size else float("nan"),
            "n_measurable": int(finite.size),
            "n_total": int(values.size),
            "committed_dependence": pairs["per_system"]["naive"]["values"][pair],
        }

    most = np.concatenate(
        [pair_excess_scores(excess, p, sources) for p in pairs["most_dependent"]]
    )
    least = np.concatenate(
        [pair_excess_scores(excess, p, sources) for p in pairs["least_dependent"]]
    )
    return {
        "label": NATURAL_DEPENDENCE_LABEL,
        "auc_most_over_least": detector_auc(most, least),
        "per_pair": per_pair,
        "most_dependent": pairs["most_dependent"],
        "least_dependent": pairs["least_dependent"],
        "construction": (
            "AUC that a most-dependent pair's per-sample excess exceeds a "
            "least-dependent pair's, on CLEAN data. Pair-against-pair, NOT "
            "condition-against-clean, so it is not comparable in scale with "
            "Part 10's clone_k2 AUC of 0.8743 +/- 0.0583."
        ),
        "not_the_same_as_clone_k2": True,
    }


# ---------------------------------------------------------------------------
# One (direction, system): every cell
# ---------------------------------------------------------------------------

ATTACK_KEYS = (
    "attack_success",
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
    "brier",
    "prob_ece",
    "attack_recall",
    "benign_recall",
    "eniv_eigen",
    "eniv_design_effect",
    "mean_dependence_eigen",
    "dependence_compromised",
    "suspicion_mean_score",
    "detector_auc",
    "detect_at_5pct",
    "detect_at_1pct",
)

NATURAL_KEYS = (
    "eniv_eigen",
    "eniv_design_effect",
    "mean_dependence_eigen",
    "mean_dependence_design_effect",
    "efficiency_ratio_eigen",
    "suspicion_mean_score",
)


def run_system(direction: dict, system: dict, prepared: dict, cfg: dict, part28: dict) -> dict:
    """Every test, for one direction and one system, over all seeds."""
    sources = list(prepared["test"].sources)
    populations = window_populations(prepared, direction["test"])
    pairs = ranked_pairs(part28, direction["name"], cfg)
    subsets = enumerate_subsets(sources, cfg["attack"]["k_grid"])
    budgets = cfg["detector"]["fpr_budgets"]

    per_seed: dict[str, list[dict]] = {}
    natural_per_seed: dict[str, list[dict]] = {}
    natural_dep_per_seed: list[dict] = []
    clean_rows: list[dict] = []

    def record(bucket: dict, key: str, row: dict, seed: int) -> None:
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        clean_row["seed"] = seed
        bucket.setdefault(key, []).append(clean_row)

    for seed in cfg["seeds"]:
        started = time.time()
        model = train_ait(prepared["train"], prepared["train_index"], seed, system["use_discount"], cfg)
        defended = DefendedPrismFlow(
            model,
            suspicion_permutations=cfg["detector"]["permutations"],
            suspicion_seed=seed,
        )

        # --- clean, first: it calibrates every threshold for this seed ------
        clean = run_cell(defended, prepared["test"], prepared["test_index"], cfg, kind="clean", seed=seed)
        clean_scores = clean["_scores"]
        clean_row = {k: v for k, v in clean.items() if not k.startswith("_")}
        clean_row["seed"] = seed
        has_detector = bool(np.isfinite(clean_scores).any())
        # 0.5 is what a detector scores against its own reference, BY
        # CONSTRUCTION. Naive fusion has no detector at all, so writing 0.5
        # there would invent a measurement for a system that produced none.
        clean_row["detector_auc"] = 0.5 if has_detector else float("nan")
        clean_row["detector_available"] = has_detector
        if not has_detector:
            clean_row["detector_status"] = NOT_MEASURED
            clean_row["detector_reason"] = NO_DETECTOR_REASON
        clean_row.update(detection_at_budgets(clean_scores, clean_scores, budgets))
        clean_rows.append(clean_row)

        # --- TEST A: natural window populations, never pooled ---------------
        for population in (
            "benign", "dnsteal_only", "multi_step", "exfiltration_service_only"
        ):
            rows = populations[population]
            if rows.size == 0:
                record(
                    natural_per_seed,
                    population,
                    {"status": NOT_MEASURED, "reason": "no windows in this population", "n_windows": 0},
                    seed,
                )
                continue
            cell = run_cell(defended, prepared["test"], rows, cfg, kind="clean", seed=seed)
            cell["n_windows"] = int(rows.size)
            record(natural_per_seed, population, cell, seed)

        # --- TEST C: natural dependence on clean data -----------------------
        natural_dep = natural_dependence_block(clean["_excess"], pairs, sources, cfg)
        natural_dep["seed"] = seed
        natural_dep_per_seed.append(natural_dep)

        # --- TEST B: every subset, every grid point -------------------------
        for position, entry in enumerate(subsets, start=1):
            subset, k = entry["subset"], entry["k"]
            for kind, epsilon, beta in attack_grid(k, cfg):
                cell = run_cell(
                    defended, prepared["test"], prepared["test_index"], cfg,
                    kind=kind, subset=subset, epsilon=epsilon, beta=beta, seed=seed,
                )
                cell["detector_auc"] = detector_auc(cell["_scores"], clean_scores)
                cell["detector_available"] = has_detector
                if not has_detector:
                    cell["detector_status"] = NOT_MEASURED
                    cell["detector_reason"] = NO_DETECTOR_REASON
                cell.update(detection_at_budgets(cell["_scores"], clean_scores, budgets))
                key = cell_key(kind, entry, epsilon, beta)
                LOGGER.info(
                    "  seed %d subset %d/%d %s [%s] dns=%s -> success %s",
                    seed, position, len(subsets), key.split("_[")[0],
                    "+".join(entry["names"]), entry["dns_compromised"],
                    f"{cell['attack_success']:.4f}",
                )
                cell.update(
                    {
                        "k": k,
                        "dns_compromised": entry["dns_compromised"],
                        "compromised_views": entry["names"],
                        "kind": kind,
                        "epsilon": epsilon,
                        "beta": beta if kind == "chorus" else None,
                    }
                )
                record(per_seed, key, cell, seed)

        # --- TEST D: extra missingness --------------------------------------
        for rate in cfg["missingness"]["rates"]:
            cell = run_cell(
                defended, prepared["test"], prepared["test_index"], cfg,
                kind="missing", seed=seed, missing_rate=rate,
            )
            cell["detector_auc"] = detector_auc(cell["_scores"], clean_scores)
            cell["detector_available"] = has_detector
            if not has_detector:
                cell["detector_status"] = NOT_MEASURED
                cell["detector_reason"] = NO_DETECTOR_REASON
            cell.update(detection_at_budgets(cell["_scores"], clean_scores, budgets))
            cell.update({"kind": "missing", "rate": rate, "dns_compromised": None})
            record(per_seed, f"missing_{int(round(rate * 100))}", cell, seed)

        LOGGER.info(
            "%s / %s seed %d done in %.1fs",
            direction["name"], system["name"], seed, time.time() - started,
        )

    return {
        "direction": direction["name"],
        "system": system["name"],
        "sources": sources,
        "populations": {
            k: v for k, v in populations.items()
            if not isinstance(v, np.ndarray)
        },
        "feature_scale": feature_scale(prepared),
        "pairs": pairs,
        "subset_sampling": {
            "enumerated": "exhaustive",
            "n_subsets": len(subsets),
            "n_with_dns": sum(1 for s in subsets if s["dns_compromised"]),
            "n_without_dns": sum(1 for s in subsets if not s["dns_compromised"]),
            "sampled": False,
            "note": (
                "All subsets of every k were run. No cell was sampled, so no "
                "cell is presented as exhaustive when it is not."
            ),
        },
        "clean": {"per_seed": clean_rows, "summary": summarise_rows(clean_rows, ATTACK_KEYS)},
        "natural_windows": {
            population: {
                "per_seed": rows,
                "n_windows": populations[f"n_{population}"],
                "summary": summarise_rows(rows, NATURAL_KEYS),
                "descriptive_only": population in ("multi_step", "exfiltration_service_only"),
                "caveat": (
                    HANDFUL_CAVEAT
                    if population in ("multi_step", "exfiltration_service_only")
                    else None
                ),
            }
            for population, rows in natural_per_seed.items()
        },
        "natural_dependence": {
            "per_seed": natural_dep_per_seed,
            "summary": summarise_rows(natural_dep_per_seed, ("auc_most_over_least",)),
            "label": NATURAL_DEPENDENCE_LABEL,
        },
        "cells": {
            key: {
                "per_seed": rows,
                "summary": summarise_rows(rows, ATTACK_KEYS),
                "k": rows[0].get("k"),
                "dns_compromised": rows[0].get("dns_compromised"),
                "compromised_views": rows[0].get("compromised_views"),
                "kind": rows[0].get("kind"),
                "epsilon": rows[0].get("epsilon"),
                "beta": rows[0].get("beta"),
            }
            for key, rows in per_seed.items()
        },
    }


def cell_key(kind: str, entry: dict, epsilon: float, beta: float) -> str:
    views = "+".join(entry["names"])
    if kind == "pgd":
        return f"pgd_k{entry['k']}_eps{epsilon}_[{views}]"
    return f"chorus_k{entry['k']}_eps{epsilon}_beta{beta}_[{views}]"


# ---------------------------------------------------------------------------
# Aggregation across subsets, strictly within one dns side
# ---------------------------------------------------------------------------


def aggregate_by_dns_side(system_record: dict, cfg: dict) -> dict:
    """Per (kind, epsilon, beta, k, dns side) means over subsets AND seeds.

    `assert_no_pooling` is called on every group before it is summarised, so a
    bug that mixed the two sides would raise rather than print.
    """
    groups: dict[str, list[dict]] = {}
    for key, cell in system_record["cells"].items():
        if cell.get("kind") not in ("chorus", "pgd"):
            continue
        side = "dns_compromised" if cell["dns_compromised"] else "dns_not_compromised"
        label = (
            f"{cell['kind']}_k{cell['k']}_eps{cell['epsilon']}"
            + (f"_beta{cell['beta']}" if cell["kind"] == "chorus" else "")
            + f"__{side}"
        )
        for row in cell["per_seed"]:
            groups.setdefault(label, []).append(row)

    out = {}
    for label, rows in groups.items():
        assert_no_pooling(rows, f"aggregate_by_dns_side[{label}]")
        out[label] = {
            "dns_compromised": bool(rows[0]["dns_compromised"]),
            "k": rows[0]["k"],
            "kind": rows[0]["kind"],
            "epsilon": rows[0]["epsilon"],
            "beta": rows[0].get("beta"),
            "n_subsets": len({tuple(r["compromised_views"]) for r in rows}),
            "n_rows": len(rows),
            "summary": summarise_rows(rows, ATTACK_KEYS),
            "per_subset": {
                views: summarise_rows(
                    [r for r in rows if "+".join(r["compromised_views"]) == views], ATTACK_KEYS
                )
                for views in sorted({"+".join(r["compromised_views"]) for r in rows})
            },
        }
    return out


def paired_by_cell(naive: dict, prismflow: dict) -> dict:
    """prismflow - naive within each seed, per cell. Never across the dns split."""
    out = {}
    for key, cell in prismflow["cells"].items():
        other = naive["cells"].get(key)
        if other is None:
            continue
        if cell.get("dns_compromised") != other.get("dns_compromised"):
            raise AssertionError(f"{key}: dns side differs between systems")
        treatment = [
            {k: v for k, v in r.items() if isinstance(v, (int, float)) or k == "seed"}
            for r in cell["per_seed"]
        ]
        control = [
            {k: v for k, v in r.items() if isinstance(v, (int, float)) or k == "seed"}
            for r in other["per_seed"]
        ]
        lower = set(LOWER_IS_BETTER) | {"attack_success", "dependence_compromised"}
        out[key] = {
            "dns_compromised": cell.get("dns_compromised"),
            "k": cell.get("k"),
            "kind": cell.get("kind"),
            "epsilon": cell.get("epsilon"),
            "beta": cell.get("beta"),
            "diff": paired_difference(treatment, control, lower),
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def json_safe(value):
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (np.floating,)):
        return json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 29: attack and detection on real AIT data")
    parser.add_argument("--quick", action="store_true",
                        help="2 seeds, k=1 only, one epsilon. SMOKE TEST, NOT EVIDENCE.")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["attack"]["k_grid"] = [1]
        cfg["attack"]["epsilon_grid"] = [1.0]
        cfg["attack"]["pgd_epsilon_grid"] = [1.0]
        cfg["attack"]["beta_grid"] = [1.0]
        cfg["missingness"]["rates"] = [0.3]
        LOGGER.warning("--quick: %d seeds. BELOW the contract's 5-seed minimum; "
                       "every number is a plumbing check and is NOT EVIDENCE.", len(cfg["seeds"]))

    part28 = check_part28(cfg)
    root = Path(cfg["output"]["results_root"])
    root.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": cfg["experiment"],
        "commit": git_commit(),
        "seeds": cfg["seeds"],
        "label_meaning": LABEL_MEANING,
        "not_evidence": len(cfg["seeds"]) < 5,
        "config": json_safe(cfg),
        "sensor_group_check": {
            "requested_group": cfg["natural_dependence"]["same_sensor_group"],
            "sensor_groups": sensor_groups(),
            "n_members": len(sensor_groups().get(cfg["natural_dependence"]["same_sensor_group"], [])),
            "verdict": "NO EVIDENCE",
            "reason": (
                "sensor_groups() returns {'network': [4]} -- one member, suricata -- "
                "so no within-group PAIR exists and the strict same-sensor comparison "
                "has no value to report. There is no packet-capture view yet "
                "(docs/part27-ait-datacard.md section 9, limitation 9). Nothing was "
                "substituted under this name; see natural_dependence for a DIFFERENT "
                "and separately labelled quantity."
            ),
        },
        "directions": {},
    }

    cache: dict = {}
    started = time.time()
    for direction in cfg["directions"]:
        prepared = prepare_direction(direction, cfg, cache)
        record = {
            "label": direction["label"],
            "train_scenario": direction["train"],
            "test_scenario": direction["test"],
            "n_train": int(prepared["train_index"].size),
            "n_test": int(prepared["test_index"].size),
            "label_meaning": LABEL_MEANING,
            "systems": {},
        }
        # Registered BEFORE the system loop so the mid-direction checkpoints
        # below actually contain the completed systems. Registering it after
        # would write an empty `directions` until a whole direction finished,
        # and an interrupted run would look as though it had done nothing.
        results["directions"][direction["name"]] = record
        for system in cfg["systems"]:
            LOGGER.info("=== %s / %s", direction["name"], system["name"])
            system_record = run_system(direction, system, prepared, cfg, part28)
            system_record["by_dns_side"] = aggregate_by_dns_side(system_record, cfg)
            record["systems"][system["name"]] = system_record
            # Checkpoint after every (direction, system).
            (root / "results.json").write_text(json.dumps(json_safe(results), indent=2))
        record["paired_prismflow_minus_naive"] = paired_by_cell(
            record["systems"]["naive"], record["systems"]["prismflow"]
        )
        (root / "results.json").write_text(json.dumps(json_safe(results), indent=2))

    results["runtime_seconds"] = time.time() - started
    (root / "results.json").write_text(json.dumps(json_safe(results), indent=2))
    LOGGER.info("wrote %s in %.1fs", root / "results.json", results["runtime_seconds"])


if __name__ == "__main__":
    main()
