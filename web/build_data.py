"""Copy specific committed result files into web/data/ for the Part 25 site.

    python web/build_data.py

WHY A COPY STEP AND NOT A LIVE READ
------------------------------------
``serve.py`` roots the document tree at ``web/`` so that ``.env`` is not one URL
away from a browser. The site therefore cannot reach ``results/`` at runtime,
which is the point: what the site can read is exactly what this script copied,
and listing ``web/data/`` answers "where does every number on this page come
from" without reading any JavaScript.

WHAT THIS SCRIPT IS NOT ALLOWED TO DO
--------------------------------------
Compute anything. It copies fields and records provenance. Where a committed
file has no value, it emits ``null`` and the page renders "not measured" -- it
never fills a gap, averages across a gap, or carries a value sideways from a
different experiment. The one derived quantity is a sha256 of each source file,
so a stale ``web/data/`` is detectable rather than silent.

THE JOIN THE BRIEF ASKED FOR DOES NOT EXIST IN ANY COMMITTED FILE
------------------------------------------------------------------
Part 25's brief asks for "k copies x rho -> ENIV, coloured by attack success".
The first half is real and is emitted here. The colour channel is not, and the
reason is structural rather than missing data:

  results/clone_eigen/summary.json   rho in {0.0, 0.5}, k in {0..4}
                                     k = DUPLICATE COPIES of view 0
                                     has eniv_mean; has NO attack or success
                                     field of any kind

  results/chorus/attack_metrics.json rho = 0.3, fixed, not a swept axis
                                     k = COLLUDING COMPROMISED views
                                     has success_rate

The two k axes count different things, and chorus's single rho = 0.3 is not one
of clone_eigen's two values, so no committed cell gives an attack success rate
at any (k, rho) on the ENIV grid. Colouring that surface by attack success
would invent a correspondence between two experiments. The surface is therefore
emitted with ``attack_success: null`` on every cell and the page says so; the
chorus attack numbers are emitted separately, on their own real axes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "data"

# Any of these appearing in emitted JSON aborts the build. The result files
# should never contain a credential, and this is the cheap check that keeps it
# true if one ever does.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"\b[0-9a-f]{32}\b"),          # NewsAPI-shaped keys
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    """JSON cannot hold NaN or Infinity; both become null -> "not measured".

    The committed summaries DO contain NaN (``dependence_clone_pairs_mean`` at
    k=0, where there are no clone pairs to average). Letting json.dump emit a
    bare NaN produces a file that ``JSON.parse`` rejects outright, so the whole
    page would fail rather than one cell. null is the honest mapping: the value
    genuinely does not exist.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def provenance(*paths: Path) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [
            {"path": str(p.relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in paths
        ],
    }


# --------------------------------------------------------------------------
# 1. V1 -- the ENIV surface over (k copies, rho)
# --------------------------------------------------------------------------

def build_v1_surface() -> dict:
    src = ROOT / "results/clone_eigen/summary.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    cfg = raw.get("config", {})

    cells = []
    for row in raw["summary"]:
        cells.append({
            "rho": clean(row.get("rho")),
            "k": clean(row.get("k")),
            "n_views": clean(row.get("n_views")),
            "system": row.get("system"),
            "n_seeds": clean(row.get("n_seeds")),
            "eniv_mean": clean(row.get("eniv_mean")),
            "eniv_std": clean(row.get("eniv_std")),
            "confidence_mean": clean(row.get("mean_confidence_mean")),
            "confidence_std": clean(row.get("mean_confidence_std")),
            "accuracy_mean": clean(row.get("accuracy_mean")),
            # Emitted explicitly as null rather than omitted, so the page can
            # distinguish "this experiment has no attack" from "the build
            # forgot a field". See the module docstring.
            "attack_success": None,
        })

    return {
        "title": "ENIV under view duplication",
        "axes": {
            "k": {"label": "duplicate copies of view 0",
                  "values": sorted({c["k"] for c in cells if c["k"] is not None})},
            "rho": {"label": "generator rho (set, not estimated)",
                    "values": sorted({c["rho"] for c in cells if c["rho"] is not None})},
        },
        "seeds": cfg.get("seeds"),
        "base_views": cfg.get("base_views"),
        "attack_success_available": False,
        "attack_success_note": (
            "Not measured on this grid. results/clone_eigen/ contains no attack "
            "or success field: it is the duplication experiment, where k counts "
            "COPIES of a view. The attack numbers live in results/chorus/, where "
            "k counts COLLUDING COMPROMISED views and rho is fixed at 0.3 -- "
            "which is not one of this grid's rho values. No committed file gives "
            "an attack success rate at any (k, rho) cell here, so none is shown."
        ),
        "cells": cells,
        "provenance": provenance(src),
    }


# --------------------------------------------------------------------------
# 2. V1 -- attack success, on its own real axes
# --------------------------------------------------------------------------

def build_v1_attack() -> dict:
    src = ROOT / "results/chorus/attack_metrics.json"
    meta_src = ROOT / "results/chorus/metadata.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    meta = json.loads(meta_src.read_text(encoding="utf-8"))

    rows = []
    for key, fields in raw.get("summary", {}).items():
        condition, _, system = key.partition("|")
        sr = fields.get("success_rate") or {}
        rows.append({
            "condition": condition,
            "system": system,
            "success_mean": clean(sr.get("mean")),
            "success_std": clean(sr.get("std")),
            "n_seeds": clean(sr.get("n_seeds")),
            "eniv_measured": clean((fields.get("eniv_measured") or {}).get("mean")),
            "dependence_compromised": clean(
                (fields.get("dependence_compromised") or {}).get("mean")),
        })

    rho = None
    found = set(re.findall(r'"rho":\s*([0-9.]+)', json.dumps(meta)))
    if len(found) == 1:
        rho = float(found.pop())

    return {
        "title": "Attack success rate (chorus)",
        "rho_fixed": rho,
        "rho_note": "rho is a single fixed value in this experiment, not an axis.",
        "rows": rows,
        "provenance": provenance(src, meta_src),
    }


# --------------------------------------------------------------------------
# 2b. V1 -- the conditions where PrismFlow was fooled MORE than naive
# --------------------------------------------------------------------------

def build_v1_failures() -> dict:
    """Every non-clean attack condition where PrismFlow lost to the baseline.

    SELECTION IS MECHANICAL, NOT CHOSEN. A condition is included when
    prismflow's mean success rate exceeds naive's, over the same seeds. Nothing
    is picked for being dramatic and nothing is dropped for being small: the
    near-zero cells are the ones that show the effect is not uniform, so
    omitting them would flatter the result in the same way that showing only
    eps 1.0 does.

    THE PER-SEED SIGN IS CARRIED, NOT JUST THE MEAN. A mean difference over 5
    seeds can be positive while two seeds went the other way, and a reader who
    sees only the mean cannot tell a consistent effect from a coin flip. Each
    condition therefore reports how many of its seeds individually went the
    wrong way, and the per-seed differences themselves.

    NO FOOLED EXAMPLE IS AVAILABLE. The committed run records aggregates,
    per-seed rows and a perturbation tensor -- results/chorus/perturbations/
    holds `delta` of shape (300, 4, 16) and the compromised view indices, and
    nothing anywhere records which sample was fooled, what it was predicted as,
    or what its true label was. Every example field is therefore emitted as
    null and the page prints "not recorded". Reconstructing one would mean
    re-running the attack, which is a new experiment and not a page build.
    """
    src = ROOT / "results/chorus/attack_metrics.json"
    meta_src = ROOT / "results/chorus/metadata.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    meta = json.loads(meta_src.read_text(encoding="utf-8"))

    summary = raw.get("summary", {})
    per_seed = raw.get("per_seed", {})
    cells = {c["name"]: c for c in
             (((meta.get("config") or {}).get("cells")) or []) if "name" in c}

    def stat(cond, system, field):
        f = (summary.get("%s|%s" % (cond, system)) or {}).get(field) or {}
        return {"mean": clean(f.get("mean")), "std": clean(f.get("std")),
                "n_seeds": clean(f.get("n_seeds"))}

    conditions = sorted({k.split("|")[0] for k in summary})
    included, excluded = [], []

    for cond in conditions:
        naive = stat(cond, "naive", "success_rate")
        pf = stat(cond, "prismflow", "success_rate")
        nodisc = stat(cond, "prismflow_nodiscount", "success_rate")

        if naive["mean"] is None or pf["mean"] is None:
            excluded.append({"condition": cond,
                             "why": "a success rate is missing for one system"})
            continue
        if pf["mean"] <= naive["mean"]:
            excluded.append({
                "condition": cond,
                "why": ("PrismFlow was not fooled more than naive here "
                        "(%.4f against %.4f)" % (pf["mean"], naive["mean"])),
            })
            continue

        # Paired per seed, by seed number, so a missing seed drops the pair
        # rather than shifting one series against the other.
        n_by_seed = {r["seed"]: r.get("success_rate")
                     for r in per_seed.get("%s|naive" % cond, [])}
        p_by_seed = {r["seed"]: r.get("success_rate")
                     for r in per_seed.get("%s|prismflow" % cond, [])}
        seeds = sorted(s for s in set(n_by_seed) & set(p_by_seed)
                       if n_by_seed[s] is not None and p_by_seed[s] is not None)
        deltas = [{"seed": s, "naive": n_by_seed[s], "prismflow": p_by_seed[s],
                   "delta": p_by_seed[s] - n_by_seed[s]} for s in seeds]
        worse = sum(1 for d in deltas if d["delta"] > 0)

        cell = cells.get(cond, {})
        included.append({
            "condition": cond,
            "attack": cell.get("attack"),
            "k": clean(cell.get("k")),
            "epsilon": clean(cell.get("epsilon")),
            "beta": clean(cell.get("beta")),
            "naive": naive,
            "prismflow": pf,
            "prismflow_nodiscount": nodisc,
            "delta_mean": pf["mean"] - naive["mean"],
            "n_seeds": len(deltas),
            "seeds_prismflow_worse": worse,
            "per_seed": deltas,
            # What the attack achieved where it succeeded, and what the
            # pipeline believed while it did.
            "target_prob_on_success": stat(cond, "prismflow", "target_prob_on_success"),
            "target_belief_on_success": stat(cond, "prismflow", "target_belief_on_success"),
            "vacuity": stat(cond, "prismflow", "vacuity"),
            "eniv_measured": stat(cond, "prismflow", "eniv_measured"),
            "dependence_compromised": stat(cond, "prismflow", "dependence_compromised"),
            "dependence_all_pairs": stat(cond, "prismflow", "dependence_all_pairs"),
            # No committed file records a fooled sample. See the docstring.
            "example": None,
            "example_note": "not recorded",
        })

    included.sort(key=lambda r: -r["delta_mean"])

    cfg = meta.get("config") or {}
    return {
        "title": "Where PrismFlow lost to the baseline",
        "criterion": ("every non-clean condition in which PrismFlow's mean "
                      "attack success rate is HIGHER than naive's, over the "
                      "same seeds. Nothing is selected for size."),
        "n_conditions_total": len(conditions),
        "n_included": len(included),
        "excluded": excluded,
        "seeds": (cfg.get("seeds") or None),
        "training": (cfg.get("training") or None),
        "attack_config": (cfg.get("attack") or None),
        "rho_fixed": ((cfg.get("data") or {}).get("rho")),
        "example_availability": (
            "No committed file records an individual fooled sample. "
            "results/chorus/perturbations/ holds the perturbation tensor and "
            "the compromised view indices; no file anywhere holds which sample "
            "was fooled, its prediction or its true label. Every example is "
            "shown as \"not recorded\" rather than reconstructed."),
        "label": ("5 seeds per cell, mean and standard deviation, from the "
                  "committed Part 09 run. Higher success rate is worse."),
        "conditions": included,
        "provenance": provenance(src, meta_src),
    }


# --------------------------------------------------------------------------
# 2c. V1 -- calibration under view duplication
# --------------------------------------------------------------------------

CALIBRATION_K = (0, 2, 4)
CALIBRATION_SYSTEMS = ("naive", "prismflow", "naive_weights_discounted")
CALIBRATION_METRICS = (
    "accuracy", "prob_mean_confidence", "prob_ece", "prob_mce", "brier",
    "brier_reliability", "brier_resolution", "prob_aurc", "eniv",
    "efficiency_ratio", "mean_dependence",
)


def build_calibration() -> dict:
    """Calibration under view duplication -- results/calibration_duplicated/.

    Same shape as build_v1_surface(): a flat grid of cells, one per (k,
    system), each carrying mean/std/n_seeds per metric straight out of the
    committed metrics.json. naive's eniv/efficiency_ratio/mean_dependence are
    genuinely absent (ENIV is not defined for the undiscounted baseline) --
    metrics.json itself records them as {"mean": null, "n_seeds": 0}, so they
    are copied through as null and the page renders "not measured", not 0.
    """
    base = ROOT / "results/calibration_duplicated"
    sources = [base / "config_used.json"]
    cells = []
    for k in CALIBRATION_K:
        for system in CALIBRATION_SYSTEMS:
            src = base / ("k%d_%s" % (k, system)) / "metrics.json"
            if not src.exists():
                continue
            sources.append(src)
            raw = json.loads(src.read_text(encoding="utf-8"))
            summary = raw.get("summary", {})
            cell = {
                "k": k, "system": system,
                "n_views": 4 + k,
                "n_seeds": clean((summary.get("accuracy") or {}).get("n_seeds")),
            }
            for metric in CALIBRATION_METRICS:
                m = summary.get(metric) or {}
                cell[metric + "_mean"] = clean(m.get("mean"))
                cell[metric + "_std"] = clean(m.get("std"))
            cells.append(cell)

    return {
        "title": "Calibration under view duplication",
        "axes": {
            "k": {"label": "duplicate copies of view 0", "values": list(CALIBRATION_K)},
            "system": {"label": "system", "values": list(CALIBRATION_SYSTEMS)},
        },
        "metrics": list(CALIBRATION_METRICS),
        "cells": cells,
        "label": ("5 seeds per cell, mean and standard deviation, from the "
                  "committed calibration_duplicated run. naive has no ENIV, "
                  "efficiency_ratio or mean_dependence by construction -- "
                  "those cells are \"not measured\", not zero."),
        "provenance": provenance(*sources),
    }


# --------------------------------------------------------------------------
# 3. V2 -- per-row angle coverage over the 48 rows
# --------------------------------------------------------------------------

ANGLES = ("tech", "market", "financial", "regulatory", "sentiment")


def build_v2_coverage() -> dict:
    # CORRECTED 2026-10-09. The grid used to be built from
    # part24_pipeline_verification.json, the run that PREDATES the V2-L2 and
    # V2-L3 connector fixes. That run fused no rows, so the page said "no row
    # reached two angles" -- which stopped being true once the fixes were
    # re-run over all 48 rows on 2026-10-01. The grid is now the postfix run;
    # the pre-fix run stays in the file as a labelled comparison, because the
    # zeros it recorded were real and it is still the source for V2-L2's
    # original evidence.
    ds_src = ROOT / "data/v2/evaluation_queries.json"
    run_src = ROOT / "results/v2/part24_pipeline_verification_postfix.json"
    prefix_src = ROOT / "results/v2/part24_pipeline_verification.json"
    pass1_src = ROOT / "results/v2/part24_pipeline_verification_pass1.json"

    dataset = {q["id"]: q for q in json.loads(ds_src.read_text(encoding="utf-8"))}
    run = json.loads(run_src.read_text(encoding="utf-8"))

    rows = []
    for rec in run["rows"]:
        meta = dataset.get(rec["id"], {})
        angles = {}
        for name in ANGLES:
            entry = rec["angles"].get(name, {})
            records = entry.get("records")
            claims = entry.get("claims")
            angles[name] = {
                # A retrieval exception leaves a string here, not a count. It is
                # not zero, and must not be shown as zero.
                "records": records if isinstance(records, int) else None,
                "claims": claims if isinstance(claims, int) else None,
                "reasoner_error": entry.get("reasoner_error"),
                "retrieval_error": entry.get("retrieval"),
                "provenance": entry.get("provenance") or {},
                # The row-detail view needs the per-angle fields the grid has no
                # room for. Only two numbers exist beyond the counts, and only on
                # the angles that reached a reasoner:
                #
                #   mean_claim_conf  a real value on 20 of 240 angle cells --
                #                    the key exists on 55, but is null wherever
                #                    the reasoner produced no claims, so the
                #                    detail view shows "not measured" on 220
                #   tokens           the evidence bundle handed to the reasoner
                #
                # CLAIM TEXT DOES NOT EXIST IN ANY COMMITTED FILE. The run wrote
                # counts and a mean confidence, not the claims themselves, so the
                # detail view says so rather than leaving a blank that looks like
                # a value that failed to load. No field is invented here.
                "mean_claim_conf": clean(entry.get("mean_claim_conf")),
                "tokens": clean(entry.get("tokens")),
                "note": entry.get("note"),
                "warnings": entry.get("warnings") or [],
            }
        rows.append({
            "id": rec["id"],
            "domain": meta.get("domain"),
            "outcome": meta.get("actual_outcome"),
            "conflict_expected": meta.get("conflict_expected"),
            # Dataset side of the row, for the detail view. Copied verbatim from
            # evaluation_queries.json -- the same file this build already reads
            # for domain and outcome, so the detail view adds no source. The
            # pitch is the anonymised text the pipeline was actually given; the
            # ground-truth source is the URL the label was taken from, and it is
            # rendered as a real link so a reader can check the label.
            "idea_pitch": meta.get("idea_pitch"),
            "outcome_date": meta.get("outcome_date"),
            "ground_truth_source": meta.get("ground_truth_source"),
            "dataset_notes": meta.get("notes"),
            "angles": angles,
            # Why this row did or did not fuse, in the run's own words.
            "run_note": rec.get("note"),
            "fused": rec.get("fused"),
            "seconds": clean(rec.get("seconds")),
            "angles_with_claims": rec.get("angles_with_claims"),
            "angles_with_reasoner_error": rec.get("angles_with_reasoner_error"),
            "zero_claim_cause": rec.get("zero_claim_cause"),
            # Present only on rows that actually fused. Absent stays null and the
            # page prints "not measured" -- it never shows 0 for "did not fuse".
            "eniv": rec.get("eniv"),
            "confidence": rec.get("confidence"),
            "discount": rec.get("discount"),
            "contradictions": rec.get("contradictions"),
        })

    sources = [ds_src, run_src]

    # Rows that reached >=2 claim-producing angles, read off the run rather than
    # restated. ENIV exists on exactly these rows and nowhere else.
    fused = [{"id": r["id"], "eniv": r.get("eniv"), "n_angles": r.get("n_angles"),
              "discount": clean(r.get("discount")),
              "confidence": clean(r.get("confidence"))}
             for r in run["rows"] if (r.get("angles_with_claims") or 0) >= 2]

    prefix_note = None
    if prefix_src.exists():
        sources.append(prefix_src)
        pre = json.loads(prefix_src.read_text(encoding="utf-8"))
        fused_pre = [r["id"] for r in pre["rows"]
                     if (r.get("angles_with_claims") or 0) >= 2]
        prefix_note = {
            "rows": pre.get("n_rows"),
            "fused_rows": fused_pre,
            "note": ("SUPERSEDED, kept as a record. The grid above is the "
                     "post-fix re-run of 2026-10-01. The earlier run, before "
                     "the GitHub and arXiv query fixes, fused %d of %d rows: "
                     "GitHub returned 0 records on every row and arXiv "
                     "answered unrelated queries with the same ten records "
                     "(V2-L2, V2-L3)."
                     % (len(fused_pre), pre.get("n_rows") or 0)),
        }

    pass1_note = None
    if pass1_src.exists():
        sources.append(pass1_src)
        p1 = json.loads(pass1_src.read_text(encoding="utf-8"))
        fused1 = [r["id"] for r in p1["rows"] if (r.get("angles_with_claims") or 0) >= 2]
        pass1_note = {
            "rows": p1.get("n_rows"),
            "fused_rows": fused1,
            "note": ("Also pre-fix: the first of that run's two passes. "
                     "Retrieval was identical across both, but CLAIM counts "
                     "were not -- the second pass ran into an exhausted daily "
                     "Groq quota. The first pass fused %d row(s)." % len(fused1)),
        }

    # The one sentence that was wrong on the page, now derived from the run it
    # describes instead of written by hand.
    fused_note = (
        "%d of %d rows reached two claim-producing angles, so ENIV exists on "
        "%s and on no other row: %s. Below two angles the dependence matrix is "
        "1x1 and ENIV is degenerate, so an empty ENIV cell means the row never "
        "fused -- it is not a zero. ONE PASS, no seeds: not a Part 24 finding "
        "under CONTRACT.md section 5, and 2 of 48 is not a distribution to "
        "calibrate against (V2-L6)."
        % (len(fused), run.get("n_rows") or 0,
           "that row" if len(fused) == 1 else "those rows",
           ", ".join("row %s at ENIV %s" % (f["id"], f["eniv"]) for f in fused)
           or "none")
    )

    return {
        "title": "Per-row angle coverage",
        "angle_order": list(ANGLES),
        "angle_notes": {
            "regulatory": "No connector has been built. Retrieves nothing on every row.",
            "sentiment": "StackExchange only. Reddit is excluded: an IP-scoped 403.",
            "financial": "yfinance resolves tickers; the pitches are anonymised.",
        },
        "n_rows": run.get("n_rows"),
        "label": run.get("label"),
        "run_source": run_src.relative_to(ROOT).as_posix(),
        "zero_claim_rows_provider_error": run.get("zero_claim_rows_provider_error"),
        "zero_claim_rows_genuine": run.get("zero_claim_rows_genuine"),
        "fused_rows": fused,
        "fused_note": fused_note,
        "prefix": prefix_note,
        "pass1": pass1_note,
        "rows": rows,
        "provenance": provenance(*sources),
    }


# --------------------------------------------------------------------------
# 4. Committed FIGURES, copied as files
# --------------------------------------------------------------------------
#
# serve.py roots the document tree at web/, so the site cannot reach results/.
# These are therefore copied, not linked, and each carries the sha256 of the
# file it came from so a stale copy is detectable. Nothing is re-plotted: the
# PNG on the page is byte-identical to the PNG the experiment wrote.
#
# The selection is deliberately small. results/ holds 141 reliability diagrams
# and 26 other figures; putting all 167 on a page would be a file listing, not
# an argument. One figure per claim, and the claim is named.

FIGURES = [
    ("v1_eniv_vs_duplicates.png", "results/clone_eigen/eniv_vs_duplicates.png",
     "ENIV against duplicate copies of one view",
     "Duplicating a view adds no information, so a correct effective-view count "
     "should barely move. It rises. This is V1-L1 and V1-L2 as a picture.",
     "results/clone_eigen/summary.json"),
    ("v1_confidence_vs_duplicates.png", "results/clone_eigen/confidence_vs_duplicates.png",
     "Confidence against duplicate copies",
     "The same duplication, seen through confidence rather than ENIV.",
     "results/clone_eigen/summary.json"),
    ("v1_estimator_vs_truth.png", "results/eniv_validation/estimator_vs_truth.png",
     "ENIV estimator against known truth",
     "The only place ENIV is VALIDATED rather than applied: synthetic data "
     "where rho is a parameter we set.",
     "results/eniv_validation/summary.json"),
    ("v1_reliability_prismflow.png", "results/calibration/prismflow/reliability_diagram.png",
     "Reliability diagram -- PrismFlow",
     "Calibration on synthetic data, 5 seeds.",
     "results/calibration/prismflow/metrics.json"),
    ("v1_reliability_naive.png", "results/calibration/naive/reliability_diagram.png",
     "Reliability diagram -- naive baseline",
     "The same axes for the undiscounted baseline, so the pair can be read "
     "against each other.",
     "results/calibration/naive/metrics.json"),
    ("v1_reliability_chorus_prismflow.png",
     "results/chorus/chorus_k2_eps1.0_prismflow/reliability_diagram.png",
     "Reliability under the Chorus attack -- PrismFlow, k=2, eps 1.0",
     "The condition where PrismFlow is fooled MORE often than naive. See the "
     "failures section.",
     "results/chorus/chorus_k2_eps1.0_prismflow/metrics.json"),
    ("v1_reliability_chorus_naive.png",
     "results/chorus/chorus_k2_eps1.0_naive/reliability_diagram.png",
     "Reliability under the Chorus attack -- naive, k=2, eps 1.0",
     "The baseline for the same condition.",
     "results/chorus/chorus_k2_eps1.0_naive/metrics.json"),
    ("v1_missing_overview.png", "results/robustness/missing_overview.png",
     "Robustness to missing views",
     "Behaviour as views are dropped, across rates.",
     "results/robustness/missing_r0.30_prismflow/metrics.json"),
    ("v1_tail_vs_correlation.png", "results/tail/tail_vs_correlation.png",
     "Tail dependence against correlation",
     "Why linear correlation is not the quantity the discount needs.",
     "results/tail/tail.json"),
    ("v1_audit_gap.png", "results/per_sample/audit_gap.png",
     "Trained against audited dependence -- the gap",
     "The per-sample audit: what the model trained on versus what an "
     "independent audit measures.",
     "results/per_sample/per_sample.json"),
]


def build_figures() -> dict:
    """Copy the selected PNGs into web/data/figures/ and describe each one."""
    dest_dir = OUT / "figures"
    dest_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for name, rel, title, caption, numbers_from in FIGURES:
        src = ROOT / rel
        if not src.exists():
            # A missing figure is recorded as missing. The page renders "not
            # measured" rather than a broken image, and never silently drops it.
            entries.append({
                "file": None, "title": title, "caption": caption,
                "source": rel, "sha256": None, "bytes": None,
                "numbers_from": numbers_from,
                "missing": "not present in results/ at build time",
            })
            print("    MISSING  %s" % rel)
            continue
        shutil.copyfile(src, dest_dir / name)
        entries.append({
            "file": "figures/%s" % name,
            "title": title,
            "caption": caption,
            "source": rel,
            "sha256": sha256(src),
            "bytes": src.stat().st_size,
            "numbers_from": numbers_from,
            "missing": None,
        })

    return {
        "title": "Committed V1 figures",
        "note": ("copied byte-for-byte from results/. The site cannot read "
                 "results/ at runtime, so these are files in web/data/figures/ "
                 "and each carries the sha256 of its source."),
        "figures": entries,
        "not_shown": ("results/ holds %d reliability diagrams and %d other "
                      "figures. One figure per claim is shown, not all of them."
                      % (len(list((ROOT / "results").rglob("reliability_diagram.png"))),
                         len([p for p in (ROOT / "results").rglob("*.png")
                              if p.name != "reliability_diagram.png"]))),
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": [
                {"path": e["source"], "sha256": e["sha256"], "bytes": e["bytes"]}
                for e in entries if e["sha256"]
            ],
        },
    }


# --------------------------------------------------------------------------
# 4b. The limitations, read from the two docs rather than retyped
# --------------------------------------------------------------------------

# `## L7. Tail dependence is not the sharper instrument ...`
LIMIT_HEADING = re.compile(r"^##\s+((?:V2-)?L\d+)\.\s+(.*?)\s*$")
# A title may open with a status marker the docs add when an entry changes:
#   "RESOLVED 2026-09-28. GitHub returned 0 records because ..."
#   "CLOSED 2026-10-09, blocked on funding. Part 20's Claude and gpt-4o ..."
#   "NEW, created by the V2-L2 fix: some GitHub records exceed ..."
#
# RESOLVED and CLOSED are deliberately NOT the same marker and are never
# merged. RESOLVED means the defect was fixed. CLOSED means the entry reached a
# terminal state WITHOUT a fix -- blocked on funding, or not measured and
# premature at current coverage -- and collapsing the two would read as
# progress that did not happen. Both are nevertheless not-open, which is the
# only thing the counts need from them.
STATUS_MARKER = re.compile(r"^(RESOLVED\s+[\d-]+|CLOSED\s+[\d-]+|NEW)\b[.,:]?\s*(.*)$")


def read_limitations(path: Path, group_at: int | None = None,
                     real_data_at: int | None = None) -> list:
    """Every `## Lx. title` heading in a limitations doc, verbatim.

    THE HEADINGS ARE READ, NOT RETYPED, and that is the whole point of this
    step. The page carried V2-L2 and V2-L3 as open for the entire session in
    which they were fixed, because two hand-written copies of the list in
    index.html had no connection to the document that records their state, and
    V2-L7 did not appear at all. A list with no link to its source drifts
    silently; one that is read cannot.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LIMIT_HEADING.match(line)
        if not m:
            continue
        ident, title = m.group(1), m.group(2)
        status, sm = None, STATUS_MARKER.match(title)
        if sm:
            status = sm.group(1)
            rest = sm.group(2) or ""
            # The marker moves into a field of its own so the page can style it,
            # but ONLY when what is left still reads as a title. V2-L7's heading
            # is "NEW, created by the V2-L2 fix: ...", and lifting the marker out
            # of that leaves a sentence starting mid-clause, so it keeps its
            # whole heading. Nothing is ever reworded either way.
            if rest[:1].isupper():
                title = rest
        n = int(ident.split("L")[-1])
        out.append({
            "id": ident,
            "n": n,
            "title": title,
            "status": status,
            # Three groups, checked most-specific first. L16-L30 came out of
            # Parts 28-30 on real datasets; they are NOT standing limitations
            # of the project as a whole, and calling them that would widen
            # fifteen dataset-specific findings into claims about everything.
            # L1-L15 keep exactly the grouping they had.
            "group": ("real_data"
                      if real_data_at is not None and n >= real_data_at
                      else "standing"
                      if group_at is not None and n >= group_at
                      else "per_part"),
        })
    return out


def build_limitations() -> dict:
    """V1's L1-L15 and V2's V2-L1 onwards, from the documents that own them."""
    v1_src = ROOT / "docs/KNOWN_LIMITATIONS.md"
    v2_src = ROOT / "docs/v2-known-limitations.md"

    v1 = read_limitations(v1_src, group_at=9, real_data_at=16)
    v2 = read_limitations(v2_src)

    def counts(rows):
        def marked(r, word):
            return (r["status"] or "").startswith(word)
        return {
            "total": len(rows),
            "resolved": sum(1 for r in rows if marked(r, "RESOLVED")),
            # Closed without a fix. Counted apart from "resolved" so the page
            # cannot present a funding blocker as a repaired defect.
            "closed": sum(1 for r in rows if marked(r, "CLOSED")),
            "new": sum(1 for r in rows if r["status"] == "NEW"),
            "open": sum(1 for r in rows
                        if not marked(r, "RESOLVED") and not marked(r, "CLOSED")),
        }

    return {
        "title": "Known limitations",
        "note": ("Headings are read from the documents at build time, never "
                 "retyped, so an entry that is resolved there cannot stay open "
                 "here. Resolved means fixed; closed means the entry reached a "
                 "terminal state without a fix, and the two are counted "
                 "separately."),
        "v1": {
            "source": str(v1_src.relative_to(ROOT)).replace("\\", "/"),
            "entries": v1,
            "counts": counts(v1),
            "group_labels": {
                "per_part": "discovered as each Part ran",
                "standing": "standing limitations of the project as a whole",
                "real_data": "real-data findings (Parts 28-30)",
            },
        },
        "v2": {
            "source": str(v2_src.relative_to(ROOT)).replace("\\", "/"),
            "entries": v2,
            "counts": counts(v2),
        },
        "provenance": provenance(v1_src, v2_src),
    }


# --------------------------------------------------------------------------
# 5. The evaluation dataset itself, all 48 rows
# --------------------------------------------------------------------------

# The commit that brought the dataset to 48 rows and left 040 out. The page has
# to say WHY a row is missing, and the only honest text for that is the repo's
# own, so it is read from the commit rather than retyped here.
ROW_040_COMMIT = "81c9550"


def commit_paragraph(commit: str, heading: str) -> dict:
    """One section of a commit message, quoted verbatim.

    Verbatim matters: this is the page's justification for a gap in a benchmark.
    Paraphrasing it here would put a second, drifting version of the reason in
    the repository. If the commit or the heading cannot be found the build
    fails, because a footer that silently loses its citation is worse than no
    footer.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%B", commit],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:      # pragma: no cover
        raise SystemExit("cannot read commit %s: %s" % (commit, exc))
    if out.returncode != 0:
        raise SystemExit("cannot read commit %s: %s" % (commit, out.stderr.strip()))

    lines = out.stdout.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    except StopIteration:
        raise SystemExit("commit %s has no section %r" % (commit, heading))

    body, i = [], start + 1
    if i < len(lines) and set(lines[i].strip()) <= {"-"} and lines[i].strip():
        i += 1                                   # the ---- underline
    while i < len(lines):
        ln = lines[i]
        # A section ends at the next ALL-CAPS heading, which is this message's
        # own convention, not at a blank line -- the section has blank lines in it.
        if ln.strip() and ln == ln.upper() and ln[:1].isalpha() and len(ln.strip()) > 8:
            break
        body.append(ln)
        i += 1

    full = subprocess.run(["git", "rev-parse", commit], cwd=str(ROOT),
                          capture_output=True, text=True).stdout.strip()
    return {
        "commit": commit,
        "commit_full": full or None,
        "subject": lines[0] if lines else None,
        "heading": heading,
        "text": "\n".join(body).strip("\n"),
    }


def build_v2_dataset() -> dict:
    """All 48 evaluation rows, as the exporter wrote them.

    The source is data/v2/evaluation_queries.json -- the derived file the
    harness actually loads -- not the workbook, so the table shows exactly the
    rows the evaluation would run on. The workbook is offered separately as a
    download, and the two can be compared because both carry a sha256.

    Nothing here is computed except counts over the rows that are present, and
    the id gaps, which are found by comparing the ids to the range they sit in
    rather than being asserted.
    """
    src = ROOT / "data/v2/evaluation_queries.json"
    rows_raw = json.loads(src.read_text(encoding="utf-8"))

    rows = [{
        "id": r.get("id"),
        "domain": clean(r.get("domain")),
        "outcome": clean(r.get("actual_outcome")),
        "outcome_date": clean(r.get("outcome_date")),
        "conflict_expected": clean(r.get("conflict_expected")),
        "ground_truth_source": clean(r.get("ground_truth_source")),
        "idea_pitch": clean(r.get("idea_pitch")),
        "notes": clean(r.get("notes")),
    } for r in rows_raw]

    present = {r["id"] for r in rows}
    numeric = sorted(int(i) for i in present if str(i).isdigit())
    gaps = [] if not numeric else [
        "%03d" % n for n in range(min(numeric), max(numeric) + 1)
        if "%03d" % n not in present
    ]

    def tally(field):
        counts = {}
        for r in rows:
            counts[r[field] or "not recorded"] = counts.get(r[field] or "not recorded", 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # The two denominators are different and are not interchangeable. Both are
    # counted here rather than quoted, so they cannot drift from the file.
    conflict_n = sum(1 for r in rows if (r["conflict_expected"] or "") in ("Yes", "No"))

    return {
        "title": "The Part 24 evaluation dataset",
        "n_rows": len(rows),
        "columns": ["id", "domain", "outcome", "outcome_date",
                    "conflict_expected", "ground_truth_source"],
        "domains": tally("domain"),
        "outcomes": tally("outcome"),
        "conflict": tally("conflict_expected"),
        "denominators": {
            "calibration_n": len(rows),
            "conflict_n": conflict_n,
            "note": ("the two denominators differ and are not interchangeable: "
                     "conflict excludes the rows labelled Unsure. Every figure "
                     "computed from this dataset carries its own n."),
        },
        "id_gaps": gaps,
        "gap_note": ("id 001 does not exist: the EX example row occupies that "
                     "slot in the sheet. 040 is blank on purpose -- see below."),
        "row_040": commit_paragraph(ROW_040_COMMIT, "ROW 040 IS BLANK ON PURPOSE"),
        "rows": rows,
        "label": ("This is the dataset, not a result. No prediction, score or "
                  "calibration metric appears in this table, because none has "
                  "been computed on these rows -- see V2-L6."),
        "provenance": provenance(src),
    }


# --------------------------------------------------------------------------
# 6. The labelled workbook, offered as a download
# --------------------------------------------------------------------------

def build_dataset_download() -> dict:
    """Copy the committed workbook into web/data/ so the page can offer it."""
    src = ROOT / "data/v2/part24_labeled_dataset.xlsx"
    if not src.exists():
        return {"file": None, "missing": "workbook not present at build time"}
    shutil.copyfile(src, OUT / "part24_labeled_dataset.xlsx")
    return {
        "file": "part24_labeled_dataset.xlsx",
        "title": "Part 24 labelled dataset (workbook)",
        "source": "data/v2/part24_labeled_dataset.xlsx",
        "sha256": sha256(src),
        "bytes": src.stat().st_size,
        "note": ("the sheet the exporter reads. Row 040 is present in the sheet "
                 "with every field empty, and id 001 does not exist because the "
                 "EX example row occupies that slot."),
        "missing": None,
    }


# --------------------------------------------------------------------------
# 7. V2-L2 / V2-L3: the N=6 re-verification, before against after
# --------------------------------------------------------------------------

REVERIFY_ANGLES = ("tech", "market", "financial", "regulatory", "sentiment")


def build_v2_reverify() -> dict:
    """Per-angle retrieval before and after the connector fix, on the same 6 rows.

    BEFORE is the committed 48-row run, restricted to those six rows -- it IS
    the pre-fix code on those rows. AFTER is the N=6 re-verification. Both are
    committed files; nothing here is recomputed, and the two are never averaged
    together.
    """
    before_src = ROOT / "results/v2/part24_pipeline_verification.json"
    after_src = ROOT / "results/v2/part24_reverify_n6.json"
    if not after_src.exists():
        return None

    before_raw = json.loads(before_src.read_text(encoding="utf-8"))
    after_raw = json.loads(after_src.read_text(encoding="utf-8"))
    before = {r["id"]: r for r in before_raw["rows"]}
    after = {r["id"]: r for r in after_raw["rows"]}
    selection = after_raw.get("selection") or {}
    ids = selection.get("rows") or [r["id"] for r in after_raw["rows"]]

    rows = []
    for rid in ids:
        b, a = before.get(rid, {}), after.get(rid, {})

        def angle(rec, name, field):
            entry = (rec.get("angles") or {}).get(name) or {}
            value = entry.get(field)
            return value if isinstance(value, int) else None

        rows.append({
            "id": rid,
            "domain": (selection.get("by_domain") and next(
                (d for d, v in selection["by_domain"].items() if rid in v), None)),
            "records_before": {n: angle(b, n, "records") for n in REVERIFY_ANGLES},
            "records_after": {n: angle(a, n, "records") for n in REVERIFY_ANGLES},
            "github_before": ((b.get("angles") or {}).get("tech") or {}).get("provenance", {}).get("github"),
            "github_after": ((a.get("angles") or {}).get("tech") or {}).get("provenance", {}).get("github"),
            "arxiv_before": ((b.get("angles") or {}).get("tech") or {}).get("provenance", {}).get("arxiv"),
            "arxiv_after": ((a.get("angles") or {}).get("tech") or {}).get("provenance", {}).get("arxiv"),
            "tech_tokens_before": clean(((b.get("angles") or {}).get("tech") or {}).get("tokens")),
            "tech_tokens_after": clean(((a.get("angles") or {}).get("tech") or {}).get("tokens")),
            "angles_with_claims_before": b.get("angles_with_claims"),
            "angles_with_claims_after": a.get("angles_with_claims"),
            "eniv_before": clean(b.get("eniv")),
            "eniv_after": clean(a.get("eniv")),
            "confidence_after": clean(a.get("confidence")),
            "note_after": a.get("note"),
        })

    tb = [r["tech_tokens_before"] for r in rows]
    ta = [r["tech_tokens_after"] for r in rows]
    return {
        "title": "V2-L2 and V2-L3: retrieval before and after the connector fix",
        "label": ("ONE PASS, N=6, GROQ, LEXICAL FUSION, NOT A PART 24 FINDING. "
                  "Six rows of the 48 committed, two per domain. No seeds, no "
                  "mean, no std -- see docs/CONTRACT.md section 5."),
        "is_part24_evidence": False,
        "angle_order": list(REVERIFY_ANGLES),
        "rows": rows,
        "summary": {
            "fusible_before": sum(1 for r in rows if (r["angles_with_claims_before"] or 0) >= 2),
            "fusible_after": sum(1 for r in rows if (r["angles_with_claims_after"] or 0) >= 2),
            "github_rows_with_records_before": sum(1 for r in rows if (r["github_before"] or 0) > 0),
            "github_rows_with_records_after": sum(1 for r in rows if (r["github_after"] or 0) > 0),
            "distinct_tech_token_totals_before": len({t for t in tb if t is not None}),
            "distinct_tech_token_totals_after": len({t for t in ta if t is not None}),
            "rows_on_collapsed_1076_before": sum(1 for t in tb if t == 1076),
            "rows_on_collapsed_1076_after": sum(1 for t in ta if t == 1076),
            "n_rows": len(rows),
        },
        "caveats": [
            "The tech angle SWAPPED sources rather than gaining both: GitHub is "
            "the primary slot, so on rows where it now returns records the "
            "fallback chain stops before arXiv.",
            "Row 020's tech records fell 10 -> 7 because three GitHub records "
            "each exceed the entire 2000-token angle budget. That is V2-L7, a "
            "new limitation created by this fix.",
            "Row 046 produced no claims on a Groq TPD 429 (V2-L5). Its "
            "retrieval returned 10 records; the reasoner never ran.",
            "One fused row is not a calibration result. V2-L6 is unchanged.",
        ],
        "provenance": provenance(before_src, after_src),
    }


# --------------------------------------------------------------------------
# 7. Part 31 -- the real datasets (AIT Log Data Set V2.0, LANL cyber1)
# --------------------------------------------------------------------------
#
# Three rules this block adds to the ones above.
#
# GZIPPED RESULTS ARE VERIFIED BEFORE USE. Part 29's results.json is committed
# only as gzip, and its digest is pinned in results_manifest.json against the
# DECOMPRESSED bytes. read_gz_checked() recomputes that digest and raises on a
# mismatch, so a corrupted or re-gzipped archive fails the build instead of
# quietly shipping different numbers to the page.
#
# EVERY CLAIM CELL CARRIES ITS OWN RULE. The claim table's verdict words are
# the strongest thing on the page, so a cell is not a word: it is
# {verdict, rule, source_file, sha256}. The rule is the written test the cited
# file had to satisfy for that word, so a reader can check the verdict against
# the evidence instead of trusting it. Where the rule is not satisfied, or
# where a limitation says the measurement cannot bear the claim, the cell is
# NO EVIDENCE -- never a softened CONFIRMS.
#
# NO LANL CITATION TEXT EXISTS IN THIS REPOSITORY. configs/data/lanl.yaml and
# experiments/lanl/README.md name the dataset and nothing else: no licence, no
# DOI, no required-attribution paragraph. The card therefore prints the name
# the repo uses and says the citation text is absent, rather than inventing
# one. AIT's attribution IS in the repo and is quoted verbatim.

AIT_DATACARD = "docs/part27-ait-datacard.md"


def dig(obj, *path):
    """Nested get that returns None rather than raising on a missing key.

    A result file that loses a field should render "not measured", not crash
    the build -- the same policy clean() applies to NaN.
    """
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def read_gz_checked(gz_path: Path, manifest_path: Path, key: str) -> dict:
    """Decompress a committed .gz result and verify its pinned sha256.

    The manifest records the digest of the decompressed json, because a gzip
    stream's own digest is not stable across implementations. A mismatch is a
    build failure, not a warning: the alternative is a page of numbers from a
    file nobody can identify.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][key]
    import gzip
    raw = gzip.decompress(gz_path.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    expected = entry["sha256_decompressed_json"]
    if digest != expected:
        raise SystemExit(
            "ABORT: %s decompresses to sha256 %s, but %s pins %s. Nothing "
            "written." % (gz_path.name, digest, manifest_path.name, expected))
    if len(raw) != entry["raw_bytes"]:
        raise SystemExit(
            "ABORT: %s decompresses to %d bytes, manifest says %d."
            % (gz_path.name, len(raw), entry["raw_bytes"]))
    return json.loads(raw)


def summary_pair(summary: dict, field: str) -> dict:
    """mean / std / n_seeds for one metric, or nulls where it does not exist."""
    block = (summary or {}).get(field) or {}
    return {
        "mean": clean(block.get("mean")),
        "std": clean(block.get("std")),
        "n_seeds": clean(block.get("n_seeds")),
    }


def quote_block(path: Path, start: str, stop: str) -> str:
    """The blockquote lines between two markers, verbatim.

    Attribution text is a licence obligation, so it is read from the document
    that records it rather than retyped here, exactly as the limitations
    headings are. If the markers cannot be found the build fails.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out, collecting = [], False
    for line in lines:
        if not collecting:
            if line.strip().startswith(start):
                collecting = True
            continue
        if line.strip().startswith(stop):
            break
        if line.startswith(">"):
            out.append(line.lstrip("> ").rstrip())
        elif out and not line.strip():
            break
    if not out:
        raise SystemExit("ABORT: no quote found in %s after %r" % (path, start))
    return " ".join(out)


def build_real_ait() -> dict:
    """AIT Log Data Set V2.0 -- the clean split and the attack grid.

    Both directions are kept apart everywhere. Part 29's own rule is that the
    two scenarios are never pooled and never averaged, and the page follows it:
    there is no row anywhere below that mixes A and B.
    """
    clean_src = ROOT / "results/ait_clean/results.json"
    gz_src = ROOT / "results/ait_attack/results.json.gz"
    manifest_src = ROOT / "results/ait_attack/results_manifest.json"
    card_src = ROOT / AIT_DATACARD

    attack = read_gz_checked(gz_src, manifest_src, "results.json")

    directions = []
    for name, direction in attack["directions"].items():
        reference = direction["systems"]["prismflow"]

        # Dependence heatmap: the committed Part 28 pair values, read through
        # Part 29's reference block which records where it got them.
        pairs = reference["pairs"]
        heat = []
        for system, block in pairs["per_system"].items():
            for pair, value in block["values"].items():
                heat.append({
                    "system": system,
                    "pair": pair,
                    "mean": clean(value.get("mean")),
                    "std": clean(value.get("std")),
                })

        # Detector AUC and ENIV by k, split by whether `dns` was compromised.
        # L17 requires that split: `dns` alone solves this label, so a row
        # that does not say which side it is on is uninterpretable.
        by_k = []
        for key, cell in reference["by_dns_side"].items():
            s = cell["summary"]
            by_k.append({
                "cell": key,
                "kind": cell.get("kind"),
                "k": clean(cell.get("k")),
                "epsilon": clean(cell.get("epsilon")),
                "beta": clean(cell.get("beta")),
                "dns_compromised": cell.get("dns_compromised"),
                "detector_auc": summary_pair(s, "detector_auc"),
                "eniv_eigen": summary_pair(s, "eniv_eigen"),
                "attack_success": summary_pair(s, "attack_success"),
            })
        by_k.sort(key=lambda r: (r["kind"] or "", r["k"] or 0,
                                 r["epsilon"] or 0, bool(r["dns_compromised"])))

        clean_summary = reference["clean"]["summary"]
        directions.append({
            "id": name,
            "label": direction.get("label"),
            "train_scenario": direction.get("train_scenario"),
            "test_scenario": direction.get("test_scenario"),
            "n_train": clean(direction.get("n_train")),
            "n_test": clean(direction.get("n_test")),
            "label_meaning": direction.get("label_meaning"),
            "views": reference.get("sources"),
            "clean": {
                "eniv_eigen": summary_pair(clean_summary, "eniv_eigen"),
                "detector_auc": summary_pair(clean_summary, "detector_auc"),
                "roc_auc": summary_pair(clean_summary, "roc_auc"),
            },
            "dependence": {
                "label": pairs.get("label"),
                "source": pairs.get("source"),
                "systems_agree_on_ordering": pairs.get("systems_agree_on_ordering"),
                "most_dependent": pairs.get("most_dependent"),
                "least_dependent": pairs.get("least_dependent"),
                "caveat": pairs.get("caveat"),
                "cells": heat,
            },
            "attack_by_k": by_k,
        })

    # Test C. The committed verdict word is read, not decided here.
    sensor = attack["sensor_group_check"]

    return {
        "title": "AIT Log Data Set V2.0",
        "dataset": {
            "name": "AIT Log Data Set V2.0",
            "licence": "CC BY-NC-SA 4.0",
            "licence_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "source_url": "https://zenodo.org/records/5789064",
            "doi": "10.5281/zenodo.5789064",
            "attribution": quote_block(card_src, "> AIT Log Data Set V2.0",
                                       "**The record asks"),
            "attribution_source": AIT_DATACARD,
            "scenarios": sorted({d["test_scenario"] for d in directions}),
            "views": directions[0]["views"] if directions else None,
            "label_method": directions[0]["label_meaning"] if directions else None,
        },
        "directions": directions,
        "honest_duplication": {
            "verdict": sensor["verdict"],
            "reason": sensor["reason"],
            "requested_group": sensor.get("requested_group"),
            "n_members": clean(sensor.get("n_members")),
        },
        "seeds": attack.get("seeds"),
        "gz_note": (
            "results/ait_attack/results.json is committed as gzip only. This "
            "build decompressed it and checked its sha256 against "
            "results/ait_attack/results_manifest.json before reading a number "
            "from it."),
        "provenance": provenance(clean_src, gz_src, manifest_src, card_src),
    }


def build_real_lanl() -> dict:
    """LANL cyber1 -- per arm, naive against PrismFlow, with what bounds it.

    The card carries three things the AUCs cannot be read without: the
    case-control prevalence (L23), the untrained scalar that reaches 0.6843 on
    the same test block, and the seen/unseen split that removes the models'
    lead on computers not already seen attacked (L26).
    """
    src = ROOT / "results/lanl/results.json"
    manifest_src = ROOT / "results/lanl/results_manifest.json"
    raw = json.loads(src.read_text(encoding="utf-8"))

    # samples.npz and the run-1 gzip are pinned in the same manifest. Nothing
    # here reads them, but the digests are emitted so the page can state that
    # the arrays every diagnostic below was computed from are the committed
    # ones.
    manifest = json.loads(manifest_src.read_text(encoding="utf-8"))

    bootstrap = dig(raw, "phase3_diagnostics", "cluster_bootstrap") or {}
    trivial = dig(raw, "phase3_diagnostics", "trivial_baselines") or {}
    seen_unseen = dig(raw, "phase3_diagnostics", "seen_unseen") or {}
    split = raw["split_report"]["per_split"]

    arms = []
    for name, arm in raw["configurations"].items():
        ci = (bootstrap.get("per_arm") or {}).get(name) or {}
        paired = (arm.get("paired_prismflow_minus_naive") or {}).get("roc_auc") or {}
        arms.append({
            "arm": name,
            "label": arm.get("label"),
            "views": arm.get("views"),
            "naive_roc_auc": summary_pair(
                dig(arm, "systems", "naive", "summary"), "roc_auc"),
            "prismflow_roc_auc": summary_pair(
                dig(arm, "systems", "prismflow", "summary"), "roc_auc"),
            "paired_difference": {
                "mean": clean(paired.get("mean")),
                "std": clean(paired.get("std")),
                "n_seeds": clean(paired.get("n_seeds")),
                "n_seeds_favouring_prismflow": clean(
                    paired.get("n_seeds_favouring_prismflow")),
            },
            "bootstrap_ci95": {
                "observed": clean(ci.get("observed_mean_paired_auc_difference")),
                "low": clean(ci.get("ci95_low")),
                "high": clean(ci.get("ci95_high")),
                "replicates": clean(ci.get("n_replicates_used")),
                "n_test_computers": clean(ci.get("n_test_computers")),
            },
        })
    arms.sort(key=lambda r: r["arm"])

    scalar = (trivial.get("baselines") or {}).get("total_records_across_views") or {}

    return {
        "title": "LANL cyber1 (days 1-16)",
        "dataset": {
            # The repository names the dataset and nothing else. See the
            # module comment: no citation text is invented here.
            "name": raw.get("data_config") and "LANL Comprehensive Multi-Source "
                    "Cyber-Security Events (cyber1)",
            "name_source": "configs/data/lanl.yaml",
            "licence": None,
            "source_url": None,
            "attribution": None,
            "attribution_absent_note": (
                "No licence text, DOI, source link or required-attribution "
                "paragraph for LANL cyber1 exists anywhere in this "
                "repository. configs/data/lanl.yaml and "
                "experiments/lanl/README.md name the dataset and stop there. "
                "None is shown here rather than one being written for the "
                "occasion, and no download link is published."),
            "views": sorted({v for arm in raw["config"].get("view_configurations") or []
                             for v in arm["views"]}) or None,
            "label_method": raw.get("label_meaning"),
            "window_seconds": clean(raw.get("window_seconds")),
            "days_train": dig(split, "train", "days"),
            "days_test": dig(split, "test", "days"),
        },
        "sample": {
            "design": "declared case-control: all cases, deterministic 1/%s "
                      "hash of controls" % dig(raw, "population", "control_divisor"),
            "n_test": clean(dig(split, "test", "n_samples")),
            "n_test_attack": clean(dig(split, "test", "n_attack_samples")),
            "test_prevalence": clean(dig(split, "test", "attack_prevalence_in_sample")),
            "limitation": "L23",
            "limitation_note": (
                "The reported prevalence is a property of the sample, not of "
                "the network: accuracy, Brier and ECE describe this sample "
                "only and no population operating point has been measured "
                "(docs/KNOWN_LIMITATIONS.md L23)."),
        },
        "arms": arms,
        "trivial_scalar": {
            "name": "total_records_across_views",
            "test_auc": clean(scalar.get("test_auc")),
            "train_auc": clean(scalar.get("train_auc_as_measured")),
            "orientation": scalar.get("orientation_fixed_on_train"),
            "note": trivial.get("note"),
        },
        "seen_unseen": {
            "limitation": "L26",
            "note": seen_unseen.get("note"),
            "limitation_note": (
                "The models' ranking advantage over that scalar is confined "
                "to attack computers already seen as attacked in train. On "
                "unseen computers the scalar reaches 0.7563 and no arm "
                "clears it by more than 0.01 "
                "(docs/KNOWN_LIMITATIONS.md L26)."),
        },
        "bootstrap_note": bootstrap.get("note"),
        "evidence_digests": {
            "samples_npz": dig(manifest, "samples", "sha256"),
            "scores_n_files": dig(manifest, "scores", "n_files"),
            "run1_decompressed": dig(manifest, "run1", "decompressed_sha256"),
        },
        "seeds": raw["config"].get("seeds"),
        "provenance": provenance(src, manifest_src),
    }


# --------------------------------------------------------------------------
# 7c. The claim table
# --------------------------------------------------------------------------
#
# THE VERDICT WORD IS EARNED, NOT ASSIGNED. Each cell carries the written rule
# the cited file had to satisfy. Where a rule is testable from a committed
# file, this function tests it and derives the word; where a limitation says
# the measurement cannot bear the claim at all, the word is NO EVIDENCE and
# the rule says which limitation and why. A cell is never softened from
# CONTRADICTS to "needs more data", and never hardened from NO EVIDENCE to
# CONFIRMS because a number happens to point the right way.

VERDICTS = ("CONFIRMS", "CONTRADICTS", "NO EVIDENCE", "needs more data")

# The two negative words are NOT interchangeable, and collapsing them would
# lose the only distinction a reader needs to decide what to do next:
#
#   NO EVIDENCE       the quantity WAS measured and the rule is not satisfied
#                     by the measured values. More of the same measurement
#                     will not settle it; the design has to change.
#   needs more data   the quantity was not run, or cannot be measured on this
#                     dataset at all. There is nothing to be inconclusive
#                     about yet.
#
# A cell that was never run must never read NO EVIDENCE: that would report an
# absent experiment as a negative finding.
VERDICT_MEANING = {
    "CONFIRMS": "the cited file satisfies the rule",
    "CONTRADICTS": "the cited file satisfies the rule's opposite",
    "NO EVIDENCE": ("measured, but the rule is not satisfied by the measured "
                    "values"),
    "needs more data": "not run on this dataset, or not measurable on it",
}


def cell(verdict: str, rule: str, source_file: Path | None, detail: str = "") -> dict:
    if verdict not in VERDICTS:
        raise SystemExit("ABORT: %r is not one of %s" % (verdict, VERDICTS))
    return {
        "verdict": verdict,
        "rule": rule,
        "detail": detail,
        "source_file": (str(source_file.relative_to(ROOT)).replace("\\", "/")
                        if source_file else None),
        "sha256": sha256(source_file) if source_file else None,
    }


def build_real_summary() -> dict:
    chorus_src = ROOT / "results/chorus/attack_metrics.json"
    detector_src = ROOT / "results/comparison/detector.json"
    gz_src = ROOT / "results/ait_attack/results.json.gz"
    manifest_src = ROOT / "results/ait_attack/results_manifest.json"
    lanl_src = ROOT / "results/lanl/results.json"
    limits_src = ROOT / "docs/KNOWN_LIMITATIONS.md"

    chorus = json.loads(chorus_src.read_text(encoding="utf-8"))["summary"]
    detector = json.loads(detector_src.read_text(encoding="utf-8"))["summary"]
    attack = read_gz_checked(gz_src, manifest_src, "results.json")
    lanl = json.loads(lanl_src.read_text(encoding="utf-8"))

    # ---- detection, synthetic: measured, not asserted --------------------
    base = chorus["clean|prismflow"]["dependence_all_pairs"]
    comp = chorus["chorus_k2_eps1.0|prismflow"]["dependence_compromised"]
    margin = (base.get("std") or 0) + (comp.get("std") or 0)
    rose = (comp["mean"] - base["mean"]) > margin
    det_syn = cell(
        "CONFIRMS" if rose else "NO EVIDENCE",
        "CONFIRMS only if mean dependence among the compromised views under "
        "chorus k=2 eps=1.0 exceeds clean all-pairs dependence by more than "
        "the two 5-seed standard deviations combined.",
        chorus_src,
        "%.4f +/- %.4f compromised against %.4f +/- %.4f clean: a rise of "
        "%.4f against a combined std of %.4f, over 5 seeds."
        % (comp["mean"], comp["std"], base["mean"], base["std"],
           comp["mean"] - base["mean"], margin))

    # ---- detection, AIT: both directions, never pooled -------------------
    signs = {}
    for name, direction in attack["directions"].items():
        nat = direction["systems"]["prismflow"]["natural_windows"]
        atk = nat["dnsteal_only"]["summary"]["mean_dependence_eigen"]
        ben = nat["benign"]["summary"]["mean_dependence_eigen"]
        signs[name] = (atk["mean"] - ben["mean"], atk, ben)
    agree = len({d > 0 for d, _, _ in signs.values()}) == 1
    det_ait = cell(
        "CONFIRMS" if agree and all(d > 0 for d, _, _ in signs.values())
        else "NO EVIDENCE",
        "CONFIRMS only if attack-window dependence exceeds benign-window "
        "dependence in BOTH directions, reported separately (Part 29 forbids "
        "pooling the two scenarios), AND the comparison is informative about "
        "fusion at all. L17 says it is not: one view (`dns`) solves this "
        "label, so no AIT comparison between fused systems is evidence about "
        "fusion. NO EVIDENCE rather than `needs more data`: both directions "
        "WERE measured over 5 seeds and they disagree, so more runs of the "
        "same design will not settle it.",
        gz_src,
        "; ".join(
            "%s: attack %.4f +/- %.4f against benign %.4f +/- %.4f (%s)"
            % (name, atk["mean"], atk["std"], ben["mean"], ben["std"],
               "rises" if d > 0 else "falls")
            for name, (d, atk, ben) in sorted(signs.items()))
        + ". The two directions disagree, and L17 conditions both.")

    # ---- detection, LANL: L30 ------------------------------------------
    lanl_signs = []
    for name, arm in sorted(lanl["configurations"].items()):
        pops = dig(arm, "part29_test_a_natural_attack_windows", "per_system",
                   "prismflow", "populations") or {}
        atk = dig(pops, "attack_windows", "summary", "mean_dependence_eigen")
        ben = dig(pops, "benign_windows", "summary", "mean_dependence_eigen")
        if atk and ben and atk.get("mean") is not None:
            lanl_signs.append((name, atk["mean"] - ben["mean"]))
    reversing = [n for n, d in lanl_signs if d <= 0]
    det_lanl = cell(
        "NO EVIDENCE",
        "CONFIRMS only if Part 29's test A holds its sign across all four "
        "view configurations. NO EVIDENCE rather than `needs more data`: "
        "test A WAS run on all four arms over 5 seeds, and it does not hold "
        "its sign -- docs/KNOWN_LIMITATIONS.md L30 records it as not a "
        "stable signal on LANL. (Tests B, D and the detector half of C were "
        "not run, which is why the other two LANL cells read `needs more "
        "data` instead.)",
        lanl_src,
        "Attack-minus-benign mean dependence per arm: "
        + ", ".join("%s %+.4f" % (n, d) for n, d in lanl_signs)
        + ". %s reverse%s the sign."
        % (", ".join(reversing) or "No arm", "" if len(reversing) != 1 else "s"))

    # ---- defence, synthetic: mechanical over every non-clean condition ---
    worse = []
    for key, fields in chorus.items():
        condition, _, system = key.partition("|")
        if condition == "clean" or system != "prismflow":
            continue
        naive = chorus.get(condition + "|naive")
        if not naive:
            continue
        diff = (fields["success_rate"]["mean"]
                - naive["success_rate"]["mean"])
        worse.append((condition, diff))
    all_worse = worse and all(d > 0 for _, d in worse)
    def_syn = cell(
        "CONTRADICTS" if all_worse else "NO EVIDENCE",
        "CONFIRMS if PrismFlow's mean attack success is below naive's in "
        "every non-clean condition; CONTRADICTS if it is at or above naive's "
        "in every one of them; otherwise NO EVIDENCE.",
        chorus_src,
        "PrismFlow is above naive in %d of %d non-clean conditions "
        "(range %+.4f to %+.4f), 5 seeds each."
        % (sum(1 for _, d in worse if d > 0), len(worse),
           min(d for _, d in worse), max(d for _, d in worse)))

    # ---- defence, AIT: paired within seed, by direction ------------------
    per_direction = {}
    for name, direction in attack["directions"].items():
        below = above = 0
        for _, paired in direction["paired_prismflow_minus_naive"].items():
            if paired.get("kind") not in ("chorus", "pgd"):
                continue
            mean = dig(paired, "diff", "attack_success", "mean")
            if mean is None:
                continue
            if mean < 0:
                below += 1
            elif mean > 0:
                above += 1
        per_direction[name] = (below, above)
    def_ait = cell(
        "NO EVIDENCE",
        "CONFIRMS only if PrismFlow's paired attack success is below naive's "
        "in every cell of both directions, with the compromised set named "
        "(L17: a cell that does not say whether `dns` was compromised is "
        "uninterpretable). Mixed signs in either direction fail the rule. "
        "NO EVIDENCE rather than `needs more data`: all 185 cells per "
        "direction WERE measured, and their signs are mixed.",
        gz_src,
        "; ".join("%s: PrismFlow below naive in %d cells, above in %d"
                  % (name, below, above)
                  for name, (below, above) in sorted(per_direction.items()))
        + ". Signs are mixed in both directions.")

    # ---- defence, LANL: the chorus grid was never run --------------------
    not_run = dig(lanl, "configurations", "with_auth", "part29_tests_not_run")
    def_lanl = cell(
        "needs more data",
        "CONFIRMS or CONTRADICTS requires an attack-success measurement. "
        "Part 29's chorus grid was not run on LANL, so no cell exists to "
        "test either way. Not NO EVIDENCE: nothing was measured and found "
        "inconclusive here, so there is no negative result to report -- the "
        "experiment is simply absent.",
        lanl_src,
        "Part 29's chorus grid was not run on LANL, and no partial grid is "
        "reported.")

    # ---- honest duplication, synthetic: L22 blocks the figure ------------
    clone = detector["clone_k2"]["auc"]
    best_chorus = max(detector[k]["auc"]["mean"] for k in detector
                      if k.startswith("chorus"))
    dup_syn = cell(
        "needs more data",
        "CONFIRMS only if the clone condition and its clean reference run the "
        "SAME number of views, so the detector's score can be attributed to "
        "duplication rather than to width. docs/KNOWN_LIMITATIONS.md L22 "
        "records that they do not: `clone_k2` runs 6 views against a 4-view "
        "reference, and the suspicion score is a maximum over off-diagonal "
        "entries, which grows with width. The figure therefore cannot be "
        "differenced against any fixed-width result.",
        detector_src,
        "clone_k2 detector AUC %.4f +/- %.4f over 5 seeds, above every "
        "chorus condition (best %.4f) -- but at 6 views against 4, so part "
        "of the gap is the width change itself (L22)."
        % (clone["mean"], clone["std"], best_chorus))

    # ---- honest duplication, AIT: the committed verdict word -------------
    sensor = attack["sensor_group_check"]
    dup_ait = cell(
        "needs more data",
        "CONFIRMS requires a within-sensor-group PAIR to score the detector "
        "on, and on these five views no such pair exists, so the quantity is "
        "not measurable here rather than measured and inconclusive. Part 29 "
        "recorded the word `%s` in `sensor_group_check.verdict`; under this "
        "page's vocabulary an unmeasurable quantity is `needs more data`, and "
        "the run's own word is quoted below unchanged."
        % sensor["verdict"],
        gz_src,
        sensor["reason"])

    # ---- honest duplication, LANL: half-run, and the half that ran is NO
    # EVIDENCE by LANL's own record ---------------------------------------
    dup_lanl = cell(
        "needs more data",
        "CONFIRMS requires a detector AUC on a duplicated-but-honest "
        "condition. LANL ships no sensor-group definition, so no strict "
        "same-sensor pair exists to score, and the detector was not scored "
        "on the naturally-dependent pairs either. Nothing was measured and "
        "found inconclusive, so this is not NO EVIDENCE.",
        lanl_src,
        (not_run or {}).get("C_honest_duplication", ""))

    return {
        "title": "What each dataset says about each claim",
        "note": (
            "Columns are datasets, rows are claims. A cell is a verdict plus "
            "the written RULE that verdict had to satisfy, plus the file it "
            "was read from and that file's sha256. Nothing is pooled across "
            "datasets, nothing is averaged across AIT's two directions, and "
            "where a rule is not satisfied the cell says NO EVIDENCE rather "
            "than a weaker version of the claim. NO EVIDENCE and needs more "
            "data are different findings: the first means the quantity was "
            "measured and the rule is not satisfied by the measured values, "
            "the second that it was not run or cannot be measured on that "
            "dataset at all."),
        "verdict_vocabulary": list(VERDICTS),
        "verdict_meaning": VERDICT_MEANING,
        "columns": [
            {"id": "synthetic", "label": "synthetic",
             "note": "rho is set, so dependence has a ground truth"},
            {"id": "ait", "label": "AIT Log Data Set V2.0",
             "note": "two directions, never pooled"},
            {"id": "lanl", "label": "LANL cyber1",
             "note": "clean data only; Part 29's tests B, D and half of C "
                     "were not run"},
        ],
        "rows": [
            {"id": "detection",
             "claim": "Detection: under attack, dependence among the "
                      "implicated views rises above its clean reference.",
             "cells": {"synthetic": det_syn, "ait": det_ait, "lanl": det_lanl}},
            {"id": "defence",
             "claim": "Defence: the dependence discount lowers attack "
                      "success relative to the undiscounted baseline.",
             "cells": {"synthetic": def_syn, "ait": def_ait, "lanl": def_lanl}},
            {"id": "honest_duplication",
             "claim": "Honest-duplication false positive: the detector fires "
                      "on redundancy that no attacker created.",
             "cells": {"synthetic": dup_syn, "ait": dup_ait, "lanl": dup_lanl}},
        ],
        "limitations_source": str(limits_src.relative_to(ROOT)).replace("\\", "/"),
        "provenance": provenance(chorus_src, detector_src, gz_src,
                                 manifest_src, lanl_src, limits_src),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "v1_surface.json": build_v1_surface,
        "v1_attack.json": build_v1_attack,
        "v1_failures.json": build_v1_failures,
        "calibration_duplicated.json": build_calibration,
        "v2_coverage.json": build_v2_coverage,
        "v2_dataset.json": build_v2_dataset,
        "v1_figures.json": build_figures,
        "limitations.json": build_limitations,
        "real_ait.json": build_real_ait,
        "real_lanl.json": build_real_lanl,
        "real_summary.json": build_real_summary,
    }
    reverify = build_v2_reverify()
    if reverify is not None:
        artifacts["v2_reverify.json"] = lambda: reverify
    else:
        print("note: results/v2/part24_reverify_n6.json is absent, so no "
              "before/after panel is emitted. The page omits it rather than "
              "showing one side of a comparison.")

    for name, builder in artifacts.items():
        payload = builder()
        text = json.dumps(payload, indent=2, allow_nan=False)

        for pattern in SECRET_PATTERNS:
            hit = pattern.search(text)
            if hit:
                print("ABORT: %s matches a credential pattern (%s). Nothing written."
                      % (name, pattern.pattern))
                return 1

        (OUT / name).write_text(text, encoding="utf-8")
        srcs = payload["provenance"]["sources"]
        print("wrote web/data/%-18s  %6d bytes  from %d source(s)"
              % (name, len(text), len(srcs)))
        for s in srcs:
            print("    %s  sha256 %s" % (s["path"], s["sha256"][:16]))

    download = build_dataset_download()
    (OUT / "v2_dataset_download.json").write_text(
        json.dumps(download, indent=2), encoding="utf-8")
    print("wrote web/data/%-18s  %s"
          % ("v2_dataset_download.json",
             download["file"] or "workbook MISSING at build time"))

    print("\nweb/data/ now holds every file the site can read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
