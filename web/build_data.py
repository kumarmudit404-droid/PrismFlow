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
# 3. V2 -- per-row angle coverage over the 48 rows
# --------------------------------------------------------------------------

ANGLES = ("tech", "market", "financial", "regulatory", "sentiment")


def build_v2_coverage() -> dict:
    ds_src = ROOT / "data/v2/evaluation_queries.json"
    run_src = ROOT / "results/v2/part24_pipeline_verification.json"
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
    pass1_note = None
    if pass1_src.exists():
        sources.append(pass1_src)
        p1 = json.loads(pass1_src.read_text(encoding="utf-8"))
        fused1 = [r["id"] for r in p1["rows"] if (r.get("angles_with_claims") or 0) >= 2]
        pass1_note = {
            "rows": p1.get("n_rows"),
            "fused_rows": fused1,
            "note": ("Retrieval is identical in both passes. The committed run "
                     "above is the second pass, whose CLAIM counts are limited "
                     "by an exhausted daily Groq quota; the first pass fused "
                     "%d row(s)." % len(fused1)),
        }

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
        "zero_claim_rows_provider_error": run.get("zero_claim_rows_provider_error"),
        "zero_claim_rows_genuine": run.get("zero_claim_rows_genuine"),
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "v1_surface.json": build_v1_surface,
        "v1_attack.json": build_v1_attack,
        "v1_failures.json": build_v1_failures,
        "v2_coverage.json": build_v2_coverage,
        "v2_dataset.json": build_v2_dataset,
        "v1_figures.json": build_figures,
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
