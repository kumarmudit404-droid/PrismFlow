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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "v1_surface.json": build_v1_surface,
        "v1_attack.json": build_v1_attack,
        "v2_coverage.json": build_v2_coverage,
    }

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

    print("\nweb/data/ now holds every file the site can read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
