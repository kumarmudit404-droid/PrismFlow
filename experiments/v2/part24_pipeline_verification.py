"""Partial pipeline-verification run over the real evaluation rows.

    .venv\\Scripts\\python.exe experiments/v2/part24_pipeline_verification.py
    .venv\\Scripts\\python.exe experiments/v2/part24_pipeline_verification.py --limit 2

WHAT THIS IS NOT
----------------
NOT Part 24 evidence under docs/CONTRACT.md section 5. Not partially, not
"pending more rows". The standard needs the full dataset and the reasoners this
project intends to report on, and this run has neither:

* n = 17 of the eventual 50 rows, and all 17 are Tech-OSS but one. There is no
  domain balance, so nothing here generalises past Swift/PEP decisions.
* The reasoner is Groq (openai/gpt-oss-120b) through the OpenAIReasoner CODE
  PATH. Claude and OpenAI are both unfunded. Groq is OpenAI-*compatible*, not
  OpenAI: this says nothing about gpt-4o and nothing about Claude.
* Fusion adjudication is the offline LexicalAdjudicator, because
  ClaudeAdjudicator needs Anthropic credentials this checkout cannot spend. The
  "Fusion Layer via Claude API" is therefore NOT exercised.
* ONE pass. No seeds, no mean, no standard deviation -- see below.

WHY THERE IS NO SEED LOOP
-------------------------
Because on this path a seed would select nothing, and a loop that selects
nothing is how the 5-SEED RULE gets faked.

Traced stage by stage: connector results come from Part 18's cache, which
guarantees a cached result is indistinguishable from a fresh one; the reranker
sorts stably and keeps retrieval order on ties, deliberately, so dependence is
not a tie-breaking artefact; the sentence-transformer encoder is deterministic;
Part 21's dependence and Part 22's ENIV contain no RNG; fusion's candidate
generation is encoder-based and the offline adjudicator is deterministic.

That leaves the reasoner. prismflow/v2/reasoners/openai_reasoner.py sends
model, max_tokens, messages and response_format -- no temperature, no top_p and
no seed. So the provider samples at its own default and repeated runs DO
differ, but a seed index would not select that variation and could not
reproduce it. Reporting such runs as "5 seeds" would dress uncontrolled sampler
noise as a reproducible measurement, which is a worse error than the std=0.0
defect this project already caught twice: that one made an absent measurement
look stable, this one would make noise look controlled.

Where the seeds in this repo DO select something -- experiments/v2's fusion e2e
and ENIV redundancy runs -- they draw synthetic fixtures: banked queries,
planted contradictions, confidence bands. Over 17 fixed real rows there is no
fixture left to draw.

WHAT IT IS
----------
A code-correctness check that the five angles retrieve, the reasoner returns
parseable claims, dependence and ENIV compute over more than one angle, fusion
aggregates, and the Part 24 metrics consume the result on the right
denominators. Per-row and per-angle output, so a defect surfaces at the row it
came from instead of averaging into a headline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prismflow.v2.angles import build_angles, load_config  # noqa: E402
from prismflow.v2.angles import financial_angle, market_angle  # noqa: E402
from prismflow.v2.angles import sentiment_angle  # noqa: E402
from prismflow.v2.cache import init_cache_db  # noqa: E402
from prismflow.v2.dependence import aggregate_dependence  # noqa: E402
from prismflow.v2.evaluation import (  # noqa: E402
    compute_calibration,
    compute_conflict_metrics,
    load_evaluation_dataset,
)
from prismflow.v2.fusion import PrismFusion  # noqa: E402
from prismflow.v2.reasoners.openai_reasoner import OpenAIReasoner  # noqa: E402
from prismflow.v2.statistics import compute_semantic_eniv  # noqa: E402

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
ANGLE_ORDER = ("tech", "market", "financial", "regulatory", "sentiment")

BANNER = "=" * 78
LABEL = (
    "PARTIAL PIPELINE-VERIFICATION RUN -- NOT PART 24 EVIDENCE\n"
    "  n=17 of the eventual 50 rows; reasoner is GROQ (openai/gpt-oss-120b)\n"
    "  via the OpenAIReasoner code path -- NOT Claude, NOT OpenAI (unfunded).\n"
    "  Fusion adjudication is the offline LexicalAdjudicator, so Part 23's\n"
    "  Claude path is NOT exercised.\n"
    "  ONE pass. No seeds: nothing on this path varies as a function of a\n"
    "  seed (see module docstring), so no mean and no std are reported.\n"
    "  Not evidence under docs/CONTRACT.md section 5."
)


def groq_key() -> str:
    """GROQ_API_KEY from the environment or .env. Never printed."""
    from_env = os.getenv("GROQ_API_KEY")
    if from_env:
        return from_env.strip()
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return ""
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        if line.startswith("GROQ_API_KEY="):
            return line.partition("=")[2].strip()
    return ""


def all_live_sources() -> Dict[str, tuple]:
    """Every angle wired to the connectors that actually exist today.

    build_angles leaves an angle without connectors when its slot is omitted.
    Dependence excludes angles that produced no claims, so a one-angle run
    yields a 1x1 matrix and a degenerate ENIV -- wiring everything that CAN be
    wired is what makes the middle of the pipeline mean anything.

    FOUR angles, not five. ``regulatory`` is not omitted by oversight: no
    connector for it has been built. Its module says so outright -- "No
    connector exists yet; inject one when it is built", with data.gov, EUR-Lex
    and RBI all unimplemented. It will retrieve nothing on every row, and that
    is a property of this checkout rather than a fault of the run.

    ``tech`` has no ``default_connectors`` helper of its own, so its GitHub and
    arXiv chain is constructed here directly.
    """
    from prismflow.v2.connectors.arxiv import ArxivConnector
    from prismflow.v2.connectors.github import GitHubConnector

    return {
        "tech": (GitHubConnector(), ArxivConnector(), None),
        "market": market_angle.default_connectors(),
        "financial": financial_angle.default_connectors(),
        "sentiment": sentiment_angle.default_connectors(),
        # "regulatory": no connector exists to wire. Left unwired deliberately.
    }


async def run_row(query, angles, reasoners, fusion):
    """One row through angles -> reasoners -> dependence -> ENIV -> fusion."""
    started = time.perf_counter()
    record: Dict = {"id": query.id, "angles": {}, "errors": []}

    evidences = await asyncio.gather(
        *(angles[name].retrieve(query.query_text) for name in ANGLE_ORDER),
        return_exceptions=True,
    )

    claimsets = []
    for name, evidence in zip(ANGLE_ORDER, evidences):
        entry: Dict = {}
        if isinstance(evidence, BaseException):
            entry["retrieval"] = "ERROR %s: %s" % (
                type(evidence).__name__, str(evidence)[:90])
            record["angles"][name] = entry
            record["errors"].append("%s retrieval" % name)
            continue

        entry["records"] = len(evidence.ranked_records)
        entry["tokens"] = evidence.total_tokens
        entry["provenance"] = dict(evidence.provenance)
        entry["warnings"] = list(evidence.warnings)[:2]

        if evidence.is_empty:
            entry["claims"] = 0
            entry["note"] = "no evidence retrieved; reasoner not called"
            record["angles"][name] = entry
            continue

        try:
            claimset = await reasoners[name].reason(evidence)
        except Exception as exc:                      # provider or parse error
            entry["claims"] = "ERROR %s: %s" % (type(exc).__name__, str(exc)[:90])
            record["angles"][name] = entry
            record["errors"].append("%s reasoner" % name)
            continue

        entry["claims"] = len(claimset.claims)
        entry["mean_claim_conf"] = (
            round(sum(c.confidence for c in claimset.claims) / len(claimset.claims), 4)
            if claimset.claims else None)
        record["angles"][name] = entry
        if claimset.claims:
            claimsets.append(claimset)

    record["angles_with_claims"] = len(claimsets)
    if len(claimsets) < 2:
        record["fused"] = None
        record["note"] = ("fewer than 2 angles produced claims; dependence "
                          "and ENIV are not defined for this row")
        record["seconds"] = round(time.perf_counter() - started, 2)
        return record, None

    report = aggregate_dependence(claimsets)
    eniv = compute_semantic_eniv(report, method="eigen")
    fused = await fusion.fuse(query.query_text, claimsets, report, eniv=eniv)

    record["eniv"] = round(eniv, 4)
    record["n_angles"] = fused.n_angles
    record["discount"] = round(fused.discount_factor, 4)
    record["confidence"] = round(fused.overall_confidence, 4)
    record["undiscounted"] = round(fused.undiscounted_confidence, 4)
    record["conflicts"] = len(fused.conflicts)
    record["contradictions"] = len(fused.contradictions)
    record["adjudicator"] = fused.adjudicator
    record["seconds"] = round(time.perf_counter() - started, 2)
    return record, fused


async def main_async(limit: Optional[int]) -> int:
    key = groq_key()
    if not key:
        print("GROQ_API_KEY is not set. Nothing was called; this run did not "
              "happen. An absent credential is not a result.")
        return 1

    dataset = load_evaluation_dataset(str(ROOT / "data/v2/evaluation_queries.json"))
    rows = dataset[:limit] if limit else dataset

    print(BANNER)
    print(LABEL)
    print(BANNER)
    print("rows: %d of %d loaded  |  angles: 4 of 5 wired live (regulatory has no connector)  |  one pass"
          % (len(rows), len(dataset)))
    print()

    db = os.path.join(tempfile.mkdtemp(prefix="prismflow_part24_verify_"), "cache.db")
    init_cache_db(db)

    config = load_config(str(ROOT / "configs/v2/angle_defaults.yaml"))
    angles = build_angles(config=config, sources=all_live_sources(),
                          use_cache=True, cache_db_path=db)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=key, base_url=GROQ_BASE_URL)
    reasoners = {name: OpenAIReasoner(name, model=GROQ_MODEL, client=client)
                 for name in ANGLE_ORDER}
    fusion = PrismFusion()          # adjudicator=None -> LexicalAdjudicator

    records: List[Dict] = []
    confidence_by_id: Dict[str, float] = {}
    detected_by_id: Dict[str, bool] = {}

    try:
        for index, query in enumerate(rows, start=1):
            print("--- row %s (%d/%d) %s" % (query.id, index, len(rows),
                                             query.actual_outcome))
            record, fused = await run_row(query, angles, reasoners, fusion)
            records.append(record)

            for name in ANGLE_ORDER:
                entry = record["angles"].get(name, {})
                print("      %-11s records=%-4s claims=%-6s %s"
                      % (name, entry.get("records", "-"), entry.get("claims", "-"),
                         ((entry.get("warnings") or [""])[0]
                          or entry.get("retrieval") or entry.get("note")
                          or "")[:100]))
            if fused is None:
                print("      -> %s" % record.get("note"))
            else:
                print("      -> ENIV %.3f  discount %.3f  confidence %.4f  "
                      "conflicts %d (contradictions %d)  %.1fs"
                      % (record["eniv"], record["discount"], record["confidence"],
                         record["conflicts"], record["contradictions"],
                         record["seconds"]))
                confidence_by_id[query.id] = fused.overall_confidence
                detected_by_id[query.id] = len(fused.contradictions) > 0
    finally:
        for angle in angles.values():
            try:
                await angle.aclose()
            except Exception:
                pass

    print()
    print(BANNER)
    print("PER-ROW SUMMARY (still not Part 24 evidence)")
    print(BANNER)
    print("%-5s %-9s %-7s %-8s %-7s %-6s %-5s %s"
          % ("id", "outcome", "angles", "ENIV", "conf", "confl", "err", "note"))
    for record in records:
        print("%-5s %-9s %-7s %-8s %-7s %-6s %-5s %s"
              % (record["id"],
                 next(q.actual_outcome for q in rows if q.id == record["id"]),
                 record.get("angles_with_claims", 0),
                 record.get("eniv", "-"),
                 record.get("confidence", "-"),
                 record.get("contradictions", "-"),
                 len(record["errors"]),
                 (record.get("note") or "")[:30]))

    # Persisted BEFORE the metrics gate: on a run too degraded to produce
    # metrics the per-row detail is the whole result, and an early return
    # used to discard it.
    out = ROOT / "results" / "v2" / "part24_pipeline_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": LABEL,
        "is_part24_evidence": False,
        "reasoner": {"provider": "groq", "model": GROQ_MODEL,
                     "note": "OpenAIReasoner code path; not OpenAI, not Claude"},
        "adjudicator": "LexicalAdjudicator (offline; Claude path not exercised)",
        "passes": 1,
        "seeds": None,
        "seed_note": "no seed loop: nothing on this path varies per seed",
        "n_rows": len(rows),
        "rows": records,
    }, indent=2), encoding="utf-8")
    print("\nper-row detail written to %s" % out)

    scored = [q for q in rows if q.id in confidence_by_id]
    print()
    if len(scored) < 2:
        print("Too few rows completed for metrics. Reporting no ECE, no Brier "
              "and no conflict figures rather than computing them on a "
              "handful of rows.")
        return 0

    cal = compute_calibration(scored, confidence_by_id)
    con = compute_conflict_metrics(
        [q for q in scored if q.id in detected_by_id], detected_by_id)
    print(BANNER)
    print("METRICS -- NOT A PART 24 FINDING")
    print("  Single run, no seeds (nothing varies per seed on this path), no")
    print("  comparison condition, n=%d of 50, Groq not Claude/OpenAI, and")
    print("  fusion adjudicated offline. Under CONTRACT.md section 5 and the")
    print("  5-SEED RULE these numbers are a smoke test, not a result.")
    print(BANNER)
    print("  calibration %s" % cal.summary())
    print("              excluded: %s" % (list(cal.excluded_ids) or "none"))
    print("  %s" % con.summary())
    print("              excluded: %s" % (list(con.excluded_ids) or "none"))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N rows (for a dry check)")
    args = parser.parse_args()
    return asyncio.run(main_async(args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
