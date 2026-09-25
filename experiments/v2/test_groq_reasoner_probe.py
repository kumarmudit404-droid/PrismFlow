"""Live smoke test for the Part 20 OpenAIReasoner CODE PATH, via Groq.

Not a pytest module despite the name (Part 17 set the convention). Run it
directly:

    .venv\\Scripts\\python.exe experiments/v2/test_groq_reasoner_probe.py

This is CONNECTIVITY VERIFICATION, not evidence. One query, one pass, no
seeds. docs/CONTRACT.md section 5 requires five seeds and mean/sd for anything
that counts as a finding, and confirming that a request shape is accepted is
not that. It writes NO results JSON, because a file under results/ would be
citable as a Part 24 measurement and this is not one. Claim text produced by a
model is not a measurement.

INJECTION ONLY -- NOTHING IN prismflow/v2/reasoners/ IS MODIFIED
---------------------------------------------------------------
``OpenAIReasoner.__init__`` already accepts a ``client`` and only constructs
``AsyncOpenAI()`` when none is given, so a client pointed at Groq's
OpenAI-compatible endpoint is pure injection -- the same pattern the angle
wirings use for connectors. No frozen module is touched to run this.

WHAT THIS CAN AND CANNOT ESTABLISH
----------------------------------
It exercises the OpenAIReasoner code path against a real OpenAI-compatible
server: request shape, ``response_format={"type": "json_object"}`` handling,
``usage`` field names, ``finish_reason``, the JSON parsing in parsing.py, and
error mapping. It says NOTHING about claude_reasoner.py, which it never calls,
and NOTHING about gpt-4o -- a different model on a different vendor's server.
Groq is OpenAI-*compatible*, not OpenAI. A pass here is never "Part 20
verified"; see docs/part20-groq-verification.md.

CREDENTIALS -- WHAT A RE-RUN SPENDS
-----------------------------------
By default this script spends ONE Groq completion and nothing else. It does
NOT touch NEWSAPI_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY, and it reads no
rotating credential: the evidence it reasons over is the static fixture below,
so re-running it cannot burn NewsAPI's 100-requests-per-day free-tier quota.
``--live-newsapi`` opts back in to the live Market retrieval the original run
used, which costs NewsAPI requests; it is off by default for that reason.
``GROQ_API_KEY`` is read from .env and never printed.

THE FIXTURE IS A FIXTURE
------------------------
``FIXTURE_RECORDS`` is placeholder text written for this script. It is NOT a
transcript of the articles the original run retrieved, and nothing derived
from it -- claim text, citation counts -- should be reported as reproducing
that run. What it reproduces is the CODE PATH: an AngleEvidence of the right
shape reaching the reasoner. To reproduce the original run's inputs, pass
``--live-newsapi``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, str(ROOT))

from prismflow.v2.angles import AngleEvidence, MarketAngle, load_config  # noqa: E402
from prismflow.v2.angles.market_angle import default_connectors  # noqa: E402
from prismflow.v2.cache import init_cache_db  # noqa: E402
from prismflow.v2.connectors.base import NormalizedRecord, count_tokens  # noqa: E402
from prismflow.v2.reasoners import OpenAIReasoner  # noqa: E402

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
QUERY = "semiconductor export controls"

#: Static stand-in evidence. See "THE FIXTURE IS A FIXTURE" above: this text is
#: written for this script, not retrieved from anywhere, and exists so the
#: reasoner can be exercised without spending a NewsAPI request.
FIXTURE_RECORDS = [
    (
        "fixture-1",
        "Export controls tightened on advanced logic nodes",
        "A fixture record. Controls are described as extending to sub-14nm "
        "logic and to the tooling used to produce it, with a compliance "
        "deadline later in the year.",
    ),
    (
        "fixture-2",
        "Equipment vendors report order deferrals",
        "A fixture record. Vendors are described as deferring shipments into "
        "the affected region, with guidance withdrawn for the coming quarter.",
    ),
    (
        "fixture-3",
        "Licence exemptions narrowed for memory fabs",
        "A fixture record. Exemptions previously granted to memory producers "
        "are described as narrowed, contradicting fixture-2's framing of the "
        "scope as equipment-only.",
    ),
]


def fixture_evidence() -> AngleEvidence:
    """An AngleEvidence of the right shape, retrieved from nowhere."""
    now = datetime.now(timezone.utc)
    records = [
        NormalizedRecord(
            id=record_id,
            title=title,
            url=f"https://example.invalid/{record_id}",
            snippet=snippet,
            snippet_tokens=count_tokens(snippet),
            source="fixture",
            published_date=now,
            author=None,
            relevance_score=1.0 - index * 0.1,
        )
        for index, (record_id, title, snippet) in enumerate(FIXTURE_RECORDS)
    ]
    return AngleEvidence(
        query_text=QUERY,
        angle_name="market",
        raw_records=list(records),
        ranked_records=list(records),
        total_tokens=sum(r.snippet_tokens for r in records),
        provenance={"fixture": len(records)},
        warnings=["evidence is a static fixture, not a retrieval"],
    )


async def live_evidence(db_path: str) -> AngleEvidence:
    """The original run's input: a live Market retrieval. Spends NEWSAPI_KEY."""
    config = load_config(str(ROOT / "configs/v2/angle_defaults.yaml"))
    angle = MarketAngle(*default_connectors(), config=config["market"],
                        use_cache=True, cache_db_path=db_path)
    try:
        return await angle.retrieve(QUERY)
    finally:
        await angle.aclose()


def groq_key() -> str:
    """GROQ_API_KEY from the environment or .env. Never printed.

    Minimal on purpose, matching the reader in connectors/newsapi.py: no
    dependency on python-dotenv and no mutation of os.environ.
    """
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


async def smoke(use_live_newsapi: bool) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    key = groq_key()
    if not key:
        print("GROQ_API_KEY is not set (in the environment or .env).")
        print("UNVERIFIED: nothing was called. An absent credential is not a")
        print("defect in the code, and this script does not pretend the check")
        print("ran.")
        return 0

    db = os.path.join(tempfile.mkdtemp(prefix="prismflow_groq_probe_"), "p.db")
    init_cache_db(db)

    if use_live_newsapi:
        print("=== retrieval (Part 19, market angle, LIVE NewsAPI) ===")
        print("  spending NewsAPI requests against NEWSAPI_KEY, as asked")
        evidence = await live_evidence(db)
    else:
        print("=== evidence (static fixture, no retrieval, no NewsAPI) ===")
        print("  pass --live-newsapi to reproduce the original run's inputs")
        evidence = fixture_evidence()
    print(f"  {evidence.summary()}")
    print(f"  provenance {evidence.provenance}")
    if evidence.is_empty:
        print("  FAILED: no evidence; cannot exercise the reasoner")
        return 1

    # --- Part 20: the existing reasoner, client injected ----------------
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=key, base_url=GROQ_BASE_URL)
    reasoner = OpenAIReasoner("market", model=GROQ_MODEL, client=client)
    print("\n=== reasoner ===")
    print(f"  class     {type(reasoner).__name__} (unmodified)")
    print(f"  PROVIDER  {reasoner.PROVIDER!r}")
    print(f"  model     {reasoner.model!r}  (injected, not the default "
          f"{OpenAIReasoner.default_model()!r})")
    print(f"  base_url  {GROQ_BASE_URL}")

    xml = reasoner.format_evidence_xml(evidence)
    print(f"  evidence XML: {len(xml)} chars, {xml.count('<record ')} record(s)")

    print(f"\ncalling {GROQ_MODEL} ...")
    result = await reasoner.reason(evidence)
    print(f"  {result.summary()}")
    if not result.succeeded:
        print(f"  FAILED: {result.error}")
        for line in result.audit_trail:
            print(f"    audit: {line}")
        return 1

    known = {r.id for r in evidence.ranked_records}
    ungrounded = 0
    for index, claim in enumerate(result.claims, 1):
        print(f"\n  claim {index} [confidence {claim.confidence:.2f}]")
        print(f"    {_ascii(claim.text, 300)}")
        print(f"    cites: {[_ascii(c, 70) for c in claim.cited_ids]}")
        for conflict in claim.conflicts_noted:
            print(f"    conflict: {_ascii(conflict, 160)}")
        for caveat in claim.caveats:
            print(f"    caveat: {_ascii(caveat, 160)}")
        if not set(claim.cited_ids) <= known:
            ungrounded += 1
            print("    !! a citation is NOT in the evidence")

    print("\naudit trail:")
    for line in result.audit_trail:
        print(f"    {_ascii(line, 200)}")
    print(f"\nclaims: {result.total_claims()}, ungrounded citations: {ungrounded}")
    print("\nA pass here verifies the OpenAIReasoner CODE PATH only.")
    print("claude_reasoner.py and gpt-4o remain UNVERIFIED.")
    return 1 if ungrounded else 0


def _ascii(text, limit: int) -> str:
    text = "" if text is None else str(text)
    return text[:limit].encode("ascii", "replace").decode("ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live-newsapi",
        action="store_true",
        help="retrieve real Market evidence instead of using the fixture; "
             "this spends NewsAPI requests against NEWSAPI_KEY",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(smoke(args.live_newsapi)))
