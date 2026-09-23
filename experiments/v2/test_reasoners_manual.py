"""Live smoke test for the Part 20 reasoners: retrieval through to claims.

Not a pytest module despite the name (Part 17 set the convention). Run it with a
credential present:

    .venv\\Scripts\\python.exe experiments/v2/test_reasoners_manual.py

WHAT THIS COVERS THAT THE TEST SUITE CANNOT
-------------------------------------------
tests/v2/test_reasoners.py drives both providers through doubles, which reaches
every failure mode in the code around the call. The one thing a double cannot
establish is that the request SHAPE is accepted by the real API -- the model id,
the thinking parameter, the usage field names, the content block types. That is
what this script is for, and it is the reason it is not skipped quietly: an
unverified request shape is a real gap, not a passing test.

CREDENTIALS
-----------
This repository has none: .env ships ANTHROPIC_API_KEY and OPENAI_API_KEY as
placeholders and neither is exported. The script reports that as UNVERIFIED and
exits 0 -- an absent credential is not a defect in the code -- but it does not
pretend the check ran.

An unset ANTHROPIC_API_KEY does not by itself mean there is no credential: the
Anthropic SDK also resolves ANTHROPIC_AUTH_TOKEN and an `ant auth login`
profile. The script therefore tries the call rather than pre-judging from the
environment, and reports an auth failure distinctly from a code failure.

DIAGNOSTIC, NOT EVIDENCE. One query, one pass; per the 5-SEED RULE nothing here
is a finding. Claim text produced by a model is not a measurement.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.angles import TechAngle, load_config  # noqa: E402
from prismflow.v2.cache import init_cache_db  # noqa: E402
from prismflow.v2.connectors import ArxivConnector, GitHubConnector  # noqa: E402
from prismflow.v2.reasoners import (  # noqa: E402
    ClaudeReasoner,
    get_reasoner,
    provider_clusters,
)

QUERY = "adversarial robustness"


def _placeholder(value: str) -> bool:
    return not value or value.startswith(("sk-ant-...", "sk-...", "ghp_...")) or value == "..."


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    db = os.path.join(
        tempfile.mkdtemp(prefix="prismflow_reasoners_manual_"), "manual.db"
    )
    init_cache_db(db)

    print("provider assignment (angles sharing a model are not independent):")
    for provider, angles in sorted(provider_clusters().items()):
        print(f"    {provider:8s} {', '.join(angles)}")
    print()

    # --- Part 19: retrieve real evidence --------------------------------
    config = load_config("configs/v2/angle_defaults.yaml")
    github, arxiv = GitHubConnector(), ArxivConnector()
    angle = TechAngle(
        github, arxiv, config=config["tech"], use_cache=True, cache_db_path=db,
    )
    try:
        evidence = await angle.retrieve(QUERY)
    finally:
        await angle.aclose()
    print(f"evidence: {evidence.summary()}")
    if evidence.is_empty:
        print("FAILED: no evidence retrieved; cannot exercise the reasoner")
        return 1

    # --- Part 20: the prompt, which needs no credential ------------------
    reasoner = ClaudeReasoner("tech")
    xml = reasoner.format_evidence_xml(evidence)
    print(f"\nevidence XML: {len(xml)} chars, "
          f"{xml.count('<record ')} record(s), model {reasoner.model}")
    assert "<angle_evidence" in xml
    print("    first 3 lines:")
    for line in xml.splitlines()[:3]:
        print(f"      {line}")

    # --- the live call ---------------------------------------------------
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if _placeholder(key) and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("\n" + "=" * 68)
        print("UNVERIFIED: no Anthropic credential is present, so the request")
        print("shape (model id, thinking parameter, usage fields, content")
        print("blocks) has NOT been checked against the live API.")
        print("Everything above ran; everything below did not.")
        print("To verify:  export ANTHROPIC_API_KEY=sk-ant-...  and re-run.")
        print("=" * 68)
        return 0

    print(f"\ncalling {reasoner.PROVIDER}/{reasoner.model}...")
    result = await get_reasoner("tech").reason(evidence)
    print(f"    {result.summary()}")
    if not result.succeeded:
        print(f"    FAILED: {result.error}")
        for line in result.audit_trail:
            print(f"      audit: {line}")
        return 1

    known = {r.id for r in evidence.ranked_records}
    for index, claim in enumerate(result.claims, 1):
        print(f"\n  claim {index} [confidence {claim.confidence:.2f}]")
        print(f"    {claim.text}")
        print(f"    cites: {claim.cited_ids}")
        for conflict in claim.conflicts_noted:
            print(f"    conflict: {conflict}")
        for caveat in claim.caveats:
            print(f"    caveat: {caveat}")
        if not set(claim.cited_ids) <= known:
            print("    FAILED: a citation is not in the evidence")
            return 1

    print("\naudit trail:")
    for line in result.audit_trail:
        print(f"    {line}")
    print(f"\nclaims: {result.total_claims()}, all citations grounded")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
