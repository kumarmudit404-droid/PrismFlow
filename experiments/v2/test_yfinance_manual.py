"""Live smoke test for the yfinance connector (TASK 2a).

Not a pytest module despite the name, matching the Part 17 manual-script
convention. Run it directly:

    .venv\\Scripts\\python.exe experiments/v2/test_yfinance_manual.py

This is CONNECTIVITY VERIFICATION, not evidence. One query per case on a
single pass, nothing aggregated, no seeds. docs/CONTRACT.md section 5 requires
five seeds and mean/sd for anything that counts as a finding, and confirming
that bytes arrive is not that. It deliberately writes NO results JSON, because
a file under results/ would be citable as a Part 24 measurement and this is
not one. Nothing printed here should ever be quoted as a result.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.angles import build_angles  # noqa: E402
from prismflow.v2.angles.financial_angle import default_connectors  # noqa: E402
from prismflow.v2.connectors import ConnectorError, YFinanceConnector  # noqa: E402

QUERIES = [
    "nvidia",
    "semiconductor",
    "NVDA@6mo",
    "artificial intelligence chips",  # expected: zero, see module docstring
]


async def smoke() -> int:
    connector = YFinanceConnector()
    failures = 0
    try:
        for query in QUERIES:
            print(f"\n=== {query!r} ===")
            try:
                records = await connector.get_records(query, k=3)
            except ConnectorError as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                failures += 1
                continue

            print(f"  raw record count: {len(records)}")
            if not records:
                print("  (zero is a real answer here -- query not rewritten)")
                continue
            record = records[0]
            print("  example NormalizedRecord:")
            print(f"    id              {record.id!r}")
            print(f"    title           {record.title!r}")
            print(f"    url             {record.url!r}")
            print(f"    snippet         {record.snippet!r}")
            print(f"    snippet_tokens  {record.snippet_tokens}")
            print(f"    source          {record.source!r}")
            print(f"    published_date  {record.published_date!r}")
            print(f"    author          {record.author!r}")
            print(f"    relevance_score {record.relevance_score}")
    finally:
        await connector.aclose()
    return failures


async def through_the_angle() -> None:
    """The same connector reached through FinancialAngle, as Part 19 will."""
    print("\n=== financial angle, end to end ===")
    angles = build_angles(sources={"financial": default_connectors()})
    evidence = await angles["financial"].retrieve("nvidia")
    print(f"  angle_name       {evidence.angle_name!r}")
    print(f"  provenance       {evidence.provenance!r}")
    print(f"  raw records      {len(evidence.raw_records)}")
    print(f"  ranked records   {len(evidence.ranked_records)}")
    print(f"  total tokens     {evidence.total_tokens}")
    print(f"  warnings         {evidence.warnings}")
    if evidence.ranked_records:
        top = evidence.ranked_records[0]
        print(f"  top record       {top.id!r} -- {top.snippet[:80]!r}")


async def main() -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    failures = await smoke()
    await through_the_angle()
    print(
        "\nconnectivity verification only -- not a finding, no results JSON "
        "written"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
