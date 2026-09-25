"""Live smoke test for the NewsAPI connector (Market angle).

Not a pytest module despite the name, matching the Part 17 manual-script
convention. Run it directly:

    .venv\\Scripts\\python.exe experiments/v2/test_newsapi_manual.py

CONNECTIVITY VERIFICATION, not evidence. One query per case on a single pass,
nothing aggregated, no seeds. docs/CONTRACT.md section 5 requires five seeds
and mean/sd for anything that counts as a finding, and confirming that bytes
arrive is not that. It writes NO results JSON, because a file under results/
would be citable as a Part 24 measurement and this is not one.

QUOTA: the free Developer plan allows 100 requests per DAY, and NewsAPI reports
no remaining quota in any header or body field -- unlike StackExchange, there
is nothing to print. This run costs 4 requests plus 1 for the angle pass, so 5
of 100. The connector's own request tally is printed instead, with the caveat
that it counts this process only.

``through_the_angle`` needs the wiring commit (``market_angle.default_connectors``)
to be present; the same was true of the StackExchange smoke script against the
sentiment angle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.angles import build_angles  # noqa: E402
from prismflow.v2.angles.market_angle import default_connectors  # noqa: E402
from prismflow.v2.connectors import ConnectorError, NewsAPIConnector  # noqa: E402
from prismflow.v2.connectors.newsapi import (  # noqa: E402
    ARTICLE_DELAY_HOURS,
    DAILY_REQUEST_CAP,
)

QUERIES = [
    "inflation",
    "semiconductor export controls",
    "Zinsen@de",          # a different language, a different cache key
    "intelligence artificielle@fr",
]


async def smoke() -> int:
    connector = NewsAPIConnector()
    if not connector.api_key:
        print("NEWSAPI_KEY is not set (or is still a template placeholder).")
        print("Nothing to smoke-test; this is not a connector failure.")
        return 1
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
            print(f"  totalResults:     {connector.total_results} "
                  f"(corpus size, not page size)")
            print(f"  staleness:        "
                  f"{_fmt(connector.staleness_hours)}h for the freshest record "
                  f"(documented delay {ARTICLE_DELAY_HOURS:.0f}h)")
            print(f"  requests so far:  {connector.request_count} of "
                  f"{DAILY_REQUEST_CAP}/day -- this process only, NewsAPI "
                  f"reports no remaining quota")
            if not records:
                print("  (zero is a real answer -- query not rewritten)")
                continue
            record = records[0]
            print("  example NormalizedRecord:")
            print(f"    id              {_short(record.id)}")
            print(f"    title           {_short(record.title, 72)}")
            print(f"    url             {_short(record.url)}")
            print(f"    snippet         {_short(record.snippet, 120)}")
            print(f"    snippet_tokens  {record.snippet_tokens}")
            print(f"    source          {record.source!r}")
            print(f"    published_date  {record.published_date!r}")
            print(f"    author          {_short(record.author, 40)}")
            print(f"    relevance_score {record.relevance_score}")
    finally:
        await connector.aclose()
    return failures


async def through_the_angle() -> None:
    """The same connector reached through MarketAngle, as Part 19 will."""
    print("\n=== market angle, end to end ===")
    # default_connectors() constructs a connector the angle does not own, so
    # this scope closes it -- otherwise aiohttp logs an unclosed-session ERROR
    # at interpreter exit, which reads like a failure and is not one.
    chain = default_connectors()
    angles = build_angles(sources={"market": chain})
    try:
        evidence = await angles["market"].retrieve("inflation")
    finally:
        for connector in chain:
            if connector is not None:
                await connector.aclose()
    print(f"  angle_name       {evidence.angle_name!r}")
    print(f"  provenance       {evidence.provenance!r}")
    print(f"  raw records      {len(evidence.raw_records)}")
    print(f"  ranked records   {len(evidence.ranked_records)}")
    print(f"  total tokens     {evidence.total_tokens} (budget 1500)")
    print(f"  warnings         {evidence.warnings}")
    if evidence.ranked_records:
        top = evidence.ranked_records[0]
        print(f"  top record       {_short(top.id)} -- {_short(top.snippet, 80)}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _short(text, limit: int = 88) -> str:
    """Repr, ascii-safe: this console is cp1252 and headlines carry emoji."""
    text = "" if text is None else str(text)
    return repr(text[:limit].encode("ascii", "replace").decode("ascii"))


async def main() -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    failures = await smoke()
    if not failures:
        await through_the_angle()
    print(
        "\nconnectivity verification only -- not a finding, no results JSON "
        "written"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
