"""Live smoke test for the Part 17 connectors.

Not a pytest module despite the name (the Part 17 brief names the file). Run
it directly to see real traffic, real throttling and real records:

    .venv\\Scripts\\python.exe experiments/v2/test_connectors_manual.py

This is DIAGNOSTIC, not evidence. It runs one query per source on a single
pass and reports nothing aggregated; docs/CONTRACT.md section 5 requires five
seeds and mean/sd for anything that counts as a finding, and a connector
smoke test is not that. Nothing here should ever be quoted as a result.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.connectors import (  # noqa: E402
    ArxivConnector,
    ConnectorError,
    GitHubConnector,
)

QUERIES = [
    ("github", "machine learning"),
    ("github", "evidential deep learning"),
    ("arxiv", "adversarial robustness"),
    ("arxiv", "multi-view fusion"),
]


def _build(source: str):
    return GitHubConnector() if source == "github" else ArxivConnector()


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    connectors = {"github": _build("github"), "arxiv": _build("arxiv")}
    gh = connectors["github"]
    print(f"GitHub auth: {'token' if gh.authenticated else 'UNAUTHENTICATED'} "
          f"({gh.rate_limit} req/min)")
    print(f"arXiv: no credential required ({connectors['arxiv'].rate_limit} "
          "req/min)\n")

    failures = 0
    try:
        for source, query in QUERIES:
            connector = connectors[source]
            print(f"--- {source}: {query!r}")
            try:
                records = await connector.get_records(query, k=3)
            except ConnectorError as exc:
                failures += 1
                print(f"    FAILED: {type(exc).__name__}: {exc}\n")
                continue
            if not records:
                failures += 1
                print("    FAILED: zero records returned\n")
                continue
            for record in records:
                date = (record.published_date.date()
                        if record.published_date else "unknown")
                print(f"    [{record.snippet_tokens:>4} tok] {record.title[:58]}")
                print(f"           {record.author or 'unknown'} | {date} | "
                      f"{record.url}")
            print()
    finally:
        for connector in connectors.values():
            await connector.aclose()

    total = sum(c.request_count for c in connectors.values())
    print(f"requests issued: {total}, failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
