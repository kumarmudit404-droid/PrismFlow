"""Shared fixtures for V2 tests, and the fixture factories Parts 18-24 reuse.

WHY THE FACTORIES ARE MODULE-LEVEL FUNCTIONS AND NOT ONLY FIXTURES
------------------------------------------------------------------
Later parts need these records outside a pytest fixture context -- Part 19
builds retrieval pipelines over them, Part 21 estimates dependence between
angles from them, and both have experiment scripts that are not tests. So the
data lives in plain functions (``make_github_records`` and friends) and the
pytest fixtures are thin wrappers. Import the function; do not copy the data.

The records are deliberately NOT randomised. Part 21 measures dependence
between angles, and a fixture whose contents shift per run would make that
measurement unreproducible in exactly the way docs/CONTRACT.md section 5
forbids.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence

import pytest

from prismflow.v2.connectors.base import (
    NormalizedRecord,
    RawResponse,
    Record,
)
from prismflow.v2.connectors.github import GitHubConnector


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test performs a real network request to an external API",
    )


# --- transport doubles --------------------------------------------------


class MockHTTPClient:
    """An HTTPClient that replays queued responses and records its calls.

    Satisfies the ``HTTPClient`` protocol structurally -- no inheritance --
    which is the point of making the transport a Protocol.
    """

    def __init__(self, responses: Sequence[RawResponse]) -> None:
        self._responses: List[RawResponse] = list(responses)
        self.calls: List[Dict[str, object]] = []

    async def get(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
    ) -> RawResponse:
        self.calls.append({"url": url, "headers": dict(headers or {})})
        if not self._responses:
            raise AssertionError(
                f"MockHTTPClient exhausted after {len(self.calls)} call(s); "
                "queue more responses"
            )
        if len(self._responses) == 1:
            return self._responses[0]  # last response repeats
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


def ok(body: str, **headers: str) -> RawResponse:
    return RawResponse(200, {k.lower(): v for k, v in headers.items()}, body)


def status(code: int, body: str = "", **headers: str) -> RawResponse:
    return RawResponse(code, {k.lower(): v for k, v in headers.items()}, body)


# --- raw bodies ---------------------------------------------------------


GITHUB_BODY = """
{
  "total_count": 2,
  "incomplete_results": false,
  "items": [
    {
      "id": 1296269,
      "name": "hello-world",
      "full_name": "octocat/hello-world",
      "html_url": "https://github.com/octocat/hello-world",
      "description": "My first repository on GitHub, for testing purposes.",
      "stargazers_count": 2310,
      "created_at": "2011-01-26T19:01:12Z",
      "owner": {"login": "octocat", "id": 583231}
    },
    {
      "id": 7654321,
      "name": "transformers",
      "full_name": "acme/transformers",
      "html_url": "https://github.com/acme/transformers",
      "description": null,
      "stargazers_count": 44,
      "created_at": "2019-06-02T08:15:00Z",
      "owner": {"login": "acme", "id": 99}
    }
  ]
}
"""

ARXIV_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/1412.6572v3</id>
    <updated>2015-03-20T18:36:58Z</updated>
    <published>2014-12-20T00:24:02Z</published>
    <title>Explaining and Harnessing
      Adversarial Examples</title>
    <summary>  Several machine learning models are vulnerable to
      adversarial examples.  </summary>
    <author><name>Ian J. Goodfellow</name></author>
    <author><name>Jonathon Shlens</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1706.06083v4</id>
    <published>2017-06-19T17:55:24Z</published>
    <title>Towards Deep Learning Models Resistant to Adversarial Attacks</title>
    <summary>Recent work has demonstrated that deep networks are
      vulnerable.</summary>
    <author><name>Aleksander Madry</name></author>
  </entry>
</feed>
"""


# --- normalized fixture factories (used by Parts 18-24) -----------------


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def make_github_records() -> List[NormalizedRecord]:
    """Five normalized GitHub records. Stable across runs."""
    rows = [
        ("1296269", "hello-world", "octocat", 12, _dt(2011, 1, 26)),
        ("7654321", "transformers", "acme", 9, _dt(2019, 6, 2)),
        ("1010101", "flax", "google", 7, _dt(2020, 2, 14)),
        ("2020202", "vllm", "vllm-project", 15, _dt(2023, 3, 9)),
        ("3030303", "keploy", "keploy", 11, _dt(2021, 8, 30)),
    ]
    return [
        NormalizedRecord(
            id=rid,
            title=title,
            url=f"https://github.com/{author}/{title}",
            snippet_tokens=tokens,
            source="github",
            published_date=published,
            author=author,
            relevance_score=1.0,
        )
        for rid, title, author, tokens, published in rows
    ]


def make_arxiv_records() -> List[NormalizedRecord]:
    """Five normalized arXiv records. Stable across runs."""
    rows = [
        ("1412.6572v3", "Explaining and Harnessing Adversarial Examples",
         "Ian J. Goodfellow", 24, _dt(2014, 12, 20)),
        ("1706.06083v4", "Towards Deep Learning Models Resistant to "
         "Adversarial Attacks", "Aleksander Madry", 19, _dt(2017, 6, 19)),
        ("1312.6199v4", "Intriguing properties of neural networks",
         "Christian Szegedy", 17, _dt(2013, 12, 21)),
        ("1802.00420v2", "Obfuscated Gradients Give a False Sense of "
         "Security", "Anish Athalye", 22, _dt(2018, 2, 1)),
        ("2006.11239v2", "Denoising Diffusion Probabilistic Models",
         "Jonathan Ho", 14, _dt(2020, 6, 19)),
    ]
    return [
        NormalizedRecord(
            id=rid,
            title=title,
            url=f"https://arxiv.org/abs/{rid}",
            snippet_tokens=tokens,
            source="arxiv",
            published_date=published,
            author=author,
            relevance_score=1.0,
        )
        for rid, title, author, tokens, published in rows
    ]


def make_raw_records(source: str = "github") -> List[Record]:
    """Un-normalized Records, for tests that exercise ``normalize`` directly."""
    fetched = _dt(2026, 9, 23)
    return [
        Record(
            id="1296269",
            title="hello-world",
            url="https://github.com/octocat/hello-world",
            snippet="My first repository on GitHub, for testing purposes.",
            source_name=source,
            fetched_at=fetched,
            extra={"author": "octocat", "published_date": _dt(2011, 1, 26)},
        ),
        Record(
            id="7654321",
            title="transformers",
            url="https://github.com/acme/transformers",
            snippet="",
            source_name=source,
            fetched_at=fetched,
            extra={"author": None, "published_date": None},
        ),
    ]


class MockGitHubConnector(GitHubConnector):
    """A GitHubConnector that never touches the network.

    Downstream parts that need records without HTTP should use this rather
    than monkeypatching ``fetch`` at each call site.
    """

    def __init__(self, records: Optional[Sequence[NormalizedRecord]] = None):
        super().__init__(token=None, http_client=MockHTTPClient([ok("{}")]))
        self._records = list(records) if records is not None \
            else make_github_records()

    async def get_records(self, query: str, k: int) -> List[NormalizedRecord]:
        return self._records[:k]


# --- pytest fixtures ----------------------------------------------------


@pytest.fixture
def github_records_fixture() -> List[NormalizedRecord]:
    """5 sample GitHub records (normalized)."""
    return make_github_records()


@pytest.fixture
def arxiv_records_fixture() -> List[NormalizedRecord]:
    """5 sample arXiv records (normalized)."""
    return make_arxiv_records()


@pytest.fixture
def raw_records_fixture() -> List[Record]:
    return make_raw_records()


@pytest.fixture
def mock_http_client():
    """Factory: ``mock_http_client([...responses])``."""
    return lambda responses: MockHTTPClient(responses)


@pytest.fixture
def mock_github_connector() -> MockGitHubConnector:
    """A GitHubConnector returning fixture records without HTTP."""
    return MockGitHubConnector()


@pytest.fixture
def run_async():
    """Run a coroutine to completion.

    V2 is async but the repo has no pytest-asyncio and V1 added no pytest
    plugins. Rather than take a new test dependency for Part 17, async code is
    driven through ``asyncio.run`` from ordinary sync tests.
    """
    def _run(coro):
        return asyncio.run(coro)
    return _run
