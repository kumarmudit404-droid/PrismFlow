# /part17 — Connector Interface (PrismFlow V2)

**Depends:** V1 sealed (v1-final tag), all V2 dependencies installed  
**Estimate:** 4 hours  
**Difficulty:** Medium  
**Gate:** All tests pass; no modification to V1 files

---

## Context

PrismFlow V2 uses independent external data sources per angle (Tech, Market, Financial, Regulatory, Sentiment) to achieve data-layer independence. Each source is accessed via a **Connector** — a standardized interface with fetch, parse, and normalize steps.

**Frozen discipline:** Do NOT modify `prismflow/v1/`, `prismflow/models/`, `prismflow/statistics/`, or `prismflow/evaluation/`. They are sealed. Defects in V1 are reported, not patched.

---

## Goals

1. Define `AngleConnector` abstract base class in `prismflow/v2/connectors/base.py`
   - Methods: `fetch(query: str, k: int) -> RawResponse`, `parse(raw) -> List[Record]`, `normalize(records) -> List[NormalizedRecord]`
   - Rate limiting, retry logic, user-agent headers
   - Dependency injection for HTTPClient

2. Implement two concrete connectors:
   - `GitHubConnector` — query GitHub REST API (`/search/repositories`, `/search/code`)
   - `ArxivConnector` — query arXiv Atom feed (no API key needed)

3. Define data structures:
   - `RawResponse` — (status_code, headers, body_text)
   - `Record` — {id, title, url, snippet, source_name, fetched_at}
   - `NormalizedRecord` — {id, title, url, snippet_tokens, source, published_date, author, relevance_score}

4. Write tests:
   - Unit tests for parse/normalize logic (mock HTTP)
   - Integration test for GitHubConnector (live API, 2–3 queries, use GITHUB_TOKEN from .env)
   - Integration test for ArxivConnector (live API, 2–3 queries)

5. Create fixture factories for testing downstream parts (Part 19, 20, 21)

---

## Specifications

### `AngleConnector` (ABC)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import asyncio

@dataclass
class RawResponse:
    status_code: int
    headers: dict
    body_text: str

@dataclass
class Record:
    id: str                    # Unique within source
    title: str
    url: str
    snippet: str              # Raw text fragment
    source_name: str          # "github", "arxiv", etc.
    fetched_at: datetime

@dataclass
class NormalizedRecord:
    id: str
    title: str
    url: str
    snippet_tokens: int       # Token count (will be refined in Part 19)
    source: str
    published_date: Optional[datetime]
    author: Optional[str]
    relevance_score: float    # [0, 1] — initially all 1.0; reranked in Part 19

class AngleConnector(ABC):
    def __init__(self, name: str, rate_limit_per_min: int = 30):
        self.name = name
        self.rate_limit = rate_limit_per_min
        self.last_request_time = 0.0
        self.request_count = 0
    
    @abstractmethod
    async def fetch(self, query: str, k: int) -> RawResponse:
        """
        Fetch up to k results for query. Handle rate limiting (backoff if needed).
        Raise HTTPError for 4xx/5xx; ConnectionError for network issues.
        """
        pass
    
    @abstractmethod
    def parse(self, raw: RawResponse) -> List[Record]:
        """Parse raw response into typed Record objects. Raise ValueError if malformed."""
        pass
    
    @abstractmethod
    def normalize(self, records: List[Record]) -> List[NormalizedRecord]:
        """Normalize Records: extract author, date, count tokens, set relevance."""
        pass
    
    async def get_records(self, query: str, k: int) -> List[NormalizedRecord]:
        """Orchestrate: fetch → parse → normalize. Re-raise exceptions."""
        raw = await self.fetch(query, k)
        records = self.parse(raw)
        normalized = self.normalize(records)
        return normalized
```

### `GitHubConnector`

- Rate limit: 30 req/min for unauthenticated; 60 req/min with token. Enforce via `time.sleep()` or async semaphore.
- Endpoints:
  - `/search/repositories?q={query}&sort=stars&order=desc` (k results)
  - Optional: `/search/code?q={query}&sort=stars` for code snippets
- Extract: `id` (repo.id), `title` (repo.name), `url` (repo.html_url), `snippet` (repo.description or top README chunk), `author` (repo.owner.login), `published_date` (repo.created_at)
- Handle: 422 Unprocessable Entity (too many results → narrow), 401 Unauthorized (no token), 403 Forbidden (rate limit hit → backoff exponentially)

### `ArxivConnector`

- Free API: https://arxiv.org/api/query?search_query={query}&start=0&max_results=k&sortBy=relevance&sortOrder=descending
- Returns Atom XML; parse with `xml.etree.ElementTree` or `feedparser`
- Extract: `id` (arxiv_id), `title`, `url` (arxiv.org/abs/{id}), `snippet` (summary[:500]), `author` (first author), `published_date` (published field)
- Handle: connection timeouts (backoff), malformed XML (ValueError)

---

## Deliverables

### File structure
```
prismflow/v2/
├── connectors/
│   ├── __init__.py
│   ├── base.py          # AngleConnector ABC, dataclasses
│   ├── github.py        # GitHubConnector
│   ├── arxiv.py         # ArxivConnector
│   └── errors.py        # ConnectorError, RateLimitError, ParseError

tests/v2/
├── __init__.py
├── conftest.py          # Fixtures (mock_http_client, github_fixture_factory, etc.)
├── test_connectors.py   # Unit + integration tests

experiments/v2/
├── test_connectors_manual.py  # (Optional) live API smoke test with logging
```

### Tests to write

1. **test_parse_github_response()** — mock HTTP, call parse, verify Records
2. **test_parse_arxiv_response()** — mock Atom XML, verify extraction
3. **test_normalize_records()** — verify token counts, dates, author extraction
4. **test_github_connector_live()** — query "machine learning" with GITHUB_TOKEN, k=5, verify 5 records returned
5. **test_arxiv_connector_live()** — query "adversarial robustness", k=5, verify 5 records returned
6. **test_rate_limiting()** — spawn 10 rapid requests, measure time, assert backoff applied
7. **test_error_handling()** — mock 401, 403, 422, 429 responses; verify exceptions raised/retried

### Fixture factories (for downstream parts)

```python
# tests/v2/conftest.py

@pytest.fixture
def github_records_fixture():
    """5 sample GitHub records (normalized) for testing."""
    return [...]  # Pre-created NormalizedRecord list

@pytest.fixture
def arxiv_records_fixture():
    """5 sample arXiv records (normalized) for testing."""
    return [...]  # Pre-created NormalizedRecord list

@pytest.fixture
def mock_github_connector():
    """Mock GitHubConnector that returns fixture records without HTTP."""
    ...
```

---

## Standing Rules

1. **No V1 modifications.** Sealed. If a V1 bug emerges, report in KNOWN_LIMITATIONS.md; do not patch.
2. **Document rate limits.** Log every request; include backoff duration in stderr/logging for Part 19 to observe.
3. **Token counts (tiktoken).** Use `tiktoken.encoding_for_model("gpt-3.5-turbo")` to count snippet tokens; store in NormalizedRecord.
4. **Error handling.** Distinguish user error (invalid query) from system error (API down, rate limit). Log fully.
5. **Tests pass before closing.** `pytest tests/v2/test_connectors.py -v` must show 7/7 passing.

---

## Success Criteria

- [ ] `AngleConnector` ABC created with all three methods
- [ ] `GitHubConnector` and `ArxivConnector` implemented and tested
- [ ] Live API integration tests pass (GitHub, arXiv queries return ≥ 1 result)
- [ ] Rate limiting tested and working
- [ ] Fixture factories created and usable by Parts 18–24
- [ ] No V1 files modified
- [ ] All tests pass: `pytest tests/v2/ -v` → 7/7 passing

---

## Notes

- **Async vs. sync:** Use `async/await` (asyncio) for HTTP to enable concurrent fetches in Part 19. Start simple with `aiohttp.ClientSession`.
- **Credentials:** Read GITHUB_TOKEN from `os.getenv("GITHUB_TOKEN")` inside GitHubConnector.__init__(); handle None gracefully (falls back to unauthenticated, slower rate limit).
- **Timeouts:** Set 10s connection timeout, 30s read timeout in aiohttp/requests.
- **User-Agent:** Always include; some APIs reject requests without one. Use `"PrismFlow/2.0 (Mudit Kumar, kumarmudit404-droid/PrismFlow)"`.

---

## Checklist for Closing

- [ ] code committed to git
- [ ] git log shows new files in prismflow/v2/ and tests/v2/
- [ ] `pytest tests/v2/test_connectors.py -v` passes cleanly
- [ ] GITHUB_TOKEN and other env vars set in .env (not in code)
- [ ] Manual smoke test: run a single query with each connector; log output
- [ ] Ready for Part 18 (Cache Layer)
