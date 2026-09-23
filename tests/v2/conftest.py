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
    """Five normalized GitHub records. Stable across runs.

    ``snippet_tokens`` is a fixed number rather than ``count_tokens(snippet)``
    on purpose. Part 19 budgets against the count and Part 21 needs identical
    fixtures on every machine, but ``count_tokens`` falls back to a character
    heuristic when tiktoken cannot be reached -- so deriving the count here
    would make the fixtures vary with network conditions. Tests that need the
    two to agree should compute it themselves.

    The snippets are worded so a query like "adversarial robustness" matches
    some records far better than others; a reranking test against a corpus of
    interchangeable text cannot show that reranking does anything.
    """
    rows = [
        ("1296269", "hello-world", "octocat", 12, _dt(2011, 1, 26),
         "My first repository on GitHub, for testing purposes only."),
        ("7654321", "transformers", "acme", 9, _dt(2019, 6, 2),
         "State of the art transformer models for natural language "
         "processing and text classification."),
        ("1010101", "flax", "google", 7, _dt(2020, 2, 14),
         "A neural network library for JAX designed for flexibility and "
         "functional programming."),
        ("2020202", "vllm", "vllm-project", 15, _dt(2023, 3, 9),
         "A high throughput and memory efficient inference and serving "
         "engine for large language models."),
        ("3030303", "keploy", "keploy", 11, _dt(2021, 8, 30),
         "Adversarial robustness testing toolkit that generates test cases "
         "and mocks from real traffic."),
    ]
    return [
        NormalizedRecord(
            id=rid,
            title=title,
            url=f"https://github.com/{author}/{title}",
            snippet=snippet,
            snippet_tokens=tokens,
            source="github",
            published_date=published,
            author=author,
            relevance_score=1.0,
        )
        for rid, title, author, tokens, published, snippet in rows
    ]


def make_arxiv_records() -> List[NormalizedRecord]:
    """Five normalized arXiv records. Stable across runs."""
    rows = [
        ("1412.6572v3", "Explaining and Harnessing Adversarial Examples",
         "Ian J. Goodfellow", 24, _dt(2014, 12, 20),
         "Several machine learning models are vulnerable to adversarial "
         "examples, inputs formed by applying small perturbations."),
        ("1706.06083v4", "Towards Deep Learning Models Resistant to "
         "Adversarial Attacks", "Aleksander Madry", 19, _dt(2017, 6, 19),
         "We study the adversarial robustness of neural networks through the "
         "lens of robust optimization and min-max training."),
        ("1312.6199v4", "Intriguing properties of neural networks",
         "Christian Szegedy", 17, _dt(2013, 12, 21),
         "Deep neural networks learn input-output mappings that are fairly "
         "discontinuous to a significant extent."),
        ("1802.00420v2", "Obfuscated Gradients Give a False Sense of "
         "Security", "Anish Athalye", 22, _dt(2018, 2, 1),
         "We identify obfuscated gradients, a phenomenon that leads to a "
         "false sense of adversarial robustness in defences."),
        ("2006.11239v2", "Denoising Diffusion Probabilistic Models",
         "Jonathan Ho", 14, _dt(2020, 6, 19),
         "We present high quality image synthesis results using diffusion "
         "probabilistic models and a variational bound."),
    ]
    return [
        NormalizedRecord(
            id=rid,
            title=title,
            url=f"https://arxiv.org/abs/{rid}",
            snippet=snippet,
            snippet_tokens=tokens,
            source="arxiv",
            published_date=published,
            author=author,
            relevance_score=1.0,
        )
        for rid, title, author, tokens, published, snippet in rows
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


# --- Part 21: planted-redundancy claim corpus ---------------------------
#
# Synthetic claims used to probe the dependence estimators under known
# conditions. They are PROBES, not model output and not results: the point of a
# planted-redundancy experiment is that the true answer is known in advance, so
# the input has to be constructed rather than observed. Nothing here is ever
# reported as a finding about the world.
#
# The corpus is a pool rather than a single hardcoded pair because the Part 21
# brief's experiment loops over five seeds while varying nothing -- its claims
# are literals and the encoder is deterministic, so every seed returns the same
# number and the reported standard deviation is 0.0 by construction. Seeds have
# to select something for a mean and sd to mean anything; here they select which
# topics are used, which findings within a topic, and which unrelated topic is
# paired against which.
#
# Each topic carries two findings, each phrased twice. The two phrasings say the
# same thing in deliberately different words, so the paraphrase condition tests
# semantic similarity rather than string overlap.

_TOPIC_CORPUS = [
    ("vector databases",
     (("Vector database adoption is consolidating around a few open-source engines.",
       "A small number of open-source engines now account for most vector search deployments."),
      ("Hybrid keyword and vector retrieval beats pure vector search on recall.",
       "Combining lexical matching with embeddings retrieves more relevant documents than embeddings alone."))),
    ("model quantisation",
     (("Four-bit quantisation preserves accuracy on most instruction-following benchmarks.",
       "Cutting weights to four bits leaves benchmark scores for instruction tasks largely intact."),
      ("Quantised inference shifts the bottleneck from compute to memory bandwidth.",
       "After quantisation, memory throughput rather than arithmetic limits serving speed."))),
    ("container orchestration",
     (("Kubernetes operator patterns have replaced bespoke deployment scripts.",
       "Custom deployment scripting has given way to operator-based control loops."),
      ("Cluster autoscaling reduces idle capacity but increases cold-start latency.",
       "Scaling nodes on demand trims wasted capacity at the cost of slower first responses."))),
    ("static analysis",
     (("Incremental type checking makes large codebases tractable for gradual typing.",
       "Checking only changed files lets gradual typing scale to very large repositories."),
      ("False positive rates dominate developer adoption of static analysers.",
       "Whether engineers keep using a linter is driven mainly by how often it cries wolf."))),
    ("differential privacy",
     (("Privacy budgets are rarely accounted for across repeated queries in practice.",
       "Deployments seldom track cumulative epsilon spend over many releases."),
      ("Adding calibrated noise degrades utility most sharply on small subgroups.",
       "Minority cohorts lose the most accuracy when privacy noise is applied."))),
    ("build systems",
     (("Remote caching cuts median build times more than parallelism does.",
       "Sharing cached artefacts shortens typical builds further than adding workers."),
      ("Non-hermetic builds are the main cause of cache misses at scale.",
       "Builds that leak environment state are why large caches fail to hit."))),
    ("time series forecasting",
     (("Gradient-boosted trees remain competitive with deep models on tabular forecasts.",
       "Boosted decision trees still match neural networks for tabular time series."),
      ("Forecast accuracy degrades fastest around regime changes.",
       "Predictions break down most severely when the underlying process shifts."))),
    ("API versioning",
     (("Consumers migrate off deprecated endpoints only when forced by shutdown dates.",
       "Callers move to new API versions when a sunset deadline compels them, not before."),
      ("Additive schema changes cause fewer breakages than field renames.",
       "Adding fields is far safer for clients than renaming existing ones."))),
    ("graph neural networks",
     (("Message passing depth beyond three layers yields diminishing returns.",
       "Stacking more than about three propagation steps adds little accuracy."),
      ("Sampling neighbourhoods is required for graphs that exceed device memory.",
       "Large graphs must subsample adjacency to fit in accelerator memory."))),
    ("observability",
     (("Trace sampling strategy determines whether rare failures are ever observed.",
       "Whether uncommon faults appear in telemetry depends on how traces are sampled."),
      ("Cardinality growth in labels is the main driver of metrics storage cost.",
       "Unbounded label values are what makes time series storage expensive."))),
    ("federated learning",
     (("Client drift under non-identical data distributions slows convergence.",
       "Heterogeneous local datasets pull updates apart and lengthen training."),
      ("Secure aggregation adds communication overhead that dominates small models.",
       "For small networks the cost of private aggregation outweighs the training traffic."))),
    ("compiler optimisation",
     (("Profile-guided optimisation gives larger gains than aggressive inlining alone.",
       "Using runtime profiles beats inlining heuristics for end-to-end speed."),
      ("Auto-vectorisation fails silently on loops with unpredictable control flow.",
       "Loops with data-dependent branches quietly miss out on vector instructions."))),
    ("reinforcement learning",
     (("Reward misspecification is a more common failure than exploration collapse.",
       "Badly specified objectives break agents more often than insufficient exploration does."),
      ("Offline policy evaluation is unreliable without overlap in the behaviour policy.",
       "Estimating a new policy from logged data fails when the logging policy never tried those actions."))),
    ("edge computing",
     (("Inference at the edge is limited by thermal budget rather than raw compute.",
       "Heat dissipation, not processor speed, caps what edge devices can run."),
      ("Intermittent connectivity forces reconciliation logic into every client.",
       "Unreliable networks mean each device must resolve conflicting state itself."))),
    ("supply chain security",
     (("Dependency confusion attacks exploit resolution order rather than code flaws.",
       "These attacks abuse how package managers choose a registry, not bugs in the packages."),
      ("Reproducible builds make tampering detectable but are rarely achieved.",
       "Bit-identical rebuilds would reveal interference, yet few projects manage them."))),
    ("database indexing",
     (("Learned indexes outperform B-trees only on predictable key distributions.",
       "Model-based indexes beat balanced trees when key layouts are smooth and stable."),
      ("Write amplification limits index density on log-structured storage.",
       "How much data each write rewrites bounds how dense indexes can get on LSM engines."))),
    ("speech recognition",
     (("Word error rate understates failure on accented and code-switched speech.",
       "Aggregate error rates hide how badly systems handle accents and mixed languages."),
      ("Streaming recognition trades accuracy for latency through limited right context.",
       "Real-time transcription sees less future audio and is less accurate as a result."))),
    ("property testing",
     (("Shrinking quality determines whether a failing case is actionable.",
       "How well a generator minimises a counterexample decides if developers can debug it."),
      ("Stateful property tests find concurrency bugs that unit tests miss.",
       "Model-based sequential testing surfaces race conditions that example tests never reach."))),
    ("recommendation systems",
     (("Popularity bias compounds when feedback loops are not corrected.",
       "Uncorrected feedback makes already-popular items steadily more dominant."),
      ("Offline ranking metrics correlate weakly with online engagement.",
       "Improvements in held-out ranking scores often fail to show up in live tests."))),
    ("memory safety",
     (("Rewriting parsers in memory-safe languages removes most exploitable bugs.",
       "Porting input-handling code to safe languages eliminates the majority of vulnerabilities."),
      ("Unsafe blocks concentrate risk rather than eliminating it.",
       "Escape hatches localise danger but do not remove it from the program."))),
    ("data labelling",
     (("Annotator disagreement is signal about task ambiguity, not just noise.",
       "When labellers differ it often reveals that the task itself is underspecified."),
      ("Active learning gains shrink once the labelled pool is broadly representative.",
       "Selecting informative examples stops helping once coverage is already good."))),
    ("serverless computing",
     (("Cold start cost dominates for infrequently invoked functions.",
       "Rarely called functions spend most of their latency budget on initialisation."),
      ("Per-invocation billing penalises long-running synchronous workloads.",
       "Charging by call makes sustained synchronous jobs expensive."))),
    ("code review",
     (("Review latency predicts defect escape rate better than review depth.",
       "How long a change waits for review forecasts escaped bugs more than how thoroughly it is read."),
      ("Large changesets receive systematically shallower review.",
       "The bigger the diff, the less carefully each line is actually examined."))),
    ("model evaluation",
     (("Benchmark contamination inflates reported scores on public test sets.",
       "Training data overlap with public benchmarks makes published numbers look better than they are."),
      ("Single-number leaderboards hide per-slice regressions.",
       "Aggregate rankings conceal subgroups where performance has got worse."))),
]

#: Conditions the planted-redundancy experiment measures.
PLANTED_CONDITIONS = ("identical", "paraphrase", "unrelated", "low_rank")


def topic_count() -> int:
    """How many distinct topics the corpus holds."""
    return len(_TOPIC_CORPUS)


def topic_name(topic_index: int) -> str:
    return _TOPIC_CORPUS[topic_index % len(_TOPIC_CORPUS)][0]


def topic_claims(topic_index: int, *, paraphrase: bool = False):
    """The claims for one topic, in base or paraphrased phrasing.

    Both phrasings cite the SAME record ids: a paraphrase of a finding rests on
    the evidence the finding rested on. That is what makes the paraphrase
    condition a test of semantic similarity rather than of citation overlap.
    """
    from prismflow.v2.reasoners.models import Claim

    name, findings = _TOPIC_CORPUS[topic_index % len(_TOPIC_CORPUS)]
    claims = []
    for finding_index, phrasings in enumerate(findings):
        claims.append(
            Claim(
                text=phrasings[1 if paraphrase else 0],
                confidence=0.8,
                cited_ids=[
                    f"t{topic_index}-r{finding_index}",
                    f"t{topic_index}-r{finding_index + 1}",
                ],
                conflicts_noted=[],
                caveats=[f"synthetic probe for {name}"],
            )
        )
    return claims


def make_claimset(
    angle_name: str,
    query_text: str,
    claims,
    *,
    provider: str = "claude",
    model_name: str = "claude-sonnet-5",
):
    """A ClaimSet wrapping given claims, with plausible non-claim fields."""
    from prismflow.v2.reasoners.models import ClaimSet

    return ClaimSet(
        angle_name=angle_name,
        query_text=query_text,
        claims=list(claims),
        provider=provider,
        model_name=model_name,
        tokens_input=100,
        tokens_output=50,
        latency_seconds=1.0,
    )


def planted_pair(condition: str, topic_index: int, other_topic_index: int):
    """Two ClaimSets standing in a known relationship.

    ``identical``  angle B repeats angle A exactly -- perfect redundancy.
    ``paraphrase`` angle B restates A's findings in different words, citing the
                   same records.
    ``unrelated``  angle B reports a different topic entirely, citing records
                   from that topic, so the id spaces are disjoint.
    ``low_rank``   the same inputs as ``identical``; the condition is applied to
                   the ESTIMATOR (truncated embeddings), not to the claims.
    """
    name = topic_name(topic_index)
    left = make_claimset("tech", name, topic_claims(topic_index))

    if condition in ("identical", "low_rank"):
        right_claims = topic_claims(topic_index)
    elif condition == "paraphrase":
        right_claims = topic_claims(topic_index, paraphrase=True)
    elif condition == "unrelated":
        right_claims = topic_claims(other_topic_index)
    else:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of "
            f"{', '.join(PLANTED_CONDITIONS)}"
        )

    right = make_claimset(
        "market", name, right_claims, provider="openai", model_name="gpt-4o",
    )
    return left, right
