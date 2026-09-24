"""End-to-end fusion test: 30 queries x 4 domains x 5 seeds.

    .venv\\Scripts\\python.exe experiments/v2/test_fusion_e2e.py

This one IS evidence for one claim and NOT evidence for another, and the
distinction is the most important thing in this file.

IT IS EVIDENCE that the corrected conflict detector finds planted
contradictions and the brief's rule does not. The relationships are planted, so
the ground truth is known before measuring, and what is measured is the
DETECTOR.

IT IS NOT EVIDENCE about latency, cost, or live API behaviour. This checkout has
no Anthropic credentials, so no request is sent: the adjudicator is the offline
``LexicalAdjudicator`` and the ClaimSets are fixtures. Their token counts are
made up, so a dollar figure computed from them would be a fabricated
measurement. ``cost_usd`` is therefore reported as null with the reason
attached, alongside the rate table and the token counts a real run would
multiply. The latency reported is local fusion compute and is labelled as such.

CLAUDE.md's NEVER FABRICATE rule is why this file reports a null instead of a
number it could easily have produced.


WHAT THE SEEDS SELECT
---------------------
The brief's e2e test has no seeds at all, and its query list is
``[4 literals] * 6`` sliced to 30 -- which yields 24, not 30, so the headline
"30 queries" is wrong by six in its own code.

Here each seed draws 30 of the 40 banked queries without replacement, picks the
subject each query disputes, picks which two angles carry the planted
contradiction and which two carry the planted agreement, and draws every claim's
confidence from a band. The spread across seeds therefore describes variation in
content, in structure and in confidence, which is what a standard deviation over
seeds is supposed to describe.

TWO KINDS OF ZERO STANDARD DEVIATION
------------------------------------
The recall rows report 1.0000 +/- 0.0000 for the corrected detector and
0.0000 +/- 0.0000 for the brief's. Those zeros are NOT the defect described
above. The inputs varied across all 150 runs; the outcome was the same in every
one of them, because the effect is total in both directions. A zero from an
input that never changed is an absent measurement; a zero from 150 varying
inputs that all landed the same way is a result. The distinction is worth
stating because the two look identical in a results table.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.dependence import aggregate_dependence  # noqa: E402
from prismflow.v2.dependence.estimators import (  # noqa: E402
    EmbeddingEstimator,
    cosine,
)
from prismflow.v2.fusion import (  # noqa: E402
    MODEL_PRICING,
    LexicalAdjudicator,
    PrismFusion,
)
from prismflow.v2.reasoners.models import Claim, ClaimSet  # noqa: E402
from prismflow.v2.statistics import compute_semantic_eniv  # noqa: E402

N_SEEDS = 5
N_QUERIES = 30
ANGLE_NAMES = ["tech", "market", "financial", "regulatory", "sentiment"]
OUTPUT = Path("results/v2/fusion_e2e_test.json")

#: The brief's threshold and comparison, replicated for the contrast.
BRIEF_THRESHOLD = 0.3

#: Fixture token counts. Plausible, invented, and never turned into a dollar
#: figure -- see the module docstring.
FIXTURE_TOKENS_INPUT = 1800
FIXTURE_TOKENS_OUTPUT = 420

#: 40 queries, 10 per domain. Each seed draws 30.
QUERY_BANK: Dict[str, List[str]] = {
    "tech": [
        "adoption of rust in systems programming",
        "maturity of webassembly runtimes",
        "vector database performance at scale",
        "state of on-device model inference",
        "kubernetes operator complexity",
        "typescript adoption in backend services",
        "serverless cold start latency",
        "open source licence changes in infrastructure",
        "build tooling consolidation",
        "observability cost growth",
    ],
    "market": [
        "semiconductor demand this quarter",
        "enterprise software renewal rates",
        "cloud infrastructure price competition",
        "venture funding in developer tools",
        "consumer hardware shipment trends",
        "data centre capital expenditure",
        "subscription churn in saas",
        "advertising spend in technology media",
        "secondary market activity in private tech",
        "currency exposure for software exporters",
    ],
    "career": [
        "demand for machine learning engineers",
        "remote work policy in large technology firms",
        "junior developer hiring conditions",
        "compensation trends in platform engineering",
        "contractor versus permanent hiring",
        "internal mobility in engineering organisations",
        "certification value for cloud roles",
        "technical interview format changes",
        "attrition in site reliability teams",
        "graduate intake in software engineering",
    ],
    "policy": [
        "ai regulation in europe",
        "data residency requirements for cloud providers",
        "antitrust scrutiny of app stores",
        "export controls on advanced chips",
        "privacy enforcement actions this year",
        "accessibility mandates for public sector software",
        "open banking implementation deadlines",
        "content moderation obligations",
        "cross border data transfer rulings",
        "procurement rules for government software",
    ],
}

#: Planted claim triples per domain: an assertion, its direct reversal, and a
#: paraphrase of the assertion. The reversal is a CONTRADICTION with the
#: assertion; the paraphrase is an AGREEMENT with it.
SUBJECTS: Dict[str, List[Tuple[str, str, str]]] = {
    "tech": [
        (
            "Production adoption rose sharply over the last four quarters.",
            "Production adoption fell sharply over the last four quarters.",
            "Uptake in production increased steeply across the past four quarters.",
        ),
        (
            "Benchmark latency improved after the runtime migration.",
            "Benchmark latency worsened after the runtime migration.",
            "The runtime migration made measured latency better.",
        ),
        (
            "The toolchain is ready for production workloads.",
            "The toolchain is not ready for production workloads.",
            "The toolchain can be relied on for production use.",
        ),
    ],
    "market": [
        (
            "Quarterly revenue increased against the prior year.",
            "Quarterly revenue decreased against the prior year.",
            "Year over year quarterly revenue was up.",
        ),
        (
            "Unit shipments grew through the second half.",
            "Unit shipments shrank through the second half.",
            "Shipment volumes rose over the second half of the year.",
        ),
        (
            "Margins strengthened as input costs settled.",
            "Margins weakened as input costs settled.",
            "Margin performance improved once input costs stabilised.",
        ),
    ],
    "career": [
        (
            "Hiring in the specialism is accelerating.",
            "Hiring in the specialism is slowing.",
            "Recruitment for the specialism is picking up pace.",
        ),
        (
            "Median compensation rose for mid level roles.",
            "Median compensation fell for mid level roles.",
            "Mid level roles saw median pay increase.",
        ),
        (
            "Attrition improved after the policy change.",
            "Attrition worsened after the policy change.",
            "Staff retention got better following the policy change.",
        ),
    ],
    "policy": [
        (
            "The measure was approved by the legislature.",
            "The measure was rejected by the legislature.",
            "Lawmakers passed the measure.",
        ),
        (
            "Enforcement activity increased over the period.",
            "Enforcement activity decreased over the period.",
            "There was more enforcement action during the period.",
        ),
        (
            "The rule was enacted on schedule.",
            "The rule was repealed before taking effect.",
            "The rule came into force as planned.",
        ),
    ],
}

#: Filler claims, unrelated to any planted subject and to each other. These are
#: the pairs the brief's rule flags and the corrected rule correctly ignores.
FILLER = [
    "The published dataset covers twelve months of records.",
    "Documentation for the interface is versioned separately.",
    "Two of the cited sources are preprints rather than journal articles.",
    "Regional coverage in the sample is uneven.",
    "The measurement window excludes the holiday period.",
    "Reporting granularity changed midway through the series.",
    "Several records lack an explicit publication date.",
    "The sample is weighted toward larger organisations.",
    "Methodology notes were revised in the latest release.",
    "Coverage of smaller vendors is sparse.",
]


def make_claim(text: str, confidence: float, cid: str) -> Claim:
    return Claim(text=text, confidence=confidence, cited_ids=[cid])


def make_claimset(angle: str, query: str, claims: Sequence[Claim]) -> ClaimSet:
    return ClaimSet(
        angle_name=angle,
        query_text=query,
        claims=list(claims),
        provider="claude",
        model_name="claude-sonnet-5",
        tokens_input=FIXTURE_TOKENS_INPUT,
        tokens_output=FIXTURE_TOKENS_OUTPUT,
        latency_seconds=1.0,
    )


def build_case(
    rng: np.random.Generator, domain: str, query: str
) -> Tuple[List[ClaimSet], Tuple[str, str], Tuple[str, str]]:
    """Five angles with one planted contradiction and one planted agreement.

    Returns ``(claimsets, contradiction_pair, agreement_pair)`` where each pair
    is the two claim TEXTS that stand in that relationship.
    """
    assertion, reversal, paraphrase = SUBJECTS[domain][
        int(rng.integers(len(SUBJECTS[domain])))
    ]

    # Which angles carry which relationship varies per query and per seed.
    order = list(rng.permutation(ANGLE_NAMES))
    contra_a, contra_b, agree_a, agree_b, _spare = order

    # Confidences are DRAWN, not fixed. The first version of this file used the
    # literals 0.80/0.75/0.70/0.65/0.55, which meant every query carried the
    # identical multiset of confidences and the reported standard deviation of
    # undiscounted_confidence came out at exactly 0.0 -- the same defect this
    # project has now found in three consecutive briefs, here self-inflicted.
    # A seed has to select something, and confidence is one of the things it
    # selects.
    def band(low: float, high: float) -> float:
        return float(rng.uniform(low, high))

    planted: Dict[str, List[Claim]] = {name: [] for name in ANGLE_NAMES}
    planted[contra_a].append(make_claim(assertion, band(0.70, 0.90), "p_assert"))
    planted[contra_b].append(make_claim(reversal, band(0.65, 0.85), "p_reverse"))
    planted[agree_a].append(make_claim(assertion, band(0.60, 0.80), "p_assert2"))
    planted[agree_b].append(make_claim(paraphrase, band(0.55, 0.75), "p_para"))

    # One unrelated filler per angle, drawn without replacement.
    filler_idx = rng.choice(len(FILLER), size=len(ANGLE_NAMES), replace=False)
    for name, index in zip(ANGLE_NAMES, filler_idx):
        planted[name].append(
            make_claim(FILLER[int(index)], band(0.40, 0.70), f"f_{int(index)}")
        )

    claimsets = [make_claimset(name, query, planted[name]) for name in ANGLE_NAMES]
    return claimsets, (assertion, reversal), (assertion, paraphrase)


# --- the brief's rule, replicated for the contrast -------------------------


def brief_rule_pairs(
    claimsets: Sequence[ClaimSet], encoder
) -> List[Tuple[str, str, float]]:
    """``sim < 0.3 and both cited``, over every pair, exactly as specified.

    The brief flattens every angle's claims into one list and compares all
    pairs, same-angle pairs included; that is reproduced here rather than
    tidied, because the point is what its rule does.
    """
    claims = [claim for cs in claimsets for claim in cs.claims]
    if len(claims) < 2:
        return []
    vectors = np.asarray(encoder([c.text for c in claims]), dtype=float)
    flagged = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            similarity = cosine(vectors[i], vectors[j])
            if similarity < BRIEF_THRESHOLD and claims[i].cited_ids and claims[
                j
            ].cited_ids:
                flagged.append((claims[i].text, claims[j].text, float(similarity)))
    return flagged


def pair_key(text_a: str, text_b: str) -> frozenset:
    return frozenset((text_a, text_b))


# --- the run ---------------------------------------------------------------


async def run_one_seed(
    seed: int, encoder, fusion: PrismFusion, embedding: EmbeddingEstimator
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)

    domains = list(QUERY_BANK)
    pool = [(domain, query) for domain in domains for query in QUERY_BANK[domain]]
    chosen = rng.choice(len(pool), size=N_QUERIES, replace=False)

    per_query: List[Dict[str, object]] = []
    for index in chosen:
        domain, query = pool[int(index)]
        claimsets, contradiction, agreement = build_case(rng, domain, query)

        # The shared estimator is passed in: aggregate_dependence builds its own
        # EmbeddingEstimator otherwise, which reloads the weights once per query.
        report = aggregate_dependence(
            claimsets, estimators={"embedding": embedding}
        )
        eniv = compute_semantic_eniv(report)

        started = time.time()
        recommendation = await fusion.fuse(query, claimsets, report, eniv)
        fusion_seconds = time.time() - started

        found = {
            pair_key(c.text_a, c.text_b) for c in recommendation.contradictions
        }
        candidates = {pair_key(c.text_a, c.text_b) for c in recommendation.conflicts}
        brief_flagged = {
            pair_key(a, b) for a, b, _ in brief_rule_pairs(claimsets, encoder)
        }

        truth_contradiction = pair_key(*contradiction)
        truth_agreement = pair_key(*agreement)

        per_query.append(
            {
                "domain": domain,
                "query": query,
                "n_claims": len(recommendation.fused_claims),
                "eniv": recommendation.eniv,
                "discount_factor": recommendation.discount_factor,
                "overall_confidence": recommendation.overall_confidence,
                "undiscounted_confidence": recommendation.undiscounted_confidence,
                "n_candidates": len(recommendation.conflicts),
                "n_contradictions": len(recommendation.contradictions),
                "fusion_seconds": fusion_seconds,
                # corrected detector against the planted truth
                "found_planted_contradiction": truth_contradiction in found,
                "nominated_planted_contradiction": truth_contradiction in candidates,
                "false_contradiction_on_agreement": truth_agreement in found,
                "false_contradictions": len(found - {truth_contradiction}),
                # the brief's rule against the same planted truth
                "brief_found_planted_contradiction": truth_contradiction
                in brief_flagged,
                "brief_total_flagged": len(brief_flagged),
                "brief_false_flags": len(brief_flagged - {truth_contradiction}),
                # every claim must come out discounted
                "discount_applied_to_all": all(
                    fc.discounted_confidence <= fc.raw_confidence + 1e-12
                    and fc.discount_applied <= 1.0
                    for fc in recommendation.fused_claims
                ),
                "tokens_input": recommendation.tokens_input,
                "tokens_output": recommendation.tokens_output,
            }
        )

    return {"seed": seed, "queries": per_query}


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean": float(statistics.fmean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarise(seed_runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def seed_means(key: str) -> List[float]:
        return [
            float(statistics.fmean([float(q[key]) for q in run["queries"]]))
            for run in seed_runs
        ]

    keys = [
        "eniv",
        "discount_factor",
        "overall_confidence",
        "undiscounted_confidence",
        "n_candidates",
        "n_contradictions",
        "fusion_seconds",
        "found_planted_contradiction",
        "nominated_planted_contradiction",
        "false_contradiction_on_agreement",
        "false_contradictions",
        "brief_found_planted_contradiction",
        "brief_total_flagged",
        "brief_false_flags",
    ]
    return {
        key: {**mean_sd(seed_means(key)), "per_seed_mean": seed_means(key)}
        for key in keys
    }


async def run_fusion_e2e_test(
    num_seeds: int = N_SEEDS, num_queries: int = N_QUERIES
) -> Dict[str, object]:
    global N_QUERIES
    N_QUERIES = num_queries

    started = time.time()
    # One encoder, loaded once. The brief loads the weights inside the conflict
    # detector, so its 30-query run loads them 30 times.
    embedding = EmbeddingEstimator()
    encoder = embedding.encoder
    fusion = PrismFusion(encoder=encoder, adjudicator=LexicalAdjudicator())

    seed_runs = [
        await run_one_seed(seed, encoder, fusion, embedding)
        for seed in range(num_seeds)
    ]
    summary = summarise(seed_runs)

    all_queries = [q for run in seed_runs for q in run["queries"]]
    errors = [q for q in all_queries if not q["discount_applied_to_all"]]

    gate = {
        "queries_completed": len(all_queries),
        "queries_expected": num_seeds * num_queries,
        "all_queries_ran": len(all_queries) == num_seeds * num_queries,
        "discount_applied_everywhere": not errors,
        "contradiction_recall": summary["found_planted_contradiction"]["mean"],
        "brief_contradiction_recall": summary["brief_found_planted_contradiction"][
            "mean"
        ],
        "false_contradiction_rate_on_planted_agreement": summary[
            "false_contradiction_on_agreement"
        ]["mean"],
        "conflicts_flagged": summary["n_contradictions"]["mean"] > 0,
    }
    gate["passed"] = bool(
        gate["all_queries_ran"]
        and gate["discount_applied_everywhere"]
        and gate["conflicts_flagged"]
        and gate["contradiction_recall"] > gate["brief_contradiction_recall"]
    )

    return {
        "part": 23,
        "experiment": "fusion_e2e",
        "n_seeds": num_seeds,
        "n_queries": num_queries,
        "domains": sorted(QUERY_BANK),
        "adjudicator": "lexical",
        "live_api_calls": 0,
        "cost_usd": None,
        "cost_usd_reason": (
            "No Anthropic credentials in this checkout, so no request was sent "
            "and the ClaimSet token counts are fixture values. A dollar figure "
            "computed from invented token counts would be a fabricated "
            "measurement (CLAUDE.md: NEVER FABRICATE). The rate table and the "
            "token counts a real run would multiply are reported instead."
        ),
        "latency_meaning": (
            "fusion_seconds is local compute -- embedding, discounting, "
            "lexical adjudication. It contains no API latency."
        ),
        "model_pricing_usd_per_mtok": {
            name: {"input": rates[0], "output": rates[1]}
            for name, rates in MODEL_PRICING.items()
        },
        "fixture_tokens_per_angle": {
            "input": FIXTURE_TOKENS_INPUT,
            "output": FIXTURE_TOKENS_OUTPUT,
            "note": "invented; see cost_usd_reason",
        },
        "summary": summary,
        "gate": gate,
        "passed": gate["passed"],
        "elapsed_seconds": round(time.time() - started, 2),
    }


def _print(payload: Dict[str, object]) -> None:
    summary = payload["summary"]
    n_seeds = payload["n_seeds"]
    print(
        f"Part 23 -- fusion e2e: {payload['n_queries']} queries x "
        f"{len(payload['domains'])} domains x {n_seeds} seeds "
        f"({payload['gate']['queries_completed']} runs)"
    )
    print(f"adjudicator: {payload['adjudicator']}, live API calls: "
          f"{payload['live_api_calls']}")
    print()
    print(f"  {'metric':42s} {'mean +/- sd over seeds':>26s}")
    for key in (
        "eniv",
        "discount_factor",
        "undiscounted_confidence",
        "overall_confidence",
        "n_candidates",
        "n_contradictions",
        "fusion_seconds",
    ):
        s = summary[key]
        print(f"  {key:42s} {s['mean']:>14.4f} +/- {s['sd']:<9.4f}")
    print()
    print("  planted-contradiction detection (1.0 = found in every query)")
    for key, label in (
        ("nominated_planted_contradiction", "nominated as a candidate"),
        ("found_planted_contradiction", "CORRECTED rule, adjudicated"),
        ("brief_found_planted_contradiction", "BRIEF's rule (sim < 0.3)"),
    ):
        s = summary[key]
        print(f"    {label:40s} {s['mean']:>8.4f} +/- {s['sd']:<8.4f}")
    print()
    print("  false positives")
    for key, label in (
        ("false_contradiction_on_agreement", "corrected: planted paraphrase called a contradiction"),
        ("false_contradictions", "corrected: other pairs called contradictions"),
        ("brief_false_flags", "brief: pairs flagged that were not the contradiction"),
    ):
        s = summary[key]
        print(f"    {label:56s} {s['mean']:>8.4f} +/- {s['sd']:<8.4f}")
    print()
    print(f"  cost_usd: {payload['cost_usd']}")
    print(f"    {payload['cost_usd_reason']}")
    print()
    gate = payload["gate"]
    print(
        f"gate: all {gate['queries_completed']}/{gate['queries_expected']} runs "
        f"completed={gate['all_queries_ran']}, discount applied everywhere="
        f"{gate['discount_applied_everywhere']}, conflicts flagged="
        f"{gate['conflicts_flagged']}, recall {gate['contradiction_recall']:.3f} "
        f"> brief {gate['brief_contradiction_recall']:.3f}"
    )
    print()
    if payload["passed"]:
        print("GATE PASSED -- Part 24 may proceed.")
    else:
        print("GATE FAILED -- STOP. Do not proceed to Part 24.")


def main() -> int:
    payload = asyncio.run(run_fusion_e2e_test())
    _print(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
