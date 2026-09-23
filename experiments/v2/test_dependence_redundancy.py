"""Planted-redundancy experiment for the Part 21 dependence estimators.

    .venv\\Scripts\\python.exe experiments/v2/test_dependence_redundancy.py

Unlike the other experiments/v2 scripts, this one IS evidence, and it is run
under the 5-SEED RULE: 20 queries x 4 conditions x 5 seeds, reporting mean and
standard deviation. Its inputs are synthetic by design -- a planted-redundancy
test works precisely because the true relationship is known before measuring, so
the claims are probes, not observations. What is measured is the ESTIMATOR, and
that measurement is a finding.

WHY THE SEED LOOP HAD TO BE REDESIGNED
--------------------------------------
The brief's experiment calls ``np.random.seed(seed)`` and then builds the same
two hardcoded claim lists on every iteration. Nothing downstream is stochastic
-- sentence-transformer encoding is deterministic -- so all five seeds compute
an identical number and the reported standard deviation is exactly 0.0.

That is worse than reporting one seed, because 0.0 reads as a remarkably stable
estimator rather than as an absent measurement, and CLAUDE.md's 5-SEED RULE is
satisfied in form while producing no information. A seed has to select
something. Here each seed draws 20 of the 24 corpus topics without replacement
and, for each, an unrelated partner topic. The spread across seeds then reflects
genuine variation in how the estimator behaves across content, which is the
quantity a standard deviation is supposed to describe.

The brief also lists a fourth condition (low-rank embeddings) in its goals,
initialises ``results["low_rank"]`` and never fills it; ``np.mean([])`` is nan,
so the saved JSON would carry a nan for a required condition. It is implemented
here.

WHAT THE GATE ASKS AND WHERE IT IS CHECKED
------------------------------------------
"Three estimators agree on ranking" cannot be tested on the planted pairs: two
angles give one pair, and a rank correlation over one point is not a
measurement. So the script runs a second configuration of five angles with a
known dependence structure, which yields ten pairs, and reports Spearman
agreement between the estimators over those.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.dependence import (  # noqa: E402
    DEFAULT_WEIGHTS,
    EmbeddingEstimator,
    aggregate_dependence,
    estimator_agreement,
)
from tests.v2.conftest import (  # noqa: E402
    make_claimset,
    planted_pair,
    topic_claims,
    topic_count,
    topic_name,
)

N_SEEDS = 5
N_QUERIES = 20
LOW_RANK_DIMS = 4
OUTPUT = Path("results/v2/dependence_redundancy_experiment.json")

#: Each reported condition, as (planted relationship, which embedding estimator).
#:
#: The brief's fourth condition is "low-rank embedding (truncated) -> verify
#: graceful degradation", applied to the identical pair. On identical claims that
#: is uninformative: the inputs are the same text, so the similarity is 1.0 at
#: any dimensionality and the condition cannot show degradation of anything.
#: What truncation actually damages is SEPARABILITY -- its ability to tell a
#: paraphrase from an unrelated claim -- so the low-rank estimator is run over
#: all three relationships and the margin is compared against the full-rank one.
CONDITION_SPECS = {
    "identical": ("identical", "embedding"),
    "paraphrase": ("paraphrase", "embedding"),
    "unrelated": ("unrelated", "embedding"),
    "low_rank": ("identical", "embedding_low_rank"),
    "low_rank_paraphrase": ("paraphrase", "embedding_low_rank"),
    "low_rank_unrelated": ("unrelated", "embedding_low_rank"),
}
CONDITIONS = tuple(CONDITION_SPECS)

#: The bands the brief specifies. Checked and reported, not assumed.
THRESHOLDS = {
    "identical": ("mean > 0.95", lambda m: m > 0.95),
    "paraphrase": ("0.6 < mean < 0.95", lambda m: 0.6 < m < 0.95),
    "unrelated": ("mean < 0.3", lambda m: m < 0.3),
}


class CachingEncoder:
    """Encodes each distinct text once.

    The conditions reuse the same claim texts heavily -- identical and low_rank
    share their inputs entirely -- so without this the script spends most of its
    time re-encoding strings it has already seen. Caching cannot change any
    result: the encoder is deterministic, which is the same property that made
    the brief's seed loop a no-op.
    """

    def __init__(self, inner):
        self._inner = inner
        self._cache: Dict[str, np.ndarray] = {}
        self.calls = 0
        self.encoded = 0

    def __call__(self, texts):
        texts = list(texts)
        self.calls += 1
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vectors = np.asarray(self._inner(missing), dtype=float)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            self.encoded += len(missing)
            for text, vector in zip(missing, vectors):
                self._cache[text] = vector
        return np.vstack([self._cache[t] for t in texts])


def dependence_for(condition, topic, other, estimators) -> float:
    """The single off-diagonal dependence for one planted pair."""
    relationship, estimator_key = CONDITION_SPECS[condition]
    left, right = planted_pair(relationship, topic, other)
    report = aggregate_dependence(
        [left, right], estimators={"embedding": estimators[estimator_key]},
    )
    return float(report.dependence_matrix[0, 1])


def agreement_configuration(rng, estimators) -> Tuple[float, Dict[str, float], List[str]]:
    """Five angles with a known dependence structure, giving ten pairs.

    Two paraphrase couples plus one unrelated angle: A~B and C~D should rank
    high, everything else low. Every estimator should recover that ORDER, which
    is what the gate asks. Their magnitudes are not comparable and are not
    compared.
    """
    a, c, e = rng.choice(topic_count(), size=3, replace=False)
    claimsets = [
        make_claimset("tech", topic_name(int(a)), topic_claims(int(a))),
        make_claimset("market", topic_name(int(a)),
                      topic_claims(int(a), paraphrase=True)),
        make_claimset("financial", topic_name(int(c)), topic_claims(int(c))),
        make_claimset("regulatory", topic_name(int(c)),
                      topic_claims(int(c), paraphrase=True)),
        make_claimset("sentiment", topic_name(int(e)), topic_claims(int(e))),
    ]
    report = aggregate_dependence(
        claimsets, estimators={"embedding": estimators["embedding"]},
    )
    return estimator_agreement(report)


def main() -> int:
    started = time.perf_counter()

    base = EmbeddingEstimator()
    encoder = CachingEncoder(base.encoder)  # forces the model load, once
    estimators = {
        "embedding": EmbeddingEstimator(encoder=encoder),
        "embedding_low_rank": EmbeddingEstimator(
            encoder=encoder, truncate_dims=LOW_RANK_DIMS
        ),
    }

    print(f"corpus: {topic_count()} topics; drawing {N_QUERIES} per seed")
    print(f"weights: {DEFAULT_WEIGHTS}")
    print(f"low-rank condition truncates embeddings to {LOW_RANK_DIMS} dims\n")

    # per_seed[condition] = [seed-level mean over that seed's 20 queries]
    per_seed: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
    # pooled[condition] = every individual measurement, for separability
    pooled: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
    agreements: List[float] = []
    agreement_notes: List[str] = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        topics = rng.choice(topic_count(), size=N_QUERIES, replace=False)
        # An unrelated partner for each drawn topic, never the topic itself.
        partners = [
            int(rng.choice([t for t in range(topic_count()) if t != int(topic)]))
            for topic in topics
        ]

        for condition in CONDITIONS:
            values = [
                dependence_for(condition, int(topic), partner, estimators)
                for topic, partner in zip(topics, partners)
            ]
            per_seed[condition].append(float(np.mean(values)))
            pooled[condition].extend(values)

        rho, _, notes = agreement_configuration(rng, estimators)
        if rho is not None:
            agreements.append(rho)
        agreement_notes.extend(notes)

        print(
            f"seed {seed}: "
            + "  ".join(
                f"{c}={per_seed[c][-1]:.3f}" for c in CONDITIONS
            )
            + (f"  rho={rho:.3f}" if rho is not None else "  rho=undefined")
        )

    # --- summarise ------------------------------------------------------
    summary: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        seed_means = per_seed[condition]
        every = pooled[condition]
        summary[condition] = {
            "mean_of_seed_means": float(np.mean(seed_means)),
            "std_of_seed_means": float(statistics.stdev(seed_means))
            if len(seed_means) > 1 else 0.0,
            "per_seed_means": [float(v) for v in seed_means],
            "pooled_min": float(np.min(every)),
            "pooled_max": float(np.max(every)),
            "n_measurements": len(every),
        }

    print("\n--- per condition, mean +/- sd over 5 seeds ---")
    for condition in CONDITIONS:
        row = summary[condition]
        print(
            f"  {condition:12s} {row['mean_of_seed_means']:.4f} +/- "
            f"{row['std_of_seed_means']:.4f}   "
            f"[{row['pooled_min']:.3f}, {row['pooled_max']:.3f}] "
            f"over {row['n_measurements']} measurements"
        )

    # --- gate checks ----------------------------------------------------
    checks: Dict[str, Dict[str, object]] = {}
    for condition, (description, predicate) in THRESHOLDS.items():
        mean = summary[condition]["mean_of_seed_means"]
        checks[condition] = {
            "requirement": description,
            "observed_mean": mean,
            "passed": bool(predicate(mean)),
        }

    separable = (
        summary["unrelated"]["pooled_max"] < summary["paraphrase"]["pooled_min"]
        and summary["paraphrase"]["pooled_max"] < summary["identical"]["pooled_min"]
    )
    checks["separability"] = {
        "requirement": "unrelated < paraphrase < identical with no overlap, "
                       "pooled across all seeds and queries",
        "passed": bool(separable),
        "unrelated_max": summary["unrelated"]["pooled_max"],
        "paraphrase_min": summary["paraphrase"]["pooled_min"],
        "paraphrase_max": summary["paraphrase"]["pooled_max"],
        "identical_min": summary["identical"]["pooled_min"],
    }

    # Graceful degradation, measured where it can actually be seen: the margin
    # between a paraphrase and an unrelated pair, full-rank against truncated.
    full_margin = (
        summary["paraphrase"]["pooled_min"] - summary["unrelated"]["pooled_max"]
    )
    low_margin = (
        summary["low_rank_paraphrase"]["pooled_min"]
        - summary["low_rank_unrelated"]["pooled_max"]
    )
    checks["low_rank_degrades_gracefully"] = {
        "requirement": (
            f"truncating embeddings to {LOW_RANK_DIMS} dims keeps identical "
            "pairs high and still yields a valid matrix, while the "
            "paraphrase-vs-unrelated margin narrows rather than the estimator "
            "failing"
        ),
        "identical_mean_full_rank": summary["identical"]["mean_of_seed_means"],
        "identical_mean_low_rank": summary["low_rank"]["mean_of_seed_means"],
        "unrelated_mean_full_rank": summary["unrelated"]["mean_of_seed_means"],
        "unrelated_mean_low_rank": summary["low_rank_unrelated"]["mean_of_seed_means"],
        "separability_margin_full_rank": full_margin,
        "separability_margin_low_rank": low_margin,
        "margin_lost": full_margin - low_margin,
        "passed": bool(
            summary["low_rank"]["mean_of_seed_means"] > 0.95
            and low_margin < full_margin
        ),
        "note": (
            "On identical claims the similarity is 1.0 at any dimensionality, so "
            "the brief's condition 4 as written cannot show degradation of "
            "anything. The informative comparison is the margin."
        ),
    }

    mean_rho = float(np.mean(agreements)) if agreements else None
    checks["estimator_agreement"] = {
        "requirement": "mean Spearman rho between estimators > 0.5 over a "
                       "5-angle configuration (10 pairs)",
        "observed_mean_rho": mean_rho,
        "per_seed_rho": [float(r) for r in agreements],
        "passed": bool(mean_rho is not None and mean_rho > 0.5),
        "notes": sorted(set(agreement_notes)),
    }

    print("\n--- gate checks ---")
    for name, check in checks.items():
        state = "PASS" if check["passed"] else "FAIL"
        print(f"  [{state}] {name}: {check['requirement']}")
        if "observed_mean" in check:
            print(f"           observed mean {check['observed_mean']:.4f}")
        if check.get("observed_mean_rho") is not None:
            print(f"           observed mean rho {check['observed_mean_rho']:.4f}")

    failed = [name for name, check in checks.items() if not check["passed"]]

    payload = {
        "experiment": "planted_redundancy_dependence",
        "part": 21,
        "n_seeds": N_SEEDS,
        "n_queries_per_seed": N_QUERIES,
        "n_topics_in_corpus": topic_count(),
        "conditions": list(CONDITIONS),
        "weights": DEFAULT_WEIGHTS,
        "low_rank_dims": LOW_RANK_DIMS,
        "embedding_model": base.model_name,
        "summary": summary,
        "checks": checks,
        "encoder_calls": encoder.calls,
        "distinct_texts_encoded": encoder.encoded,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "inputs_are_synthetic": (
            "Claims are constructed probes with known relationships, not model "
            "output. The measurement is of the estimator, not of the world."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT}")
    print(f"encoded {encoder.encoded} distinct texts over {encoder.calls} calls")
    print(f"runtime {payload['runtime_seconds']}s")

    if failed:
        print(f"\nFAILED checks: {', '.join(failed)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
