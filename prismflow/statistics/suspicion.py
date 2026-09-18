"""Suspicion detector: agreement that the dependence structure cannot explain.

WHAT IT MEASURES

Two views agree for two legitimate reasons. They are both informative about the
same label, and they share structure with each other (measured dependence R_ij).
Neither is suspicious. What is suspicious is agreement in EXCESS of both.

For a pair of views the agreeing belief mass on one sample is

    a_ij = sum_k b_i[k] * b_j[k]

which is exactly the mass Dempster's rule treats as concordant -- the
complement of the conflict term that the discount operates on. It is used as a
RATIO to its own Cauchy-Schwarz bound,

    r_ij(x) = a_ij(x) / sqrt(a_ii(x) * a_jj(x))   in [0, 1]

i.e. the cosine between the two belief vectors. The ratio matters: raw agreeing
mass rises with confidence, so an unnormalised statistic flags every confident
sample rather than every colluding one. The cosine asks only WHETHER two views
point the same way, not how loudly. Two bounds are then estimated from the batch
itself:

    null_ij      mean r_ij after a WITHIN-STRATUM PERMUTATION of view j's
                 samples. Class proportions, per-view belief scale and vacuity
                 are all preserved; only the sample-level pairing is destroyed.
                 This is agreement attributable to "both views are about the
                 same classification problem".

    1            the cosine's own ceiling: two views that point identically.

Measured dependence interpolates between them:

    expected_ij = null_ij + clip(R_ij, 0, 1) * (1 - null_ij)

so a pair with R = 0 is expected to agree only as much as the permutation null,
and a pair with R = 1 is allowed to agree all the way to identical. The
per-sample excess is what is left over:

    excess_ij(x) = r_ij(x) - expected_ij

and the sample's score is the MAXIMUM excess over available pairs. Maximum, not
mean: one colluding pair should raise the alarm, and averaging over the honest
pairs would dilute exactly the signal being looked for.

The excess is deliberately NOT rescaled by (1 - null_ij). On clean data the
permutation null sits very close to the observed agreement -- almost all of it
is explained by the shared classification problem -- so that denominator is
near zero and dividing by it turns rounding noise into enormous scores. An
earlier version of this detector did exactly that and ranked clean batches as
more suspicious than attacked ones.

HARD ISOLATION (docs/CONTRACT.md, Part 10 brief)

Every input is a statistic available at inference time: beliefs, the view mask,
the estimated dependence matrix, and a stratum label taken from the model's own
pre-fusion prediction. Nothing here receives an attack label, an attack config,
a compromise mask or a corruption flag, and `tests/unit/test_suspicion.py`
asserts that against the signatures. A detector that is told where the attack is
would be a lookup table, and would measure nothing.

The threshold is likewise calibrated against a REFERENCE batch the caller
asserts is uncompromised -- `calibrate_threshold` sees scores and a target rate,
never a label. Choosing reference data is the caller's responsibility.

WHAT IT CANNOT SEE (measured, Part 09)

This detector reads coordination. Part 09 established that independent PGD
raises no dependence signal at all (rho_bar moved -0.0290 +/- 0.0251 while the
attack succeeded 46% of the time), so an attacker who does not coordinate is
expected to evade it. `experiments/comparison/` measures how badly. See
`docs/KNOWN_LIMITATIONS.md` L3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prismflow.statistics.dependence import (
    _to_numpy,
    _within_stratum_permutation,
    predicted_classes,
)

# `_within_stratum_permutation` and `_to_numpy` are imported rather than
# reimplemented so this null is the SAME null the dependence estimator uses.
# dependence.py is frozen and is only read from here, never modified.

DEFAULT_PERMUTATIONS = 32
DEFAULT_THRESHOLD = 0.05

# Below this the belief vector carries no direction worth comparing (a wholly
# vacuous view), and its cosine against anything is noise.
_MIN_NORM = 1e-6


@dataclass
class SuspicionReport:
    """score [B], flag [B], and the batch-level quantities behind them.

    score    per-sample maximum unexplained agreement, in cosine units. 0 means
             "exactly as much agreement as the dependence structure accounts
             for"; negative means less. Bounded in [-1, 1].
    flag     score > threshold
    excess   [B, V, V] per-pair excess, NaN where a pair is not measurable
    expected / null   [V, V] batch-level bounds described above
    """

    score: np.ndarray
    flag: np.ndarray
    excess: np.ndarray
    expected: np.ndarray
    null: np.ndarray
    threshold: float

    @property
    def flag_rate(self) -> float:
        return float(self.flag.mean()) if self.flag.size else float("nan")


def pairwise_agreement(belief, view_mask=None) -> np.ndarray:
    """Cosine agreement r_ij = a_ij / sqrt(a_ii a_jj), as [B, V, V].

    Pairs with either view absent, or either belief vector too close to wholly
    vacuous to have a direction, are NaN. The diagonal is 1 where measurable.
    """
    belief = _to_numpy(belief)
    if belief.ndim != 3:
        raise ValueError(f"belief must be [B, V, K], got {belief.shape}")
    n_samples, n_views, _ = belief.shape

    mass = np.einsum("bik,bjk->bij", belief, belief)
    norm = np.sqrt(np.einsum("bik,bik->bi", belief, belief))  # [B, V]
    usable = norm > _MIN_NORM

    with np.errstate(invalid="ignore", divide="ignore"):
        agreement = mass / (norm[:, :, None] * norm[:, None, :])
    agreement = np.where(usable[:, :, None] & usable[:, None, :], agreement, np.nan)

    if view_mask is not None:
        mask = _to_numpy(view_mask).astype(bool)
        if mask.shape != (n_samples, n_views):
            raise ValueError(f"view_mask must be [{n_samples}, {n_views}], got {mask.shape}")
        both = mask[:, :, None] & mask[:, None, :]
        agreement = np.where(both, agreement, np.nan)
    return agreement


def agreement_null(
    belief,
    view_mask=None,
    strata=None,
    seed: int = 0,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> np.ndarray:
    """Mean cosine agreement [V, V] after permuting view j within strata.

    This is the agreement attributable to both views addressing the same
    classification problem, with the sample-level pairing destroyed.
    """
    belief = _to_numpy(belief)
    n_samples, n_views, _ = belief.shape
    if strata is None:
        strata = np.zeros(n_samples, dtype=np.int64)
    strata = np.asarray(_to_numpy(strata), dtype=np.int64)

    norm = np.sqrt(np.einsum("bik,bik->bi", belief, belief))  # [B, V]
    has_direction = norm > _MIN_NORM

    if view_mask is not None:
        mask = _to_numpy(view_mask).astype(bool)
        if mask.shape != (n_samples, n_views):
            raise ValueError(f"view_mask must be [{n_samples}, {n_views}], got {mask.shape}")
    else:
        mask = np.ones((n_samples, n_views), dtype=bool)
    mask = mask & has_direction

    rng = np.random.default_rng(seed)
    null = np.full((n_views, n_views), np.nan)
    for i in range(n_views):
        for j in range(n_views):
            if i == j:
                null[i, j] = 1.0
                continue
            values = []
            for _ in range(n_permutations):
                order = _within_stratum_permutation(strata, rng)
                # view j's opinion attached to another sample of the same stratum
                dot = np.einsum("bk,bk->b", belief[:, i, :], belief[order, j, :])
                with np.errstate(invalid="ignore", divide="ignore"):
                    cosine = dot / (norm[:, i] * norm[order, j])
                usable = mask[:, i] & mask[order, j]
                if usable.any():
                    values.append(float(cosine[usable].mean()))
            null[i, j] = float(np.mean(values)) if values else np.nan
    # Symmetrise: the permutation is directional but the statistic is not.
    return np.where(np.isnan(null), null.T, (null + null.T) / 2.0)


def suspicion_report(
    belief,
    dependence,
    view_mask=None,
    evidence=None,
    strata=None,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 0,
    n_permutations: int = DEFAULT_PERMUTATIONS,
) -> SuspicionReport:
    """Per-sample unexplained-agreement score and flag.

    belief      [B, V, K] the opinions actually fused (post-discount if on)
    dependence  [V, V] estimated inter-view dependence
    view_mask   [B, V] True = present
    evidence    [B, V, K] optional, only to derive strata when none are given
    strata      [B] conditioning stratum; defaults to the pre-fusion predicted
                class, the same label-free stratification the dependence
                estimator uses

    Every argument is an inference-time statistic. See the module docstring's
    HARD ISOLATION note.
    """
    belief = _to_numpy(belief)
    if belief.ndim != 3:
        raise ValueError(f"belief must be [B, V, K], got {belief.shape}")
    n_samples, n_views, _ = belief.shape

    matrix = _to_numpy(dependence)
    if matrix.shape != (n_views, n_views):
        raise ValueError(f"dependence must be [{n_views}, {n_views}], got {matrix.shape}")

    if strata is None:
        source = evidence if evidence is not None else belief
        strata = predicted_classes(source, view_mask)

    null = agreement_null(
        belief, view_mask, strata=strata, seed=seed, n_permutations=n_permutations
    )
    agreement = pairwise_agreement(belief, view_mask)

    explained = np.clip(np.nan_to_num(matrix, nan=0.0), 0.0, 1.0)
    expected = null + explained * (1.0 - null)
    excess = agreement - expected[None, :, :]

    # Off-diagonal pairs only: a view always agrees perfectly with itself.
    off_diagonal = ~np.eye(n_views, dtype=bool)
    usable = np.broadcast_to(off_diagonal, excess.shape) & np.isfinite(excess)
    masked = np.where(usable, excess, -np.inf)

    score = masked.max(axis=(1, 2))
    score = np.where(np.isfinite(score), score, np.nan)

    flag = np.isfinite(score) & (score > threshold)
    return SuspicionReport(
        score=score,
        flag=flag,
        excess=np.where(usable, excess, np.nan),
        expected=expected,
        null=null,
        threshold=float(threshold),
    )


def calibrate_threshold(reference_scores, target_rate: float = 0.05) -> float:
    """Threshold giving `target_rate` flags on a batch the caller says is clean.

    Takes scores and a rate. It never sees a label of any kind: which data
    counts as a clean reference is the caller's assertion, not something this
    function can or should verify.
    """
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError(f"target_rate must be in [0, 1], got {target_rate}")
    scores = np.asarray(_to_numpy(reference_scores), dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float(DEFAULT_THRESHOLD)
    if target_rate == 0.0:
        return float(np.nextafter(finite.max(), np.inf))
    return float(np.quantile(finite, 1.0 - target_rate))


def detection_rates(scores, threshold: float) -> float:
    """Fraction of finite scores above `threshold`. Scores only, no labels."""
    scores = np.asarray(_to_numpy(scores), dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    return float((finite > threshold).mean()) if finite.size else float("nan")
