"""Clique contrast, pinned against CONSTRUCTED dependence structures.

WHY THIS FILE GATES THE EXPERIMENT

Part 14's Arm B asks whether four signals jointly carry enough information to
separate honest duplication from adversarial collusion. A bug in
`clique_contrast` could produce either of the two ways that question can be
answered wrongly:

  - reading ~0 on both structures would look like "information absent" and would
    stop the Part on a false negative;
  - reading large on both would look like "information present" and would send a
    signal into the oracle fit that generalises to nothing.

So the signal is checked against structures built to order, where the right
answer is known by construction, BEFORE it is allowed near real data. These are
sanity checks on an estimator, not results about attacks.

The two structures, in the project's own terms:

  honest duplication   a clique that agrees internally AND carries the rest with
                       it -- every view agrees with every other
  chorus collusion     a clique that agrees internally and is near-independent
                       of the remaining views
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from prismflow.statistics.clique_contrast import (
    clique_contrast,
    clique_contrast_matrix,
    per_view_contrast,
)

N_VIEWS = 4


def honest_duplication(level: float = 0.8) -> np.ndarray:
    """Every pair equally dependent: a clique correlated with EVERYTHING.

    This is the structure that must read ~0. It is the case that has repeatedly
    tripped the project's detectors (L5: clone_k2 scored higher than any attack).
    """
    matrix = np.full((N_VIEWS, N_VIEWS), level, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def chorus_collusion(
    clique=(0, 1, 2), internal: float = 0.8, external: float = 0.05
) -> np.ndarray:
    """A clique agreeing internally and near-independent of the rest."""
    matrix = np.full((N_VIEWS, N_VIEWS), external, dtype=np.float64)
    for i in clique:
        for j in clique:
            if i != j:
                matrix[i, j] = internal
    np.fill_diagonal(matrix, 1.0)
    return matrix


# --------------------------------------------------------------------------
# THE GATE: the two structures must read differently, in the right directions
# --------------------------------------------------------------------------

def test_honest_duplication_reads_near_zero():
    """A clique correlated with everything is NOT a dissenting clique."""
    assert clique_contrast_matrix(honest_duplication()) == pytest.approx(0.0, abs=1e-9)


def test_honest_duplication_reads_near_zero_at_every_level():
    """The reading must not depend on HOW correlated the views are, only on
    whether the correlation is differentiated."""
    for level in (0.1, 0.3, 0.5, 0.8, 0.95):
        assert abs(clique_contrast_matrix(honest_duplication(level))) < 1e-9, level


def test_chorus_collusion_reads_clearly_positive():
    value = clique_contrast_matrix(chorus_collusion())
    assert value > 0.4, f"collusion must read clearly positive, got {value}"
    assert value == pytest.approx(0.75, abs=0.01)  # 0.8 - 0.05


def test_collusion_separates_from_honest_duplication_by_a_wide_margin():
    """The comparison Arm B depends on, stated as a single assertion."""
    honest = clique_contrast_matrix(honest_duplication(0.8))
    collusion = clique_contrast_matrix(chorus_collusion(internal=0.8, external=0.05))
    assert collusion - honest > 0.5


@pytest.mark.parametrize("clique", [(0, 1), (0, 1, 2), (1, 2, 3), (2, 3)])
def test_collusion_detected_for_every_clique_size_and_membership(clique):
    """k is not known in advance, so the best-split form must cover k = 2 and 3
    and must not depend on WHICH views collude."""
    value = clique_contrast_matrix(chorus_collusion(clique=clique))
    assert value > 0.4, f"clique {clique} read {value}"


def test_contrast_grows_as_the_clique_separates_from_the_rest():
    previous = -np.inf
    for external in (0.75, 0.6, 0.4, 0.2, 0.05):
        value = clique_contrast_matrix(chorus_collusion(internal=0.8, external=external))
        assert value > previous
        previous = value


def test_independent_views_read_near_zero():
    """No clique at all, nothing to report."""
    matrix = np.full((N_VIEWS, N_VIEWS), 0.02)
    np.fill_diagonal(matrix, 1.0)
    assert abs(clique_contrast_matrix(matrix)) < 1e-9


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------

def test_per_view_contrast_identifies_the_colluding_views():
    """Clique members must be the ones scoring high, not the honest views."""
    contrasts = per_view_contrast(chorus_collusion(clique=(0, 1)))
    assert contrasts[0] > 0.4 and contrasts[1] > 0.4
    assert contrasts[2] < 0.2 and contrasts[3] < 0.2


def test_per_sample_form_matches_the_matrix_form():
    matrix = chorus_collusion()
    batch = np.repeat(matrix[None, :, :], 7, axis=0)
    per_sample = clique_contrast(batch)
    assert per_sample.shape == (7,)
    assert np.allclose(per_sample, clique_contrast_matrix(matrix))


def test_per_sample_form_handles_a_mixed_batch():
    batch = np.stack([honest_duplication(), chorus_collusion(), honest_duplication()])
    scores = clique_contrast(batch)
    assert scores[1] > scores[0] + 0.5
    assert scores[1] > scores[2] + 0.5


def test_unmeasurable_pairs_yield_nan_not_zero():
    """Refusing to score is not the same as scoring 'no clique'."""
    matrix = chorus_collusion().astype(float)
    matrix[:] = np.nan
    assert np.isnan(clique_contrast_matrix(matrix))


def test_view_mask_excludes_absent_views():
    batch = np.repeat(chorus_collusion(clique=(0, 1))[None, :, :], 2, axis=0)
    mask = np.ones((2, N_VIEWS), dtype=bool)
    mask[1, 1] = False  # break the clique on the second sample
    scores = clique_contrast(batch, view_mask=mask)
    assert scores[0] > scores[1]


def test_two_views_have_no_contrast():
    """With two views there is no 'rest' to contrast against."""
    assert np.isnan(clique_contrast_matrix(np.array([[1.0, 0.9], [0.9, 1.0]])))


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        clique_contrast_matrix(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        clique_contrast(np.zeros((4, 4)))


def test_only_two_usable_views_is_nan_and_does_not_warn():
    """The Arm B drop case: 4 views, 2 of them with vacuous belief.

    `pairwise_agreement` returns NaN for a view whose belief vector is too
    close to vacuous to have a direction. When that leaves only two usable
    views, no internal/external split exists and the sample is undefined --
    but it must arrive as a NaN the caller can count, not as a RuntimeWarning
    raised from inside numpy. Seeds 3 and 4 of the Arm B run hit this on
    clone_k3, 1 row each.
    """
    sample = np.full((1, N_VIEWS, N_VIEWS), np.nan)
    usable = (1, 3)
    for i in usable:
        for j in usable:
            sample[0, i, j] = 1.0 if i == j else 0.9999

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails this test
        scores = clique_contrast(sample)

    assert np.isnan(scores[0]), "two usable views cannot define a contrast"


def test_three_usable_views_still_scores():
    """One vacuous view out of four is survivable -- only the third is fatal."""
    batch = chorus_collusion(clique=(0, 1))[None, :, :].astype(float).copy()
    batch[0, 3, :] = np.nan
    batch[0, :, 3] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = clique_contrast(batch)

    assert np.isfinite(scores[0])
