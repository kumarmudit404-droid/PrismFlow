"""``DependenceReport``: how much each pair of angles is saying the same thing.

An ENGINEERING component per docs/CONTRACT.md section 3 -- deterministic given
its inputs, no learnable parameters of its own (the embedding model is a fixed
pretrained artefact, not something this project fits).

This is the measurement the thesis rests on. Agreement between views should
contribute evidence in proportion to the effective independence of those views,
so everything Part 22 discounts is computed from this matrix. A dependence
estimate that is quietly wrong does not announce itself: it shows up as
confidence that is too high, in a system whose whole purpose is to stop exactly
that.

VALIDATION RAISES, IT DOES NOT ASSERT
-------------------------------------
The brief validates shape, symmetry, diagonal and range with bare ``assert``.
Python invoked with ``-O`` removes assert statements entirely, so the one
guarantee Part 22 relies on -- that this matrix is a well-formed symmetric
correlation-like object -- would silently disappear under an optimisation flag
nobody remembers setting. These are contract checks on a value that flows into
a published number, so they raise ValueError and stay.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT
-------------------------------------------
An entry is a similarity in [0, 1], not a correlation and not a mutual
information in bits. 1.0 means "these two angles said the same thing, citing the
same evidence"; 0.0 means "nothing measurable in common". It is an ordinal
instrument: the ranking of pairs is meaningful, the absolute value is only as
meaningful as the estimators behind it, and Part 22 should treat it as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Tolerance for the symmetry and diagonal checks. Floating-point averaging of
#: three matrices cannot be expected to land exactly on 1.0.
TOLERANCE = 1e-6


@dataclass
class DependenceReport:
    """Pairwise semantic dependence between angles, and how it was obtained."""

    angle_names: List[str]
    n_angles: int
    dependence_matrix: np.ndarray
    per_estimator: Dict[str, np.ndarray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)

    #: Angles left out of the matrix because they produced no claims. Part 22
    #: must not count these toward its effective view count -- they are absent
    #: from the matrix precisely so that it cannot.
    excluded_angles: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        matrix = self.dependence_matrix
        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                f"dependence_matrix must be an ndarray, got {type(matrix).__name__}"
            )
        expected = (self.n_angles, self.n_angles)
        if matrix.shape != expected:
            raise ValueError(
                f"dependence_matrix has shape {matrix.shape}, expected {expected}"
            )
        if len(self.angle_names) != self.n_angles:
            raise ValueError(
                f"{len(self.angle_names)} angle names for {self.n_angles} angles"
            )
        if len(set(self.angle_names)) != len(self.angle_names):
            raise ValueError(
                f"angle names must be unique, got {self.angle_names}; two angles "
                "with one name cannot be told apart in the matrix"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                "dependence_matrix contains non-finite values; a NaN here "
                "propagates silently into the Part 22 discount"
            )
        if not np.allclose(matrix.diagonal(), 1.0, atol=TOLERANCE):
            raise ValueError(
                f"dependence_matrix diagonal must be 1.0, got {matrix.diagonal()}"
            )
        if not np.allclose(matrix, matrix.T, atol=TOLERANCE):
            raise ValueError("dependence_matrix must be symmetric")
        # A 0x0 matrix is legitimate -- every angle was excluded for having no
        # claims -- and min() on an empty array raises rather than returning a
        # neutral value, so the range check is skipped when there is no range.
        if matrix.size and (
            matrix.min() < -TOLERANCE or matrix.max() > 1.0 + TOLERANCE
        ):
            raise ValueError(
                f"dependence_matrix values must lie in [0, 1], got "
                f"[{matrix.min():.4f}, {matrix.max():.4f}]"
            )

    # -- reading the matrix ----------------------------------------------

    def between(self, angle_a: str, angle_b: str) -> float:
        """Dependence between two named angles."""
        try:
            i = self.angle_names.index(angle_a)
            j = self.angle_names.index(angle_b)
        except ValueError as exc:
            raise KeyError(
                f"unknown angle in ({angle_a!r}, {angle_b!r}); known: "
                f"{self.angle_names}"
            ) from exc
        return float(self.dependence_matrix[i, j])

    @property
    def off_diagonal(self) -> np.ndarray:
        """Every distinct pair's dependence, in upper-triangle row order."""
        i, j = np.triu_indices(self.n_angles, k=1)
        return self.dependence_matrix[i, j]

    @property
    def pairs(self) -> List[Tuple[str, str]]:
        """The angle-name pairs matching ``off_diagonal``, in the same order."""
        i, j = np.triu_indices(self.n_angles, k=1)
        return [(self.angle_names[a], self.angle_names[b]) for a, b in zip(i, j)]

    @property
    def mean_dependence(self) -> Optional[float]:
        """Mean over distinct pairs. None when there is only one angle."""
        values = self.off_diagonal
        return float(values.mean()) if values.size else None

    def most_dependent_pair(self) -> Optional[Tuple[str, str, float]]:
        """The pair sharing the most, which is the one Part 22 discounts hardest."""
        values = self.off_diagonal
        if not values.size:
            return None
        index = int(np.argmax(values))
        a, b = self.pairs[index]
        return a, b, float(values[index])

    def summary(self) -> str:
        """One line, for logs and the experiment script."""
        top = self.most_dependent_pair()
        mean = self.mean_dependence
        head = (
            f"{self.n_angles} angles, mean pairwise dependence "
            f"{mean:.3f}" if mean is not None else f"{self.n_angles} angle"
        )
        if top is None:
            return head
        return (
            f"{head}, most dependent {top[0]}~{top[1]}={top[2]:.3f}, "
            f"{len(self.warnings)} warning(s)"
        )

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe form, for writing beside an experiment's output."""
        return {
            "angle_names": list(self.angle_names),
            "n_angles": self.n_angles,
            "dependence_matrix": self.dependence_matrix.tolist(),
            "per_estimator": {
                name: matrix.tolist() for name, matrix in self.per_estimator.items()
            },
            "weights": dict(self.weights),
            "warnings": list(self.warnings),
            "excluded_angles": list(self.excluded_angles),
        }


__all__ = ["DependenceReport", "TOLERANCE"]
