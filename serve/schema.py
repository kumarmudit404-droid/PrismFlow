"""Settings and response shapes for the scoring service.

EVERY SETTING IS AN ENVIRONMENT VARIABLE WITH A SAFE DEFAULT, and "safe"
here means closed. `ALLOWED_ORIGIN` defaults to the empty string, which
allows no browser origin at all: an unset variable produces a service that
only non-browser clients can reach, never one that answers every site on the
internet. There is no wildcard branch in this file to fall into.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from serve.gate import (
    MAX_DECOMPRESSED_BYTES,
    MAX_FEATURES,
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
    MAX_VIEWS,
    MIN_SAMPLES,
)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, read once at import.

    allowed_origins  exact origins permitted by CORS. Empty means deny every
                     browser origin -- the default, and what an unconfigured
                     deployment does.
    """

    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    seed: int = 0

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        raw = (env.get("ALLOWED_ORIGIN") or "").strip()
        # Comma-separated so a staging origin can be added without code, but
        # "*" is not a value this accepts: it is dropped with the rest of the
        # empty entries, so a deployment cannot open itself up by typo.
        origins = tuple(
            o.strip() for o in raw.split(",")
            if o.strip() and o.strip() != "*"
        )
        try:
            seed = int(env.get("SCORING_SEED", "0"))
        except ValueError:
            seed = 0
        return cls(allowed_origins=origins, seed=seed)


def limits() -> dict:
    """The caps, so a client can show them before a user picks a file."""
    return {
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_decompressed_bytes": MAX_DECOMPRESSED_BYTES,
        "max_rows": MAX_ROWS,
        "max_views": MAX_VIEWS,
        "max_features_per_view": MAX_FEATURES,
        "min_samples": MIN_SAMPLES,
        "accepted_types": [".npz", ".npy", ".csv", ".parquet"],
    }
