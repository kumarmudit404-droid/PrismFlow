"""Hit/miss/latency counters for the fetch cache.

An ENGINEERING component per docs/CONTRACT.md section 3.

The counters exist so the Part 18 gate ("hit rate >= 95% on repeated queries")
is something measured rather than asserted, and so Part 19 can tell a slow
fan-out caused by real network work from one caused by a cache that is not
being hit.

A NOTE ON WHAT THESE NUMBERS ARE AND ARE NOT
--------------------------------------------
Hit and miss *counts* are deterministic: the same sequence of queries against
the same database always produces the same tallies, so a hit rate is a fact
about the cache. Latencies are not -- they depend on the machine, the disk and
whatever else is running. Per the 5-SEED RULE in CLAUDE.md, a latency figure
from one run is not evidence and must not be reported as a finding; it is
included here as an operational signal (is the cache doing anything at all?)
and is labelled as such wherever it is printed.

Four outcomes are tracked, not two, because "miss" conflates three different
situations and Part 19 will want to tell them apart:

  hit             fresh entry, and it held enough records to answer.
  miss            nothing stored under this key.
  expired         stored, but past its TTL.
  insufficient_k  stored and fresh, but fetched with a smaller k than asked
                  for, so it cannot answer without under-serving. Counted
                  separately because a steady stream of these means some
                  caller's k is growing and the TTL is hiding it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CacheMetrics:
    """Tallies and latencies for one process's cache activity.

    Guarded by a lock because Part 19 may drive the cache from several threads
    as well as several coroutines, and a torn counter would be indistinguishable
    from a genuine cache problem.
    """

    hits: int = 0
    misses: int = 0
    expired: int = 0
    insufficient_k: int = 0
    collisions: int = 0
    hit_latencies_ms: List[float] = field(default_factory=list)
    miss_latencies_ms: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # -- recording -------------------------------------------------------

    def record_hit(self, latency_ms: float) -> None:
        with self._lock:
            self.hits += 1
            self.hit_latencies_ms.append(latency_ms)

    def record_miss(
        self,
        latency_ms: float,
        *,
        reason: str = "absent",
    ) -> None:
        """Record a fetch that had to go to the source.

        ``reason`` is one of ``absent``, ``expired``, ``insufficient_k`` or
        ``collision``; all of them count as a miss, and the specific reason is
        also tallied on its own.
        """
        with self._lock:
            self.misses += 1
            self.miss_latencies_ms.append(latency_ms)
            if reason == "expired":
                self.expired += 1
            elif reason == "insufficient_k":
                self.insufficient_k += 1
            elif reason == "collision":
                self.collisions += 1

    def reset(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.expired = 0
            self.insufficient_k = 0
            self.collisions = 0
            self.hit_latencies_ms = []
            self.miss_latencies_ms = []

    # -- derived ---------------------------------------------------------

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache. 0.0 when nothing happened."""
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.lookups if self.lookups else 0.0

    @property
    def avg_hit_latency_ms(self) -> Optional[float]:
        return _mean(self.hit_latencies_ms)

    @property
    def avg_miss_latency_ms(self) -> Optional[float]:
        return _mean(self.miss_latencies_ms)

    @property
    def latency_saved_ms(self) -> Optional[float]:
        """Average time a hit avoided, or None until both paths have run.

        This is the "latency diff" standing rule 5 asks to be logged. It is an
        observation about this machine and this run, not a benchmark.
        """
        hit, miss = self.avg_hit_latency_ms, self.avg_miss_latency_ms
        if hit is None or miss is None:
            return None
        return miss - hit

    def snapshot(self) -> Dict[str, object]:
        """A plain dict, for logging or writing beside an experiment's output."""
        with self._lock:
            return {
                "lookups": self.lookups,
                "hits": self.hits,
                "misses": self.misses,
                "expired": self.expired,
                "insufficient_k": self.insufficient_k,
                "collisions": self.collisions,
                "hit_rate": round(self.hit_rate, 4),
                "miss_rate": round(self.miss_rate, 4),
                "avg_hit_latency_ms": _round_opt(self.avg_hit_latency_ms),
                "avg_miss_latency_ms": _round_opt(self.avg_miss_latency_ms),
                "latency_saved_ms": _round_opt(self.latency_saved_ms),
            }


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _round_opt(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


#: Process-wide metrics. The decorator updates this unless handed its own.
METRICS = CacheMetrics()


def get_metrics() -> CacheMetrics:
    return METRICS


def reset_metrics() -> None:
    """Zero the process-wide counters. Tests call this in a fixture."""
    METRICS.reset()


__all__ = ["CacheMetrics", "METRICS", "get_metrics", "reset_metrics"]
