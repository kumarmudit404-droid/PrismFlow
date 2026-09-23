"""What a cached fetch is, and how it survives a round trip unchanged.

An ENGINEERING component per docs/CONTRACT.md section 3 -- no learnable
parameters, no tensors. It sits between Part 17's connectors and Part 19's
fan-out, and its only job is to hand back exactly what the connector would
have returned.

"EXACTLY WHAT THE CONNECTOR WOULD HAVE RETURNED" IS THE WHOLE CONTRACT
---------------------------------------------------------------------
A cache that returns *nearly* the right thing is worse than no cache, because
the difference only appears on the second run. The Part 18 brief specifies
deserialisation as ``NormalizedRecord(**json.loads(...))``, and dataclasses do
not coerce types, so ``published_date`` comes back as a ``str`` where a fresh
fetch gives a ``datetime``:

    fresh  published_date: datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=utc)
    cached published_date: '2026-01-02T03:04:05+00:00'

Any date arithmetic downstream then raises ``TypeError: unsupported operand
type(s) for -: 'datetime.datetime' and 'str'`` -- but only on a cache hit.
Part 19 ranks by recency and Part 21 measures dependence between angles; a
field that changes type depending on cache state would make both of them
behave differently on a warm machine than on a cold one, which is precisely
what docs/CONTRACT.md section 5 forbids. So serialisation here is explicit,
inverse-by-construction, and ``test_serialisation_covers_every_field`` fails
loudly if a later part adds a field this module does not know about.

WHY TIMESTAMPS ARE FIXED-WIDTH
------------------------------
``inserted_at`` is compared in SQL (``WHERE inserted_at < ?``) so the index on
it can be used. Text comparison is only chronological if every row shares one
format, and ``datetime.isoformat()`` omits the microseconds when they happen
to be zero, which changes the string's width. Forcing
``timespec="microseconds"`` makes every stored timestamp exactly 32
characters, so lexicographic order is chronological order. The Part 18 brief
instead lets SQLite write ``DEFAULT CURRENT_TIMESTAMP`` (``'2026-09-23
07:58:48'`` -- space separator, no offset) and compares it against
``cutoff.isoformat()`` (``'2026-09-16T07:58:48.278331+00:00'`` -- 'T'
separator); since ``' '`` (0x20) sorts below ``'T'`` (0x54), every row written
on the cutoff's own calendar date compares as older than the cutoff and is
deleted regardless of its clock time.

WHY NOT datetime.utcnow()
-------------------------
Standing rule 3 of the Part 18 brief mandates ``datetime.utcnow()``. On this
interpreter (CPython 3.13.7) that call is deprecated, and it returns a *naive*
datetime while Part 17 produces aware ones, so ``utcnow() - inserted_at``
raises ``TypeError: can't subtract offset-naive and offset-aware datetimes``.
The rule's intent -- never depend on the machine's local zone -- is satisfied
by ``prismflow.v2.connectors.base.utcnow()``, which is aware UTC and is
already what every Part 17 timestamp is built from. That helper is reused
rather than redefined so the two layers cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from prismflow.v2.connectors.base import NormalizedRecord, utcnow

#: Width-stable UTC ISO-8601. See "WHY TIMESTAMPS ARE FIXED-WIDTH" above.
ISO_TIMESPEC = "microseconds"

#: The fields this module knows how to serialise, and which of them are
#: datetimes. Kept explicit (rather than derived from ``__annotations__``)
#: because ``base.py`` uses ``from __future__ import annotations``, so its
#: field types are strings at runtime and cannot be dispatched on reliably.
_RECORD_FIELDS = (
    "id",
    "title",
    "url",
    "snippet",
    "snippet_tokens",
    "source",
    "published_date",
    "author",
    "relevance_score",
)
_RECORD_DATETIME_FIELDS = frozenset({"published_date"})


def hash_query(query: str) -> str:
    """Stable 16-hex-char digest of a query string.

    Truncated to 64 bits, as the brief specifies. That is ample for a cache
    keyed per connector, and ``query_text`` is stored alongside so a collision
    is *detected* rather than silently served -- see ``CacheEntry.matches``.
    """
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def to_iso(moment: datetime) -> str:
    """Canonical, width-stable, UTC ISO-8601 rendering of ``moment``.

    A naive datetime is assumed to be UTC rather than rejected: SQLite's own
    ``CURRENT_TIMESTAMP`` is naive UTC, so a row written by anything other
    than this module still round-trips to the right instant.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec=ISO_TIMESPEC)


def from_iso(text: str) -> datetime:
    """Parse ``to_iso`` output, or SQLite's naive ``CURRENT_TIMESTAMP`` form.

    Always returns an aware UTC datetime, so nothing downstream can end up
    mixing naive and aware values.
    """
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def record_to_dict(record: NormalizedRecord) -> Dict[str, Any]:
    """One NormalizedRecord as JSON-safe primitives."""
    out: Dict[str, Any] = {}
    for name in _RECORD_FIELDS:
        value = getattr(record, name)
        if name in _RECORD_DATETIME_FIELDS and value is not None:
            value = to_iso(value)
        out[name] = value
    return out


def record_from_dict(data: Dict[str, Any]) -> NormalizedRecord:
    """Inverse of ``record_to_dict``, restoring datetimes as datetimes."""
    kwargs: Dict[str, Any] = {}
    for name in _RECORD_FIELDS:
        value = data.get(name)
        if name in _RECORD_DATETIME_FIELDS and value is not None:
            value = from_iso(value) if isinstance(value, str) else value
        kwargs[name] = value
    return NormalizedRecord(**kwargs)


def records_to_json(records: Sequence[NormalizedRecord]) -> str:
    """Serialise records for storage. Inverse of ``records_from_json``."""
    return json.dumps([record_to_dict(r) for r in records])


def records_from_json(payload: str) -> List[NormalizedRecord]:
    """Deserialise stored records. Inverse of ``records_to_json``."""
    return [record_from_dict(d) for d in json.loads(payload)]


@dataclass
class CacheEntry:
    """One stored fetch result.

    ``requested_k`` is the ``k`` the underlying fetch was made with, and it is
    the reason a hit is not simply "the key is present". The brief's lookup
    returns ``entry.to_records()[:k]`` without recording how many records were
    ever fetched, so an entry stored at k=5 answers a later k=20 with five
    records and logs a CACHE HIT -- Part 19's fan-out would silently retrieve
    a quarter of the evidence it asked for. Storing k lets a too-small entry
    be treated as a miss and upgraded in place.

    It defaults to 0 ("unknown"), which ``satisfies`` reads as insufficient
    for any positive k. A row of unknown provenance is refetched rather than
    trusted.
    """

    connector_name: str
    query_hash: str
    query_text: str
    normalized_records_json: str
    inserted_at: datetime
    ttl_seconds: int
    requested_k: int = 0

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        """True once ``ttl_seconds`` have elapsed since ``inserted_at``."""
        moment = now or utcnow()
        inserted = self.inserted_at
        if inserted.tzinfo is None:
            inserted = inserted.replace(tzinfo=timezone.utc)
        return (moment - inserted) > timedelta(seconds=self.ttl_seconds)

    def satisfies(self, k: int) -> bool:
        """True if this entry holds at least ``k`` records' worth of fetch."""
        return self.requested_k >= k

    def matches(self, query: str) -> bool:
        """False on a 64-bit hash collision between two different queries."""
        return self.query_text == query

    def to_records(self) -> List[NormalizedRecord]:
        """Deserialise to real NormalizedRecords, datetimes included."""
        return records_from_json(self.normalized_records_json)

    @property
    def age_seconds(self) -> float:
        inserted = self.inserted_at
        if inserted.tzinfo is None:
            inserted = inserted.replace(tzinfo=timezone.utc)
        return (utcnow() - inserted).total_seconds()


def normalized_record_field_names() -> List[str]:
    """Every field of NormalizedRecord, straight from the dataclass.

    Used only by the test that asserts this module's serialisation has not
    fallen behind Part 17's data model.
    """
    return [f.name for f in fields(NormalizedRecord)]


__all__ = [
    "ISO_TIMESPEC",
    "CacheEntry",
    "hash_query",
    "to_iso",
    "from_iso",
    "record_to_dict",
    "record_from_dict",
    "records_to_json",
    "records_from_json",
    "normalized_record_field_names",
]
