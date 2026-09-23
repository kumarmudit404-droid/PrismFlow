"""Getting a JSON object out of whatever a language model actually returned.

An ENGINEERING component per docs/CONTRACT.md section 3.

WHY THIS IS ITS OWN MODULE
--------------------------
It is the part of Part 20 that can be tested exhaustively without an API key,
and it is the part most likely to decide whether a run produces evidence or
silence. Both providers share it.

WHAT THE BRIEF'S EXTRACTION DOES
--------------------------------
    json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")
    ...
    except json.JSONDecodeError as e:

Two failures, both measured:

1. ``.*`` with DOTALL is greedy, so it spans from the FIRST ``{`` anywhere in
   the reply to the LAST ``}`` anywhere in it. Given
   ``Note {see below}. Here: {"claims": []} Thanks {end}`` it captures the whole
   line and ``json.loads`` fails on it -- a well-formed answer thrown away
   because the model wrapped it in prose, which is exactly what the regex was
   introduced to tolerate.
2. ``ValueError("No JSON found")`` is raised inside a ``try`` whose only handler
   is ``except json.JSONDecodeError``. JSONDecodeError is a *subclass* of
   ValueError, not a superclass, so the bare ValueError propagates and kills the
   call. Standing rule 5 asks for a ClaimSet with zero claims; the specified
   code raises instead, precisely when the model declined to answer in JSON.

The strategies here are ordered cheapest-first and each is reported by name, so
an audit trail records how the answer had to be recovered. A model that needs
brace-scanning every time is a prompt problem worth seeing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("prismflow.v2.reasoners")

#: Longest model reply we will scan. Bounded so a runaway response cannot turn
#: into a quadratic parse.
MAX_SCAN_CHARS = 1_000_000

_FENCE = re.compile(
    r"```(?:json|JSON)?\s*(?P<body>\{.*?\})\s*```",
    re.DOTALL,
)


def extract_json_object(
    text: str,
    *,
    require_key: Optional[str] = "claims",
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """Recover a JSON object from a model reply.

    Args:
        text: the raw reply.
        require_key: prefer a candidate object containing this key. A model that
            emits an illustrative object before the real answer is common, and
            position alone does not identify the answer.

    Returns:
        ``(obj, strategy, problem)``. ``obj`` is None if nothing parsed, in which
        case ``problem`` says why. ``strategy`` names how it was found:
        ``direct``, ``fenced``, ``scan`` or ``none``.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "none", "model returned no text"
    if len(text) > MAX_SCAN_CHARS:
        return None, "none", (
            f"model reply is {len(text)} characters, above the "
            f"{MAX_SCAN_CHARS} scan limit"
        )

    stripped = text.strip()

    # 1. The whole reply is the object. The common case when the model is asked
    #    for JSON and complies.
    obj = _load_object(stripped)
    if obj is not None and _acceptable(obj, require_key):
        return obj, "direct", None

    # 2. A fenced code block. The next most common, because chat models like
    #    wrapping JSON in ```json fences.
    fenced: List[Dict[str, Any]] = []
    for match in _FENCE.finditer(text):
        candidate = _load_object(match.group("body"))
        if candidate is not None:
            fenced.append(candidate)
    chosen = _choose(fenced, require_key)
    if chosen is not None:
        return chosen, "fenced", None

    # 3. Balanced-brace scan. Unlike the greedy regex this yields each complete
    #    top-level object separately, and it tracks string literals so a brace
    #    inside a claim's text cannot end the object early.
    scanned = [
        candidate
        for candidate in (_load_object(raw) for raw in _balanced_objects(text))
        if candidate is not None
    ]
    chosen = _choose(scanned, require_key)
    if chosen is not None:
        return chosen, "scan", None

    # Nothing parsed. Report what the reply looked like, not just that it failed.
    if obj is not None:
        return obj, "direct", (
            f"parsed an object but it has no {require_key!r} key; keys: "
            f"{sorted(obj)[:8]}"
        )
    preview = stripped[:160].replace("\n", " ")
    return None, "none", f"no JSON object found in the reply (starts: {preview!r})"


def _acceptable(obj: Dict[str, Any], require_key: Optional[str]) -> bool:
    return require_key is None or require_key in obj


def _choose(
    candidates: List[Dict[str, Any]],
    require_key: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Prefer the first candidate carrying the required key, else the first."""
    if not candidates:
        return None
    if require_key is not None:
        for candidate in candidates:
            if require_key in candidate:
                return candidate
        return None
    return candidates[0]


def _load_object(raw: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield each complete top-level ``{...}`` span, respecting string literals.

    A claim reading ``the config {"a": 1} broke`` contains braces inside a JSON
    string; counting braces without tracking quotes would close the object at
    the wrong place and produce garbage that happens to parse.
    """
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : index + 1]
                    start = None


def claims_payload(obj: Optional[Dict[str, Any]]) -> Tuple[List[Any], Optional[str]]:
    """Pull the ``claims`` array out of a parsed object, tolerantly.

    A model asked for ``{"claims": [...]}`` sometimes returns the bare array, or
    a single claim object. Both are recoverable and both are reported, because a
    model that keeps doing it is a prompt to fix.
    """
    if obj is None:
        return [], None
    raw = obj.get("claims")
    if isinstance(raw, list):
        return raw, None
    if isinstance(raw, dict):
        return [raw], "'claims' was a single object rather than a list"
    if raw is None:
        return [], "parsed object has no 'claims' key"
    return [], f"'claims' is {type(raw).__name__}, not a list"


__all__ = ["extract_json_object", "claims_payload", "MAX_SCAN_CHARS"]
