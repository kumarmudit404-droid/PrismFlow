"""The labeled evaluation dataset, and the loader that refuses bad rows.

Part 24. An ENGINEERING component per docs/CONTRACT.md section 3 -- no
learnable parameters. Every Part 24 number (ECE, Brier, conflict recall and
precision) is computed against what this module returns, so a row that reaches
the metrics without being ground truth does not produce a slightly worse
result -- it produces a different claim than the one reported.

The schema is specified in ``docs/part24-dataset-schema.md``. Where that
document and ``.claude/commands/part24.md`` conflict, the document wins; its
"Why the brief's schema was not adopted" section records the reasons.

IT REJECTS, IT DOES NOT SKIP
----------------------------
A row failing any rule raises ``DatasetValidationError`` naming the offending
``id``. Skipping would let an unsourced or mislabeled row silently shrink the
benchmark, and a gate met on a quietly smaller set is a different claim than
the one being reported. This is also why the loader does not repair values --
no stripping a stray ``<``, no coercing an unknown outcome to the nearest
legal one. A row that needs repairing needs a human to look at it.

THE TWO ``None``s ARE DELIBERATE AND ARE NOT ERRORS
---------------------------------------------------
``Broke-even`` maps to ``None`` rather than 0: it is neither a success nor a
failure, and forcing it to 0 would put a fabricated label into the calibration
set. ``Unsure`` maps to ``None`` for the same reason -- coercing it to
``False`` would count an annotator's hesitation as a true negative and inflate
conflict precision. Both rows stay in the dataset; they drop out of one array
each. Callers filter with ``scored_for_calibration`` and ``true_conflict is
not None``, which is why the calibration set and the conflict set have
different denominators and both are smaller than ``len(dataset)``. Report each
figure with its own ``n``.

THE DERIVED VALUES ARE NOT STORED
---------------------------------
``outcome_binary`` and ``true_conflict`` are properties, not fields, and the
mappings live here rather than in the JSON. A mapping that lives in one
versioned place cannot drift between annotators, or between rows labeled on
different days.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# The mappings. Derived at load time, deliberately absent from the JSON.
# --------------------------------------------------------------------------
OUTCOME_BINARY: Dict[str, Optional[int]] = {
    "Funded": 1, "Adopted": 1, "Profitable": 1,
    "Failed": 0, "Rejected": 0, "Loss": 0,
    "Broke-even": None,        # scored for conflict/cost, excluded from ECE/Brier
}

DOMAINS = ("Startup", "Tech-OSS", "Financial-Product")

CONFLICT_BOOL: Dict[str, Optional[bool]] = {
    "Yes": True, "No": False, "Unsure": None,
}

DEFAULT_DATASET_PATH = "data/v2/evaluation_queries.json"

# The eight authored fields, in workbook column order.
AUTHORED_FIELDS = (
    "id", "idea_pitch", "domain", "actual_outcome",
    "outcome_date", "ground_truth_source", "conflict_expected", "notes",
)


class DatasetValidationError(ValueError):
    """A row is not ground truth. Carries the ``id`` that failed."""

    def __init__(self, row_id: str, reason: str) -> None:
        self.row_id = row_id
        self.reason = reason
        super().__init__("row %s: %s" % (row_id, reason))


@dataclass(frozen=True)
class EvaluationQuery:
    """One labeled idea: what was pitched, and what actually happened.

    Attributes:
        id: ``"001"`` .. ``"050"``, the workbook's row identifier.
        idea_pitch: the pitch as it would have read BEFORE the outcome was
            known. This is what the pipeline is pointed at, so hindsight here
            leaks the answer into the input.
        domain: one of ``DOMAINS``.
        actual_outcome: one of ``OUTCOME_BINARY``'s keys -- an observable fact
            with a citable source, not an annotator's success/fail judgment.
        outcome_date: the day the outcome became publicly verifiable, not the
            day the idea was pitched.
        ground_truth_source: one real, checkable URL. One, not a list: a
            plural field invites unsourced padding.
        conflict_expected: one of ``CONFLICT_BOOL``'s keys, judged BEFORE the
            pipeline was run. It is the only source of ``true_conflicts``,
            without which the conflict-recall gate is uncomputable.
        notes: optional; ambiguity, paywalls, mixed signals.
    """

    # --- authored, straight from the workbook -------------------------
    id: str
    idea_pitch: str
    domain: str
    actual_outcome: str
    outcome_date: date
    ground_truth_source: str
    conflict_expected: str
    notes: str = ""

    # --- derived, never hand-entered ----------------------------------
    @property
    def query_text(self) -> str:
        """What the pipeline is actually pointed at."""
        return self.idea_pitch

    @property
    def outcome_binary(self) -> Optional[int]:
        return OUTCOME_BINARY[self.actual_outcome]

    @property
    def true_conflict(self) -> Optional[bool]:
        return CONFLICT_BOOL[self.conflict_expected]

    @property
    def scored_for_calibration(self) -> bool:
        return self.outcome_binary is not None

    @property
    def reference_urls(self) -> List[str]:
        """The plural shape, for callers from the brief that expect one."""
        return [self.ground_truth_source]


# --------------------------------------------------------------------------
# The six validation rules
# --------------------------------------------------------------------------
def _validate_row(raw: Dict[str, Any], row_id: str) -> date:
    """Apply the six rules. Raise on the first failure; return the parsed date.

    The rules are checked in the order docs/part24-dataset-schema.md lists
    them, so a row with several problems reports the same one every run.
    """
    # 1. actual_outcome is a key of OUTCOME_BINARY
    outcome = raw.get("actual_outcome")
    if outcome not in OUTCOME_BINARY:
        raise DatasetValidationError(
            row_id, "actual_outcome %r is not one of %s"
            % (outcome, sorted(OUTCOME_BINARY)))

    # 2. domain is in DOMAINS
    domain = raw.get("domain")
    if domain not in DOMAINS:
        raise DatasetValidationError(
            row_id, "domain %r is not one of %s" % (domain, list(DOMAINS)))

    # 3. conflict_expected is a key of CONFLICT_BOOL
    conflict = raw.get("conflict_expected")
    if conflict not in CONFLICT_BOOL:
        raise DatasetValidationError(
            row_id, "conflict_expected %r is not one of %s"
            % (conflict, sorted(CONFLICT_BOOL)))

    # 4. ground_truth_source parses as http(s) and holds no '<' placeholder.
    #    The '<' check is what catches a template row left in by accident.
    source = raw.get("ground_truth_source")
    if not isinstance(source, str) or not source.strip():
        raise DatasetValidationError(row_id, "ground_truth_source is empty")
    parsed = urlparse(source.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise DatasetValidationError(
            row_id, "ground_truth_source %r does not parse as an http(s) URL"
            % source[:80])
    if "<" in source:
        raise DatasetValidationError(
            row_id, "ground_truth_source %r contains a '<' placeholder -- "
            "this looks like an unreplaced template row" % source[:80])

    # 5. outcome_date parses as ISO-8601
    raw_date = raw.get("outcome_date")
    if isinstance(raw_date, date):
        parsed_date = raw_date
    else:
        if not isinstance(raw_date, str) or not raw_date.strip():
            raise DatasetValidationError(row_id, "outcome_date is empty")
        try:
            parsed_date = date.fromisoformat(raw_date.strip())
        except ValueError:
            raise DatasetValidationError(
                row_id, "outcome_date %r is not an ISO-8601 date" % raw_date
            ) from None

    # 6. idea_pitch is non-empty
    pitch = raw.get("idea_pitch")
    if not isinstance(pitch, str) or not pitch.strip():
        raise DatasetValidationError(row_id, "idea_pitch is empty")

    return parsed_date


def load_evaluation_dataset(
    path: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> List[EvaluationQuery]:
    """Load the labeled queries. Raises on any row that is not ground truth.

    Args:
        path: the exported JSON, not the workbook. The workbook is the
            human-editable source of truth; this file is derived from it by
            ``prismflow.v2.evaluation.export`` and is what the pipeline reads.

    Returns:
        Every row in file order. Nothing is filtered -- a caller wanting only
        the calibration rows filters on ``scored_for_calibration``, and the
        resulting ``n`` is part of the claim.

    Raises:
        FileNotFoundError: the export has not been run.
        DatasetValidationError: a row failed one of the six rules, named by id.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "%s does not exist. It is derived from the workbook; run "
            "'python -m prismflow.v2.evaluation.export' to generate it."
            % dataset_path)

    with dataset_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list):
        raise DatasetValidationError(
            "<file>",
            "expected a JSON list of rows, got %s" % type(rows).__name__)

    queries: List[EvaluationQuery] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise DatasetValidationError(
                "<position %d>" % position,
                "expected an object, got %s" % type(raw).__name__)
        row_id = str(raw.get("id", "<position %d>" % position))

        parsed_date = _validate_row(raw, row_id)

        queries.append(EvaluationQuery(
            id=row_id,
            idea_pitch=raw["idea_pitch"],
            domain=raw["domain"],
            actual_outcome=raw["actual_outcome"],
            outcome_date=parsed_date,
            ground_truth_source=raw["ground_truth_source"].strip(),
            conflict_expected=raw["conflict_expected"],
            notes=raw.get("notes") or "",
        ))

    return queries
