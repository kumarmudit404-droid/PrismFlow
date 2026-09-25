# Part 24 evaluation dataset: the reconciled schema

**Status: authority.** Where this document and `.claude/commands/part24.md`
conflict on the dataset schema, this document wins, for the reasons recorded
under "Why the brief's schema was not adopted". The brief remains authoritative
for everything else in Part 24 -- the gates, the metrics, the run structure.

Written before any hand labeling began, so that no row has to be re-labeled
later. The human-editable source of truth is the workbook
`data/v2/part24_labeled_dataset.xlsx`; `data/v2/evaluation_queries.json` is
derived from it and is what the loader reads.

## The 8 authored fields

These are the only fields anyone fills by hand. They mirror the workbook's
columns exactly.

| field | type | rule |
|---|---|---|
| `id` | str | `"001"` .. `"050"` |
| `idea_pitch` | str | the pitch as it would have read BEFORE the outcome was known; no hindsight |
| `domain` | enum | one of `DOMAINS` |
| `actual_outcome` | enum | one of `OUTCOME_BINARY`'s keys |
| `outcome_date` | ISO-8601 date | the date the outcome became publicly verifiable, not the pitch date |
| `ground_truth_source` | URL | a real, checkable link; an unsourced label is not ground truth |
| `conflict_expected` | enum | one of `CONFLICT_BOOL`'s keys, judged BEFORE running the pipeline |
| `notes` | str | optional; ambiguity, paywalls, mixed signals |

```json
[
  {
    "id": "001",
    "idea_pitch": "<pre-outcome pitch text, as you'd query PrismFlow>",
    "domain": "Startup | Tech-OSS | Financial-Product",
    "actual_outcome": "Funded | Failed | Adopted | Rejected | Profitable | Broke-even | Loss",
    "outcome_date": "YYYY-MM-DD",
    "ground_truth_source": "https://...",
    "conflict_expected": "Yes | No | Unsure",
    "notes": ""
  }
]
```

## The mappings

Derived values are computed at load time from these tables. They are
deliberately NOT stored in the JSON: a mapping that lives in one versioned
place cannot drift between annotators, or between rows labeled on different
days.

```python
OUTCOME_BINARY = {
    "Funded": 1, "Adopted": 1, "Profitable": 1,
    "Failed": 0, "Rejected": 0, "Loss": 0,
    "Broke-even": None,          # scored for conflict/cost, excluded from ECE/Brier
}
DOMAINS = ("Startup", "Tech-OSS", "Financial-Product")
CONFLICT_BOOL = {"Yes": True, "No": False, "Unsure": None}
```

`Broke-even` maps to `None` rather than to 0. It is neither a success nor a
failure, and forcing it to 0 would put a fabricated label into the calibration
set. The row stays in the dataset and still counts for conflict metrics and
cost; it drops out of the calibration arrays only.

`Unsure` maps to `None` for the same reason. Coercing it to `False` would count
every annotator hesitation as a true negative and inflate conflict precision.

## The dataclass

```python
@dataclass(frozen=True)
class EvaluationQuery:
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
```

## `load_evaluation_dataset()` and its validation rules

```python
def load_evaluation_dataset(
    path: str = "data/v2/evaluation_queries.json",
) -> List[EvaluationQuery]:
    """Load the labeled queries. Raises on any row that is not ground truth."""
```

It **rejects, rather than skips**. A row that fails any rule below raises and
names the offending `id`. Skipping would let an unsourced or mislabeled row
silently shrink the benchmark, and a gate computed on a quietly smaller set is
a different claim than the one being reported.

Per row:

- `actual_outcome` is a key of `OUTCOME_BINARY`
- `domain` is in `DOMAINS`
- `conflict_expected` is a key of `CONFLICT_BOOL`
- `ground_truth_source` parses as `http(s)` and contains no `<` placeholder
  character (this is what catches a template row left in by accident)
- `outcome_date` parses as ISO-8601
- `idea_pitch` is non-empty

## The two call sites this schema exists to serve

```python
scored  = [q for q in dataset if q.scored_for_calibration]
labels  = np.array([q.outcome_binary for q in scored])              # ECE, Brier

conflict_rows  = [q for q in dataset if q.true_conflict is not None]
true_conflicts = np.array([q.true_conflict for q in conflict_rows]) # recall/precision
```

**Every ECE, Brier and conflict figure must be reported with its own `n`.**
The `Broke-even` and `Unsure` exclusions mean the calibration set and the
conflict set have different denominators, and both are smaller than the row
count. A gate of ECE < 0.15 met on 31 rows is not the same claim as one met on
50, and the numbers must not be presented as if they were.

`reference_urls` from the brief, if a caller needs that plural shape, is
`[ground_truth_source]`. One checkable source per row is the workbook's rule; a
list field invites unsourced padding.

## Why the brief's schema was not adopted

Field by field, against `.claude/commands/part24.md`.

### Outcome labels: 7 (workbook), not 3 (brief)

The brief specifies `ground_truth_label` as `success` / `fail` / `partial`.
**Its own metric code cannot consume that.**
`compute_expected_calibration_error` computes `labels[mask].mean()`, which
requires a binary numeric array, and `brier_score_loss` requires binary as
well. Three string categories can be averaged by neither, and `partial` has no
numeric value at all.

Beyond that, the 7 workbook values are observable facts with a citable source
("the Series B closed", "the PEP was rejected", "the segment reported a loss"),
whereas success/fail/partial is an annotator judgment that is unsourceable and
collapses the very distinction that makes `ground_truth_source` checkable.
Storing the fact lets any judgment be re-derived later; storing the judgment
loses the fact. The binary the metrics need is derived, not authored.

### Domains: 3 (workbook), not 4 (brief)

The brief names `startup, tech, career, policy`.

- **`career` cannot satisfy `ground_truth_source` by construction.** There is
  no public, checkable link recording whether an individual's career move
  succeeded. A domain that structurally fails the sourcing rule does not belong
  in the schema.
- **`policy` has no live retrieval.** The connectors that actually retrieve are
  arXiv, GitHub, StackExchange, yfinance and NewsAPI. Nothing retrieves
  legislative or regulatory text, so policy queries would be scored against
  angles with no evidence to reason over, and the resulting ECE would measure
  retrieval starvation rather than calibration.

The three workbook domains each map to a connector that retrieves and to an
outcome type with a citable source.

### `conflict_expected`: kept

It is the only source of `true_conflicts`. The brief's `EvaluationQuery` has
**no field for it**, while the brief's own `compute_conflict_metrics(detected,
true_conflicts)` requires it -- so one of Part 24's four gates, conflict recall
> 0.7, is uncomputable from the brief's own schema. The workbook's Instructions
tab already defines the term correctly: two or more angles pointing in
substantively different DIRECTIONS, not merely differing confidence numbers.

### `expected_confidence`: dropped

The brief defines it as "what we expect the model to output". That is not
ground truth. It has no source, it would have to be invented per row, and it is
the one field that would actively corrupt the metric: calibration means
comparing the system's confidence against WHAT HAPPENED, and scoring it against
a human's prediction of what the model should say measures
agreement-with-annotator instead.

Nothing in the metric code needs it.
`compute_expected_calibration_error(confidences, labels)` takes its confidences
from the pipeline output, not from the dataset.

---

# FLAGGED: Part 24 implementation defect (not a schema question)

**`sklearn.metrics.expected_calibration_error` does not exist.**

`.claude/commands/part24.md` opens `prismflow/v2/evaluation/calibration.py`
with:

```python
from sklearn.metrics import expected_calibration_error, brier_score_loss
```

Verified against the installed scikit-learn 1.9.1: `brier_score_loss` exists,
`expected_calibration_error` does not. The import raises `ImportError` and the
module cannot load.

The brief then hand-rolls the same function immediately below the import, so
the import is dead as well as broken -- deleting the
`expected_calibration_error` name from it is the whole fix, and the hand-rolled
implementation is the one to keep.

Recorded here because it was found while reconciling the schema. It is a defect
in the Part 24 implementation spec, not a property of the dataset, and it is
filed separately so that it is not mistaken for one. It does not affect any
field above.
