"""Part 24's evaluation harness: the labeled dataset and its loader.

The dataset is authored by hand in ``data/v2/part24_labeled_dataset.xlsx``
and exported to ``data/v2/evaluation_queries.json``, which is what
``load_evaluation_dataset`` reads. The schema, the mappings and the six
validation rules are specified in ``docs/part24-dataset-schema.md``, which is
authority where it and ``.claude/commands/part24.md`` disagree.

Two files, not one, because they have different dependencies and different
audiences. ``dataset`` is imported by the metric code and depends only on the
standard library; ``export`` is a one-shot authoring tool that needs
``openpyxl``, which is not in ``requirements.txt``. Importing the loader must
not require a spreadsheet library.
"""

from prismflow.v2.evaluation.calibration import (
    DEFAULT_N_BINS,
    CalibrationReport,
    MetricInputError,
    calibration_arrays,
    calibration_brier_decomposition,
    calibration_reliability,
    compute_calibration,
)
from prismflow.v2.evaluation.conflict import (
    ConflictInputError,
    ConflictReport,
    compute_conflict_metrics,
    conflict_arrays,
)
from prismflow.v2.evaluation.dataset import (
    CONFLICT_BOOL,
    DOMAINS,
    OUTCOME_BINARY,
    DatasetValidationError,
    EvaluationQuery,
    load_evaluation_dataset,
)

__all__ = [
    "CONFLICT_BOOL",
    "DEFAULT_N_BINS",
    "DOMAINS",
    "OUTCOME_BINARY",
    "CalibrationReport",
    "ConflictInputError",
    "ConflictReport",
    "DatasetValidationError",
    "EvaluationQuery",
    "MetricInputError",
    "calibration_arrays",
    "calibration_brier_decomposition",
    "calibration_reliability",
    "compute_calibration",
    "compute_conflict_metrics",
    "conflict_arrays",
    "load_evaluation_dataset",
]
