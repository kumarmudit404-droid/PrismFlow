"""Dataset-agnostic scaling and training (L25).

Part 28 wrote `fit_scaler`, `apply_scaler` and `train_ait` inside
`experiments/ait_clean/run_ait_clean.py`, and each resolved a view's native
feature width through `prismflow.data.ait_lds.FEATURES[source]` -- a table
keyed by AIT source names. Part 30 could not import them for LANL, whose
views are `auth, proc, flows, dns`, so it mirrored all three
(`experiments/lanl/run_lanl.py`). That duplication is L25.

The functions here are the same arithmetic with the lookup removed: the
caller PASSES the native widths in as a list of ints. Nothing in this package
imports a dataset module, reads a source-name table, or knows what a view is
called, so a third dataset needs no fourth copy.

Both runners keep their own copies and are not modified; this package is
what a later Part should call. `tests/unit/test_training_agnostic.py` pins it
to their output.
"""

from prismflow.training.scaling import apply_scaler, fit_scaler
from prismflow.training.trainer import encoder_configs, train

__all__ = ["fit_scaler", "apply_scaler", "train", "encoder_configs"]
