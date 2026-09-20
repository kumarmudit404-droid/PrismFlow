# Installation

Verified on Windows 11, Python 3.13.7, CPU only. Linux and macOS should work
unchanged; neither was tested.

## Requirements

- Python 3.11 or newer (3.13.7 is what the committed results were produced on)
- ~2 GB disk for the virtual environment, most of it torch
- No GPU required. Every committed result was produced on CPU.

## Setup

```bash
git clone <repository-url>
cd "PrIsM FlOw"

python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install "streamlit>=1.40"     # see the note below
```

### Why `streamlit` is installed separately

`requirements.txt` declares `numpy`, `torch`, `PyYAML`, `pytest` and
`matplotlib`. It does **not** declare `streamlit`, although `app.py` and
`prismflow/app/panels.py` both import it and the demo cannot start without it.

This is a real gap, recorded as Finding 3 in `AUDIT_REPORT.md`. It was left
unfixed rather than patched quietly because `requirements.txt` is outside the
set of files the Part 16 task was scoped to change. The correct fix is to add
one line to `requirements.txt`:

```
streamlit>=1.40
```

Until that is done, install it explicitly as shown above. The demo was verified
on streamlit 1.64.0.

## Verify the installation

```bash
python -m pytest -q
```

Expect **563 passed** in roughly 40-50 seconds. Anything less means the
environment is incomplete; the suite has no skipped or conditional tests.

Then confirm the engine is deterministic on your machine:

```bash
python -c "
from prismflow.data.dataset import split_indices
import numpy as np
a, b = split_indices(2000, seed=3), split_indices(2000, seed=3)
print('deterministic:', all(np.array_equal(a[k], b[k]) for k in a))
"
```

## Run the demo

```bash
streamlit run app.py --server.fileWatcherType none
```

**`--server.fileWatcherType none` is not optional.** Streamlit's default
watcher walks the module table and touches `torch.classes`, whose custom
`__getattr__` raises on the attribute the watcher probes. Without the flag the
app dies with:

```
Tried to instantiate class '__path__._path'
```

Disabling the watcher costs only hot-reload. See `docs/DEMO_GUIDE.md` for what
to do once it is running.

## Data

Synthetic data is generated on demand from `prismflow/data/synthetic.py` and
needs no download.

The real dataset (`handwritten`, the UCI multi-feature set, 6 views) is loaded
through `prismflow/data/real_datasets.py`. If `data/raw/` is empty, follow the
acquisition notes in `docs/DATASETS.md`. Only the Part 15 experiments and the
demo's real-data path need it; everything else runs on synthetic data alone.

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`** — the virtual environment is
not active, or you are invoking the system Python. On Windows, call the venv
interpreter directly: `./.venv/Scripts/python.exe -m pytest -q`.

**`ModuleNotFoundError: No module named 'prismflow'`** — run from the
repository root. Every experiment is a module (`python -m experiments...`) and
relies on the root being on `sys.path`.

**`Tried to instantiate class '__path__._path'`** — the missing
`--server.fileWatcherType none` flag. See above.

**The test suite passes but an experiment fails on import** — check you are on
Python 3.11+; the codebase uses `X | Y` type syntax and `str.removesuffix`.
