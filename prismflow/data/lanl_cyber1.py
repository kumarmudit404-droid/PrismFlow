"""Part 30 - LANL cyber1 loader: streaming, read-only, (computer, window) samples.

WHAT THIS MODULE DOES AND DOES NOT DO
-------------------------------------
It streams the five gzipped text files of LANL's "Comprehensive, Multi-Source
Cyber-Security Events" (cyber1) release and builds an in-memory multi-view
dataset whose interface matches `prismflow.data.ait_lds.AITMultiViewDataset`,
so the Part 28 evaluation code can score it without being edited.

It NEVER writes, extracts, renames, moves or deletes anything under
`data/raw/`. Every file is read through `gzip -dc` as a pipe and only a prefix
of each file is read: the files are sorted by time, so the loader stops at
`read_until` seconds. Timestamps are verified non-decreasing over the whole
portion scanned, and an inversion raises `TimestampInversion` naming the file,
the line number and the two values rather than sorting anything.

WHY THE SAMPLE SET IS A CASE-CONTROL SAMPLE
-------------------------------------------
`.claude/commands/part30.md` step 1 fixes the sample as `(computer, time
window)`. At the 300 s window this dataset occupies roughly 1.5 million
occupied cells PER DAY (measured; see `results/lanl/population.json`), so days
1-16 are on the order of 2e7 samples, while the red-team file marks only 440
attack cells in the train block and 410 in the test block - a prevalence near
2e-5. Training 40 epochs over 2e7 samples is not reachable in this Part's time
budget, and a uniform subsample that WOULD be reachable leaves single-digit
attack cells and so trips this Part's own stop condition of at least 10 attack
windows per split.

So the sample set is a case-control sample, declared rather than disguised:

  * every attack cell in the read range is kept (all 440 train, all 410 test);
  * a benign cell is kept iff `crc32(computer|window) % control_divisor == 0`.

The control rule is a DETERMINISTIC HASH, not a draw from a random generator.
That matters three ways. It needs no prior knowledge of the cell universe, so
one streaming pass suffices; it is identical on every rerun and on every
experiment seed, so a seed varies only model initialisation and batch order
exactly as in Part 28; and it is computed from the cell identity alone, so no
label and no feature value influences which controls appear.

The cost is that sample prevalence is an artefact of `control_divisor` and is
NOT the population prevalence. Accuracy and any other prevalence-dependent
figure computed on this sample therefore describes the sample, not the network.
Balanced accuracy, ROC AUC and per-class recall are the figures that survive
the reweighting, which is why `part28_extras` reports them.

FEATURES
--------
`part30.md` names no features. `configs/data/lanl.yaml` pre-registers the
plainest count-based per-window aggregates of each view's own columns - event
counts, distinct users and computers, and the auth failure rate - committed
before any model ran. Counts are `log1p` transformed, matching the AIT
loader's documented transform; `failure_rate` is already a fraction and is not
transformed. No feature reads the label file, and no feature combines views.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from prismflow.data.dataset import Batch, Sample

#: Class codes, same orientation as Part 28 (`run_ait_clean.CLASS_*`).
CLASS_BENIGN = 0
CLASS_ATTACK = 1

#: Per-view feature names, in tensor column order. Mirrors
#: `configs/data/lanl.yaml` `features:`; the config is the pre-registered
#: statement and `load_lanl_config` checks this constant against it.
FEATURES: dict[str, tuple[str, ...]] = {
    "auth": (
        "n_events",
        "n_distinct_src_user",
        "n_distinct_dst_user",
        "n_distinct_dst_computer",
        "n_success",
        "n_failure",
        "n_distinct_auth_type",
        "failure_rate",
    ),
    "proc": ("n_events", "n_distinct_user", "n_distinct_process", "n_start", "n_end"),
    "flows": (
        "n_events",
        "n_distinct_dst_computer",
        "n_distinct_src_port",
        "n_distinct_dst_port",
        "sum_packets",
        "sum_bytes",
        "sum_duration",
    ),
    "dns": ("n_events", "n_distinct_resolved"),
}

#: Features that are NOT counts and so are not log1p transformed.
NOT_COUNTS = frozenset({"failure_rate"})

#: 0-based column of the computer each record is attributed to, and the record
#: width used to reject a malformed line loudly instead of silently mis-parsing.
SOURCE_COLUMN = {"auth": 3, "proc": 2, "flows": 2, "dns": 1}
RECORD_WIDTH = {"auth": 9, "proc": 5, "flows": 9, "dns": 3}

LABEL_MEANING = (
    "A sample is (computer, 300 s window). attack = 1 means a redteam.txt.gz "
    "event in that window names that computer as its source or its "
    "destination. benign = 0 means NO KNOWN red-team event does, which is not "
    "the same as verified clean: the red-team file covers only the activity "
    "LANL's red team recorded, so an unlabelled compromise inside the capture "
    "is counted as benign here."
)


class TimestampInversion(RuntimeError):
    """Raised when a scanned file's timestamps decrease. Never repaired."""


class GzipStreamError(RuntimeError):
    """Raised when a `gzip -dc` pipe exits non-zero."""


@dataclasses.dataclass(frozen=True)
class LANLConfig:
    raw_root: Path
    day_seconds: int
    window_seconds: int
    read_until: int
    views: tuple[str, ...]
    feature_dim: int
    split: dict
    label_definition: str
    control_divisor: int
    log1p_counts: bool

    def window_of(self, t: int) -> int:
        return (t - 1) // self.window_seconds

    def day_of(self, t: int) -> int:
        return (t - 1) // self.day_seconds + 1

    def block_of_window(self, window: int) -> str:
        """Which split a whole window belongs to. A window never straddles."""
        start = window * self.window_seconds + 1
        for name in ("train", "gap", "test"):
            bounds = self.split[name]
            if bounds["start"] <= start <= bounds["end"]:
                return name
        return "unused"


def load_lanl_config(path: str | Path = "configs/data/lanl.yaml") -> LANLConfig:
    raw = yaml.safe_load(Path(path).read_text())
    views = tuple(raw["views"])
    for view in views:
        declared = tuple(raw["features"][view])
        if declared != FEATURES[view]:
            raise ValueError(
                f"config feature list for {view!r} does not match "
                f"prismflow.data.lanl_cyber1.FEATURES: {declared} != {FEATURES[view]}"
            )
    # Every split boundary must land on a window edge, or a window would belong
    # to two splits. Checked here so no caller can opt out.
    w = int(raw["window_seconds"])
    for name, bounds in raw["split"].items():
        if (bounds["start"] - 1) % w or bounds["end"] % w:
            raise ValueError(f"split {name} does not align to a {w}s window grid")
    return LANLConfig(
        raw_root=Path(raw["raw_root"]),
        day_seconds=int(raw["day_seconds"]),
        window_seconds=w,
        read_until=int(raw["read_until"]),
        views=views,
        feature_dim=int(raw["feature_dim"]),
        split=raw["split"],
        label_definition=raw["label"]["definition"],
        control_divisor=int(raw["control_divisor"]),
        log1p_counts=bool(raw["log1p_counts"]),
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def read_redteam(cfg: LANLConfig) -> dict:
    """Attack cells and attacked windows from redteam.txt.gz.

    Duplicate rows are kept and simply collapse, because a window is binary.
    Both label readings are returned: `cells` is the pre-registered primary
    (computer, window) reading, `windows` is the window-only alternative that
    `configs/data/lanl.yaml` also reports.
    """
    path = cfg.raw_root / "redteam.txt.gz"
    cells: set[tuple[bytes, int]] = set()
    windows: set[int] = set()
    rows = 0
    events_per_day: dict[int, int] = defaultdict(int)
    for fields, _lineno in _stream(path, "redteam", 4, until=None):
        t = int(fields[0])
        rows += 1
        events_per_day[cfg.day_of(t)] += 1
        window = cfg.window_of(t)
        windows.add(window)
        cells.add((fields[2], window))
        cells.add((fields[3], window))
    return {
        "rows": rows,
        "cells": cells,
        "windows": windows,
        "events_per_day": dict(sorted(events_per_day.items())),
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _stream(path: Path, view: str, width: int, until: int | None):
    """Yield (fields, lineno) from `gzip -dc path`, stopping once t > until.

    Verifies non-decreasing timestamps over everything it scans and raises
    rather than sorting. Closing the pipe early is a normal stop, so a SIGPIPE
    exit is not treated as a gzip failure; any other non-zero exit is.
    """
    proc = subprocess.Popen(
        ["gzip", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1 << 20,
    )
    previous = 0
    lineno = 0
    stopped_early = False
    try:
        for raw in proc.stdout:
            lineno += 1
            fields = raw.rstrip(b"\r\n").split(b",")
            if len(fields) != width:
                raise ValueError(
                    f"{path.name} line {lineno}: expected {width} fields, "
                    f"got {len(fields)}"
                )
            t = int(fields[0])
            if t < previous:
                raise TimestampInversion(
                    f"{path.name} line {lineno}: timestamp {t} follows {previous}"
                )
            previous = t
            if until is not None and t > until:
                stopped_early = True
                break
            yield fields, lineno
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if stopped_early:
            proc.terminate()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stderr is not None:
            proc.stderr.close()
        code = proc.wait()
        if not stopped_early and code != 0:
            raise GzipStreamError(f"gzip -dc {path.name} exited {code}: {stderr!r}")


def _keep(cell: tuple[bytes, int], attack_cells: set, divisor: int) -> bool:
    """All cases, plus a deterministic 1/divisor hash sample of controls."""
    if cell in attack_cells:
        return True
    key = cell[0] + b"|" + str(cell[1]).encode()
    return zlib.crc32(key) % divisor == 0


def scan_view(view: str, cfg: LANLConfig, attack_cells: set) -> dict:
    """One streaming pass over one view. Returns kept cells plus population counts.

    Only KEPT cells get an accumulator, so memory is bounded by the sample
    size, not by the ~2e7 cells the capture occupies. The inclusion decision is
    cached per window, so crc32 runs about once per cell rather than once per
    record.

    `population_cells` is the exact number of occupied (computer, window) cells
    this view has in the read range - counted with a per-window set that is
    discarded when the window advances, which the files' time ordering makes
    correct.
    """
    path = cfg.raw_root / f"{view}.txt.gz"
    source = SOURCE_COLUMN[view]
    width = RECORD_WIDTH[view]
    names = FEATURES[view]
    divisor = cfg.control_divisor

    kept: dict[tuple[bytes, int], dict] = {}
    decision: dict[bytes, bool] = {}
    seen_this_window: set[bytes] = set()
    current_window = -1
    population_cells = 0
    population_per_day: dict[int, int] = defaultdict(int)
    records = 0
    last_day = 0

    for fields, _lineno in _stream(path, view, width, cfg.read_until):
        t = int(fields[0])
        records += 1
        window = cfg.window_of(t)
        if window != current_window:
            population_cells += len(seen_this_window)
            population_per_day[last_day] += len(seen_this_window)
            seen_this_window = set()
            decision = {}
            current_window = window
            last_day = cfg.day_of(t)
        computer = fields[source]
        seen_this_window.add(computer)
        take = decision.get(computer)
        if take is None:
            take = _keep((computer, window), attack_cells, divisor)
            decision[computer] = take
        if not take:
            continue
        cell = (computer, window)
        acc = kept.get(cell)
        if acc is None:
            acc = kept[cell] = _new_accumulator(view)
        _accumulate(view, acc, fields)

    population_cells += len(seen_this_window)
    population_per_day[last_day] += len(seen_this_window)

    return {
        "view": view,
        "records": records,
        "population_cells": population_cells,
        "population_cells_per_day": dict(sorted(population_per_day.items())),
        "kept": {cell: _finalise(view, acc, names, cfg) for cell, acc in kept.items()},
    }


def _new_accumulator(view: str) -> dict:
    if view == "auth":
        return {"n": 0, "su": set(), "du": set(), "dc": set(), "ok": 0, "bad": 0, "at": set()}
    if view == "proc":
        return {"n": 0, "u": set(), "p": set(), "start": 0, "end": 0}
    if view == "flows":
        return {"n": 0, "dc": set(), "sp": set(), "dp": set(), "pk": 0, "by": 0, "dur": 0}
    return {"n": 0, "rc": set()}


def _accumulate(view: str, acc: dict, f: list[bytes]) -> None:
    acc["n"] += 1
    if view == "auth":
        acc["su"].add(f[1])
        acc["du"].add(f[2])
        acc["dc"].add(f[4])
        acc["at"].add(f[5])
        if f[8] == b"Success":
            acc["ok"] += 1
        elif f[8] == b"Fail":
            acc["bad"] += 1
    elif view == "proc":
        acc["u"].add(f[1])
        acc["p"].add(f[3])
        if f[4] == b"Start":
            acc["start"] += 1
        elif f[4] == b"End":
            acc["end"] += 1
    elif view == "flows":
        acc["dc"].add(f[4])
        acc["sp"].add(f[3])
        acc["dp"].add(f[5])
        acc["pk"] += int(f[7])
        acc["by"] += int(f[8])
        acc["dur"] += int(f[1])
    else:
        acc["rc"].add(f[2])


def _finalise(view: str, acc: dict, names: tuple[str, ...], cfg: LANLConfig) -> np.ndarray:
    if view == "auth":
        decided = acc["ok"] + acc["bad"]
        raw = [
            acc["n"], len(acc["su"]), len(acc["du"]), len(acc["dc"]),
            acc["ok"], acc["bad"], len(acc["at"]),
            (acc["bad"] / decided) if decided else 0.0,
        ]
    elif view == "proc":
        raw = [acc["n"], len(acc["u"]), len(acc["p"]), acc["start"], acc["end"]]
    elif view == "flows":
        raw = [
            acc["n"], len(acc["dc"]), len(acc["sp"]), len(acc["dp"]),
            acc["pk"], acc["by"], acc["dur"],
        ]
    else:
        raw = [acc["n"], len(acc["rc"])]
    out = np.asarray(raw, dtype=np.float64)
    if cfg.log1p_counts:
        for i, name in enumerate(names):
            if name not in NOT_COUNTS:
                out[i] = np.log1p(out[i])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LANLSamples:
    views: np.ndarray            # [N, V, D]
    view_mask: np.ndarray        # [N, V] bool
    labels: np.ndarray           # [N] int64
    label_known: np.ndarray      # [N] bool, all True (see config)
    window_start: np.ndarray     # [N] float64, seconds
    computer: list[str]          # [N]
    block: list[str]             # [N] train | test
    sources: tuple[str, ...]
    config: LANLConfig
    population: dict


def build_lanl_samples(cfg: LANLConfig, scans: dict[str, dict], redteam: dict) -> LANLSamples:
    """Join the per-view scans into one [N, V, D] tensor over the union of cells."""
    attack_cells = redteam["cells"]
    cells: set[tuple[bytes, int]] = set()
    for scan in scans.values():
        cells.update(scan["kept"])
    # A cell is only a sample if its whole window sits in train or test; the gap
    # and days 17-30 are dropped here, not filtered downstream.
    ordered = sorted(
        c for c in cells if cfg.block_of_window(c[1]) in ("train", "test")
    )

    n, v, d = len(ordered), len(cfg.views), cfg.feature_dim
    views = np.zeros((n, v, d), dtype=np.float32)
    mask = np.zeros((n, v), dtype=bool)
    labels = np.zeros(n, dtype=np.int64)
    window_start = np.zeros(n, dtype=np.float64)
    computers, blocks = [], []
    index = {cell: i for i, cell in enumerate(ordered)}

    for vi, view in enumerate(cfg.views):
        native = len(FEATURES[view])
        for cell, vector in scans[view]["kept"].items():
            i = index.get(cell)
            if i is None:
                continue
            views[i, vi, :native] = vector
            mask[i, vi] = True

    for i, (computer, window) in enumerate(ordered):
        labels[i] = CLASS_ATTACK if (computer, window) in attack_cells else CLASS_BENIGN
        window_start[i] = window * cfg.window_seconds + 1
        computers.append(computer.decode())
        blocks.append(cfg.block_of_window(window))

    population = {
        "per_view": {
            view: {
                "records_scanned": scans[view]["records"],
                "population_cells": scans[view]["population_cells"],
                "population_cells_per_day": scans[view]["population_cells_per_day"],
            }
            for view in cfg.views
        },
        "control_divisor": cfg.control_divisor,
        "redteam_rows": redteam["rows"],
        "redteam_events_per_day": redteam["events_per_day"],
    }
    return LANLSamples(
        views=views, view_mask=mask, labels=labels,
        label_known=np.ones(n, dtype=bool),
        window_start=window_start, computer=computers, block=blocks,
        sources=cfg.views, config=cfg, population=population,
    )


def load_lanl(cfg: LANLConfig, scan=None) -> LANLSamples:
    """Full read: redteam first (cheap, and it defines the cases), then the views."""
    redteam = read_redteam(cfg)
    runner = scan or (lambda view: scan_view(view, cfg, redteam["cells"]))
    scans = {view: runner(view) for view in cfg.views}
    return build_lanl_samples(cfg, scans, redteam)


def save_samples(samples: LANLSamples, path: Path) -> dict:
    """Write the whole sample table to a compressed .npz and return its sha256.

    Saved so that every later arm reads the SAME rows without rescanning 515M
    records. The four view streams are read exactly once per Part; an arm that
    drops a view drops feature columns from this table and nothing else, which
    is what makes the arms paired rather than merely similar.

    `config` is stored as the JSON of the pre-registered yaml path plus the
    resolved values, not as a pickle, so the file stays readable without this
    module.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "sources": list(samples.sources),
        "window_seconds": samples.config.window_seconds,
        "day_seconds": samples.config.day_seconds,
        "read_until": samples.config.read_until,
        "control_divisor": samples.config.control_divisor,
        "split": samples.config.split,
        "features": {v: list(FEATURES[v]) for v in samples.sources},
        "population": samples.population,
    }
    np.savez_compressed(
        path,
        views=samples.views,
        view_mask=samples.view_mask,
        labels=samples.labels,
        label_known=samples.label_known,
        window_start=samples.window_start,
        computer=np.asarray(samples.computer, dtype=object),
        block=np.asarray(samples.block, dtype=object),
        meta=np.asarray([json.dumps(meta)], dtype=object),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": str(path).replace("\\", "/"),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "n_samples": int(samples.views.shape[0]),
        "n_views": int(samples.views.shape[1]),
        "feature_dim": int(samples.views.shape[2]),
        "sources": list(samples.sources),
    }


def load_samples(path: Path, cfg: LANLConfig) -> LANLSamples:
    """Read the sample table back. Fails loudly if it disagrees with the config.

    The checks matter: an arm that silently trained on a table built under a
    different window length or control divisor would be comparing against the
    other arms on different data.
    """
    with np.load(path, allow_pickle=True) as handle:
        meta = json.loads(handle["meta"][0])
        if tuple(meta["sources"]) != tuple(cfg.views):
            raise ValueError(
                f"{path.name} holds views {meta['sources']}, config says "
                f"{list(cfg.views)}"
            )
        for key, value in (
            ("window_seconds", cfg.window_seconds),
            ("control_divisor", cfg.control_divisor),
            ("read_until", cfg.read_until),
        ):
            if meta[key] != value:
                raise ValueError(
                    f"{path.name} was built with {key}={meta[key]}, config "
                    f"says {value}"
                )
        return LANLSamples(
            views=handle["views"],
            view_mask=handle["view_mask"],
            labels=handle["labels"],
            label_known=handle["label_known"],
            window_start=handle["window_start"],
            computer=[str(c) for c in handle["computer"]],
            block=[str(b) for b in handle["block"]],
            sources=tuple(meta["sources"]),
            config=cfg,
            population=meta["population"],
        )


class LANLMultiViewDataset:
    """In-memory LANL dataset, interface-identical to `AITMultiViewDataset`.

    Same attributes and the same `get_sample` / `get_batch` return types as the
    synthetic loader, `RealMultiViewDataset` and the AIT loader, so Part 28's
    evaluation code scores it unchanged.

    `rho_matrix` is None for the reason the other real loaders give: real views
    have no ground-truth cross-view dependence, and this attribute must fail
    loudly rather than hand an estimate to code that wants a target.

    Like the AIT loader and unlike the handwritten one, this does NOT z-score.
    Scaling has to be fitted on train rows only, and that is the experiment's
    business.
    """

    def __init__(self, samples: LANLSamples):
        self.views = torch.as_tensor(samples.views, dtype=torch.float32)
        self.view_mask = torch.as_tensor(samples.view_mask, dtype=torch.bool)
        self.labels = torch.as_tensor(samples.labels, dtype=torch.long)
        self.label_known = torch.as_tensor(samples.label_known, dtype=torch.bool)
        self.window_start = torch.as_tensor(samples.window_start, dtype=torch.float64)
        self.host_id = None
        self.rho_matrix = None
        self.config = samples.config
        self.samples = samples
        self.sources = tuple(samples.sources)
        self.view_dims = tuple(len(FEATURES[s]) for s in self.sources)
        self.n_classes = 2

    def __len__(self) -> int:
        return int(self.views.shape[0])

    @property
    def n_views(self) -> int:
        return int(self.views.shape[1])

    def get_sample(self, index: int) -> Sample:
        return Sample(
            sample_id=index,
            views=self.views[index],
            label=int(self.labels[index]),
            view_mask=self.view_mask[index],
            metadata={
                "dataset": "lanl_cyber1",
                "sample_mode": "computer_window",
                "window_seconds": self.config.window_seconds,
                "window_start": float(self.window_start[index]),
                "label_known": True,
                "computer": self.samples.computer[index],
                "block": self.samples.block[index],
            },
        )

    def get_batch(self, indices) -> Batch:
        idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return Batch(
            views=self.views[idx],
            view_mask=self.view_mask[idx],
            labels=self.labels[idx],
            sample_ids=idx,
        )


def subset(dataset: LANLMultiViewDataset, views: tuple[str, ...]) -> LANLMultiViewDataset:
    """A copy carrying only `views`, for the WITHOUT-auth configuration.

    part30.md step 2 requires every experiment to run with and without the auth
    view, because the red-team labels are drawn from auth data. Dropping the
    view by rebuilding the tensor - rather than masking it - is what keeps the
    encoder stack genuinely free of it.
    """
    keep = [dataset.sources.index(v) for v in views]
    samples = dataclasses.replace(
        dataset.samples,
        views=dataset.samples.views[:, keep, :],
        view_mask=dataset.samples.view_mask[:, keep],
        sources=tuple(views),
    )
    return LANLMultiViewDataset(samples)


def view_configs(sources, hidden_dims=(64, 64), feature_dim: int = 32, dropout: float = 0.0):
    """One EncoderConfig per view, each with that view's NATIVE input_dim.

    LANL views are 2 to 8 features wide and padded to D. Declaring the native
    width per view is what keeps the padded columns unreachable, exactly as
    `ait_lds.view_configs` and `real_datasets.view_configs` do it.
    """
    from prismflow.models.encoders import EncoderConfig

    return [
        EncoderConfig(
            input_dim=len(FEATURES[s]),
            hidden_dims=list(hidden_dims),
            feature_dim=feature_dim,
            dropout=dropout,
        )
        for s in sources
    ]
