"""Part 28: naive fusion and V1 PrismFlow on real AIT log data, clean.

WHAT THE LABEL MEANS -- read this before any number below
---------------------------------------------------------
A window is attack=1 if ANY officially labelled line falls in it, and on both
scenarios that label is dominated by one continuously-running DNS exfiltration
tool: 1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on
santos carry only dnsteal-family steps (`docs/part26-ait-loader.md` sections 0
and 12). So the task is "WAS THE EXFILTRATION TOOL RUNNING IN THIS WINDOW",
NOT "detect an APT" and not "detect a multi-step intrusion". A result from
this script described as intrusion, APT or campaign detection is a misreport
even when every number in it is correct.

The multi-step intrusion is 5 windows on russellmitchell and 3 on santos. This
script counts them and labels anything about them a handful-of-windows result;
it does not train on them or report a metric over them.

TWO DIRECTIONS, NEVER POOLED
----------------------------
    A: train russellmitchell -> test santos
    B: train santos -> test russellmitchell

Neither is "the" result. Two directions are two measurements on two different
test sets with different prevalences (60.2% and 90.0% attack among labelled
windows), so a mean over them is meaningless and would hide whichever one is
worse. Every table, file and field carries its direction label, and no number
is averaged, concatenated or otherwise combined across A and B. The runner has
no code path that could: `run_direction` returns a self-contained record and
nothing reads two of them together except the writers, which emit one table
each.

WHY TRAIN AND TEST ARE WHOLE SCENARIOS
--------------------------------------
Neighbouring windows of one scenario leak into each other, and the labels are
a step function in time (`docs/part27-ait-datacard.md` section 8): a random
within-scenario split trains and tests on adjacent windows of the SAME
exfiltration run, and a chronological split on santos is close to a label
split. Cross-scenario is the only split here that is neither. Feature scaling
is fitted on the training scenario only and refitted per direction.

WHAT A SEED VARIES, AND WHAT IT DOES NOT
----------------------------------------
Train and test are whole scenarios, so a seed cannot redraw or re-partition
the data. It selects the model's parameter initialisation and the minibatch
order of every epoch, and nothing else. Every +/- here is
initialisation-and-batch-order variance on a fixed pair of scenarios -- NOT
dataset variance, and NOT a spread over independent attacks. The test sets
hold 3 (santos) and 1 (russellmitchell) independent contiguous attack blocks,
and that block count, not the attack-window count, is the positive class's
effective sample size.

ACCURACY IS NEVER A HEADLINE HERE
---------------------------------
The test sets are 60.2% and 90.0% attack among labelled windows, so an
accuracy figure on its own describes the prevalence and not the system. Plain
accuracy is emitted only inside rows that also carry balanced accuracy, ROC
AUC and the majority-class baseline, and every reported number carries its
test set's n, attack count, benign count and attack prevalence.

A NOTE ON THE EVALUATION PROTOCOL
---------------------------------
`docs/EVALUATION_PROTOCOL.md` says an experiment needing a metric that does
not exist adds it to `prismflow.evaluation.protocol.compute_metrics`. Part 28's
standing rules freeze `prismflow/` to import-only, and `compute_metrics` has
no balanced accuracy, no ROC AUC and no per-class recall. Those three are
therefore computed HERE, in `part28_extras`, by one function that every row in
every table goes through -- the systems and the majority-class baseline alike.
The protocol's own metrics still come from `compute_metrics`, and
`run_system` asserts that the row it reports agrees seed-by-seed with what
`evaluate()` wrote. The conflict is recorded in README.md for the user to
resolve; nothing in `prismflow/` was touched.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from prismflow.data.ait_lds import (
    ARRAY_FILES,
    FEATURES,
    LABEL_UNKNOWN,
    AITConfig,
    AITMultiViewDataset,
    AITSamples,
    load_processed,
    sensor_groups,
    view_configs,
)
from prismflow.data.loaders import iter_batches
from prismflow.eniv.eniv import compute_eniv
from prismflow.evaluation import evaluate
from prismflow.evaluation.protocol import collect_predictions, compute_metrics
from prismflow.models.prismflow import REPORTING_NULL_PERMUTATIONS, PrismFlow
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.train import compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger("prismflow.ait_clean")
CONFIG_PATH = Path(__file__).with_name("config.yaml")

#: The dnsteal family, taken from `docs/part26-ait-loader.md` section 0. The
#: loader takes this set from the caller because which steps belong to one
#: campaign is a reading of the scenario; this is the set that document used.
DNSTEAL_FAMILY = ("attacker", "dnsteal", "dnsteal-received", "dnsteal-dropped")

#: The dnsteal family PLUS the exfiltration tool's own service record. Two
#: counts are reported because the two source documents use two definitions
#: and the brief quotes one figure from each:
#:
#:   beyond the dnsteal family            russellmitchell 5, santos 4
#:   beyond family AND exfiltration-service  russellmitchell 4, santos 3
#:
#: `docs/part26-ait-loader.md` section 0 says "5 of 1,152 windows" for
#: russellmitchell, which is the first definition. The datacard and
#: `docs/part26-ait-loader.md` section 12 say santos's intrusion is "three
#: windows ... a fourth window carries exfiltration-service alongside", which
#: is the second. `exfiltration-service` is a SERVICE_START/SERVICE_STOP
#: record of the exfiltration tool itself, not a step of the multi-step
#: intrusion, so the second definition is the one labelled "multi-step
#: intrusion" here. The first is reported beside it, named for what it is, so
#: neither source document's figure is contradicted and neither is dropped.
EXFILTRATION_TOOL_STEPS = DNSTEAL_FAMILY + ("exfiltration-service",)

#: The one sentence that must appear in every artefact this script writes.
LABEL_MEANING = (
    "A window is attack=1 if ANY officially labelled line falls in it, and on "
    "both scenarios that label is dominated by one continuously-running DNS "
    "exfiltration tool (1,026 of 1,031 attack windows on russellmitchell and "
    "663 of 667 on santos carry only dnsteal-family steps, "
    "docs/part26-ait-loader.md sections 0 and 12), so the task is 'was the "
    "exfiltration tool running in this window', NOT 'detect an APT' and not "
    "'detect a multi-step intrusion'."
)

BLOCK_CONSEQUENCE = (
    "The block count, not the attack-window count, is the positive class's "
    "effective sample size. Every +/- in this Part is a spread over seeds "
    "(model initialisation and batch order), NOT over independent attacks."
)

CLASS_BENIGN, CLASS_ATTACK = 0, 1


# ---------------------------------------------------------------------------
# Data: read back a Part 26 processed directory
# ---------------------------------------------------------------------------


def load_scenario(scenario: str, cfg: dict) -> tuple[AITMultiViewDataset, dict]:
    """Rebuild one scenario's `AITSamples` from `data/processed/` and wrap it.

    Part 27 re-scanned the raw logs; this reads the committed processed arrays
    instead, which is the same bytes by `write_processed`'s contract and does
    not re-read 17 GB. Nothing is pickled: `load_processed` leaves
    `allow_pickle` at False.
    """
    data = cfg["data"]
    name = f"ait_{scenario}_{data['sample_mode']}_{data['window_seconds']}s"
    root = Path(data["processed_root"])
    directory = root / name
    if not directory.is_dir():
        raise SystemExit(
            f"missing processed scenario {directory}. Part 28 needs TWO processed "
            f"scenarios -- run /part26 on {scenario} before this script. Falling "
            f"back to a random within-scenario split is forbidden."
        )

    arrays, manifest = load_processed(directory, processed_root=root)
    for key in ("views", "view_mask", "labels", "label_known", "window_start"):
        if key not in arrays:
            raise SystemExit(f"{directory} is missing {ARRAY_FILES[key]}")

    config_fields = dict(manifest["config"])
    config_fields["sources"] = tuple(config_fields["sources"])
    config = AITConfig(**config_fields)
    if config.window_seconds != data["window_seconds"] or config.sample_mode != data["sample_mode"]:
        raise SystemExit(
            f"{directory} was built at {config.window_seconds}s/{config.sample_mode}, "
            f"config asks for {data['window_seconds']}s/{data['sample_mode']}"
        )

    n = int(arrays["labels"].shape[0])
    steps_by_row = json.loads((directory / "attack_steps.json").read_text())
    attack_steps = [list(steps_by_row.get(str(row), [])) for row in range(n)]

    samples = AITSamples(
        views=arrays["views"],
        view_mask=arrays["view_mask"],
        labels=arrays["labels"],
        label_known=arrays["label_known"],
        window_start=arrays["window_start"],
        host_id=arrays.get("host_id"),
        hosts=tuple(manifest["hosts"]),
        sources=config.sources,
        attack_steps=attack_steps,
        config=config,
    )

    # labels == -1 must mean exactly "not labelled", or every count below is
    # wrong in a way no metric would reveal.
    unknown = samples.labels == LABEL_UNKNOWN
    if not np.array_equal(unknown, ~samples.label_known):
        raise SystemExit(f"{directory}: labels == -1 and ~label_known disagree")

    return AITMultiViewDataset(samples), manifest


def labelled_indices(dataset: AITMultiViewDataset) -> np.ndarray:
    """The only index set this experiment ever uses. labels == -1 is dropped.

    `label_known` False means no label-covered file contributed an event to the
    window. That is not evidence of benign, so such a window is neither
    trained on nor scored.
    """
    return np.flatnonzero(dataset.label_known.numpy())


# ---------------------------------------------------------------------------
# Feature scaling: fitted on the TRAIN scenario only, refitted per direction
# ---------------------------------------------------------------------------


def fit_scaler(dataset: AITMultiViewDataset, indices: np.ndarray, cfg: dict) -> dict:
    """Per (view, native feature) mean and std over rows where the view is present.

    An absent view's row is all zeros and carries no observation, so including
    it would pull the mean toward zero by an amount that depends only on the
    missing rate. Padded columns beyond a view's native width are not fitted
    and are left at 0 by `apply_scaler`.
    """
    views = dataset.views.numpy()[indices]       # [n, V, D]
    mask = dataset.view_mask.numpy()[indices]    # [n, V]
    n_views, dim = views.shape[1], views.shape[2]
    mean = np.zeros((n_views, dim), dtype=np.float64)
    std = np.ones((n_views, dim), dtype=np.float64)
    native = [len(FEATURES[s]) for s in dataset.sources]
    present_counts = []
    for v in range(n_views):
        rows = mask[:, v]
        present_counts.append(int(rows.sum()))
        if rows.sum() < 2:
            # Cannot fit a scale from fewer than two observations. Leaving
            # mean 0 / std 1 is the identity, not an estimate.
            continue
        block = views[rows, v, : native[v]].astype(np.float64)
        mean[v, : native[v]] = block.mean(axis=0)
        std[v, : native[v]] = block.std(axis=0, ddof=0)
    epsilon = float(cfg["scaling"]["epsilon"])
    std = np.where(std > epsilon, std, 1.0)
    return {
        "mean": mean,
        "std": std,
        "native_dims": native,
        "fit_rows": int(len(indices)),
        "present_rows_per_view": present_counts,
    }


def apply_scaler(dataset: AITMultiViewDataset, scaler: dict) -> AITMultiViewDataset:
    """z-score the native columns of PRESENT views; leave everything else at 0."""
    views = dataset.views.numpy().copy()
    mask = dataset.view_mask.numpy()
    for v, native in enumerate(scaler["native_dims"]):
        rows = mask[:, v]
        if not rows.any():
            continue
        block = views[rows, v, :native].astype(np.float64)
        block = (block - scaler["mean"][v, :native]) / scaler["std"][v, :native]
        views[rows, v, :native] = block.astype(np.float32)
        # An absent view contributes nothing (MultiViewEncoder never feeds it
        # forward), so its row stays the zero vector it arrived as.
        views[~rows, v] = 0.0
    scaled = dataclasses.replace(dataset.samples, views=views.astype(np.float32))
    return AITMultiViewDataset(scaled)


# ---------------------------------------------------------------------------
# Training: the Part 15 entry point, with the AIT view widths
# ---------------------------------------------------------------------------


def train_ait(dataset: AITMultiViewDataset, indices: np.ndarray, seed: int, use_discount: bool, cfg: dict):
    """Train one model on `indices` of `dataset`. Same shape as Part 15's `train_real`.

    Optimiser, loss, annealing schedule, per-view loss weight and the
    same-seed-same-init discipline are Part 15's unchanged; only the encoder
    widths differ, because AIT views are 8-10 features wide and
    `ait_lds.view_configs` declares each one's native width so the padded
    columns stay unreachable.

    There is no validation split and no early stopping: a split inside the
    training scenario would be a within-scenario split, which this Part
    forbids. The epoch count is fixed by config, as in Part 15.
    """
    set_seed(seed)
    model = PrismFlow(
        view_configs(
            dataset.sources,
            hidden_dims=cfg["model"]["hidden_dims"],
            feature_dim=cfg["model"]["feature_dim"],
            dropout=cfg["model"]["dropout"],
        ),
        n_classes=dataset.n_classes,
        use_discount=use_discount,
    )
    training = cfg["training"]
    optimiser = torch.optim.Adam(model.parameters(), lr=training["lr"])
    for epoch in range(training["epochs"]):
        model.train()
        for batch in iter_batches(
            dataset, indices, training["batch_size"], shuffle=True, seed=seed + epoch
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=dataset.n_classes
            ).to(batch.views.dtype)
            loss = compute_loss(
                model(batch.views, batch.view_mask),
                targets,
                epoch=epoch,
                anneal_epochs=training["anneal_epochs"],
                per_view_loss_weight=training["per_view_loss_weight"],
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics. One function for the systems and the baseline alike.
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with tied values sharing their mean rank (scipy's 'average')."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC AUC for the attack class, ties credited 0.5.

    A constant score -- which is what the majority-class baseline produces --
    ties every pair and gives exactly 0.5, the honest reading of a predictor
    that cannot rank anything.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == CLASS_ATTACK).sum())
    n_neg = int((labels == CLASS_BENIGN).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[labels == CLASS_ATTACK].sum())
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def part28_extras(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Balanced accuracy, ROC AUC, per-class recall and the prevalence block.

    These are the metrics `compute_metrics` does not have and `prismflow/`
    is frozen against gaining (see the module docstring). Every row in every
    Part 28 table -- both systems and the majority-class baseline -- is built
    by this function plus `compute_metrics`, so the comparison is like for
    like.

    The prevalence block travels WITH the metrics rather than beside them,
    because a metric without its prevalence is not reportable here.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    predictions = probs.argmax(axis=1)

    recalls = {}
    for name, klass in (("benign", CLASS_BENIGN), ("attack", CLASS_ATTACK)):
        actual = labels == klass
        recalls[name] = (
            float((predictions[actual] == klass).mean()) if actual.any() else float("nan")
        )
    present = [r for r in recalls.values() if not math.isnan(r)]

    n = int(labels.size)
    n_attack = int((labels == CLASS_ATTACK).sum())
    n_benign = int((labels == CLASS_BENIGN).sum())
    return {
        "balanced_accuracy": float(np.mean(present)) if present else float("nan"),
        "roc_auc": roc_auc_binary(labels, probs[:, CLASS_ATTACK]),
        "attack_recall": recalls["attack"],
        "benign_recall": recalls["benign"],
        "n_test": float(n),
        "n_attack": float(n_attack),
        "n_benign": float(n_benign),
        "attack_prevalence": float(n_attack / n) if n else float("nan"),
    }


#: Keys that mean nothing for a predictor with no evidential output. A
#: majority-class baseline has no Dirichlet vacuity, so rather than feed it a
#: stand-in and report the result, those keys are dropped from its row.
_VACUITY_PREFIX = "vacuity_"


def metric_row(labels: np.ndarray, probs: np.ndarray, vacuity: np.ndarray | None, cfg: dict) -> dict:
    """One full metric row: `compute_metrics` + `part28_extras`.

    `vacuity=None` is the baseline: `compute_metrics` requires the array, so
    the top-label probability stands in for the call and every `vacuity_` key
    is then removed, because the baseline has no such quantity to report.
    """
    evaluation = cfg["evaluation"]
    stand_in = vacuity is None
    confidence = probs.max(axis=1) if stand_in else vacuity
    row = compute_metrics(
        labels, probs, confidence,
        n_bins=evaluation["n_bins"], coverages=evaluation["coverages"],
    )
    if stand_in:
        row = {k: v for k, v in row.items() if not k.startswith(_VACUITY_PREFIX)}
    row.update(part28_extras(labels, probs))
    return row


def majority_baseline(labels: np.ndarray, cfg: dict) -> dict:
    """The majority-class baseline on ONE direction's test set.

    It predicts that test set's majority class for every window, with a
    constant probability vector equal to the test set's class prevalence. So
    its accuracy IS the prevalence, its balanced accuracy is 0.5, and its ROC
    AUC is 0.5 because a constant score ranks nothing. That is the point: it
    is the number an accuracy figure has to beat before it describes anything
    but the class balance.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.size)
    prevalence = np.array(
        [float((labels == CLASS_BENIGN).sum()) / n, float((labels == CLASS_ATTACK).sum()) / n]
    )
    probs = np.tile(prevalence, (n, 1))
    row = metric_row(labels, probs, None, cfg)
    row["majority_class"] = float(int(prevalence.argmax()))
    return row


def summarise(rows: list[dict]) -> dict:
    """mean, sample std (ddof=1) and n_seeds per key, over seeds. NaN excluded."""
    keys = sorted({k for row in rows for k in row})
    out = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and not _is_nan(row[key])]
        out[key] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "n_seeds": len(values),
        }
    return out


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def paired_difference(treatment: list[dict], control: list[dict], lower_is_better: set[str]) -> dict:
    """prismflow - naive WITHIN each seed, with the per-seed signs kept.

    Paired, not two independent means: both systems see the same seed, so the
    same initialisation draw and the same batch order, and the difference
    removes that shared variance. On this data the unpaired bars overlap
    heavily in direction B, so an unpaired reading would say "no effect"
    where a paired one may not.

    `n_seeds_favouring` is reported beside every mean because a population in
    which the effect cancels reads as "no effect" in the mean alone. A mean
    near zero with 5/5 seeds on one side is a different statement from a mean
    near zero with 2/5.
    """
    by_seed_control = {row["seed"]: row for row in control}
    keys = sorted(set(treatment[0]) & set(control[0]) - {"seed"})
    out = {}
    for key in keys:
        diffs, favouring = [], 0
        for row in treatment:
            other = by_seed_control.get(row["seed"])
            if other is None or _is_nan(row.get(key)) or _is_nan(other.get(key)):
                continue
            delta = float(row[key]) - float(other[key])
            diffs.append(delta)
            better = delta < 0 if key in lower_is_better else delta > 0
            favouring += int(better)
        if not diffs:
            continue
        out[key] = {
            "mean": statistics.mean(diffs),
            "std": statistics.stdev(diffs) if len(diffs) > 1 else float("nan"),
            "n_seeds": len(diffs),
            "n_seeds_favouring_prismflow": favouring,
            "per_seed": diffs,
            "lower_is_better": key in lower_is_better,
        }
    return out


#: Which metrics improve downward. Used only to read the SIGN of a paired
#: difference; it changes no value.
LOWER_IS_BETTER = {
    "brier", "brier_reliability", "prob_ece", "prob_mce", "prob_aurc",
    "vacuity_ece", "vacuity_mce", "vacuity_aurc", "mean_dependence_eigen",
    "mean_dependence_design_effect",
}


# ---------------------------------------------------------------------------
# Test-set structure: independent contiguous blocks
# ---------------------------------------------------------------------------


def contiguous_blocks(flags: np.ndarray, window_start: np.ndarray) -> list[dict]:
    """Maximal runs of True on the scenario's own time grid.

    The grid is the window index, so a window the loader dropped (empty, or
    outside the declared interval) breaks a run rather than joining the
    windows either side of it -- the same convention Part 27 used, which is
    why the counts here reproduce `results/ait_window/window_table.md`
    section 2.
    """
    blocks, i, n = [], 0, int(flags.size)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            blocks.append(
                {
                    "start_index": i,
                    "length_windows": j - i + 1,
                    "start_time": dt.datetime.fromtimestamp(
                        float(window_start[i]), dt.timezone.utc
                    ).isoformat(),
                }
            )
            i = j + 1
        else:
            i += 1
    return blocks


def test_structure(dataset: AITMultiViewDataset, scenario: str) -> dict:
    """Blocks, counts and the multi-step-intrusion window count for one test set.

    Measured at the window length and sample mode ACTUALLY USED, from this
    scenario's own `labels.npy` and `window_start.npy`; Part 26's figures are
    not carried over.
    """
    labels = dataset.labels.numpy()
    window_start = dataset.window_start.numpy()
    attack = contiguous_blocks(labels == CLASS_ATTACK, window_start)
    benign = contiguous_blocks(labels == CLASS_BENIGN, window_start)

    steps = dataset.samples.attack_steps

    def beyond_set(allowed: tuple[str, ...]) -> list[int]:
        allowed_set = set(allowed)
        return [
            i for i, names in enumerate(steps)
            if labels[i] == CLASS_ATTACK and names and not set(names) <= allowed_set
        ]

    beyond_family = beyond_set(DNSTEAL_FAMILY)
    intrusion = beyond_set(EXFILTRATION_TOOL_STEPS)

    n_known = int(dataset.label_known.numpy().sum())
    n_attack = int((labels == CLASS_ATTACK).sum())
    return {
        "scenario": scenario,
        "window_seconds": dataset.config.window_seconds,
        "sample_mode": dataset.config.sample_mode,
        "n_windows_on_grid": int(labels.size),
        "n_label_known": n_known,
        "n_attack": n_attack,
        "n_benign": int((labels == CLASS_BENIGN).sum()),
        "n_unlabelled": int((labels == LABEL_UNKNOWN).sum()),
        "attack_prevalence_labelled": n_attack / n_known if n_known else float("nan"),
        "n_attack_blocks": len(attack),
        "attack_blocks": attack,
        "n_benign_blocks": len(benign),
        "benign_blocks": benign,
        "dnsteal_family_counted": list(DNSTEAL_FAMILY),
        "exfiltration_tool_steps_counted": list(EXFILTRATION_TOOL_STEPS),
        "n_attack_windows_dnsteal_only": n_attack - len(beyond_family),
        "n_attack_windows_beyond_dnsteal_family": len(beyond_family),
        "multi_step_intrusion_windows": [
            {
                "index": i,
                "start_time": dt.datetime.fromtimestamp(
                    float(window_start[i]), dt.timezone.utc
                ).isoformat(),
                "steps": sorted(steps[i]),
            }
            for i in intrusion
        ],
        "n_multi_step_intrusion_windows": len(intrusion),
        "multi_step_definition": (
            "attack windows carrying at least one step outside "
            f"{list(EXFILTRATION_TOOL_STEPS)}. The looser count, outside "
            f"{list(DNSTEAL_FAMILY)} alone, is "
            f"n_attack_windows_beyond_dnsteal_family = {len(beyond_family)} and "
            "differs only by the window holding the exfiltration tool's own "
            "SERVICE_START/SERVICE_STOP record."
        ),
        "multi_step_caveat": (
            f"{len(intrusion)} windows of {int(labels.size)}. Any claim about the "
            f"multi-step intrusion is a handful-of-windows result; no metric in "
            f"this Part is computed over them."
        ),
        "consequence": BLOCK_CONSEQUENCE,
        "label_meaning": LABEL_MEANING,
    }


def missingness(dataset: AITMultiViewDataset, indices: np.ndarray) -> dict:
    """Per-view missing fraction over `indices` (the label_known rows)."""
    mask = dataset.view_mask.numpy()[indices]
    return {
        source: float(1.0 - mask[:, v].mean())
        for v, source in enumerate(dataset.sources)
    }


# ---------------------------------------------------------------------------
# Diagnostics: ENIV and the full [V, V] dependence matrix
# ---------------------------------------------------------------------------


@torch.no_grad()
def dependence_and_eniv(model, dataset: AITMultiViewDataset, indices: np.ndarray, cfg: dict) -> dict:
    """The reporting-grade [V, V] matrix and ENIV on one direction's test set.

    `REPORTING_NULL_PERMUTATIONS` (12), not the forward pass's 4: the upward
    bias the permutation correction removes scales with
    feature_dim / n_samples, and this is the number that goes in a table. The
    same call Part 15 makes in `experiments/real/run_real.py::dependence_of`.

    Computed for BOTH systems by the same code. Naive fusion's forward pass
    does not produce an ENIV (`use_discount=False` leaves `output.eniv` None),
    so reading it off the model would compare one system's measurement with
    the other's absence.
    """
    views = dataset.views[indices]
    view_mask = dataset.view_mask[indices]
    output = model(views, view_mask)
    matrix = feature_dependence_matrix(
        model.encoder(views, view_mask),
        output.per_view_evidence,
        view_mask,
        method=model.dependence_method,
        conditioning=model.dependence_conditioning,
        seed=model.dependence_seed,
        null_permutations=REPORTING_NULL_PERMUTATIONS,
    )
    available = available_views(view_mask)
    record = {
        "matrix": [[_safe(x) for x in row] for row in np.asarray(matrix).tolist()],
        "views": list(dataset.sources),
        "available": available.tolist(),
        "null_permutations": REPORTING_NULL_PERMUTATIONS,
        "method": model.dependence_method,
        "conditioning": model.dependence_conditioning,
    }
    for method in cfg["diagnostics"]["eniv_methods"]:
        result = compute_eniv(matrix, available=available, method=method)
        record[f"eniv_{method}"] = _safe(result.effective_views)
        record[f"mean_dependence_{method}"] = _safe(result.mean_dependence)
        record[f"efficiency_ratio_{method}"] = _safe(result.efficiency_ratio)
    return record


def _safe(value) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def pair_table(records: list[dict], sources: list[str]) -> list[dict]:
    """Every off-diagonal pair, mean +/- std over seeds.

    Step 4 of the brief asks for the `same_sensor_group="network"` PAIR against
    the mean of unrelated pairs. `sensor_groups()` returns `{'network': [4]}` --
    ONE member, suricata -- so that pair does not exist and the paired
    difference the brief asks for is not computable on this data. The full
    pair table is reported in its place, explicitly as a different quantity.
    """
    matrices = [np.asarray(r["matrix"], dtype=np.float64) for r in records]
    rows = []
    n_views = len(sources)
    for a in range(n_views):
        for b in range(a + 1, n_views):
            values = [m[a, b] for m in matrices if math.isfinite(m[a, b])]
            rows.append(
                {
                    "pair": f"{sources[a]}-{sources[b]}",
                    "indices": [a, b],
                    "mean": statistics.mean(values) if values else float("nan"),
                    "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
                    "n_seeds": len(values),
                }
            )
    return rows


def sensor_group_check(sources: list[str], cfg: dict) -> dict:
    """Why step 4's paired difference has no value on this data."""
    groups = sensor_groups(tuple(sources))
    name = cfg["diagnostics"]["same_sensor_group"]
    members = groups.get(name, [])
    computable = len(members) >= 2
    return {
        "requested_group": name,
        "sensor_groups": {k: v for k, v in groups.items()},
        "group_members": [sources[i] for i in members],
        "n_members": len(members),
        "computable": computable,
        "status": "computed" if computable else "NOT COMPUTABLE",
        "reason": (
            None
            if computable
            else (
                f"the {name!r} same-sensor group has {len(members)} member(s) "
                f"({[sources[i] for i in members]}), so no within-group PAIR exists. "
                f"There is no packet-capture view yet "
                f"(docs/part27-ait-datacard.md section 9, limitation 9), so the "
                f"paired difference step 4 asks for has no value to report. "
                f"Substituting a different pair would answer a different question; "
                f"the full pair table is reported instead."
            )
        ),
        "same_sensor_pair_dependence": None if not computable else "see pair_table",
        "unrelated_pairs_mean": None,
    }


# ---------------------------------------------------------------------------
# One system, one direction
# ---------------------------------------------------------------------------


def run_system(
    system: dict,
    train_dataset: AITMultiViewDataset,
    train_indices: np.ndarray,
    test_dataset: AITMultiViewDataset,
    test_indices: np.ndarray,
    direction: dict,
    cfg: dict,
    seeds: list[int],
    results_dir: str,
    allow_fewer_seeds: bool = False,
) -> dict:
    """Train and score one system over every seed, on one direction."""
    batch_size = cfg["evaluation"]["batch_size"]
    models, rows, diagnostics = {}, [], []

    for seed in seeds:
        LOGGER.info(
            "direction=%s system=%s seed=%d: training on %d %s windows",
            direction["name"], system["name"], seed, len(train_indices), direction["train"],
        )
        model = train_ait(train_dataset, train_indices, seed, system["use_discount"], cfg)
        models[seed] = model

        collected = collect_predictions(
            model, iter_batches(test_dataset, test_indices, batch_size, shuffle=False)
        )
        row = {"seed": seed}
        row.update(metric_row(
            collected["labels"], collected["probs"], collected["vacuity_confidence"], cfg
        ))
        rows.append(row)

        record = {"seed": seed}
        record.update(dependence_and_eniv(model, test_dataset, test_indices, cfg))
        diagnostics.append(record)

    # The protocol's own writer, so this Part's output sits beside every other
    # Part's in the standard shape, and so the rows above are checked against
    # code this experiment does not own.
    experiment_id = f"{cfg['experiment']['id']}/{direction['name']}/{system['name']}"
    protocol = evaluate(
        models,
        lambda seed: iter_batches(test_dataset, test_indices, batch_size, shuffle=False),
        experiment_id,
        seeds,
        results_dir=results_dir,
        n_bins=cfg["evaluation"]["n_bins"],
        coverages=cfg["evaluation"]["coverages"],
        allow_fewer_seeds=allow_fewer_seeds,
        title=f"Part 28 {direction['label']} - {system['label']}",
    )
    _cross_check(rows, protocol["per_seed"], direction["name"], system["name"])

    return {
        "system": system["name"],
        "label": system["label"],
        "use_discount": system["use_discount"],
        "experiment_id": experiment_id,
        "protocol_output": f"{results_dir}/{experiment_id}/metrics.json",
        "per_seed": rows,
        "summary": summarise(rows),
        "diagnostics_per_seed": diagnostics,
        "diagnostics_summary": summarise(
            [{k: v for k, v in d.items() if isinstance(v, float)} for d in diagnostics]
        ),
        "pair_table": pair_table(diagnostics, list(test_dataset.sources)),
    }


def _cross_check(rows: list[dict], protocol_rows: list[dict], direction: str, system: str) -> None:
    """Every metric this script shares with `evaluate()` must agree seed by seed.

    Both read the same unshuffled loader, so disagreement means the two paths
    are not scoring the same predictions and no number from either is safe.
    """
    shared = ("accuracy", "macro_f1", "brier", "prob_ece", "prob_aurc", "n_samples")
    by_seed = {row["seed"]: row for row in protocol_rows}
    for row in rows:
        other = by_seed[row["seed"]]
        for key in shared:
            mine, theirs = row[key], other[key]
            if _is_nan(mine) and _is_nan(theirs):
                continue
            if abs(mine - theirs) > 1e-9:
                raise AssertionError(
                    f"{direction}/{system} seed {row['seed']}: {key} "
                    f"{mine!r} from part28 path != {theirs!r} from evaluate()"
                )


def run_direction(
    direction: dict,
    datasets: dict,
    cfg: dict,
    seeds: list[int],
    results_dir: str,
    allow_fewer_seeds: bool = False,
) -> dict:
    """Everything for ONE direction. Nothing in here reads the other direction."""
    train_dataset, train_manifest = datasets[direction["train"]]
    test_dataset, test_manifest = datasets[direction["test"]]

    if direction["train"] == direction["test"]:
        raise SystemExit(
            f"direction {direction['name']} has train == test == {direction['train']}; "
            f"train and test must come from DIFFERENT scenarios"
        )

    train_indices = labelled_indices(train_dataset)
    test_indices = labelled_indices(test_dataset)

    scaler = fit_scaler(train_dataset, train_indices, cfg)
    train_scaled = apply_scaler(train_dataset, scaler)
    test_scaled = apply_scaler(test_dataset, scaler)

    structure = test_structure(test_dataset, direction["test"])
    train_missing = missingness(train_dataset, train_indices)
    test_missing = missingness(test_dataset, test_indices)
    threshold = float(cfg["missingness"]["threshold"])
    over = sorted(
        {s for s, f in train_missing.items() if f > threshold}
        | {s for s, f in test_missing.items() if f > threshold}
    )

    systems = [
        run_system(
            system, train_scaled, train_indices, test_scaled, test_indices,
            direction, cfg, seeds, results_dir, allow_fewer_seeds,
        )
        for system in cfg["systems"]
    ]

    test_labels = test_scaled.labels.numpy()[test_indices]
    baseline = majority_baseline(test_labels, cfg)

    by_name = {s["system"]: s for s in systems}
    paired = {
        "comparison": "prismflow - naive, within seed",
        "metrics": paired_difference(
            by_name["prismflow"]["per_seed"], by_name["naive"]["per_seed"], LOWER_IS_BETTER
        ),
        "diagnostics": paired_difference(
            [{k: v for k, v in d.items() if isinstance(v, (int, float))}
             for d in by_name["prismflow"]["diagnostics_per_seed"]],
            [{k: v for k, v in d.items() if isinstance(v, (int, float))}
             for d in by_name["naive"]["diagnostics_per_seed"]],
            LOWER_IS_BETTER,
        ),
        "note": (
            "Paired within seed: both systems see the same initialisation draw and "
            "the same batch order, so the difference removes that shared variance. "
            "n_seeds_favouring_prismflow is reported beside every mean, because a "
            "mean near zero with 5/5 seeds on one side is a different statement "
            "from a mean near zero with 2/5."
        ),
    }

    return {
        "direction": direction["name"],
        "label": direction["label"],
        "train_scenario": direction["train"],
        "test_scenario": direction["test"],
        "label_meaning": LABEL_MEANING,
        "n_train": int(len(train_indices)),
        "n_train_attack": int((train_scaled.labels.numpy()[train_indices] == CLASS_ATTACK).sum()),
        "n_train_benign": int((train_scaled.labels.numpy()[train_indices] == CLASS_BENIGN).sum()),
        "train_attack_prevalence": float(
            (train_scaled.labels.numpy()[train_indices] == CLASS_ATTACK).mean()
        ),
        "scaler": {
            "fitted_on": direction["train"],
            "fit_rows": scaler["fit_rows"],
            "present_rows_per_view": scaler["present_rows_per_view"],
            "note": (
                "fitted on the training scenario's label_known rows only, refitted "
                "per direction, never shared between directions"
            ),
        },
        "test_structure": structure,
        "missingness": {
            "threshold": threshold,
            "train": train_missing,
            "test": test_missing,
            "views_over_threshold": over,
            "second_configuration_run": bool(over),
            "note": (
                f"no view exceeds {threshold:.0%} missing on this direction's train or "
                f"test set, so the brief's second configuration was not triggered"
                if not over
                else f"views over threshold: {over}"
            ),
        },
        "majority_class_baseline": baseline,
        "paired_prismflow_minus_naive": paired,
        "sensor_group_check": sensor_group_check(list(test_scaled.sources), cfg),
        "systems": {s["system"]: s for s in systems},
        "manifests": {
            direction["train"]: train_manifest.get("part"),
            direction["test"]: test_manifest.get("part"),
        },
    }


# ---------------------------------------------------------------------------
# Reporting. One table per direction. There is no pooled table anywhere.
# ---------------------------------------------------------------------------

#: Order matters: balanced accuracy, ROC AUC and the baseline sit beside plain
#: accuracy in every row it appears in, so accuracy never stands alone.
METRIC_ROWS = [
    ("balanced_accuracy", "balanced accuracy", 4),
    ("roc_auc", "ROC AUC", 4),
    ("macro_f1", "macro-F1", 4),
    ("attack_recall", "attack-class recall", 4),
    ("benign_recall", "benign-class recall", 4),
    ("brier", "Brier", 4),
    ("prob_ece", "ECE (15 bins, prob_)", 4),
    ("prob_aurc", "AURC (prob_)", 4),
    ("accuracy", "accuracy (prevalence-bound, never a headline)", 4),
]

ENIV_ROWS = [
    ("eniv_eigen", "ENIV (eigen, headline)", 4),
    ("eniv_design_effect", "ENIV (design effect)", 4),
    ("mean_dependence_eigen", "mean dependence", 4),
    ("efficiency_ratio_eigen", "efficiency ratio", 4),
]


def fmt(entry: dict | None, places: int = 4) -> str:
    if entry is None:
        return "not measured"
    mean, std = entry.get("mean"), entry.get("std")
    if mean is None or _is_nan(mean):
        return "not measured"
    if std is None or _is_nan(std):
        return f"{mean:.{places}f} (1 seed)"
    return f"{mean:.{places}f} +/- {std:.{places}f}"


def prevalence_caption(record: dict) -> str:
    s = record["test_structure"]
    return (
        f"test set `{s['scenario']}` at {s['window_seconds']} s / {s['sample_mode']}: "
        f"n = {s['n_label_known']} label_known windows, {s['n_attack']} attack, "
        f"{s['n_benign']} benign, attack prevalence "
        f"{s['attack_prevalence_labelled']:.3f}; {s['n_attack_blocks']} independent "
        f"contiguous attack block(s)"
    )


def direction_table(record: dict) -> list[str]:
    """THE summary table for one direction. Never combined with the other."""
    s = record["test_structure"]
    naive = record["systems"]["naive"]
    prism = record["systems"]["prismflow"]
    baseline = record["majority_class_baseline"]

    lines = [
        f"### Direction {record['label']}",
        "",
        f"**Train** `{record['train_scenario']}`, {record['n_train']} label_known windows "
        f"({record['n_train_attack']} attack, {record['n_train_benign']} benign, "
        f"prevalence {record['train_attack_prevalence']:.3f}). "
        f"**Test** `{record['test_scenario']}`, {s['n_label_known']} label_known windows "
        f"({s['n_attack']} attack, {s['n_benign']} benign, prevalence "
        f"{s['attack_prevalence_labelled']:.3f}). Train and test are different "
        f"scenarios; the scaler is fitted on `{record['train_scenario']}` only.",
        "",
        f"**Independent contiguous attack blocks in this test set: "
        f"{s['n_attack_blocks']}** "
        f"(lengths {'/'.join(str(b['length_windows']) for b in s['attack_blocks'])} windows). "
        f"{BLOCK_CONSEQUENCE}",
        "",
        f"**What the label means.** {LABEL_MEANING}",
        "",
        "| metric | naive fusion | V1 PrismFlow | majority-class baseline |",
        "|---|---|---|---|",
    ]
    for key, name, places in METRIC_ROWS:
        lines.append(
            f"| {name} | {fmt(naive['summary'].get(key), places)} "
            f"| {fmt(prism['summary'].get(key), places)} "
            f"| {fmt({'mean': baseline.get(key), 'std': float('nan')}, places)} |"
        )
    for key, name, places in ENIV_ROWS:
        lines.append(
            f"| {name} | {fmt(naive['diagnostics_summary'].get(key), places)} "
            f"| {fmt(prism['diagnostics_summary'].get(key), places)} "
            f"| n/a (no model) |"
        )
    lines += [
        "",
        f"{len(record['systems']['naive']['per_seed'])} seeds "
        f"{record['systems']['naive']['per_seed'][0]['seed']}-"
        f"{record['systems']['naive']['per_seed'][-1]['seed']}, mean +/- sample std "
        f"(ddof = 1). The baseline is deterministic on a fixed test set, so it has "
        f"one value and no std. Caption: {prevalence_caption(record)}.",
        "",
    ]
    return lines


def write_report(results: dict, cfg: dict, path: Path, commit: str | None) -> None:
    lines = [
        "# Part 28 - PrismFlow on real data (clean)",
        "",
        f"Produced by `experiments/ait_clean/run_ait_clean.py` with "
        f"`experiments/ait_clean/config.yaml`"
        + (f", commit `{commit}`" if commit else "")
        + ".",
        "",
    ]
    if results.get("quick_smoke_run"):
        lines += [
            "> **SMOKE TEST, NOT A FINDING.** This file came from `--quick`: "
            f"{len(results['seeds'])} seeds and "
            f"{results['config']['training']['epochs']} epochs, below the "
            "contract's 5-seed minimum. Every number here is a plumbing check and "
            "must never be reported as evidence.",
            "",
        ]
    lines += [
        "## What the label means",
        "",
        LABEL_MEANING,
        "",
        "The multi-step intrusion is "
        + ", ".join(
            f"{r['test_structure']['n_multi_step_intrusion_windows']} window(s) on "
            f"`{r['test_scenario']}` "
            f"({r['test_structure']['n_attack_windows_beyond_dnsteal_family']} "
            f"counting the exfiltration tool's own service record)"
            for r in results["directions"].values()
        )
        + ". No metric in this file is computed over those windows; any claim about "
        "them is a handful-of-windows result. The two counts per scenario are the "
        "two definitions the source documents use; see "
        "`EXFILTRATION_TOOL_STEPS` in `run_ait_clean.py`.",
        "",
        "## Two directions, reported separately and never pooled",
        "",
        "Neither direction is \"the\" result. Two directions are two measurements on "
        "two different test sets with different prevalences, so a mean over them "
        "would be meaningless and would hide whichever one is worse. There is no "
        "pooled table in this file, no averaged number across A and B, and no "
        "concatenated test set.",
        "",
        "## One summary table per direction",
        "",
    ]
    for record in results["directions"].values():
        lines += direction_table(record)

    lines += [
        "## Paired within-seed comparison, per direction",
        "",
        "prismflow - naive computed WITHIN each seed, so the shared "
        "initialisation-and-batch-order variance cancels. The seed count on one "
        "side of zero is given beside every mean: a mean near zero with 5/5 seeds "
        "on one side is a different statement from a mean near zero with 2/5. "
        "Not pooled across directions.",
        "",
    ]
    for record in results["directions"].values():
        lines += [
            f"{record['label']} ({prevalence_caption(record)}):",
            "",
            "| metric | prismflow - naive | seeds favouring prismflow | better when |",
            "|---|---|---|---|",
        ]
        paired = record["paired_prismflow_minus_naive"]
        for key, name, places in METRIC_ROWS + ENIV_ROWS:
            entry = paired["metrics"].get(key) or paired["diagnostics"].get(key)
            if entry is None:
                lines.append(f"| {name} | not measured | - | - |")
                continue
            lines.append(
                f"| {name} | {fmt(entry, places)} "
                f"| {entry['n_seeds_favouring_prismflow']}/{entry['n_seeds']} "
                f"| {'lower' if entry['lower_is_better'] else 'higher'} |"
            )
        lines.append("")

    lines += [
        "## Test-set structure, per direction",
        "",
        "| direction | test scenario | n (label_known) | attack | benign | prevalence "
        "| attack blocks | attack block lengths | benign blocks | benign block lengths |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in results["directions"].values():
        s = record["test_structure"]
        lines.append(
            f"| {record['direction']} | `{s['scenario']}` | {s['n_label_known']} "
            f"| {s['n_attack']} | {s['n_benign']} "
            f"| {s['attack_prevalence_labelled']:.3f} | **{s['n_attack_blocks']}** "
            f"| {'/'.join(str(b['length_windows']) for b in s['attack_blocks'])} "
            f"| {s['n_benign_blocks']} "
            f"| {'/'.join(str(b['length_windows']) for b in s['benign_blocks'][:8])}"
            f"{'/...' if s['n_benign_blocks'] > 8 else ''} |"
        )
    lines += [
        "",
        BLOCK_CONSEQUENCE,
        "",
        "Attack block start times:",
        "",
    ]
    for record in results["directions"].values():
        for block in record["test_structure"]["attack_blocks"]:
            lines.append(
                f"- {record['direction']} (`{record['test_scenario']}`): "
                f"{block['start_time']}, {block['length_windows']} windows"
            )
    lines += [
        "",
        "Measured at 300 s / network from each scenario's own `labels.npy` and "
        "`window_start.npy`; Part 26's figures are not carried over. The counts "
        "reproduce `results/ait_window/window_table.md` section 2 for the same rows.",
        "",
        "## Per-view missing fraction, per direction (train and test)",
        "",
        "| direction | set | scenario | "
        + " | ".join(f"`{s}`" for s in results["sources"])
        + " |",
        "|---|---|---|" + "---|" * len(results["sources"]),
    ]
    for record in results["directions"].values():
        for which, scenario in (("train", record["train_scenario"]), ("test", record["test_scenario"])):
            values = record["missingness"][which]
            lines.append(
                f"| {record['direction']} | {which} | `{scenario}` | "
                + " | ".join(f"{values[s]:.4f}" for s in results["sources"])
                + " |"
            )
    thresholds = {r["direction"]: r["missingness"] for r in results["directions"].values()}
    lines += [
        "",
        "Threshold for the brief's step 5 is "
        f"{float(cfg['missingness']['threshold']):.0%}. "
        + "; ".join(f"{d}: {m['note']}" for d, m in thresholds.items())
        + ".",
        "",
        "## Dependence pairs, per direction",
        "",
    ]
    check = next(iter(results["directions"].values()))["sensor_group_check"]
    lines += [
        f"**Step 4 of the brief is {check['status']}.** {check['reason']}",
        "",
        f"Full off-diagonal pair table, mean +/- sample std over "
        f"{len(results['seeds'])} seeds, measured on each direction's test set with "
        f"`REPORTING_NULL_PERMUTATIONS = {REPORTING_NULL_PERMUTATIONS}`:",
        "",
    ]
    for record in results["directions"].values():
        for system in record["systems"].values():
            lines += [
                f"{record['direction']} / {system['label']} "
                f"({prevalence_caption(record)}):",
                "",
                "| pair | dependence |",
                "|---|---|",
            ]
            for row in system["pair_table"]:
                lines.append(f"| `{row['pair']}` | {fmt(row, 4)} |")
            lines.append("")

    lines += [
        "## Files",
        "",
        f"- `results/{results['experiment_id']}/results.json` - every number above, "
        f"both directions, {len(results['seeds'])} seeds per metric per direction, "
        f"with each direction's prevalence block.",
        f"- `results/{results['experiment_id']}/summary.md` - this file.",
        f"- `results/{results['experiment_id']}/<direction>/<system>/"
        f"metrics.json|csv` and `reliability_diagram.png` - the standard "
        f"`prismflow.evaluation.evaluate()` output, per direction per system.",
        "",
        f"Every number in this file is a {len(results['seeds'])}-seed mean +/- sample "
        f"std except the majority-class baseline (deterministic on a fixed test set) "
        f"and the structural counts (properties of the data, not of a run).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def json_safe(value):
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return json_safe(float(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--quick", action="store_true",
        help="2 epochs, 2 seeds, into results/ait_clean_quick/. A SMOKE TEST: its "
             "output is gitignored and is never a finding.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seeds = list(cfg["seeds"])
    results_dir = "results"
    experiment_id = cfg["experiment"]["id"]
    if args.quick:
        cfg["training"]["epochs"] = 2
        seeds = seeds[:2]
        experiment_id = f"{experiment_id}_quick"
        cfg["experiment"]["id"] = experiment_id

    scenarios = sorted({d[k] for d in cfg["directions"] for k in ("train", "test")})
    if len(scenarios) < 2:
        raise SystemExit(
            "Part 28 needs TWO processed scenarios. Download and run /part26 on a "
            "second scenario; a random within-scenario split is forbidden and a "
            "single direction must not be run."
        )
    datasets = {s: load_scenario(s, cfg) for s in scenarios}

    first = next(iter(datasets.values()))[0]
    for scenario, (dataset, _) in datasets.items():
        if tuple(dataset.sources) != tuple(first.sources):
            raise SystemExit(
                f"{scenario} view order {dataset.sources} != {first.sources}; the two "
                f"scenarios must present the same views in the same order"
            )

    results = {
        "part": 28,
        "experiment_id": experiment_id,
        "commit": git_commit(),
        "config": cfg,
        "seeds": seeds,
        "not_evidence": args.quick,
        "quick_smoke_run": args.quick,
        "sources": list(first.sources),
        "label_meaning": LABEL_MEANING,
        "pooling_policy": (
            "Direction A and direction B are reported separately and are never "
            "pooled, averaged or concatenated. Two directions are two measurements "
            "on two different test sets with different prevalences."
        ),
        "seed_meaning": (
            "A seed selects the model's parameter initialisation and the minibatch "
            "order of every epoch. Train and test are whole scenarios, so a seed "
            "cannot redraw or re-partition the data. Every +/- is "
            "initialisation-and-batch-order variance, not a spread over "
            "independent attacks."
        ),
        "directions": {},
    }
    for direction in cfg["directions"]:
        results["directions"][direction["name"]] = run_direction(
            direction, datasets, cfg, seeds, results_dir, allow_fewer_seeds=args.quick
        )

    out_dir = Path(results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(json_safe(results), indent=2) + "\n", encoding="utf-8"
    )
    write_report(results, cfg, out_dir / "summary.md", results["commit"])
    LOGGER.info("wrote %s", out_dir)
    if args.quick:
        LOGGER.warning("--quick: 2 epochs, 2 seeds. SMOKE TEST, not a finding.")


if __name__ == "__main__":
    main()
