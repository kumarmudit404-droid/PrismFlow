"""Pre-compute the V1 Streamlit app's real outputs for the static site.

    .venv\\Scripts\\python.exe web/build_v1.py

WHY THIS IMPORTS app.py RATHER THAN REIMPLEMENTING IT
-----------------------------------------------------
``import app`` gives this script the app's OWN wiring -- ``load_synthetic``,
``load_real``, ``apply_scenario``, ``clean_threshold``, ``run_engine``,
``most_redundant_pair``, and the ``SCENARIOS`` / ``DEMO_TRAINING`` / ``DEMO_DATA``
constants. So the site cannot drift from the app: there is no second copy of the
engine call to keep in step, and a change to app.py changes this output too.

``app.py`` is NOT edited. It is imported. Importing it pulls in streamlit, which
prints one "missing ScriptRunContext" warning in bare mode and is harmless --
``st.cache_resource`` still memoises, so a repeated (seed, clone_k) trains once.

WHAT THIS SCRIPT IS NOT ALLOWED TO DO
--------------------------------------
Invent an output. Every number written here is a return value of a V1 engine
function, reached through app.py. Where a quantity does not exist -- a scenario
whose dataset could not be built, a detector that returned no suspicion score --
it emits ``null`` and the page renders "not measured". It never substitutes a
value from another scenario, another seed, or another dataset.

WHY THESE RUNS ARE NOT EVIDENCE, AND ARE LABELLED SO
-----------------------------------------------------
``DEMO_TRAINING`` is 12 epochs on ONE seed. The experiments in ``results/`` run
40 epochs over 5 seeds. Under ``docs/CONTRACT.md`` section 5 and the 5-SEED RULE
in CLAUDE.md, a single-seed run is not a finding and must never be reported as
one. Every record here therefore carries ``is_evidence: false`` and the label
the page prints verbatim, and the per-scenario payload records the seed that
produced it so the run can be repeated exactly.

The 5-seed results stay where they belong: ``results/`` , surfaced by
``build_data.py`` as the V1 charts in Chapter 2.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "data"
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import app as V1  # the Streamlit app itself; imported, never edited  # noqa: E402
from prismflow.statistics.tail_dependence import MIN_TAIL_SAMPLES  # noqa: E402
from prismflow.utils.seed import set_seed  # noqa: E402

#: One recorded seed for every run on the page. Written into each record so a
#: reader can reproduce it, and named in the label so nobody mistakes it for a
#: mean over seeds.
SEED = 0

#: Samples scored per scenario. app.py's slider default.
N_SAMPLES = 256

#: The label every single-scenario output on the page must carry, verbatim.
ILLUSTRATIVE_LABEL = (
    "Illustrative run, seed %d -- the findings are the 5-seed results in "
    "Chapter 2. %d epochs on one seed (the experiments use 40 epochs over 5 "
    "seeds), so this is not evidence under docs/CONTRACT.md section 5."
    % (SEED, V1.DEMO_TRAINING["epochs"])
)

#: Slider values app.py offers. These are its own defaults, not new choices.
SCENARIO_OPTIONS = {
    "clean": {},
    "clone": {"clone_k": 2},
    "missing": {"rate": 0.3},
    "noisy": {"sigma": 2.0},
    "chorus": {"k": 2, "epsilon": 1.0, "steps": 20},
    "adaptive": {"k": 2, "epsilon": 1.0, "steps": 20, "gamma": 5.0, "tau": 0.2},
}

#: How many real per-example predictions to record per scenario. Part 25 phase
#: (e) step 4 needs actual wrong answers; no committed results file holds
#: per-example predictions, so this build is the only honest source for them.
MAX_EXAMPLES = 12

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),
]


def clean(value):
    """NaN and Infinity cannot live in JSON; both become null -> not measured."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def matrix(array):
    """A 2-D array as nested lists, with NaN mapped to null rather than dropped.

    The dependence matrix legitimately contains NaN -- a pair with too few tail
    samples has no estimate -- and that must reach the page as "not measured",
    not as 0.0, which would read as "measured, and independent".
    """
    if array is None:
        return None
    out = []
    for row in np.asarray(array, dtype=float).tolist():
        out.append([clean(v) for v in row])
    return out


def per_example(output, labels, limit=MAX_EXAMPLES):
    """Real predictions against the true label, wrong ones first.

    This is the only place in the project where a per-example prediction exists:
    every committed metrics.json holds aggregates and reliability bins, and
    results/per_sample/audit_*.json holds per-epoch dependence traces. So the
    "where it goes wrong" section either reads these, or says "not recorded".
    """
    predicted = output.prediction.detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    confidence = output.confidence.detach().cpu().numpy()
    probs = output.probs.detach().cpu().numpy()
    uncertainty = (output.uncertainty.detach().cpu().numpy()
                   if output.uncertainty is not None else None)

    wrong = np.flatnonzero(predicted != truth)
    right = np.flatnonzero(predicted == truth)
    # Wrong examples first, then correct ones for contrast, both in index order
    # rather than sorted by confidence -- ranking by confidence would let the
    # page show the most embarrassing cases and call it a sample.
    chosen = list(wrong[: limit - min(3, len(right))]) + list(right[:3])
    records = []
    for i in chosen:
        i = int(i)
        records.append({
            "index": i,
            "predicted": int(predicted[i]),
            "true": int(truth[i]),
            "correct": bool(predicted[i] == truth[i]),
            "confidence": clean(float(confidence[i])),
            "max_probability": clean(float(probs[i].max())),
            "uncertainty": clean(float(uncertainty[i])) if uncertainty is not None else None,
        })
    return {
        "n_scored": int(truth.shape[0]),
        "n_wrong": int(wrong.size),
        "n_correct": int(right.size),
        "examples": records,
        "selection": ("wrong examples in index order, then three correct ones; "
                      "NOT ranked by confidence"),
    }


def run_scenario(dataset_key, loader, scenario, options):
    """One scenario, start to finish, through app.py's own functions."""
    synthetic = dataset_key == "synthetic"
    clone_k = options.get("clone_k", 0)
    opts = dict(options)
    opts["synthetic"] = synthetic
    opts["base_views"] = V1.DEMO_DATA["n_views"] if synthetic else 6

    set_seed(SEED)
    data, splits, model, names, _spec = loader(SEED, clone_k)
    clean_batch = V1.build_batch(data, splits, N_SAMPLES)
    # Calibrated on the CLEAN batch before corruption, exactly as app.py does.
    threshold = V1.clean_threshold(model, clean_batch)
    batch, states, compromised = V1.apply_scenario(
        model, clean_batch, scenario, opts, SEED)
    engine = V1.run_engine(model, batch, threshold=threshold)

    output, eniv = engine["output"], engine["eniv"]
    pair, lambda_u = V1.most_redundant_pair(
        engine["dependence"], names, engine["tail_matrix"])

    off = ~np.eye(engine["tail_matrix"].shape[0], dtype=bool)
    finite = engine["tail_matrix"][off]
    counts = engine["tail_counts"][off]

    evidence = engine["evidence"][0]
    suspicion = output.suspicion

    return {
        "scenario": scenario,
        "description": V1.SCENARIOS[scenario],
        "options": {k: v for k, v in options.items()},
        "seed": SEED,
        "n_samples_scored": int(batch.labels.shape[0]),
        "is_evidence": False,
        "label": ILLUSTRATIVE_LABEL,

        "view_names": list(names),
        "view_states": list(states),
        "compromised_views": [int(v) for v in compromised],
        "present_fraction": [clean(v) for v in engine["present_fraction"]],
        "per_view_alpha": ([clean(v) for v in eniv.per_view_alpha]
                           if eniv.per_view_alpha is not None else None),

        "nominal_views": clean(eniv.nominal_views),
        "effective_views": clean(eniv.effective_views),
        "efficiency_ratio": clean(eniv.efficiency_ratio),
        "mean_dependence": clean(eniv.mean_dependence),
        "dependence": matrix(engine["dependence"]),
        "most_redundant_pair": list(pair) if pair else None,
        "most_redundant_lambda_u": clean(lambda_u),

        "tail_lambda_u_mean": (clean(float(np.nanmean(finite))) if finite.size else None),
        "tail_n_min": (int(counts.min()) if counts.size else 0),
        "tail_min_samples_required": MIN_TAIL_SAMPLES,
        "tail_quantile": V1.TAIL_QUANTILE,

        "accuracy": clean(engine["accuracy"]),
        "sample_0": {
            "prediction": int(output.prediction[0]),
            "true_label": int(batch.labels[0]),
            "confidence": clean(float(output.confidence[0])),
            "uncertainty": clean(float(output.uncertainty[0])),
            "max_probability": clean(float(output.probs[0].max())),
            "per_view_evidence": [[clean(v) for v in row]
                                  for row in np.asarray(evidence).tolist()],
        },
        "predictions": per_example(output, batch.labels),
        "suspicion": (None if suspicion is None else {
            "score_mean": clean(float(np.nanmean(suspicion.score))),
            "flag_rate": clean(suspicion.flag_rate),
            "threshold": clean(suspicion.threshold),
            "calibrated_at": 0.05,
            "threshold_note": ("calibrated on THIS configuration's clean scores "
                               "at a 5% target rate, the same protocol Part 10 "
                               "uses per seed"),
        }),
    }


def build() -> dict:
    datasets = {}
    for key, loader, title in (
        ("synthetic", V1.load_synthetic, "synthetic (known rho)"),
        ("handwritten", V1.load_real, "handwritten (real, 6 views)"),
    ):
        scenarios = {}
        errors = {}
        for scenario, options in SCENARIO_OPTIONS.items():
            print("  %-12s %-9s ..." % (key, scenario), end="", flush=True)
            try:
                scenarios[scenario] = run_scenario(key, loader, scenario, options)
                print(" ok  acc %.3f  ENIV %.3f/%d"
                      % (scenarios[scenario]["accuracy"],
                         scenarios[scenario]["effective_views"],
                         scenarios[scenario]["nominal_views"]))
            except Exception as exc:  # noqa: BLE001
                # A scenario that could not run is recorded as a failure with
                # its exception, NOT omitted and NOT filled from elsewhere.
                scenarios[scenario] = None
                errors[scenario] = "%s: %s" % (type(exc).__name__, exc)
                print(" FAILED  %s: %s" % (type(exc).__name__, str(exc)[:70]))
                traceback.print_exc(limit=2)
        datasets[key] = {"title": title, "scenarios": scenarios, "errors": errors}

    return {
        "title": "V1 engine, run per scenario",
        "label": ILLUSTRATIVE_LABEL,
        "is_evidence": False,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "training": dict(V1.DEMO_TRAINING),
        "data": dict(V1.DEMO_DATA),
        "scenario_order": list(SCENARIO_OPTIONS),
        "scenario_descriptions": dict(V1.SCENARIOS),
        "engine": {
            "note": ("every value is a return value of these functions, reached "
                     "through app.py, which this build imports and does not edit"),
            "synthetic_loader": "experiments.calibration.run_calibration.build_dataset + train",
            "real_loader": "prismflow.data.real_datasets.build_real_dataset + experiments.real.run_real.train_real",
            "corruptions": "prismflow.data.corruption.drop_views / add_noise",
            "attacks": "prismflow.attacks.chorus.chorus_attack / prismflow.attacks.adaptive.adaptive_attack",
            "forward": "prismflow.models.defended.DefendedPrismFlow",
            "dependence": "prismflow.statistics.dependence.feature_dependence_matrix",
            "eniv": "prismflow.eniv.eniv.compute_eniv",
            "tail": "prismflow.statistics.tail_dependence.tail_dependence_matrix",
            "threshold": "prismflow.statistics.suspicion.calibrate_threshold",
        },
        "datasets": datasets,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "computed_not_copied": True,
            "sources": [{
                "path": "app.py",
                "sha256": hashlib.sha256((ROOT / "app.py").read_bytes()).hexdigest(),
                "bytes": (ROOT / "app.py").stat().st_size,
            }],
            "note": ("unlike the other web/data files, this one is COMPUTED by "
                     "running the V1 engine rather than copied from results/. "
                     "The sha256 is of app.py, the wiring it was computed "
                     "through, so a stale file is detectable."),
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("running the V1 engine, seed %d, %d scenarios x 2 datasets" % (SEED, len(SCENARIO_OPTIONS)))
    payload = build()
    text = json.dumps(payload, indent=2, allow_nan=False)

    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            print("ABORT: output matches a credential pattern (%s). Nothing written."
                  % pattern.pattern)
            return 1

    (OUT / "v1_scenarios.json").write_text(text, encoding="utf-8")
    ran = sum(1 for d in payload["datasets"].values()
              for s in d["scenarios"].values() if s is not None)
    failed = sum(len(d["errors"]) for d in payload["datasets"].values())
    print("\nwrote web/data/v1_scenarios.json  %d bytes  %d scenario(s) ran, %d failed"
          % (len(text), ran, failed))
    if failed:
        print("failed scenarios are recorded as null with their exception; "
              "the page will render them as not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
