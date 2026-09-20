"""PrismFlow interactive demo (Part 16).

    streamlit run app.py --server.fileWatcherType none

The `--server.fileWatcherType none` is not optional. Streamlit's default
watcher walks the module table and touches `torch.classes`, whose custom
`__getattr__` raises on the attribute the watcher probes; the app then dies
with "Tried to instantiate class '__path__._path'". Disabling the watcher
costs only hot-reload.

WHAT THIS FILE IS ALLOWED TO DO

Pick a scenario, hand it to the engine, and hand the engine's answers to
`prismflow.app.panels`. That is all. There is no model logic here: no fusion,
no dependence estimation, no discount, no attack objective, and no training
loop of its own. Training is delegated to the very functions the experiments
use -- `experiments.calibration.run_calibration.train` for synthetic data and
`experiments.real.run_real.train_real` for real data -- so a number on this
screen is produced by the same code that produced the numbers in `results/`.

Neither this file nor `prismflow/app/panels.py` imports `torch.nn`, and
`tests/unit/test_app.py` asserts it. The rule exists because a demo that
computes anything its own way is how a figure nobody can reproduce gets
published.

`torch` itself is imported, for `no_grad` and tensor plumbing only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from experiments.calibration.run_calibration import build_dataset, train
from experiments.real.run_real import clone_dataset, train_real
from prismflow.app import panels
from prismflow.attacks.adaptive import AdaptiveConfig, adaptive_attack
from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.data.corruption import add_noise, drop_views, duplicate_view
from prismflow.data.dataset import Batch
from prismflow.data.real_datasets import build_real_dataset, get_spec
from prismflow.eniv.eniv import compute_eniv
from prismflow.models.defended import DefendedPrismFlow
from prismflow.models.prismflow import REPORTING_NULL_PERMUTATIONS
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.statistics.suspicion import DEFAULT_THRESHOLD as DEFAULT_SUSPICION_THRESHOLD
from prismflow.statistics.suspicion import calibrate_threshold
from prismflow.statistics.tail_dependence import MIN_TAIL_SAMPLES, tail_dependence_matrix
from prismflow.utils.seed import set_seed

RESULTS_DIR = Path("results")
TAIL_QUANTILE = 0.90

SCENARIOS = {
    "clean": "No corruption. The baseline every other scenario is read against.",
    "clone": "k exact duplicates of one view are appended -- redundancy with no new information.",
    "missing": "Each (sample, view) is dropped independently at the given rate.",
    "noisy": "Gaussian noise on one view. Part 08: this makes a view LESS dependent, not more.",
    "chorus": "Part 09 collusive attack: k views optimised jointly to agree on one wrong class.",
    "adaptive": "Part 14: the Chorus attacker plus an evasion term that penalises measured dependence.",
}

# Training is kept short so the demo stays interactive. These are NOT the
# experiment settings -- experiments run 40 epochs over 5 seeds, and anything
# shown here is one seed and therefore not evidence under docs/CONTRACT.md.
DEMO_TRAINING = {
    "epochs": 12,
    "batch_size": 64,
    "lr": 1e-3,
    "anneal_epochs": 6,
    "per_view_loss_weight": 1.0,
}
DEMO_DATA = {"n_views": 4, "rho": 0.3}


# ---------------------------------------------------------------------------
# Engine calls (cached so the sliders stay responsive)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Training the model (engine call)...")
def load_synthetic(seed: int, clone_k: int):
    """Synthetic dataset + model, trained by the calibration experiment's own
    `train`. `clone_k > 0` widens the model, because the clone scenario
    appends views and the encoder stack must match."""
    data_cfg = dict(DEMO_DATA, n_views=DEMO_DATA["n_views"] + clone_k)
    dataset, splits = build_dataset(seed, data_cfg)
    model = train(dataset, splits, seed, True, data_cfg, DEMO_TRAINING)
    model.use_discount = True
    model.eval()
    names = [f"view {v}" for v in range(data_cfg["n_views"])]
    return dataset, splits, model, names, None


@st.cache_resource(show_spinner="Loading Handwritten and training (engine call)...")
def load_real(seed: int, clone_k: int):
    """UCI Handwritten + model, trained by Part 15's own `train_real`."""
    spec = get_spec("handwritten")
    dataset, splits = build_real_dataset("handwritten", seed=seed)
    source = clone_dataset(dataset, spec, source=0, k=clone_k)
    cfg = {
        "model": {"hidden_dims": [64, 64], "feature_dim": 32, "dropout": 0.0},
        "training": DEMO_TRAINING,
    }
    model = train_real(source, splits, spec, seed, True, cfg, clone_k=clone_k)
    model.use_discount = True
    model.eval()
    names = [v.name for v in spec.views] + [f"{spec.views[0].name} copy {i + 1}" for i in range(clone_k)]
    return source, splits, model, names, spec


def build_batch(dataset, splits, n_samples: int) -> Batch:
    indices = splits["test"][:n_samples]
    return dataset.get_batch(indices)


def apply_scenario(model, batch: Batch, scenario: str, options: dict, seed: int):
    """Corruptions and attacks, every one of them an existing engine function.

    Returns (batch, per-view state labels, compromised view indices).
    """
    n_views = batch.views.shape[1]
    states = ["available"] * n_views
    compromised: tuple = ()

    if scenario == "clone":
        # The dataset and model were already built at the cloned width by the
        # loader, so the extra columns are present; they are labelled here.
        for v in range(DEMO_DATA["n_views"] if options["synthetic"] else options["base_views"], n_views):
            states[v] = "duplicate"
        return batch, states, compromised

    if scenario == "missing":
        batch = drop_views(batch, rate=options["rate"], seed=seed)
        present = batch.view_mask.float().mean(dim=0).tolist()
        states = ["missing" if p < 0.5 else "available" for p in present]
        return batch, states, compromised

    if scenario == "noisy":
        batch = add_noise(batch, view_idx=0, sigma=options["sigma"], seed=seed)
        states[0] = "noisy"
        return batch, states, compromised

    if scenario in ("chorus", "adaptive"):
        if scenario == "chorus":
            config = ChorusConfig(
                k=options["k"], epsilon=options["epsilon"], beta=1.0,
                steps=options["steps"], seed=seed,
            )
            result = chorus_attack(model, batch.views, batch.view_mask, config)
        else:
            config = AdaptiveConfig(
                k=options["k"], epsilon=options["epsilon"], beta=1.0,
                steps=options["steps"], seed=seed,
                gamma=options["gamma"], tau=options["tau"],
            )
            result = adaptive_attack(model, batch.views, batch.view_mask, config)
        compromised = result.compromised
        for v in compromised:
            states[v] = "compromised"
        batch = Batch(
            views=batch.views + result.delta,
            view_mask=batch.view_mask.clone(),
            labels=batch.labels.clone(),
            sample_ids=batch.sample_ids.clone(),
        )
    return batch, states, compromised


@torch.no_grad()
def clean_threshold(model, batch: Batch, target_rate: float = 0.05) -> float:
    """Suspicion threshold calibrated on THIS configuration's clean scores.

    Without this the demo reports the module default (0.05 cosine units)
    against a model trained for 12 epochs, and flags 84-100% of CLEAN samples
    -- a false alarm that would make the detector look broken when it is
    merely uncalibrated. Part 10 calibrates per seed on that seed's own clean
    scores and this follows the same protocol, through the same
    `calibrate_threshold` function, so the number on the screen means what the
    number in `results/comparison/` means.
    """
    defended = DefendedPrismFlow(model, suspicion_seed=0)
    output = defended(batch.views, batch.view_mask)
    if output.suspicion is None:
        return DEFAULT_SUSPICION_THRESHOLD
    return float(calibrate_threshold(output.suspicion.score, target_rate=target_rate))


@torch.no_grad()
def run_engine(model, batch: Batch, threshold: float | None = None):
    """One defended forward pass plus the project's reporting-grade diagnostics.

    Every quantity returned is the return value of an existing function.
    """
    defended = DefendedPrismFlow(
        model,
        suspicion_threshold=DEFAULT_SUSPICION_THRESHOLD if threshold is None else threshold,
        suspicion_seed=0,
    )
    output = defended(batch.views, batch.view_mask)

    features = model.encoder(batch.views, batch.view_mask)
    dependence = feature_dependence_matrix(
        features, output.per_view_evidence, batch.view_mask,
        method=model.dependence_method, conditioning=model.dependence_conditioning,
        seed=model.dependence_seed, null_permutations=REPORTING_NULL_PERMUTATIONS,
    )
    present = available_views(batch.view_mask, n_views=model.n_views)
    eniv = compute_eniv(dependence, present)

    evidence = output.per_view_evidence.detach().cpu().numpy()
    predicted = output.prediction.detach().cpu().numpy()
    scalar = evidence[np.arange(evidence.shape[0]), :, predicted]
    tail_matrix, tail_counts = tail_dependence_matrix(scalar, quantile=TAIL_QUANTILE)

    return {
        "output": output,
        "dependence": dependence,
        "eniv": eniv,
        "tail_matrix": tail_matrix,
        "tail_counts": tail_counts,
        "evidence": evidence,
        "accuracy": float((output.prediction == batch.labels).float().mean()),
        "present_fraction": batch.view_mask.float().mean(dim=0).tolist(),
    }


def most_redundant_pair(matrix: np.ndarray, names, tail_matrix=None):
    """Select the largest off-diagonal entry. Selection, not measurement."""
    matrix = np.asarray(matrix, dtype=float)
    masked = matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    masked[~np.isfinite(masked)] = -np.inf
    if not np.isfinite(masked).any():
        return None, None
    i, j = np.unravel_index(np.argmax(masked), masked.shape)
    lambda_u = None if tail_matrix is None else float(np.asarray(tail_matrix)[i, j])
    return (names[i], names[j]), lambda_u


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="PrismFlow", page_icon=":material/hub:", layout="wide")
    st.title("PrismFlow")
    st.caption(
        "Dependence-discounted evidential fusion. Agreement between views should "
        "count in proportion to how independent those views are, not how many "
        "of them there are."
    )

    with st.sidebar:
        st.header("Scenario")
        source = st.radio(
            "Dataset",
            ["synthetic (known rho)", "handwritten (real, 6 views)"],
            help="ENIV is VALIDATED only on synthetic data, where rho is a "
            "parameter we set. On Handwritten it is applied, not validated.",
        )
        synthetic = source.startswith("synthetic")
        scenario = st.selectbox("Scenario", list(SCENARIOS), index=0)
        st.caption(SCENARIOS[scenario])

        seed = st.number_input("Seed", 0, 999, 0, 1)
        n_samples = st.slider("Samples to score", 64, 512, 256, 64)

        options = {"synthetic": synthetic, "base_views": 6 if not synthetic else DEMO_DATA["n_views"]}
        clone_k = 0
        if scenario == "clone":
            clone_k = st.slider("Duplicate copies (k)", 1, 4, 2)
        elif scenario == "missing":
            options["rate"] = st.slider("Drop rate", 0.0, 0.9, 0.3, 0.1)
        elif scenario == "noisy":
            options["sigma"] = st.slider("Noise sigma", 0.0, 4.0, 2.0, 0.5)
        elif scenario in ("chorus", "adaptive"):
            options["k"] = st.slider("Compromised views (k)", 1, 3, 2)
            options["epsilon"] = st.slider("Attack budget epsilon", 0.0, 2.0, 1.0, 0.1)
            options["steps"] = st.slider("Attack steps", 5, 40, 20, 5)
            if scenario == "adaptive":
                options["gamma"] = st.slider("Evasion weight gamma", 0.0, 20.0, 5.0, 1.0)
                options["tau"] = st.slider("Evasion threshold tau", 0.0, 1.0, 0.2, 0.05)

        run = st.button("Run", type="primary", width="stretch")
        st.divider()
        st.caption(
            f"Demo training: {DEMO_TRAINING['epochs']} epochs, one seed. "
            "Experiments use 40 epochs over 5 seeds. Nothing on this screen is "
            "evidence under docs/CONTRACT.md."
        )

    budget_tab, results_tab, figures_tab = st.tabs(
        ["Independence budget", "Run and results", "Saved figures"]
    )

    if run:
        set_seed(int(seed))
        loader = load_synthetic if synthetic else load_real
        dataset, splits, model, names, _ = loader(int(seed), clone_k)
        clean_batch = build_batch(dataset, splits, int(n_samples))
        # Calibrate on the clean batch BEFORE corrupting it, exactly as Part 10
        # calibrates on its own clean condition.
        threshold = clean_threshold(model, clean_batch)
        batch, states, compromised = apply_scenario(
            model, clean_batch, scenario, options, int(seed)
        )
        st.session_state["result"] = {
            "engine": run_engine(model, batch, threshold=threshold),
            "threshold": threshold,
            "names": names,
            "states": states,
            "compromised": compromised,
            "scenario": scenario,
            "batch_labels": batch.labels,
            "source": source,
        }

    payload = st.session_state.get("result")

    with budget_tab:
        if payload is None:
            st.info("Choose a scenario in the sidebar and press **Run**.")
        else:
            engine = payload["engine"]
            eniv = engine["eniv"]
            pair, lambda_u = most_redundant_pair(
                engine["dependence"], payload["names"], engine["tail_matrix"]
            )
            panels.render_independence_budget(
                nominal=eniv.nominal_views,
                effective=eniv.effective_views,
                redundant_pair=pair,
                lambda_u=lambda_u,
                rho_bar=eniv.mean_dependence,
                dataset_label=f"{payload['source']} -- scenario: {payload['scenario']}",
            )
            panels.render_dependence_heatmap(engine["dependence"], payload["names"])

    with results_tab:
        if payload is None:
            st.info("Choose a scenario in the sidebar and press **Run**.")
        else:
            engine = payload["engine"]
            output, eniv = engine["output"], engine["eniv"]
            index = 0

            panels.render_prediction(
                prediction=int(output.prediction[index]),
                confidence=float(output.confidence[index]),
                uncertainty=float(output.uncertainty[index]),
                probability=float(output.probs[index].max()),
            )
            st.caption(
                f"Batch accuracy over {len(payload['batch_labels'])} scored samples: "
                f"{engine['accuracy']:.3f}. Metrics above are for sample {index}."
            )

            panels.render_view_status(
                payload["names"], payload["states"],
                alphas=eniv.per_view_alpha,
                present_fraction=engine["present_fraction"],
            )
            panels.render_eniv(eniv.nominal_views, eniv.effective_views, eniv.efficiency_ratio)

            off_diagonal = ~np.eye(engine["tail_matrix"].shape[0], dtype=bool)
            finite = engine["tail_matrix"][off_diagonal]
            counts = engine["tail_counts"][off_diagonal]
            panels.render_tail_dependence(
                lambda_u=float(np.nanmean(finite)) if finite.size else None,
                n_tail=int(counts.min()) if counts.size else 0,
                min_tail_samples=MIN_TAIL_SAMPLES,
                quantile=TAIL_QUANTILE,
            )

            panels.render_evidence_bars(engine["evidence"][index], payload["names"])

            if output.suspicion is not None:
                panels.render_suspicion(
                    score=float(np.nanmean(output.suspicion.score)),
                    flag_rate=output.suspicion.flag_rate,
                    threshold=output.suspicion.threshold,
                    calibrated_at=0.05,
                )

    with figures_tab:
        panels.render_saved_plots(panels.find_saved_plots(RESULTS_DIR), RESULTS_DIR)

    panels.render_limitations_footer()


if __name__ == "__main__":
    main()
