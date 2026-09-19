"""Part 15: the Part 10 matrix and the Part 13 tail analysis, on real data.

WHAT THIS SCRIPT ADDS BEYOND RE-RUNNING TWO OLD PARTS

The headline is the redundancy audit in stage 1. On synthetic data the number
of independent views is a knob the generator was given; on a real benchmark it
is an unknown, and as far as we can tell nobody reports it. So before any
attack or corruption is applied, this measures how many effectively
independent views a standard multi-view benchmark actually contains:

    nominal views | measured ENIV | efficiency ratio | lambda_U

A benchmark whose six views are worth three independent ones reframes every
gain reported on it. A method claiming to exploit six views may be exploiting
three, and the headroom it is competing for is smaller than the view count
suggests.

THE CONTROL THAT MAKES THAT NUMBER MEAN ANYTHING

"ENIV is below 6" is not by itself evidence of redundancy, because the
estimator reads below the view count at finite n even when the views ARE
independent (`prismflow/eniv/eniv.py`, "finite samples"). Stage 1 therefore
measures the estimator's own reading under an explicit independence null: each
view's encoder features are permuted WITHIN CLASS, which destroys cross-view
coupling while preserving the class structure the conditional estimator
conditions on. The gap between the observed ENIV and that null is the claim;
the raw number alone is not.

WHAT IS NOT CLAIMED HERE

Real views have no ground-truth dependence, so ENIV is APPLIED on this data
and not VALIDATED by it. Validation exists only on the synthetic generator
(Part 05), where rho is known. `docs/DATASETS.md` states this plainly, and
nothing in this script's output should be read as confirming the estimator.

REUSED CODE, DELIBERATELY

The corruption, attack, detector-scoring and tail-measurement functions are
IMPORTED from `experiments.comparison.run_comparison` and
`experiments.tail.run_tail` rather than reimplemented. Re-running a Part on new
data means running the same code on new data; a reimplementation would leave
every difference ambiguous between the data and the rewrite. Those modules are
frozen and are not modified -- Part 15's brief requires the real dataset to sit
behind the existing interface precisely so that they need not be.

Usage:
    python -m experiments.real.run_real
    python -m experiments.real.run_real --stage redundancy
    python -m experiments.real.run_real --quick   # smoke test, NOT evidence
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# `run_condition` drives Part 10's own `corrupt` and `attack` internally; this
# script never calls them directly, which is the point -- the condition
# semantics stay Part 10's.
from experiments.comparison.run_comparison import (
    DETECTOR_METRICS,
    json_safe,
    roc_auc,
    run_condition,
    summarise as summarise_detector,
)
from experiments.tail.run_tail import (
    detection_at_fpr,
    measure_condition,
    roc_auc as tail_roc_auc,
    summarise as summarise_tail,
    whole_split,
)
from prismflow.data.real_datasets import (
    REGISTRY,
    RealMultiViewDataset,
    build_real_dataset,
    get_spec,
    view_configs,
)
from prismflow.data.loaders import iter_batches
from prismflow.eniv.eniv import compute_eniv
from prismflow.evaluation import evaluate
from prismflow.models.defended import DefendedPrismFlow
from prismflow.models.prismflow import REPORTING_NULL_PERMUTATIONS, PrismFlow
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.statistics.suspicion import calibrate_threshold
from prismflow.statistics.tail_dependence import MIN_TAIL_SAMPLES, tail_dependence_matrix, tail_eniv
from prismflow.train import compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger("prismflow.real")
CONFIG_PATH = Path(__file__).with_name("config.yaml")
STAGES = ("redundancy", "comparison", "tail")


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------


def clone_dataset(dataset, spec, source: int, k: int):
    """The real-data analogue of Part 10's wider synthetic dataset.

    Part 10 trains its clone-condition model on a synthetic dataset generated
    with V + k views. Real data cannot be regenerated wider, so the copies are
    made the same way the CONDITION makes them: k exact duplicates of view
    `source` appended, matching `prismflow.data.corruption.duplicate_view`.
    The model therefore sees at training time the same view structure it will
    be evaluated on, which is what the synthetic path arranges.
    """
    if k == 0:
        return dataset
    views = dataset.views.numpy()
    widened = np.concatenate([views, np.repeat(views[:, source : source + 1], k, axis=1)], axis=1)
    return RealMultiViewDataset(widened, dataset.labels.numpy(), spec, metadata={"clone_k": k})


def train_real(dataset, splits, spec, seed: int, use_discount: bool, cfg: dict, clone_k: int = 0):
    """Train one PrismFlow on real views.

    Not `experiments.calibration.run_calibration.train`: that builds its model
    through `TrainConfig`, which carries a single `d_view` for every view and a
    fixed `n_classes`. Real views have six different widths and ten classes, so
    the encoder configs are built per view by
    `prismflow.data.real_datasets.view_configs`. Everything else -- optimiser,
    loss, annealing schedule, per-view loss weight, the same-seed-same-init
    discipline -- is identical to the synthetic path.
    """
    set_seed(seed)
    model = PrismFlow(
        view_configs(
            spec,
            hidden_dims=cfg["model"]["hidden_dims"],
            feature_dim=cfg["model"]["feature_dim"],
            dropout=cfg["model"]["dropout"],
            clone_k=clone_k,
        ),
        n_classes=spec.n_classes,
        use_discount=use_discount,
    )
    training = cfg["training"]
    optimiser = torch.optim.Adam(model.parameters(), lr=training["lr"])

    for epoch in range(training["epochs"]):
        model.train()
        for batch in iter_batches(
            dataset, splits["train"], training["batch_size"], shuffle=True, seed=seed + epoch
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=spec.n_classes
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
# Stage 1: the redundancy audit
# ---------------------------------------------------------------------------


@torch.no_grad()
def encoder_state(model, views, view_mask):
    """Features, evidence and the per-sample scalar every measure below reads.

    s[n, v] = evidence view v assigns to the FUSED PREDICTED class -- Part 13's
    statistic, unchanged, so the lambda_U column here and Part 13's lambda_U
    are the same quantity computed by the same code.
    """
    output = model(views, view_mask)
    evidence = output.per_view_evidence
    predicted = output.prediction.detach().cpu().numpy()
    return {
        "features": model.encoder(views, view_mask),
        "evidence": evidence,
        "predicted": predicted,
        "scalar": evidence.detach().cpu().numpy()[np.arange(evidence.shape[0]), :, predicted],
        "output": output,
    }


def dependence_of(features, evidence, view_mask, model) -> np.ndarray:
    """The project's reporting-grade dependence matrix on encoder features.

    REPORTING_NULL_PERMUTATIONS, not the forward pass's 4: the upward bias the
    permutation correction removes scales with feature_dim / n_samples, and
    this is the number that goes in the headline table.
    """
    return feature_dependence_matrix(
        features,
        evidence,
        view_mask,
        method=model.dependence_method,
        conditioning=model.dependence_conditioning,
        seed=model.dependence_seed,
        null_permutations=REPORTING_NULL_PERMUTATIONS,
    )


def within_class_permutation(features: torch.Tensor, evidence: torch.Tensor, labels, rng):
    """Independently permute each view's rows WITHIN each class -> (features, evidence).

    This is the independence null. Permuting rows globally would also destroy
    the class signal, and the dependence estimator conditions on class -- it
    would then be measuring a different object. Permuting within class leaves
    every marginal, every class mean and the sample size untouched and removes
    only the within-class cross-view coupling, which is exactly what the
    class-conditional estimator is defined to measure.

    Each view's EVIDENCE is carried along under the same permutation as its
    features. It has to be: `feature_dependence_matrix` derives the
    conditioning stratum for a pair from the other views' evidence, so leaving
    evidence in place would stratify a permuted view's features by a class
    belonging to the row it replaced. That would centre the residuals with the
    wrong class means and put an artefact into the null the observed
    measurement does not carry.

    Every view is permuted, none privileged, so the null is symmetric in the
    views exactly as the estimator is.
    """
    labels = np.asarray(labels)
    permuted_features, permuted_evidence = features.clone(), evidence.clone()
    for view in range(features.shape[1]):
        for label in np.unique(labels):
            rows = np.flatnonzero(labels == label)
            shuffled = rows[rng.permutation(len(rows))]
            permuted_features[rows, view] = features[shuffled, view]
            permuted_evidence[rows, view] = evidence[shuffled, view]
    return permuted_features, permuted_evidence


def mean_off_diagonal(matrix: np.ndarray) -> float:
    off = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    finite = off[np.isfinite(off)]
    return float(finite.mean()) if finite.size else float("nan")


def redundancy_row(model, dataset, splits, spec, cfg, seed: int) -> dict:
    """Every redundancy measurement for one (dataset, seed), before any attack."""
    views, view_mask, labels = whole_split(
        dataset, splits["test"], cfg["tail"]["evaluation"]["batch_size"]
    )
    state = encoder_state(model, views, view_mask)
    present = available_views(view_mask, n_views=model.n_views)
    dependence = dependence_of(state["features"], state["evidence"], view_mask, model)

    row = {
        "seed": seed,
        "n_test": int(views.shape[0]),
        "nominal_views": int(model.n_views),
        "accuracy": float((state["output"].prediction == labels).float().mean()),
        "rho_bar": mean_off_diagonal(dependence),
        "dependence_matrix": dependence.tolist(),
    }

    for method in cfg["redundancy"]["eniv_methods"]:
        result = compute_eniv(dependence, present, method=method)
        row[f"eniv_{method}"] = float(result.effective_views)
        row[f"efficiency_ratio_{method}"] = float(result.efficiency_ratio)
    row["eniv"] = row["eniv_eigen"]
    row["efficiency_ratio"] = row["efficiency_ratio_eigen"]

    # lambda_U on Part 13's scalar, at the tail stage's headline quantile.
    quantile = cfg["tail"]["headline_quantile"]
    tail_matrix, counts = tail_dependence_matrix(state["scalar"], quantile=quantile)
    off_counts = counts[~np.eye(counts.shape[0], dtype=bool)]
    row["lambda_u"] = mean_off_diagonal(tail_matrix)
    row["tail_eniv"] = float(tail_eniv(tail_matrix))
    row["tail_efficiency_ratio"] = row["tail_eniv"] / model.n_views
    row["n_tail_min"] = int(off_counts.min()) if off_counts.size else 0
    row["tail_reliable"] = bool(row["n_tail_min"] >= MIN_TAIL_SAMPLES)
    row["tail_matrix"] = tail_matrix.tolist()

    # The independence null: what this estimator reads on this data, at this n
    # and this feature dimension, when the views are independent by construction.
    label_array = labels.detach().cpu().numpy()
    nulls, null_rhos = [], []
    for repeat in range(cfg["redundancy"]["independence_null_repeats"]):
        rng = np.random.default_rng(10_000 + 97 * seed + repeat)
        null_features, null_evidence = within_class_permutation(
            state["features"], state["evidence"], label_array, rng
        )
        null_dependence = dependence_of(null_features, null_evidence, view_mask, model)
        nulls.append(float(compute_eniv(null_dependence, present).effective_views))
        null_rhos.append(mean_off_diagonal(null_dependence))
    row["eniv_independence_null"] = float(np.mean(nulls))
    row["eniv_independence_null_std"] = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else float("nan")
    row["rho_bar_independence_null"] = float(np.mean(null_rhos))
    # The claim: how far below its OWN null the estimator reads on real views.
    row["eniv_deficit_vs_null"] = row["eniv"] - row["eniv_independence_null"]

    if cfg["redundancy"]["leave_one_out"]:
        # Which views are carrying the redundancy: ENIV recomputed with each
        # view removed. A view whose removal barely lowers ENIV was adding
        # little independent evidence in the first place.
        loo = {}
        for view in range(model.n_views):
            keep = np.array([v != view for v in range(model.n_views)])
            loo[spec.views[view].name if view < spec.n_views else f"view{view}"] = float(
                compute_eniv(dependence[np.ix_(keep, keep)], present[keep]).effective_views
            )
        row["eniv_leave_one_out"] = loo

    return row


def stage_redundancy(spec, models, datasets, cfg, seeds, out_dir: Path) -> dict:
    rows = []
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = datasets[seed]
        row = redundancy_row(models[seed], dataset, splits, spec, cfg, seed)
        rows.append(row)
        LOGGER.info(
            "%s seed=%d acc=%.4f ENIV=%.3f/%d ratio=%.3f null=%.3f lambda_U=%.4f (%.0fs)",
            spec.name, seed, row["accuracy"], row["eniv"], row["nominal_views"],
            row["efficiency_ratio"], row["eniv_independence_null"], row["lambda_u"],
            time.perf_counter() - started,
        )

    keys = (
        "accuracy", "rho_bar", "eniv", "efficiency_ratio", "eniv_design_effect",
        "efficiency_ratio_design_effect", "lambda_u", "tail_eniv", "tail_efficiency_ratio",
        "eniv_independence_null", "rho_bar_independence_null", "eniv_deficit_vs_null",
        "n_tail_min", "n_test",
    )
    summary = {k: summarise_tail([{"v": r[k]} for r in rows], ("v",)) for k in keys if k in rows[0]}
    summary["nominal_views"] = rows[0]["nominal_views"]
    summary["tail_reliable"] = all(r["tail_reliable"] for r in rows)
    summary["headline_quantile"] = cfg["tail"]["headline_quantile"]

    mean_matrix = np.mean([np.asarray(r["dependence_matrix"]) for r in rows], axis=0)
    summary["dependence_matrix_mean"] = mean_matrix.tolist()
    if cfg["redundancy"]["leave_one_out"]:
        summary["eniv_leave_one_out"] = {
            name: summarise_tail([{"v": r["eniv_leave_one_out"][name]} for r in rows], ("v",))
            for name in rows[0]["eniv_leave_one_out"]
        }

    payload = {"summary": summary, "per_seed": rows}
    (out_dir / "redundancy.json").write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# Stage 2: the Part 10 comparison matrix
# ---------------------------------------------------------------------------


def stage_comparison(spec, cfg, seeds, out_dir: Path, results_dir: str, experiment_id: str) -> dict:
    comparison = cfg["comparison"]
    eval_cfg = comparison["evaluation"]
    fracs = comparison["split"]

    widths = sorted(
        {spec.n_views + c["k"] if c["kind"] == "clone" else spec.n_views
         for c in comparison["conditions"]}
    )

    datasets, models = {}, {}
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = build_real_dataset(
            spec.name, seed=seed, root=cfg["data"]["root"],
            train_frac=fracs["train"], val_frac=fracs["val"], test_frac=fracs["test"],
        )
        datasets[seed] = (dataset, splits)
        models[seed] = {}
        for width in widths:
            clone_k = width - spec.n_views
            source = clone_dataset(dataset, spec, source=0, k=clone_k)
            model = train_real(source, splits, spec, seed, True, cfg, clone_k=clone_k)
            model.use_discount = True
            model.eval()
            models[seed][width] = DefendedPrismFlow(
                model,
                suspicion_threshold=comparison["detector"]["threshold"],
                suspicion_permutations=comparison["detector"]["permutations"],
                suspicion_seed=seed,
            )
        LOGGER.info("seed %d trained widths %s (%.0fs)", seed, widths, time.perf_counter() - started)

    clean_scores, protocol_summary, detector_rows = {}, {}, {}
    progress_path = out_dir / "comparison_progress.json"

    for condition in comparison["conditions"]:
        name = condition["name"]
        started = time.perf_counter()
        width = spec.n_views + condition["k"] if condition["kind"] == "clone" else spec.n_views

        batches_by_seed, rows = {}, []
        for seed in seeds:
            defended = models[seed][width]
            dataset, splits = datasets[seed]
            # `corrupt` and `attack` are Part 10's, unchanged; `run_condition`
            # drives them and scores the detector exactly as Part 10 does.
            batches, scores = run_condition(
                defended, dataset, splits[eval_cfg["split"]], condition,
                comparison["attack"], eval_cfg["batch_size"], seed,
            )
            batches_by_seed[seed] = batches

            if condition["kind"] == "clean":
                clean_scores[seed] = scores
            reference = clean_scores.get(seed)
            if reference is None:
                raise RuntimeError("the clean condition must be listed first: it calibrates the threshold")

            at_5 = calibrate_threshold(reference, target_rate=0.05)
            at_1 = calibrate_threshold(reference, target_rate=0.01)
            finite = scores[np.isfinite(scores)]
            rows.append({
                "seed": seed,
                "auc": roc_auc(scores, reference),
                "detect_at_5pct": float((finite > at_5).mean()) if finite.size else float("nan"),
                "detect_at_1pct": float((finite > at_1).mean()) if finite.size else float("nan"),
                "mean_score": float(finite.mean()) if finite.size else float("nan"),
                "flag_rate_shared": float((finite > comparison["detector"]["threshold"]).mean())
                if finite.size else float("nan"),
                "threshold_at_5pct": float(at_5),
                "threshold_at_1pct": float(at_1),
            })

        report = evaluate(
            {seed: models[seed][width] for seed in seeds},
            lambda seed: iter(batches_by_seed[seed]),
            f"{experiment_id}/comparison/{name}",
            seeds,
            results_dir=results_dir,
            n_bins=eval_cfg["n_bins"],
            coverages=eval_cfg["coverages"],
            title=f"{spec.name}/{name}",
        )
        protocol_summary[name] = report["summary"]
        detector_rows[name] = rows

        summary = summarise_detector(rows, DETECTOR_METRICS)
        LOGGER.info(
            "%-14s auc=%.3f det@5%%=%.3f det@1%%=%.3f score=%+.4f acc=%.3f ece=%.3f (%.0fs)",
            name, summary["auc"]["mean"], summary["detect_at_5pct"]["mean"],
            summary["detect_at_1pct"]["mean"], summary["mean_score"]["mean"],
            report["summary"]["accuracy"]["mean"], report["summary"]["prob_ece"]["mean"],
            time.perf_counter() - started,
        )
        progress_path.write_text(
            json.dumps(json_safe({
                "completed": list(detector_rows),
                "detector": {k: summarise_detector(v, DETECTOR_METRICS) for k, v in detector_rows.items()},
                "protocol": protocol_summary,
            }), indent=2),
            encoding="utf-8",
        )

    payload = {
        "summary": {k: summarise_detector(v, DETECTOR_METRICS) for k, v in detector_rows.items()},
        "protocol": protocol_summary,
        "per_seed": detector_rows,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# Stage 3: the Part 13 tail analysis
# ---------------------------------------------------------------------------


def stage_tail(spec, models, datasets, cfg, seeds, out_dir: Path) -> dict:
    from prismflow.attacks.chorus import ChorusConfig, chorus_attack

    tail_cfg = cfg["tail"]
    epsilons = tail_cfg["attack"]["epsilons"]
    per_seed = {f"eps{e}": [] for e in epsilons}
    roc = {f"eps{e}": [] for e in epsilons}
    # `measure_condition` reads cfg["tail"]["quantiles"], ["headline_quantile"]
    # and ["subsample_n"]; this is the shape Part 13 passes it.
    measure_cfg = {"tail": tail_cfg}

    for seed in seeds:
        dataset, splits = datasets[seed]
        model = models[seed]
        views, view_mask, _ = whole_split(
            dataset, splits["test"], tail_cfg["evaluation"]["batch_size"]
        )
        rng = np.random.default_rng(tail_cfg["subsample_seed"] + seed)
        subsample_idx = rng.choice(
            views.shape[0], size=min(tail_cfg["subsample_n"], views.shape[0]), replace=False
        )

        clean_scores = None
        for epsilon in epsilons:
            started = time.perf_counter()
            if epsilon == 0.0:
                attacked = views
                compromised = tuple(range(tail_cfg["attack"]["k"]))
                target, delta = None, None
            else:
                result = chorus_attack(
                    model, views, view_mask,
                    ChorusConfig(
                        k=tail_cfg["attack"]["k"], epsilon=epsilon,
                        beta=tail_cfg["attack"]["beta"], steps=tail_cfg["attack"]["steps"],
                        seed=seed,
                    ),
                )
                attacked = views + result.delta
                compromised, target, delta = result.compromised, result.target, result.delta

            row = measure_condition(
                model, attacked, view_mask, measure_cfg, subsample_idx, compromised, target
            )
            # The epsilon grid is Part 13's, but a per-feature L-infinity budget
            # buys far more total perturbation at d = 216 than at d = 16. The
            # realised L2 norm is recorded so the two Parts' epsilons can be
            # compared honestly rather than by name.
            row["delta_l2_mean"] = (
                0.0 if delta is None
                else float(delta.reshape(delta.shape[0], -1).norm(dim=1).mean())
            )
            scores = row.pop("_scores")
            if epsilon == 0.0:
                clean_scores = scores
            else:
                roc[f"eps{epsilon}"].append({
                    detector: {
                        "auc": tail_roc_auc(scores[detector], clean_scores[detector]),
                        "tpr_at_fpr05": detection_at_fpr(scores[detector], clean_scores[detector], 0.05),
                        "tpr_at_fpr10": detection_at_fpr(scores[detector], clean_scores[detector], 0.10),
                    }
                    for detector in ("tail", "tail_coexceed", "rho_bar")
                })
            row["seed"] = seed
            per_seed[f"eps{epsilon}"].append(row)
            LOGGER.info(
                "tail seed=%s eps=%s succ=%.3f | rho_comp=%.4f lam_comp=%.4f | "
                "rho_all=%.4f lam_all=%.4f | spread=%.3f (%.0fs)",
                seed, epsilon, row["attack_success"], row["rho_bar_compromised"],
                row["lambda_u_compromised"], row["rho_bar"], row["lambda_u"],
                row["tail_spread_compromised"], time.perf_counter() - started,
            )

    headline = str(tail_cfg["headline_quantile"])
    protocol = {}
    for cell, rows in per_seed.items():
        protocol[cell] = {
            "rho_bar": summarise_tail(rows, ("rho_bar",)),
            "rho_bar_compromised": summarise_tail(rows, ("rho_bar_compromised",)),
            "rho_bar_honest": summarise_tail(rows, ("rho_bar_honest",)),
            "lambda_u_compromised": summarise_tail(rows, ("lambda_u_compromised",)),
            "lambda_u_honest": summarise_tail(rows, ("lambda_u_honest",)),
            "attack_success": summarise_tail(rows, ("attack_success",)),
            "tail_spread_compromised": summarise_tail(rows, ("tail_spread_compromised",)),
            "rho_bar_features_part09": summarise_tail(rows, ("rho_bar_features_part09",)),
            "rho_bar_features_compromised": summarise_tail(rows, ("rho_bar_features_compromised",)),
            "eniv": summarise_tail(rows, ("eniv",)),
            "delta_l2_mean": summarise_tail(rows, ("delta_l2_mean",)),
            "lambda_u_empirical": summarise_tail(rows, ("tail", headline, "empirical", "lambda_u")),
            "lambda_u_nonparametric": summarise_tail(rows, ("tail", headline, "nonparametric", "lambda_u")),
            "tail_eniv": summarise_tail(rows, ("tail", headline, "empirical", "tail_eniv")),
            "n_tail_min": summarise_tail(rows, ("tail", headline, "empirical", "n_tail_min")),
            "subsample_lambda_u": summarise_tail(rows, ("subsample", headline, "empirical", "lambda_u")),
            "subsample_n_tail": summarise_tail(rows, ("subsample", headline, "empirical", "n_tail_min")),
            "t_loglik_gain": summarise_tail(rows, ("copula", "loglik_gain")),
            "t_df": summarise_tail(rows, ("copula", "t_df")),
            "t_lambda_u": summarise_tail(rows, ("copula", "t_lambda_u")),
        }
        for quantile in tail_cfg["quantiles"]:
            protocol[cell][f"lambda_u_at_q{quantile}"] = summarise_tail(
                rows, ("tail", str(quantile), "empirical", "lambda_u")
            )
            protocol[cell][f"n_tail_at_q{quantile}"] = summarise_tail(
                rows, ("tail", str(quantile), "empirical", "n_tail_min")
            )

    detection = {}
    for cell, entries in roc.items():
        if not entries:
            continue
        detection[cell] = {
            detector: {
                metric: summarise_tail([{"v": e[detector][metric]} for e in entries], ("v",))
                for metric in ("auc", "tpr_at_fpr05", "tpr_at_fpr10")
            }
            for detector in ("tail", "rho_bar")
        }

    payload = {"protocol": protocol, "detection": detection, "per_seed": per_seed}
    (out_dir / "tail.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt(entry, places: int = 4) -> str:
    if entry is None:
        return "n/a"
    if not isinstance(entry, dict):
        return f"{entry:.{places}f}" if isinstance(entry, float) else str(entry)
    mean, std = entry.get("mean"), entry.get("std")
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} +/- {std:.{places}f}"


def redundancy_table(results: dict, quantile) -> list[str]:
    """THE table. One row per dataset."""
    lines = [
        "| dataset | nominal views | measured ENIV | efficiency ratio | lambda_U |",
        "|---|---|---|---|---|",
    ]
    for name, payload in results.items():
        summary = payload["redundancy"]["summary"]
        lines.append(
            f"| {name} | {summary['nominal_views']} | {fmt(summary['eniv'], 2)} | "
            f"{fmt(summary['efficiency_ratio'], 3)} | {fmt(summary['lambda_u'], 3)} |"
        )
    lines += [
        "",
        f"Mean +/- sample std over seeds. ENIV is the eigenvalue form on encoder-feature "
        f"dependence; lambda_U is the mean off-diagonal upper tail dependence at "
        f"q = {quantile} on the per-view evidence for the fused predicted class.",
    ]
    return lines


PART10_DETECTOR = Path("results/comparison/detector.json")


def synthetic_side_by_side(detector: dict, protocol: dict, n_views: int) -> list[str]:
    """The same conditions, synthetic against real, when Part 10's run is on disk.

    Read from `results/comparison/detector.json` rather than transcribed, so
    the two columns cannot drift apart. If that file is absent the section is
    simply omitted -- an absent comparison is better than a remembered one.

    This is NOT a controlled experiment. The two runs differ in dataset, view
    count (4 vs 6), class count (3 vs 10) and clean accuracy (~0.83 vs ~0.99)
    as well as in baseline redundancy. The table shows that the detector is
    weaker on real data; it does not isolate why.
    """
    if not PART10_DETECTOR.exists():
        return []
    try:
        synthetic = json.loads(PART10_DETECTOR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    synthetic_views = synthetic.get("config", {}).get("data", {}).get("n_views")
    lines = [
        "",
        "### The same detector, against Part 10 on synthetic data",
        "",
        "Part 10's own numbers, read from `results/comparison/detector.json`.",
        "",
        f"| condition | AUC synthetic ({synthetic_views} views) | "
        f"AUC real ({n_views} views) | change |",
        "|---|---|---|---|",
    ]
    # The `missing` conditions are not summarised with the rest. Dropping views
    # LOWERS agreement, so both runs score them below chance and a move toward
    # 0.5 there is not the detector doing better -- averaging them in with the
    # attacks would cancel a real decline against an unrelated one.
    moved = []
    for condition, cell in detector.items():
        other = synthetic.get("summary", {}).get(condition)
        if other is None:
            continue
        delta = cell["auc"]["mean"] - other["auc"]["mean"]
        if not (condition == "clean" or condition.startswith("missing")):
            moved.append(delta)
        lines.append(
            f"| {condition} | {fmt(other['auc'], 3)} | {fmt(cell['auc'], 3)} | {delta:+.3f} |"
        )

    synthetic_clean = synthetic.get("protocol", {}).get("clean", {}).get("eniv")
    real_clean = protocol.get("clean", {}).get("eniv")
    lines.append("")
    if moved:
        lines.append(
            f"Mean change across the {len(moved)} conditions that ADD agreement "
            f"(duplication, noise, chorus, PGD): {sum(moved) / len(moved):+.3f}."
        )
        lines.append(
            "The two `missing` conditions are excluded from that mean and read below"
        )
        lines.append(
            "chance in both runs: dropping views removes agreement rather than adding"
        )
        lines.append("it, so the detector is not meant to fire on them.")
        lines.append("")
    lines += [
        "The detector flags a condition by how far its dependence sits above the",
        "CLEAN baseline's. Those baselines are not alike: "
        f"ENIV {fmt(synthetic_clean, 2)} of {synthetic_views} on synthetic data "
        f"against {fmt(real_clean, 2)} of {n_views} here -- "
        f"{synthetic_clean['mean'] / synthetic_views:.2f} of the view count against "
        f"{real_clean['mean'] / n_views:.2f}.",
        "A clean condition that already looks partly collusive leaves an attack",
        "less room to raise the score above a threshold calibrated on it.",
        "",
        "(That real figure is the CLEAN cell of this matrix, measured on the",
        "comparison split. The redundancy audit reports the same quantity on the",
        "wider split, and the two are not interchangeable: ENIV's finite-sample",
        "bias is downward, so a smaller test split reads differently.)",
        "",
        "That is a plausible mechanism, and it points the same way as "
        f"{sum(1 for d in moved if d < 0)} of the {len(moved)} agreement-adding cells,",
        "but it is NOT established here. The exception is chorus_k3, which moved up;",
        "both runs sit near chance on that cell, so it separates the two runs least",
        "well rather than contradicting them.",
        "",
        "The two runs also differ in dataset, view count, class count and clean",
        "accuracy at the same time. In particular this benchmark's near-ceiling clean",
        "accuracy compresses pairwise belief agreement on its own, and this table does",
        "not separate that from redundancy. Treat the comparison as a described",
        "difference, not an explained one.",
    ]
    return lines


def write_report(results: dict, cfg: dict, seeds, path: Path, commit) -> None:
    quantile = cfg["tail"]["headline_quantile"]
    lines = [
        "# Part 15: real multi-view data",
        "",
        f"Seeds: {list(seeds)}. Mean +/- sample std across seeds.",
        "",
        "## How much redundancy does a standard multi-view benchmark contain?",
        "",
        "Nominal view count against the effective number of INDEPENDENT views,",
        "measured before any duplication, corruption or attack is applied.",
        "",
    ]
    lines += redundancy_table(results, quantile)

    lines += [
        "",
        "### The control: what the estimator reads when the views ARE independent",
        "",
        "ENIV is biased downward at finite sample size, so a reading below the view",
        "count is not by itself evidence of redundancy. Each view's encoder features",
        "are permuted within class, which removes cross-view coupling and leaves the",
        "sample size, the feature dimension, the marginals and the class structure",
        "untouched. The deficit against that null is the claim.",
        "",
        "| dataset | n (test) | rho_bar obs | rho_bar null | ENIV obs | ENIV null | deficit |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, payload in results.items():
        summary = payload["redundancy"]["summary"]
        lines.append(
            f"| {name} | {fmt(summary['n_test'], 0)} | {fmt(summary['rho_bar'], 3)} | "
            f"{fmt(summary['rho_bar_independence_null'], 3)} | {fmt(summary['eniv'], 2)} | "
            f"{fmt(summary['eniv_independence_null'], 2)} | "
            f"{fmt(summary['eniv_deficit_vs_null'], 2)} |"
        )
    lines += [
        "",
        "`rho_bar null` near zero is what says the permutation did its job: the",
        "estimator's chance correction already removes the mean bias, so what is",
        "left in `ENIV null` below the view count is the eigenvalue form's floor,",
        "and only that part of the shortfall is an artefact.",
    ]

    for name, payload in results.items():
        summary = payload["redundancy"]["summary"]
        spec = get_spec(name)
        lines += [
            "",
            f"### {name}: where the redundancy sits",
            "",
            f"Test accuracy {fmt(summary['accuracy'], 4)}, rho_bar {fmt(summary['rho_bar'], 4)}, "
            f"design-effect ENIV {fmt(summary.get('eniv_design_effect'), 2)}, "
            f"tail ENIV {fmt(summary['tail_eniv'], 2)}.",
            "",
            "Mean cross-view dependence matrix (canonical correlation on encoder",
            "features, class-conditional, permutation-null corrected):",
            "",
            "| | " + " | ".join(v.name for v in spec.views) + " |",
            "|---" * (spec.n_views + 1) + "|",
        ]
        matrix = summary["dependence_matrix_mean"]
        for i, view in enumerate(spec.views):
            lines.append(
                f"| {view.name} | " + " | ".join(f"{matrix[i][j]:.3f}" for j in range(len(matrix))) + " |"
            )
        if "eniv_leave_one_out" in summary:
            lines += [
                "",
                "ENIV with each view removed. A view whose removal barely lowers ENIV",
                "was contributing little independent evidence.",
                "",
                "| view removed | ENIV of the remaining 5 | independent views it was carrying |",
                "|---|---|---|",
            ]
            full = summary["eniv"]["mean"]
            for view_name, entry in summary["eniv_leave_one_out"].items():
                lines.append(f"| {view_name} | {fmt(entry, 2)} | {full - entry['mean']:.2f} |")

    # --- the two re-run Parts ------------------------------------------------
    for name, payload in results.items():
        if "comparison" in payload:
            lines += [
                "",
                f"## {name}: the Part 10 detector matrix, re-run on real data",
                "",
                "Thresholds are calibrated per seed on that seed's own CLEAN scores.",
                "The detector is never told which condition it is looking at.",
                "",
                "| condition | AUC | detect@5%FPR | detect@1%FPR | mean score | accuracy | ECE |",
                "|---|---|---|---|---|---|---|",
            ]
            detector = payload["comparison"]["summary"]
            protocol = payload["comparison"]["protocol"]
            for condition in detector:
                cell, proto = detector[condition], protocol[condition]
                lines.append(
                    f"| {condition} | {fmt(cell['auc'], 3)} | {fmt(cell['detect_at_5pct'], 3)} | "
                    f"{fmt(cell['detect_at_1pct'], 3)} | {fmt(cell['mean_score'], 4)} | "
                    f"{fmt(proto['accuracy'], 3)} | {fmt(proto['prob_ece'], 3)} |"
                )
            lines += synthetic_side_by_side(detector, protocol, get_spec(name).n_views)

        if "tail" in payload:
            protocol = payload["tail"]["protocol"]
            detection = payload["tail"]["detection"]
            epsilons = cfg["tail"]["attack"]["epsilons"]
            compromised = ", ".join(
                get_spec(name).views[v].name for v in range(cfg["tail"]["attack"]["k"])
            )
            lines += [
                "",
                f"## {name}: the Part 13 tail analysis, re-run on real data",
                "",
                f"Chorus attack, k = {cfg['tail']['attack']['k']} compromised views "
                f"({compromised}) of {get_spec(name).n_views}. Headline quantile "
                f"q = {cfg['tail']['headline_quantile']}.",
                "",
                "### The colluding pair",
                "",
                "| epsilon | mean L2 of delta | success | rho_bar (comp) | lambda_U (comp) | tail spread |",
                "|---|---|---|---|---|---|",
            ]
            for epsilon in epsilons:
                cell = protocol[f"eps{epsilon}"]
                lines.append(
                    f"| {epsilon} | {fmt(cell['delta_l2_mean'], 2)} | {fmt(cell['attack_success'], 3)} | "
                    f"{fmt(cell['rho_bar_compromised'])} | {fmt(cell['lambda_u_compromised'])} | "
                    f"{fmt(cell['tail_spread_compromised'], 3)} |"
                )
            lines += [
                "",
                "### Honest pairs (the control)",
                "",
                "| epsilon | rho_bar (honest) | lambda_U (honest) |",
                "|---|---|---|",
            ]
            for epsilon in epsilons:
                cell = protocol[f"eps{epsilon}"]
                lines.append(
                    f"| {epsilon} | {fmt(cell['rho_bar_honest'])} | {fmt(cell['lambda_u_honest'])} |"
                )
            lines += [
                "",
                "### ENIV, tail ENIV, and the exceedance count behind lambda_U",
                "",
                f"MIN_TAIL_SAMPLES = {MIN_TAIL_SAMPLES}.",
                "",
                "| epsilon | ENIV | tail ENIV | lambda_U (all) | n_tail | reliable |",
                "|---|---|---|---|---|---|",
            ]
            for epsilon in epsilons:
                cell = protocol[f"eps{epsilon}"]
                n_tail = cell["n_tail_min"]["mean"]
                lines.append(
                    f"| {epsilon} | {fmt(cell['eniv'], 2)} | {fmt(cell['tail_eniv'], 2)} | "
                    f"{fmt(cell['lambda_u_empirical'])} | {n_tail:.0f} | "
                    f"{'yes' if n_tail >= MIN_TAIL_SAMPLES else 'NO'} |"
                )
            lines += [
                "",
                "### The quantile the sample size will not support",
                "",
                "Part 13's headline quantile was 0.95. On 2000 real patterns there is no",
                "`n_samples` knob to turn, so 0.95 falls below MIN_TAIL_SAMPLES here and",
                "0.90 is the headline instead. Both are shown rather than only the one",
                "that clears the bar.",
                "",
                "| epsilon | "
                + " | ".join(
                    f"lambda_U q={q} (n_tail, reliable)" for q in cfg["tail"]["quantiles"]
                )
                + " |",
                "|---" * (len(cfg["tail"]["quantiles"]) + 1) + "|",
            ]
            for epsilon in epsilons:
                cell = protocol[f"eps{epsilon}"]
                columns = []
                for quantile in cfg["tail"]["quantiles"]:
                    count = cell[f"n_tail_at_q{quantile}"]["mean"]
                    columns.append(
                        f"{fmt(cell[f'lambda_u_at_q{quantile}'])} ({count:.0f}, "
                        f"{'yes' if count >= MIN_TAIL_SAMPLES else 'NO'})"
                    )
                lines.append(f"| {epsilon} | " + " | ".join(columns) + " |")

            lines += [
                "",
                "### Detection: tail flag vs the Part 10 rho_bar flag, matched FPR",
                "",
                "| epsilon | tail AUC | rho_bar AUC | tail TPR@5%FPR | rho_bar TPR@5%FPR |",
                "|---|---|---|---|---|",
            ]
            for epsilon in epsilons:
                if f"eps{epsilon}" not in detection:
                    continue
                cell = detection[f"eps{epsilon}"]
                lines.append(
                    f"| {epsilon} | {fmt(cell['tail']['auc'], 3)} | {fmt(cell['rho_bar']['auc'], 3)} | "
                    f"{fmt(cell['tail']['tpr_at_fpr05'], 3)} | {fmt(cell['rho_bar']['tpr_at_fpr05'], 3)} |"
                )

    lines += [
        "",
        "## What this does NOT show",
        "",
        "Real views carry no ground-truth dependence, so ENIV is APPLIED here and",
        "not VALIDATED here. The only validation of the estimator is Part 05, on the",
        "synthetic generator, where rho is a parameter that was set. Nothing above",
        "is evidence that ENIV measures what it claims to measure.",
        "See `docs/DATASETS.md` for the full limitation.",
        "",
        f"Commit: {commit}. torch {torch.__version__}, numpy {np.__version__}.",
        "Controlled research simulation: own models, public benchmark data.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Part 15: real multi-view data.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--stage", choices=STAGES, action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--quick", action="store_true", help="smoke test only -- NOT evidence")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild README.md from the stage JSON already on disk; runs nothing",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = cfg["experiment"]["id"]
    stages = args.stage or list(STAGES)
    names = args.dataset or cfg["data"]["datasets"]

    if args.quick:
        cfg["training"]["epochs"] = 2
        cfg["comparison"]["attack"]["steps"] = 3
        cfg["comparison"]["detector"]["permutations"] = 4
        cfg["comparison"]["conditions"] = (
            cfg["comparison"]["conditions"][:2] + cfg["comparison"]["conditions"][-2:]
        )
        cfg["tail"]["attack"]["steps"] = 3
        cfg["tail"]["attack"]["epsilons"] = [0.0, 1.0]
        cfg["redundancy"]["independence_null_repeats"] = 2
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
        LOGGER.warning("--quick: 2 epochs, 3 attack steps, 4 conditions. NOT EVIDENCE.")

    seeds = cfg["seeds"]
    if len(seeds) < 5:
        raise ValueError("contract requires at least 5 seeds")

    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; known: {sorted(REGISTRY)}")

    results = {}
    for name in names:
        spec = get_spec(name)
        out_dir = Path(args.results_dir) / experiment_id / name
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {}

        if args.report_only:
            # Rebuild the write-up from what a previous run already measured.
            # Nothing is recomputed, so a change to the reporting code cannot
            # silently change a number -- only how it is laid out.
            for stage, filename in (
                ("redundancy", "redundancy.json"),
                ("comparison", "comparison.json"),
                ("tail", "tail.json"),
            ):
                path = out_dir / filename
                if stage in stages and path.exists():
                    payload[stage] = json.loads(path.read_text(encoding="utf-8"))
            if not payload:
                raise FileNotFoundError(f"--report-only: no stage JSON under {out_dir}")
            results[name] = payload
            continue

        # The redundancy audit and the tail analysis share a split and a model:
        # both need the wide test split (see config.yaml, `tail.split`), and
        # training one model per seed for each would be the same model twice.
        wide_models, wide_datasets = {}, {}
        if {"redundancy", "tail"} & set(stages):
            fracs = cfg["tail"]["split"]
            for seed in seeds:
                started = time.perf_counter()
                dataset, splits = build_real_dataset(
                    name, seed=seed, root=cfg["data"]["root"],
                    train_frac=fracs["train"], val_frac=fracs["val"], test_frac=fracs["test"],
                )
                wide_datasets[seed] = (dataset, splits)
                wide_models[seed] = train_real(dataset, splits, spec, seed, True, cfg)
                LOGGER.info(
                    "%s seed %d trained on the wide split (%.0fs)",
                    name, seed, time.perf_counter() - started,
                )

        if "redundancy" in stages:
            payload["redundancy"] = stage_redundancy(
                spec, wide_models, wide_datasets, cfg, seeds, out_dir
            )
        if "comparison" in stages:
            payload["comparison"] = stage_comparison(
                spec, cfg, seeds, out_dir, args.results_dir, f"{experiment_id}/{name}"
            )
        if "tail" in stages:
            payload["tail"] = stage_tail(spec, wide_models, wide_datasets, cfg, seeds, out_dir)

        results[name] = payload

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        commit = None

    root = Path(args.results_dir) / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "config_used.json").write_text(
        json.dumps(json_safe({
            "config": cfg, "stages": stages, "datasets": names, "commit": commit,
            "torch": torch.__version__, "numpy": np.__version__,
            "note": "controlled research simulation; own models, public benchmark data",
        }), indent=2),
        encoding="utf-8",
    )
    if "redundancy" in stages:
        write_report(results, cfg, seeds, root / "README.md", commit)
        LOGGER.info("wrote %s", root / "README.md")
    LOGGER.info("wrote %s", root)


if __name__ == "__main__":
    main()
