"""Part 13: average co-movement vs joint extreme agreement under the Chorus attack.

THE CLAIM BEING TESTED

Pearson correlation measures AVERAGE co-movement. A chorus attack is not an
average event -- it drives a clique of views to agree, strongly, on one wrong
class. If the standard measure is blind to that and upper tail dependence is
not, then rho_bar should stay roughly flat as epsilon rises while lambda_U
climbs. Whether that happens is the experiment; it is reported either way.

BOTH MEASURES READ THE SAME NUMBERS

This is the design decision that makes the comparison mean anything. For each
sample and view a single scalar is taken,

    s[n, v] = evidence that view v assigns to the FUSED PREDICTED class,

and rho_bar and lambda_U are both computed on that same s. Measuring one on
encoder features and the other on evidence would confound the measure with the
representation, and any difference could be blamed on the latter. The predicted
class is used rather than the target class because the suspicion detector is
forbidden ground truth by docs/CONTRACT.md, and a diagnostic that needs to know
the attack cannot run in deployment.

Part 09's own feature-space rho_bar (cca on encoder features) is reported
alongside, unchanged, so this Part remains comparable with that one.

SAMPLE SIZE -- WHY n_samples IS RAISED

Tail estimation is data-hungry because extremes are rare by construction. Part
09's 300-sample test split puts 15 points above q = 0.95, which cannot support a
conditional probability; `tail_dependence.MIN_TAIL_SAMPLES` is 50. The attack
configuration here is Part 09's exactly (k, epsilon grid, beta, steps), but
n_samples is raised so the test split carries 1200 samples and 60 exceedances.

The 300-sample case is NOT dropped. Every condition is additionally estimated on
a 300-sample subsample of the same split, so the cost of the small sample is
measured rather than asserted -- see `subsample` in the output.

Usage:
    python -m experiments.tail.run_tail
    python -m experiments.tail.run_tail --quick   # NOT evidence
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.data.dataset import MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import SyntheticConfig
from prismflow.eniv.eniv import compute_eniv
from prismflow.models.prismflow import FORWARD_NULL_PERMUTATIONS
from prismflow.statistics.copula import fit_gaussian_copula, fit_t_copula, pseudo_observations
from prismflow.statistics.dependence import (
    available_views,
    dependence_matrix,
    feature_dependence_matrix,
    predicted_classes,
)
from prismflow.statistics.suspicion import suspicion_report
from prismflow.statistics.tail_dependence import (
    MIN_TAIL_SAMPLES,
    co_exceedance_score,
    tail_agreement_score,
    tail_dependence_matrix,
    tail_eniv,
)
from prismflow.train import TrainConfig, build_model, compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

LOGGER = get_logger(__name__)
OUT_DIR = Path("results/tail")

CONFIG = {
    "experiment": {"id": "tail"},
    "seeds": [0, 1, 2, 3, 4],
    # Part 09's attack configuration, unchanged.
    "attack": {"k": 2, "beta": 1.0, "steps": 30, "epsilons": [0.0, 0.2, 0.5, 1.0, 2.0]},
    # n_samples raised so the tail estimate clears MIN_TAIL_SAMPLES; see docstring.
    "data": {"n_views": 4, "rho": 0.3, "n_samples": 8000},
    "training": {"epochs": 40, "batch_size": 64, "lr": 1e-3, "anneal_epochs": 10,
                 "per_view_loss_weight": 1.0},
    "tail": {"quantiles": [0.90, 0.95], "headline_quantile": 0.95,
             "subsample_n": 300, "subsample_seed": 7},
    "evaluation": {"batch_size": 512},
}


def build_dataset(seed: int, data_cfg: dict):
    base = TrainConfig(
        n_views=data_cfg["n_views"], rho=data_cfg["rho"], n_samples=data_cfg["n_samples"]
    )
    dataset = MultiViewDataset(
        SyntheticConfig(
            n_views=base.n_views, n_classes=base.n_classes, n_samples=base.n_samples,
            d_latent=base.d_latent, d_view=base.d_view, rho=base.rho,
            noise_std=base.noise_std, signal_strength=base.signal_strength, seed=seed,
        )
    )
    return dataset, split_indices(len(dataset), seed=base.split_seed), base


def train_model(dataset, splits, base: TrainConfig, training: dict, seed: int):
    """PrismFlow with the discount on -- the system Part 09 attacked."""
    model = build_model(
        TrainConfig(n_views=base.n_views, rho=base.rho, n_samples=base.n_samples, use_discount=True)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=training["lr"])

    for epoch in range(training["epochs"]):
        model.train()
        for batch in iter_batches(
            dataset, splits["train"], training["batch_size"], shuffle=True, seed=seed + epoch
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=base.n_classes
            ).to(batch.views.dtype)
            loss = compute_loss(
                model(batch.views, batch.view_mask), targets, epoch=epoch,
                anneal_epochs=training["anneal_epochs"],
                per_view_loss_weight=training["per_view_loss_weight"],
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return model


def whole_split(dataset, indices, batch_size: int):
    """One Batch covering the split; the estimators need a single sample set."""
    views, masks, labels = [], [], []
    for batch in iter_batches(dataset, indices, batch_size):
        views.append(batch.views)
        masks.append(batch.view_mask)
        labels.append(batch.labels)
    return torch.cat(views), torch.cat(masks), torch.cat(labels)


@torch.no_grad()
def view_statistic(model, views, view_mask):
    """s[n, v] = evidence view v puts on the fused predicted class, plus context."""
    output = model(views, view_mask)
    evidence = output.per_view_evidence
    predicted = output.prediction.detach().cpu().numpy()

    scalar = evidence.detach().cpu().numpy()[np.arange(evidence.shape[0]), :, predicted]
    return {
        "scalar": scalar,
        "evidence": evidence,
        "belief": output.per_view_belief,
        "predicted": predicted,
        "features": model.encoder(views, view_mask),
        "output": output,
    }


def rho_bar_on(scalar: np.ndarray, view_mask, strata: np.ndarray) -> float:
    """Mean off-diagonal Pearson on the SAME scalar the tail measure reads.

    Passed as [B, V, 1] with explicit strata, so the project's class-conditional
    residual convention applies to a single scalar per view.
    """
    matrix = dependence_matrix(
        scalar[:, :, None], view_mask, classes=strata, method="pearson"
    )
    off = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    finite = off[np.isfinite(off)]
    return float(finite.mean()) if finite.size else float("nan")


def pair_sets(n_views: int, compromised: tuple) -> dict[str, list]:
    """Which view pairs are colluding, which are honest, and all of them.

    Part 09 reported dependence among the COMPROMISED views separately, and it
    has to be done here too: the attack coordinates k = 2 of 4 views, so exactly
    1 of the 6 pairs carries the signal. An all-pairs mean divides it by six and
    would report a real effect as a weak one.
    """
    compromised = tuple(compromised)
    honest = [v for v in range(n_views) if v not in compromised]
    return {
        "compromised": [(i, j) for a, i in enumerate(compromised) for j in compromised[a + 1:]],
        "honest": [(i, j) for a, i in enumerate(honest) for j in honest[a + 1:]],
        "all": [(i, j) for i in range(n_views) for j in range(i + 1, n_views)],
    }


def pair_mean(matrix: np.ndarray, pairs: list) -> float:
    values = [matrix[i, j] for i, j in pairs]
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def tail_summary(scalar: np.ndarray, quantile: float, pairs: dict) -> dict:
    """lambda_U by both estimators, split by pair set, with exceedance counts."""
    out = {}
    for method in ("empirical", "nonparametric"):
        matrix, counts = tail_dependence_matrix(scalar, quantile=quantile, method=method)
        pair_counts = counts[~np.eye(counts.shape[0], dtype=bool)]
        entry = {
            "tail_eniv": tail_eniv(matrix),
            "n_tail_min": int(pair_counts.min()) if pair_counts.size else 0,
            "reliable": bool(pair_counts.size and pair_counts.min() >= MIN_TAIL_SAMPLES),
        }
        for name, group in pairs.items():
            entry[f"lambda_u_{name}"] = pair_mean(matrix, group)
        entry["lambda_u"] = entry["lambda_u_all"]
        out[method] = entry
    return out


def copula_summary(scalar: np.ndarray) -> dict:
    """Gaussian (lambda_U = 0 by construction) against t (lambda_U > 0)."""
    u = pseudo_observations(scalar)
    gaussian = fit_gaussian_copula(u)
    student = fit_t_copula(u)
    return {
        "gaussian_loglik": gaussian.log_likelihood,
        "t_loglik": student.log_likelihood,
        "t_df": student.df,
        "t_lambda_u": student.upper_tail_dependence,
        "gaussian_lambda_u": gaussian.upper_tail_dependence,
        "t_preferred": bool(student.log_likelihood > gaussian.log_likelihood),
        "loglik_gain": float(student.log_likelihood - gaussian.log_likelihood),
    }


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC via the rank identity; ties count a half."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    unique, inverse = np.unique(combined, return_inverse=True)
    if unique.size != combined.size:
        sums = np.bincount(inverse, weights=ranks)
        counts = np.bincount(inverse)
        ranks = (sums / counts)[inverse]
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def detection_at_fpr(positive: np.ndarray, negative: np.ndarray, target_fpr: float) -> float:
    """TPR at a threshold calibrated on the CLEAN scores to hit `target_fpr`.

    Matched-FPR comparison, per the brief: a single shared threshold would
    compare the two detectors at different operating points.
    """
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    threshold = float(np.quantile(neg, 1.0 - target_fpr))
    return float((pos > threshold).mean())


def saturation(scalar: np.ndarray, pairs: list) -> float:
    """Spread of the colluding views' scalar INSIDE their own upper tail.

    The leading explanation for a collapsing lambda_U at high epsilon is that
    the attack drives the colluding views to a near-constant extreme, so the
    ranks within the tail become noise and a rank-based measure loses the
    signal it needs. This quantifies that: the mean coefficient of variation of
    the colluding views' values above their own 95th percentile. Falling toward
    zero means the tail has flattened.
    """
    if not pairs:
        return float("nan")
    views = sorted({v for pair in pairs for v in pair})
    spreads = []
    for view in views:
        column = scalar[:, view]
        tail = column[column > np.quantile(column, 0.95)]
        if tail.size > 1 and abs(tail.mean()) > 1e-12:
            spreads.append(float(tail.std() / abs(tail.mean())))
    return float(np.mean(spreads)) if spreads else float("nan")


def measure_condition(model, views, view_mask, cfg, subsample_idx, compromised, target) -> dict:
    """Every diagnostic for one (seed, epsilon) cell."""
    stats = view_statistic(model, views, view_mask)
    scalar, output = stats["scalar"], stats["output"]
    strata = predicted_classes(stats["evidence"], view_mask)
    pairs = pair_sets(model.n_views, compromised)

    # Part 09's own quantity, untouched, for continuity with that Part.
    feature_matrix = feature_dependence_matrix(
        stats["features"], stats["evidence"], view_mask,
        method=model.dependence_method, conditioning=model.dependence_conditioning,
        seed=model.dependence_seed, null_permutations=FORWARD_NULL_PERMUTATIONS,
    )
    present = available_views(view_mask, n_views=model.n_views)

    rho_matrix = dependence_matrix(
        scalar[:, :, None], view_mask, classes=strata, method="pearson"
    )

    row = {
        "rho_bar": pair_mean(rho_matrix, pairs["all"]),
        "rho_bar_compromised": pair_mean(rho_matrix, pairs["compromised"]),
        "rho_bar_honest": pair_mean(rho_matrix, pairs["honest"]),
        "rho_bar_features_part09": pair_mean(feature_matrix, pairs["all"]),
        "rho_bar_features_compromised": pair_mean(feature_matrix, pairs["compromised"]),
        "eniv": float(compute_eniv(feature_matrix, present).effective_views),
        "tail": {str(q): tail_summary(scalar, q, pairs) for q in cfg["tail"]["quantiles"]},
        "copula": copula_summary(scalar),
        "tail_spread_compromised": saturation(scalar, pairs["compromised"]),
        "attack_success": (
            0.0 if target is None
            else float((stats["predicted"] == target.detach().cpu().numpy()).mean())
        ),
    }

    headline = str(cfg["tail"]["headline_quantile"])
    for name in ("all", "compromised", "honest"):
        row[f"lambda_u_{name}"] = row["tail"][headline]["empirical"][f"lambda_u_{name}"]
    row["lambda_u"] = row["lambda_u_all"]
    row["tail_eniv"] = row["tail"][headline]["empirical"]["tail_eniv"]

    # The 300-sample cost, measured rather than asserted.
    row["subsample"] = {
        str(q): tail_summary(scalar[subsample_idx], q, pairs)
        for q in cfg["tail"]["quantiles"]
    }

    # Per-sample detector scores. Neither may know WHICH views collude.
    row["_scores"] = {
        "tail": tail_agreement_score(scalar),
        "tail_coexceed": co_exceedance_score(scalar, quantile=cfg["tail"]["headline_quantile"]),
        "rho_bar": suspicion_report(
            stats["belief"], feature_matrix, view_mask,
            evidence=stats["evidence"], strata=strata,
        ).score,
    }
    row["accuracy_proxy_confidence"] = float(output.confidence.mean())
    return row


def summarise(rows: list[dict], path: tuple) -> dict:
    values = []
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        if isinstance(node, (int, float)) and not (isinstance(node, float) and math.isnan(node)):
            values.append(float(node))
    return {
        "mean": statistics.mean(values) if values else float("nan"),
        "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
        "n_seeds": len(values),
    }


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(float(value))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def fmt(entry) -> str:
    if entry is None or (isinstance(entry["mean"], float) and math.isnan(entry["mean"])):
        return "n/a"
    mean, std = entry["mean"], entry["std"]
    return f"{mean:.4f}" if math.isnan(std) else f"{mean:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, NOT evidence")
    args = parser.parse_args()

    config = json.loads(json.dumps(CONFIG))
    if args.quick:
        config["seeds"] = config["seeds"][:1]
        config["training"] = dict(config["training"], epochs=2)
        config["attack"] = dict(config["attack"], steps=3, epsilons=[0.0, 1.0])
        config["data"] = dict(config["data"], n_samples=2000)
        LOGGER.warning("--quick: 1 seed, 2 epochs, 3 attack steps. NOT EVIDENCE.")

    epsilons = config["attack"]["epsilons"]
    per_seed: dict[str, list[dict]] = {f"eps{e}": [] for e in epsilons}
    roc: dict[str, list[dict]] = {f"eps{e}": [] for e in epsilons}

    for seed in config["seeds"]:
        set_seed(seed)
        dataset, splits, base = build_dataset(seed, config["data"])
        model = train_model(dataset, splits, base, config["training"], seed)

        views, view_mask, _ = whole_split(dataset, splits["test"], config["evaluation"]["batch_size"])
        n_test = views.shape[0]
        rng = np.random.default_rng(config["tail"]["subsample_seed"] + seed)
        subsample_idx = rng.choice(
            n_test, size=min(config["tail"]["subsample_n"], n_test), replace=False
        )

        clean_scores = None
        for epsilon in epsilons:
            if epsilon == 0.0:
                # The clean cell still names the views the attack WOULD take, so
                # the "compromised pair" column is the same pair at every epsilon
                # and the clean row is its baseline rather than a different pair.
                attacked = views
                compromised = tuple(range(config["attack"]["k"]))
                target = None
            else:
                attack = chorus_attack(
                    model, views, view_mask,
                    ChorusConfig(
                        k=config["attack"]["k"], epsilon=epsilon,
                        beta=config["attack"]["beta"], steps=config["attack"]["steps"],
                        seed=seed,
                    ),
                )
                attacked = views + attack.delta
                compromised, target = attack.compromised, attack.target

            row = measure_condition(
                model, attacked, view_mask, config, subsample_idx, compromised, target
            )
            scores = row.pop("_scores")
            if epsilon == 0.0:
                clean_scores = scores
            else:
                roc[f"eps{epsilon}"].append({
                    detector: {
                        "auc": roc_auc(scores[detector], clean_scores[detector]),
                        "tpr_at_fpr05": detection_at_fpr(
                            scores[detector], clean_scores[detector], 0.05
                        ),
                        "tpr_at_fpr10": detection_at_fpr(
                            scores[detector], clean_scores[detector], 0.10
                        ),
                    }
                    for detector in ("tail", "tail_coexceed", "rho_bar")
                })
            row["seed"] = seed
            per_seed[f"eps{epsilon}"].append(row)
            LOGGER.info(
                "seed=%s eps=%s succ=%.3f | rho_comp=%.4f lam_comp=%.4f | "
                "rho_all=%.4f lam_all=%.4f | spread=%.3f",
                seed, epsilon, row["attack_success"],
                row["rho_bar_compromised"], row["lambda_u_compromised"],
                row["rho_bar"], row["lambda_u"], row["tail_spread_compromised"],
            )

    headline = str(config["tail"]["headline_quantile"])
    protocol = {}
    for cell, rows in per_seed.items():
        protocol[cell] = {
            "rho_bar": summarise(rows, ("rho_bar",)),
            "rho_bar_compromised": summarise(rows, ("rho_bar_compromised",)),
            "rho_bar_honest": summarise(rows, ("rho_bar_honest",)),
            "lambda_u_compromised": summarise(rows, ("lambda_u_compromised",)),
            "lambda_u_honest": summarise(rows, ("lambda_u_honest",)),
            "attack_success": summarise(rows, ("attack_success",)),
            "tail_spread_compromised": summarise(rows, ("tail_spread_compromised",)),
            "rho_bar_features_part09": summarise(rows, ("rho_bar_features_part09",)),
            "rho_bar_features_compromised": summarise(rows, ("rho_bar_features_compromised",)),
            "eniv": summarise(rows, ("eniv",)),
            "lambda_u_empirical": summarise(rows, ("tail", headline, "empirical", "lambda_u")),
            "lambda_u_nonparametric": summarise(
                rows, ("tail", headline, "nonparametric", "lambda_u")
            ),
            "tail_eniv": summarise(rows, ("tail", headline, "empirical", "tail_eniv")),
            "n_tail_min": summarise(rows, ("tail", headline, "empirical", "n_tail_min")),
            "subsample_lambda_u": summarise(
                rows, ("subsample", headline, "empirical", "lambda_u")
            ),
            "subsample_n_tail": summarise(
                rows, ("subsample", headline, "empirical", "n_tail_min")
            ),
            "t_loglik_gain": summarise(rows, ("copula", "loglik_gain")),
            "t_df": summarise(rows, ("copula", "t_df")),
            "t_lambda_u": summarise(rows, ("copula", "t_lambda_u")),
        }

    detection = {}
    for cell, entries in roc.items():
        if not entries:
            continue
        detection[cell] = {
            detector: {
                metric: summarise([{"v": e[detector][metric]} for e in entries], ("v",))
                for metric in ("auc", "tpr_at_fpr05", "tpr_at_fpr10")
            }
            for detector in ("tail", "rho_bar")
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tail.json").write_text(
        json.dumps(
            {
                "config": config, "torch": torch.__version__,
                "protocol": json_safe(protocol), "detection": json_safe(detection),
                "per_seed": json_safe(per_seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Tail dependence under the Chorus attack (Part 13)",
        "",
        f"Seeds: {config['seeds']}. Mean +/- sample std across seeds. "
        f"Headline quantile q = {headline}.",
        "",
        f"Test split: {config['data']['n_samples']} samples total. "
        f"MIN_TAIL_SAMPLES = {MIN_TAIL_SAMPLES}.",
        "",
        "## THE HEADLINE: the colluding pair, where the attack actually acts",
        "",
        "The attack coordinates k of V views, so 1 of 6 pairs carries the signal;",
        "an all-pairs mean divides it by six. These are the colluding pair.",
        "",
        "| epsilon | success | rho_bar (comp) | lambda_U (comp) | tail spread (comp) |",
        "|---|---|---|---|---|",
    ]
    for epsilon in epsilons:
        cell = protocol[f"eps{epsilon}"]
        lines.append(
            f"| {epsilon} | {fmt(cell['attack_success'])} | {fmt(cell['rho_bar_compromised'])} | "
            f"{fmt(cell['lambda_u_compromised'])} | {fmt(cell['tail_spread_compromised'])} |"
        )

    lines += [
        "", "## Honest pairs (the control: nothing should move here)", "",
        "| epsilon | rho_bar (honest) | lambda_U (honest) |", "|---|---|---|",
    ]
    for epsilon in epsilons:
        cell = protocol[f"eps{epsilon}"]
        lines.append(
            f"| {epsilon} | {fmt(cell['rho_bar_honest'])} | {fmt(cell['lambda_u_honest'])} |"
        )

    lines += [
        "", "## All-pairs means (diluted; kept for continuity)", "",
        "| epsilon | rho_bar (all) | lambda_U (empirical) | lambda_U (nonpar) | n_tail | reliable |",
        "|---|---|---|---|---|---|",
    ]
    for epsilon in epsilons:
        cell = protocol[f"eps{epsilon}"]
        n_tail = cell["n_tail_min"]["mean"]
        lines.append(
            f"| {epsilon} | {fmt(cell['rho_bar'])} | {fmt(cell['lambda_u_empirical'])} | "
            f"{fmt(cell['lambda_u_nonparametric'])} | {n_tail:.0f} | "
            f"{'yes' if n_tail >= MIN_TAIL_SAMPLES else 'NO'} |"
        )

    lines += ["", "## ENIV against tail-aware ENIV", "",
              "| epsilon | ENIV (Part 09 features) | tail ENIV (lambda_U) |", "|---|---|---|"]
    for epsilon in epsilons:
        cell = protocol[f"eps{epsilon}"]
        lines.append(f"| {epsilon} | {fmt(cell['eniv'])} | {fmt(cell['tail_eniv'])} |")

    lines += ["", "## Copula fit: does the data prefer a tail-dependent model?", "",
              "| epsilon | t - Gaussian loglik | fitted df | fitted lambda_U |", "|---|---|---|---|"]
    for epsilon in epsilons:
        cell = protocol[f"eps{epsilon}"]
        lines.append(
            f"| {epsilon} | {fmt(cell['t_loglik_gain'])} | {fmt(cell['t_df'])} | "
            f"{fmt(cell['t_lambda_u'])} |"
        )

    lines += ["", "## Detection: tail flag vs the Part 10 rho_bar flag, matched FPR", "",
              "Neither detector is told which views collude.", "",
              "| epsilon | tail AUC | rho_bar AUC | tail TPR@5%FPR | rho_bar TPR@5%FPR |",
              "|---|---|---|---|---|"]
    for epsilon in epsilons:
        if f"eps{epsilon}" not in detection:
            continue
        cell = detection[f"eps{epsilon}"]
        lines.append(
            f"| {epsilon} | {fmt(cell['tail']['auc'])} | {fmt(cell['rho_bar']['auc'])} | "
            f"{fmt(cell['tail']['tpr_at_fpr05'])} | {fmt(cell['rho_bar']['tpr_at_fpr05'])} |"
        )

    lines += ["", "## The cost of a 300-sample split (Part 09's size)", "",
              "| epsilon | lambda_U at n=1200 | lambda_U at n=300 | n_tail at n=300 |",
              "|---|---|---|---|"]
    for epsilon in epsilons:
        cell = protocol[f"eps{epsilon}"]
        lines.append(
            f"| {epsilon} | {fmt(cell['lambda_u_empirical'])} | {fmt(cell['subsample_lambda_u'])} | "
            f"{cell['subsample_n_tail']['mean']:.0f} |"
        )

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", OUT_DIR / "summary.md")


if __name__ == "__main__":
    main()
