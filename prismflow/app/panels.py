"""Streamlit rendering panels for the PrismFlow demo (Part 16).

DELIBERATELY DUMB, AND WHY THAT MATTERS

Every function here takes values that have ALREADY been computed by the
engine and turns them into pixels. Nothing in this file trains, fuses,
estimates dependence, attacks, or decides anything. It does not import
`torch.nn`, and `tests/unit/test_app.py` asserts that for both this module
and `app.py`.

That is not tidiness for its own sake. A demo that recomputes a number
its own way is the single most effective way to publish a figure that no
experiment produced -- the plot and the paper drift apart and nobody can
tell which is wrong. So the rule for this layer is: if a number is on the
screen, some function under `prismflow/` returned it.

The one piece of arithmetic here is `format_ratio`, which divides two
numbers the caller already has so a label can read "2.3 of 5". It is
formatting, not measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# matplotlib is used for the dependence heatmap only. Agg backend: Streamlit
# renders figures server-side and there is no display attached.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import streamlit as st  # noqa: E402

VIEW_STATE_ICON = {
    "available": ":material/check_circle:",
    "missing": ":material/cancel:",
    "noisy": ":material/warning:",
    "compromised": ":material/gpp_bad:",
    "duplicate": ":material/content_copy:",
}


def format_ratio(effective: float, nominal: float) -> str:
    """"2.3 of 5 (46%)" -- formatting of two numbers the caller already holds."""
    if nominal <= 0:
        return "n/a"
    return f"{effective:.2f} of {nominal:g} ({100.0 * effective / nominal:.0f}%)"


# ---------------------------------------------------------------------------
# Screen 1-2: scenario and per-view status
# ---------------------------------------------------------------------------


def render_view_status(view_names, states, alphas=None, present_fraction=None) -> None:
    """Per-view status table: available / missing / noisy / compromised / duplicate.

    `alphas` is the per-view discount the engine actually applied
    (`ENIVResult.per_view_alpha`), not a recomputation of it. A value below 1
    means that view's evidence was scaled down before fusion.
    """
    st.subheader("Per-view status")
    rows = []
    for index, name in enumerate(view_names):
        state = states[index]
        row = {
            "view": name,
            "state": state,
            "": VIEW_STATE_ICON.get(state, ""),
        }
        if alphas is not None and index < len(alphas):
            row["discount applied"] = f"{alphas[index]:.3f}"
        if present_fraction is not None:
            row["present in batch"] = f"{100.0 * present_fraction[index]:.0f}%"
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")

    discounted = (
        0 if alphas is None else sum(1 for a in alphas if a < 0.999)
    )
    if discounted:
        st.caption(
            f"{discounted} of {len(view_names)} views had their evidence discounted "
            "before fusion, because the dependence estimator found them partly "
            "redundant with other views."
        )


# ---------------------------------------------------------------------------
# THE INDEPENDENCE BUDGET -- its own screen
# ---------------------------------------------------------------------------


def render_independence_budget(
    nominal: int,
    effective: float,
    redundant_pair=None,
    lambda_u=None,
    rho_bar=None,
    dataset_label: str = "",
) -> None:
    """The project's most legible single output.

        Nominal detectors:      5
        Effective independent:  2.3
        Redundant pair:         View 1 <-> View 4   (lambda_U = 0.91)
        Interpretation:         one line in plain language

    `effective` is ENIV as the engine computed it; `redundant_pair` and
    `lambda_u` are the argmax pair and value the engine found. Nothing is
    recomputed here.
    """
    st.header("Independence budget")
    if dataset_label:
        st.caption(dataset_label)

    left, middle, right = st.columns(3)
    left.metric("Nominal detectors", f"{nominal:g}")
    middle.metric("Effective independent", f"{effective:.2f}")
    right.metric(
        "Independence efficiency",
        f"{100.0 * effective / nominal:.0f}%" if nominal else "n/a",
        help="ENIV divided by the nominal view count. 100% means every view is "
        "an independent witness.",
    )

    if redundant_pair is not None:
        first, second = redundant_pair
        tail = "" if lambda_u is None or not np.isfinite(lambda_u) else f"   (lambda_U = {lambda_u:.2f})"
        st.markdown(f"**Most redundant pair:** {first} &harr; {second}{tail}")
    if rho_bar is not None and np.isfinite(rho_bar):
        st.markdown(f"**Mean cross-view dependence:** rho_bar = {rho_bar:.3f}")

    lost = nominal - effective
    st.info(
        f"**In plain language.** You are paying for {nominal:g} independent sources "
        f"of evidence. You own about {effective:.1f}. Roughly {lost:.1f} of them "
        "are repeating what another source already told you, so agreement between "
        "them is worth less confidence than the count suggests."
    )
    st.caption(
        "ENIV is validated against known ground truth only on synthetic data "
        "(Part 05). On any real dataset it is applied, not validated -- see "
        "docs/DATASETS.md and docs/KNOWN_LIMITATIONS.md."
    )


# ---------------------------------------------------------------------------
# Screen 4: results
# ---------------------------------------------------------------------------


def render_prediction(prediction, confidence, uncertainty, probability=None) -> None:
    st.subheader("Prediction")
    columns = st.columns(4 if probability is not None else 3)
    columns[0].metric("Predicted class", f"{prediction}")
    columns[1].metric(
        "Confidence (1 - vacuity)",
        f"{confidence:.3f}",
        help="PrismFlowOutput.confidence. NOT a probability of being correct.",
    )
    columns[2].metric("Uncertainty (vacuity)", f"{uncertainty:.3f}")
    if probability is not None:
        columns[3].metric(
            "Top-label probability",
            f"{probability:.3f}",
            help="max_k of the expected Dirichlet probability. This is the "
            "quantity ECE is defined for.",
        )


def render_eniv(nominal: int, effective: float, efficiency: float) -> None:
    st.subheader("Nominal views against effective independent views")
    columns = st.columns(3)
    columns[0].metric("Nominal views", f"{nominal:g}")
    columns[1].metric("ENIV", f"{effective:.2f}")
    columns[2].metric("Efficiency ratio", f"{efficiency:.3f}")
    st.progress(
        min(max(effective / nominal, 0.0), 1.0) if nominal else 0.0,
        text=format_ratio(effective, nominal),
    )


def render_dependence_heatmap(matrix, view_names, title="Cross-view dependence") -> None:
    """Heatmap of the [V, V] matrix the engine measured. NaN = not measurable."""
    matrix = np.asarray(matrix, dtype=float)
    st.subheader(title)

    figure, axes = plt.subplots(figsize=(1.1 * len(view_names) + 1.6, 1.0 * len(view_names) + 1.2))
    masked = np.ma.masked_invalid(matrix)
    colormap = plt.get_cmap("magma").copy()
    colormap.set_bad(color="#cfcfcf")
    image = axes.imshow(masked, vmin=0.0, vmax=1.0, cmap=colormap)

    axes.set_xticks(range(len(view_names)), view_names, rotation=45, ha="right")
    axes.set_yticks(range(len(view_names)), view_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "n/a" if not np.isfinite(value) else f"{value:.2f}"
            axes.text(
                j, i, text, ha="center", va="center", fontsize=8,
                color="white" if np.isfinite(value) and value > 0.55 else "black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.caption(
        "Grey means the pair was not measurable and the estimator refused to "
        "guess. Refusing to estimate is not the same as estimating zero."
    )


def render_tail_dependence(lambda_u, n_tail, min_tail_samples, quantile) -> None:
    st.subheader("Tail dependence")
    columns = st.columns(2)
    columns[0].metric(
        f"lambda_U at q = {quantile}",
        "n/a" if lambda_u is None or not np.isfinite(lambda_u) else f"{lambda_u:.3f}",
        help="Probability that one view is extreme given another is -- joint "
        "extreme agreement, which mean correlation does not see.",
    )
    columns[1].metric("Exceedances behind it", f"{n_tail:g}")
    if n_tail < min_tail_samples:
        st.warning(
            f"UNRELIABLE: {n_tail:g} exceedances is below MIN_TAIL_SAMPLES = "
            f"{min_tail_samples}. Shown because hiding it would be worse, but do "
            "not read this number as evidence."
        )


def render_evidence_bars(evidence, view_names, class_labels=None) -> None:
    """Per-view evidence for each class, as the engine emitted it."""
    evidence = np.asarray(evidence, dtype=float)
    st.subheader("Per-view evidence")
    labels = class_labels or [f"class {k}" for k in range(evidence.shape[1])]
    st.bar_chart(
        {label: evidence[:, k].tolist() for k, label in enumerate(labels)},
        x_label="view",
        y_label="evidence",
    )
    st.dataframe(
        [
            {"view": name, **{labels[k]: round(float(evidence[v, k]), 3) for k in range(evidence.shape[1])}}
            for v, name in enumerate(view_names)
        ],
        hide_index=True,
        width="stretch",
    )


def render_suspicion(score, flag_rate, threshold, calibrated_at=None) -> None:
    """The detector's own output. It is never told which scenario is running.

    `calibrated_at` is the clean false-positive rate the threshold was
    calibrated to, when the caller calibrated one. Saying so matters: an
    uncalibrated default threshold flags most CLEAN samples at demo training
    lengths, which reads as a broken detector rather than an unset dial.
    """
    st.subheader("Suspicion detector")
    columns = st.columns(2)
    columns[0].metric(
        "Mean unexplained agreement",
        f"{score:+.4f}",
        help="Agreement beyond what the measured dependence structure accounts "
        "for. 0 means exactly as much as expected.",
    )
    columns[1].metric(
        "Flag rate",
        f"{flag_rate:.1%}",
        help=f"fraction of samples with score > {threshold:.4f}",
    )
    if calibrated_at is not None:
        st.caption(
            f"Threshold {threshold:.4f} was calibrated on this configuration's "
            f"CLEAN scores to a {calibrated_at:.0%} false-positive rate, the same "
            "protocol Part 10 uses. A flag rate near that number means the "
            "detector sees nothing unusual."
        )

    if flag_rate >= 0.5:
        st.error(
            f"SUSPICION RAISED on {flag_rate:.0%} of samples: views agree more "
            "than their measured dependence structure explains."
        )
    elif flag_rate > 0.0:
        st.warning(f"Suspicion raised on {flag_rate:.0%} of samples.")
    else:
        st.success("No sample exceeded the suspicion threshold.")
    st.caption(
        "The detector receives beliefs, the view mask, the dependence matrix and "
        "the model's own strata -- never the scenario, the attack, or the label. "
        "Part 09 found this signal detects collusion; it does not defend against "
        "it (docs/CONTRACT.md)."
    )


# ---------------------------------------------------------------------------
# Screen 5: saved plots
# ---------------------------------------------------------------------------


def find_saved_plots(results_dir: Path):
    """Every .png already written under results/, newest first. Read-only."""
    root = Path(results_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def render_saved_plots(plots, results_dir: Path, limit: int = 40) -> None:
    st.header("Saved experiment figures")
    if not plots:
        st.info(
            f"No .png files under {results_dir}. Run an experiment first, for "
            "example `python -m experiments.clone.run_clone`."
        )
        return
    st.caption(
        f"{len(plots)} figure(s) found under {results_dir}. These are files on "
        "disk written by experiment runs; nothing here is recomputed."
    )
    grouped: dict[str, list[Path]] = {}
    for path in plots[:limit]:
        grouped.setdefault(str(Path(path).parent.relative_to(results_dir)), []).append(path)
    for group, paths in grouped.items():
        with st.expander(f"{group}  ({len(paths)})", expanded=False):
            for path in paths:
                st.image(str(path), caption=str(path))


def render_limitations_footer() -> None:
    st.divider()
    st.caption(
        "PrismFlow is a research prototype. Two results qualify its own "
        "hypothesis: the dependence signal DETECTS collusion reliably, and the "
        "discount driven from it does NOT defend against it (Part 09); an "
        "adaptive attacker that knows the discount exists partially evades it "
        "(Part 14). See docs/KNOWN_LIMITATIONS.md before quoting any number."
    )
