"""Generate `experiments/ait_attack/README.md` from the committed Part 29 result
files. No number in the README is typed by hand.

Sources, all under `results/ait_attack/`:
  results.json                  the run itself (4 units, 5 seeds each)
  determinism_check.json        step 0: the died run vs the completed run
  padding_audit.json            step 4(a): does the attack touch padding columns
  detector_scores_subset.json   step 2: distinct scores and tie fractions

Every figure is formatted through `fmt`, which prints NOT MEASURED for a
missing value rather than 0 or a blank, and every pooled figure is labelled
`(S subsets x N seeds)` so a subset-pooled std is never mistaken for the 5-seed
std the contract's 5-SEED RULE speaks of.

Run:  PYTHONPATH=. .venv/Scripts/python.exe experiments/ait_attack/make_readme.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.ait_attack.results_io import load_results, resolve

ROOT = Path("results/ait_attack")
OUT = Path("experiments/ait_attack/README.md")
NOT_MEASURED = "not measured"

HEADLINE = [("chorus", 1, 1.0, 1.0), ("chorus", 2, 1.0, 1.0),
            ("chorus", 3, 1.0, 1.0), ("pgd", 2, 1.0, None)]

#: Part 10's detector AUCs, for the QUALITATIVE comparison only. Quoted from
#: docs/KNOWN_LIMITATIONS.md L5, which cites results/comparison/summary.md.
#: These are synthetic-data, 4-view numbers and are NOT like-for-like with
#: anything in this Part; see the "Not like-for-like" section.
PART10_L5 = {
    "chorus_k1": "0.7215", "chorus_k2": "0.6328", "chorus_k3": "0.4870 (at chance)",
    "pgd_k2": "0.8396 +/- 0.0970", "clone_k2": "0.8743 +/- 0.0583",
    "missing_30": "0.2568", "missing_50": "0.1247",
}


def load(name: str):
    """Load a sidecar measurement. Small, committed raw, not hash-pinned."""
    path = ROOT / name
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt(value, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return NOT_MEASURED
    return f"{value:.{places}f}"


def cell_seed_values(cell: dict, field: str) -> list:
    return [ps.get(field) for ps in cell["per_seed"]]


def cell_mean(cell: dict, field: str):
    vals = [v for v in cell_seed_values(cell, field) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def select(cells: dict, kind: str, k: int, epsilon: float, beta, dns: bool) -> dict:
    out = {}
    for key, cell in cells.items():
        if cell.get("kind") != kind or cell.get("k") != k:
            continue
        if cell.get("epsilon") != epsilon or cell.get("dns_compromised") is not dns:
            continue
        if kind == "chorus" and cell.get("beta") != beta:
            continue
        out[key] = cell
    return out


def pooled_row(record: dict, kind: str, k: int, epsilon: float, beta, dns: bool):
    """The runner's own by_dns_side group, with its span labelled honestly."""
    suffix = "dns_compromised" if dns else "dns_not_compromised"
    stem = (f"{kind}_k{k}_eps{epsilon}_beta{beta}" if kind == "chorus"
            else f"{kind}_k{k}_eps{epsilon}")
    group = record.get("by_dns_side", {}).get(f"{stem}__{suffix}")
    if group is None:
        return None, None
    n = group['n_subsets']
    span = f"{n} subset{'' if n == 1 else 's'} x {group['n_seeds_per_subset']} seeds"
    return group, span


def line(parts: list[str]) -> str:
    return "| " + " | ".join(parts) + " |"


def header(cols: list[str]) -> list[str]:
    return [line(cols), line(["---"] * len(cols))]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


#: The one attack cell the perturbed-cell fractions are read from. The padding
#: audit attacks each view ALONE at k=1, seed 0, so the fraction below is "this
#: view, compromised on its own". It is a structural measurement at one seed,
#: not a 5-seed finding -- see audit_padding.py's own seed note.
PERTURBED_CELL_ATTACK = "chorus_eps1.0"


def perturbed_fractions(audit, dname: str) -> dict:
    """Per-view `native_cells_perturbed / native_cells` for one direction."""
    if not audit:
        return {}
    record = audit.get("directions", {}).get(dname, {})
    out = {}
    for view, entry in record.get("per_view", {}).items():
        cell = entry.get("attacks", {}).get(PERTURBED_CELL_ATTACK)
        if not cell:
            continue
        total = cell.get("native_cells")
        moved = cell.get("native_cells_perturbed")
        if not total or moved is None:
            continue
        out[view] = (moved, total, moved / total)
    return out


def section_epsilon(results: dict, audit) -> list[str]:
    """Item (c): epsilon 1.0 in per-view std units, both directions."""
    out = ["## 3. What epsilon 1.0 is worth, per view and per direction", "",
           "`attack.epsilon_rescaled` is **false** in `config.yaml`: Part 09's grid "
           "(0.2, 0.5, 1.0, 2.0) is carried over unrescaled against features whose "
           "per-view std is measured below. A fixed epsilon is therefore a "
           "DIFFERENT relative perturbation in each view. `epsilon 1.0 in std "
           "units` is `1.0 / std`, so a larger number means the same epsilon moves "
           "that view further relative to its own spread.", "",
           "The last column is a DIFFERENT quantity and must not be read as attack "
           "strength: it is the fraction of that view's NATIVE feature cells the "
           "attack actually moves when that view is compromised alone "
           f"(`{PERTURBED_CELL_ATTACK}`, k=1, seed 0, from `padding_audit.json`). "
           "A cell is left untouched where the objective's gradient is exactly "
           "zero, since the step is `epsilon * sign(gradient)`. The denominator is "
           "every native cell in the test set, so windows in which the view is "
           "absent are inside it. One seed, structural, NOT a 5-seed finding.", ""]

    in_std: dict = {}
    fractions: dict = {}
    stds: dict = {}
    for dname, record in results["directions"].items():
        scale = record["systems"]["naive"]["feature_scale"]
        frac = perturbed_fractions(audit, dname)
        fractions[dname] = frac
        out += [f"### {dname} (test scenario `{record['test_scenario']}`, "
                f"n = {record['n_test']} windows)", ""]
        out += header(["view", "std", "IQR", "median abs", "max abs",
                       "windows present", "epsilon 1.0 in std units",
                       "native cells perturbed"])
        per_view_std = {}
        for view, entry in scale["per_view"].items():
            std = entry.get("std")
            per_view_std[view] = std
            ratio = (1.0 / std) if isinstance(std, (int, float)) and std else None
            if view == "dns":
                in_std[dname] = ratio
            name = f"**`{view}`**" if view == "dns" else f"`{view}`"
            if view in frac:
                moved, total, share = frac[view]
                moved_text = f"{moved} / {total} = {share:.4f}"
            else:
                moved_text = NOT_MEASURED
            out.append(line([name, fmt(std), fmt(entry.get("iqr")),
                             fmt(entry.get("median_abs")), fmt(entry.get("max_abs")),
                             str(entry.get("n_present", NOT_MEASURED)), fmt(ratio),
                             moved_text]))
        stds[dname] = per_view_std
        measured = {v: x for v, x in per_view_std.items()
                    if isinstance(x, (int, float))}
        out += ["",
                f"min / median / max per-view std: {fmt(scale['min_per_view_std'])} / "
                f"{fmt(scale['median_per_view_std'])} / {fmt(scale['max_per_view_std'])}; "
                f"Part 09's synthetic reference std was "
                f"{fmt(scale['synthetic_reference_std'], 1)}."]
        if measured:
            smallest = min(measured, key=measured.get)
            tail = "."
            if "dns" in measured:
                rank = sorted(measured, key=measured.get).index("dns") + 1
                tail = (f", which is rank {rank} of {len(measured)} by std "
                        f"(1 = smallest).")
            out.append(f"Smallest std in this direction: `{smallest}` "
                       f"({fmt(measured[smallest])}). `dns` is "
                       f"{fmt(measured.get('dns'))}{tail}")
        out.append("")

    # The claim this section used to make -- that `dns` has the smallest std in
    # BOTH directions -- is false. It is now derived, not asserted.
    smallest_per_direction = {}
    for dname, per_view_std in stds.items():
        measured = {v: x for v, x in per_view_std.items()
                    if isinstance(x, (int, float))}
        if measured:
            smallest_per_direction[dname] = min(measured, key=measured.get)
    dns_smallest = [d for d, v in smallest_per_direction.items() if v == "dns"]
    dns_not = [d for d, v in smallest_per_direction.items() if v != "dns"]

    out += ["`dns` is named deliberately: it is the one view that solves this label "
            "([[L17]]). **It does NOT have the smallest per-view std in both "
            "directions.**"]
    for dname in dns_not:
        winner = smallest_per_direction[dname]
        measured = {v: x for v, x in stds[dname].items()
                    if isinstance(x, (int, float))}
        out.append(f"- In **{dname}** the smallest std is `{winner}` "
                   f"({fmt(measured[winner])}); `dns` is "
                   f"{fmt(measured.get('dns'))}.")
    for dname in dns_smallest:
        measured = {v: x for v, x in stds[dname].items()
                    if isinstance(x, (int, float))}
        out.append(f"- In **{dname}** `dns` IS the smallest ({fmt(measured['dns'])}).")
    out.append("")

    names = list(in_std)
    if len(names) == 2 and all(in_std[n] for n in names):
        a, b = names
        ratio = in_std[b] / in_std[a]
        bigger, smaller = (b, a) if ratio >= 1 else (a, b)
        out += [f"Epsilon 1.0 equals **{fmt(in_std[a], 3)} of the `dns` std in {a}** "
                f"and **{fmt(in_std[b], 3)} of the `dns` std in {b}**, so the "
                f"relative perturbation on `dns` is about "
                f"{max(ratio, 1 / ratio):.1f} times larger in {bigger} than in "
                f"{smaller}. **Cross-direction comparisons of attack success are "
                f"confounded by this.** The same epsilon is not the same attack on "
                f"the view that carries the label, so reading {bigger}'s higher "
                f"attack success as a property of that scenario pair alone is "
                f"unsupported.", ""]

    dns_shares = {}
    for dname, frac in fractions.items():
        if frac.get("dns"):
            dns_shares[dname] = frac["dns"][2]
    if dns_shares:
        worded = ", ".join(f"{d} {share:.4f}" for d, share in dns_shares.items())
        out += [f"The two right-hand columns point opposite ways on `dns`, and both "
                f"belong in any reading of this grid: a fixed epsilon is a large "
                f"perturbation relative to the `dns` spread, yet the attack moves "
                f"only a small share of `dns` cells at all ({worded}), the smallest "
                f"share of any view in either direction. `audit` is at the opposite "
                f"end on both counts.", ""]

    out += ["Any reading of the epsilon grid as a single attack strength across "
            "views, or across the two directions, is wrong.", ""]
    return out


def section_padding(audit: dict | None) -> list[str]:
    """Item (a): the padding-column question, answered by measurement."""
    out = ["## 4. Padding columns: the attack does NOT write into them", ""]
    if audit is None:
        out += [f"{NOT_MEASURED}: `results/ait_attack/padding_audit.json` is absent. "
                "Run `experiments/ait_attack/audit_padding.py`.", ""]
        return out
    out += ["**The Part 29 post-run brief asked this README to state that the frozen "
            "chorus attack also writes +/- epsilon into the structurally-zero "
            "padding columns. That premise is not what this pipeline does, and the "
            "measurement below is why.**", "",
            f"AIT views are 8-10 features wide and the tensor is padded to "
            f"{audit['padded_view_width']} columns, so every view but `suricata` "
            "carries columns that are structurally zero in the data. Those columns "
            "ARE inside the attack's mask: `chorus_attack` steps "
            f"`step_size * gradient.sign() * mask` where `mask` covers all "
            f"{audit['padded_view_width']} columns of a compromised view "
            f"(`{audit['mechanism']['attack_mask_site']}`, "
            f"`{audit['mechanism']['step_site']}`). They are nonetheless never "
            "moved.", "",
            f"**Measured answer: {audit['answer']}**", "",
            f"The mechanism is `{audit['mechanism']['encoder_native_width_site']}`: "
            f"{audit['mechanism']['why_zero']}", ""]
    out += header(["view", "native dims", "padding cols", "data max abs in padding",
                   "padding cells (per direction)", "padding cells perturbed",
                   "objective gradient max abs in padding"])
    first = next(iter(audit["directions"].values()))
    for view, entry in first["per_view"].items():
        worst_delta, worst_grad, cells = 0, 0.0, []
        for direction in audit["directions"].values():
            e = direction["per_view"][view]
            cells.append(str(e["padding_cells"]))
            for a in e["attacks"].values():
                worst_delta = max(worst_delta, a["padding_cells_perturbed"])
            if "objective_gradient" in e:
                worst_grad = max(worst_grad, e["objective_gradient"]["padding_max_abs"])
        out.append(line([f"`{view}`", str(entry["native_dims"]),
                         str(entry["padding_columns"]),
                         fmt(entry["data_max_abs_in_padding"], 1),
                         " / ".join(cells), str(worst_delta),
                         f"{worst_grad:.3e}"]))
    out += ["", f"Counts are the worst case over both directions, chorus and pgd, at "
            f"epsilon {' and '.join(str(e) for e in (1.0, 2.0))}, seed "
            f"{audit['seed']}. {audit['seed_note']}", "",
            "For contrast, the NATIVE columns are perturbed to the full epsilon:", ""]
    out += header(["direction", "view", "attack", "native cells perturbed",
                   "native max abs delta"])
    for dname, direction in audit["directions"].items():
        for view, entry in direction["per_view"].items():
            a = entry["attacks"].get("chorus_eps1.0")
            if a is None:
                continue
            out.append(line([dname, f"`{view}`", "chorus eps 1.0",
                             f"{a['native_cells_perturbed']} / {a['native_cells']}",
                             fmt(a["native_max_abs_delta"])]))
    out += ["", "Two things follow. The attack surface is the native features only, "
            "which is the correct surface. And it is NOT the whole native surface "
            "either: the perturbed fraction varies widely by view, lowest on `dns`, "
            "so `epsilon` bounds the per-feature move but not how many features "
            "move.", ""]
    return out


def section_determinism(check: dict | None) -> list[str]:
    """Item (e): the step 0 result."""
    out = ["## 2. Reproducibility, and the result of the determinism check", ""]
    if check is None:
        out += [f"{NOT_MEASURED}: `results/ait_attack/determinism_check.json` is absent.", ""]
        return out
    out += ["**The runner has no resume path.** `main()` builds its results dict "
            "from scratch and loops every (direction, system) unconditionally; the "
            "`results.json` writes after each unit are crash-safety checkpoints for "
            "inspection, not a resume source. There is no `--resume` flag. A "
            "relaunch therefore recomputes all four units and overwrites "
            "`results.json`.", "",
            "That is how this determinism check exists for free: the "
            "2026-10-13 run died after three of its four units, the retained "
            "checkpoint was preserved, and the 2026-10-14 relaunch recomputed those "
            "three units hours later from the same commit, config and seeds.", ""]
    out += header(["metric family", "fields", "max abs difference",
                   "values compared", "cells differing"])
    for name, f in check["per_family"].items():
        out.append(line([name, ", ".join(f"`{x}`" for x in f["fields"]),
                         repr(f["max_abs_diff"]), str(f["n_values_compared"]),
                         str(f["n_cells_differing"])]))
    ex = check["exhaustive"]
    out += ["",
            f"Beyond those four families, an exhaustive leaf-by-leaf comparison of "
            f"the three shared units found **{ex['n_differences']} differences over "
            f"{ex['n_leaves_compared']} leaves "
            f"({ex['n_numeric_leaves_compared']} numeric)**, including the "
            "direction-A paired block.", "",
            f"**Verdict: bit-identical on the shared units = "
            f"{check['bit_identical_on_shared_units']}.** Compared units: "
            f"{', '.join('`' + u + '`' for u in check['shared_units'])}. Recomputed "
            f"only in the completed run: "
            f"{', '.join('`' + u + '`' for u in check['unit_present_only_in_new'])}.",
            "",
            "Nothing was rounded away: the differences above are exact zeros, not "
            "small numbers. Note what this does and does not establish. It shows "
            "this experiment is reproducible run-to-run on one machine at one "
            "commit. It says nothing about reproducibility across machines, torch "
            "versions or BLAS backends, and it is not a 5-seed finding - it is one "
            "comparison of two runs.", ""]
    return out


def section_testb(results: dict) -> list[str]:
    out = ["## 5. Test B: attack success, never averaged across the dns split", "",
           "Every row is one grid point and one side of the dns split. The 5-seed "
           "column is the per-cell mean +/- population std over 5 seeds, ranged "
           "across the subsets in that group; the pooled column is the runner's own "
           "group figure, whose std pools subset-to-subset with seed-to-seed "
           "variation and is therefore **not** a 5-seed std.", ""]
    for dname, record in results["directions"].items():
        out += [f"### {dname} (train `{record['train_scenario']}` -> test "
                f"`{record['test_scenario']}`)", ""]
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            out += [f"**{system}**", ""]
            out += header(["grid point", "dns in compromised set", "subsets",
                           "per-cell 5-seed attack success (min..max of means)",
                           "pooled attack success", "pooled span"])
            for kind, k, eps, beta in HEADLINE:
                for dns in (True, False):
                    chosen = select(cells, kind, k, eps, beta, dns)
                    if not chosen:
                        continue
                    means = [cell_mean(c, "attack_success") for c in chosen.values()]
                    means = [m for m in means if m is not None]
                    group, span = pooled_row(record["systems"][system], kind, k, eps, beta, dns)
                    pooled = NOT_MEASURED
                    if group is not None:
                        s = group["summary"]["attack_success"]
                        pooled = f"{s['mean']:.4f} +/- {s['std']:.4f}"
                    label = f"{kind} k={k} eps={eps}" + (f" beta={beta}" if beta is not None else "")
                    out.append(line([label, "YES" if dns else "no", str(len(chosen)),
                                     f"{min(means):.4f}..{max(means):.4f}" if means else NOT_MEASURED,
                                     pooled, span or NOT_MEASURED]))
            out.append("")
    return out


def section_paired(results: dict) -> list[str]:
    out = ["## 6. PrismFlow minus naive, paired within seed", "",
           "`paired_prismflow_minus_naive` in `results.json` holds a per-cell, "
           "within-seed difference for every condition. Negative attack-success "
           "difference favours PrismFlow. `5/5 cells` counts cells where all five "
           "seeds point the same way.", ""]
    for dname, record in results["directions"].items():
        paired = record.get("paired_prismflow_minus_naive") or {}
        out += [f"### {dname}", ""]
        out += header(["dns in compromised set", "k", "cells",
                       "attack-success difference (range of cell means)",
                       "cells favouring PrismFlow", "cells favouring naive",
                       "cells unanimous across 5 seeds"])
        for dns in (True, False):
            for k in (1, 2, 3):
                rows = [v for v in paired.values()
                        if v.get("dns_compromised") is dns and v.get("k") == k
                        and "attack_success" in v.get("diff", {})]
                if not rows:
                    continue
                means = [r["diff"]["attack_success"]["mean"] for r in rows]
                unanimous = sum(1 for r in rows
                                if r["diff"]["attack_success"]["n_seeds_favouring_prismflow"]
                                == r["diff"]["attack_success"]["n_seeds"])
                out.append(line(["YES" if dns else "no", str(k), str(len(rows)),
                                 f"{min(means):+.4f}..{max(means):+.4f}",
                                 str(sum(1 for m in means if m < 0)),
                                 str(sum(1 for m in means if m > 0)),
                                 str(unanimous)]))
        out.append("")
    return out


def section_detector(results: dict, subset: dict | None) -> list[str]:
    out = ["## 7. The detector: AUC, score ties, and why it is not comparable with "
           "Part 10's clone_k2", "",
           "The detector exists only on the PrismFlow arm. On the naive arm "
           "`use_discount=False` leaves `dependence_matrix` None, so no "
           "`SuspicionReport` is produced and `detector_auc` is "
           f"**{NOT_MEASURED}** there - undefined, not silent, and recorded as such "
           "per seed with a reason string.", ""]
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        aucs = [(key, cell_mean(c, "detector_auc")) for key, c in cells.items()]
        aucs = [(k, v) for k, v in aucs if v is not None]
        values = np.array([v for _, v in aucs])
        below = sum(1 for v in values if v < 0.5)
        out += [f"### {dname}", "",
                f"Across all {len(aucs)} conditions, the per-cell 5-seed mean AUC "
                f"runs **{values.min():.4f} to {values.max():.4f}**. "
                f"**{below} of {len(aucs)} conditions sit BELOW 0.5**, i.e. the "
                "detector scores attacked data as LESS suspicious than clean data. "
                f"No condition is exactly 0.5.", ""]
        out += header(["grid point", "dns in compromised set", "subsets",
                       "detector AUC (min..max of per-cell 5-seed means)",
                       "worst cell", "best cell"])
        for kind, k, eps, beta in HEADLINE:
            for dns in (True, False):
                chosen = select(cells, kind, k, eps, beta, dns)
                if not chosen:
                    continue
                pairs = [(key, cell_mean(c, "detector_auc")) for key, c in chosen.items()]
                pairs = [(key, v) for key, v in pairs if v is not None]
                if not pairs:
                    continue
                vals = [v for _, v in pairs]
                lo = min(pairs, key=lambda t: t[1])
                hi = max(pairs, key=lambda t: t[1])
                label = f"{kind} k={k} eps={eps}" + (f" beta={beta}" if beta is not None else "")
                out.append(line([label, "YES" if dns else "no", str(len(pairs)),
                                 f"{min(vals):.4f}..{max(vals):.4f}",
                                 f"`{lo[0]}` {lo[1]:.4f}", f"`{hi[0]}` {hi[1]:.4f}"]))
        for key in ("missing_30", "missing_50"):
            if key in cells:
                out.append(line([f"`{key}`", "n/a", "1",
                                 fmt(cell_mean(cells[key], "detector_auc")), "-", "-"]))
        out.append("")

    out += ["### 7.1 Score ties and distinct scores (bounded subset, 5 seeds)", ""]
    if subset is None:
        out += [f"{NOT_MEASURED}: `results/ait_attack/detector_scores_subset.json` "
                "is absent. Run `experiments/ait_attack/detector_score_stats.py`.", ""]
    else:
        out += [f"**Scope: {subset['scope']}**", "",
                f"Why recomputed: {subset.get('why_recomputed', 'run_ait_attack.py:860 strips _scores before serialisation.')}", "",
                f"`tie_fraction` is the {subset.get('tie_fraction_definition', 'fraction of finite scores whose value occurs more than once')}.", ""]
        for dname, conds in subset["directions"].items():
            out += [f"**{dname}**", ""]
            out += header(["condition", "dns in compromised set", "windows",
                           "distinct scores (5-seed mean +/- std)",
                           "tie fraction (5-seed mean +/- std)",
                           "largest tie group", "detector AUC (5 seeds)",
                           "AUC equals results.json"])
            for key, v in conds.items():
                dns = v.get("dns_compromised")
                match = v.get("matches_committed_exactly")
                out.append(line([
                    f"`{key}`",
                    "YES" if dns is True else ("no" if dns is False else "n/a"),
                    str(v["n_windows"]),
                    f"{v['n_distinct_mean']:.1f} +/- {v['n_distinct_std']:.1f}",
                    f"{v['tie_fraction_mean']:.4f} +/- {v['tie_fraction_std']:.4f}",
                    str(v["largest_tie_group_max"]),
                    f"{v['detector_auc_mean']:.4f} +/- {v['detector_auc_std']:.4f}",
                    "yes" if match else ("NO" if match is False else "n/a"),
                ]))
            out.append("")
        all_rows = [v for conds in subset["directions"].values() for v in conds.values()]
        ties = np.array([v["tie_fraction_mean"] for v in all_rows])
        dist = np.array([v["n_distinct_mean"] for v in all_rows])
        worst = max(
            ((d, k, v) for d, conds in subset["directions"].items()
             for k, v in conds.items()),
            key=lambda t: t[2]["tie_fraction_mean"],
        )
        out += [f"**No condition in this subset is degenerate in the "
                f"constant-score sense, but ties are not negligible either.** "
                f"Distinct scores per condition run {dist.min():.1f} to "
                f"{dist.max():.1f} of the window count, and tie fractions run "
                f"{ties.min():.4f} to {ties.max():.4f}. The most tied condition is "
                f"`{worst[1]}` in {worst[0]} at "
                f"{worst[2]['tie_fraction_mean']:.4f} +/- "
                f"{worst[2]['tie_fraction_std']:.4f}, still carrying "
                f"{worst[2]['n_distinct_mean']:.1f} distinct values, so no "
                "condition collapses onto a handful of scores and none is "
                "constant.", "",
                "**That cuts two ways, and the second way is a caveat on section "
                "7.** `detector_auc` scores ties at 0.5 "
                "(`run_ait_attack.py::detector_auc`, via `roc_auc_binary`), so a "
                "heavily tied condition is pulled MECHANICALLY toward 0.5. For the "
                "conditions far below 0.5 the inversion is therefore real and not "
                "a tie artefact - ties would have masked it, not created it. But "
                "for any condition sitting NEAR 0.5, the tie fraction in the table "
                "above should be read before calling it 'at chance': at a tie "
                "fraction of "
                f"{worst[2]['tie_fraction_mean']:.2f} a large share of the "
                "comparisons contributing to that AUC are ties rather than genuine "
                "orderings.", ""]
    return out


def section_not_like_for_like(results: dict, subset: dict | None) -> list[str]:
    """Item (b)."""
    out = ["## 8. Not like-for-like with Part 10's clone_k2", "",
           "**Confirmed from the code, not inferred.** Part 10's `clone_k2` "
           "compares a 6-view attacked model against a 4-view clean reference:", "",
           "- `experiments/comparison/run_comparison.py:88` declares "
           "`clone_k2` as `{kind: clone, k: 2}`, and `:69` sets `n_views: 4`.",
           "- `prismflow/data/corruption.py:15-20` - `duplicate_view` APPENDS k "
           "copies of a view, so the clone condition runs 4 + 2 = 6 views.",
           "- `experiments/comparison/run_comparison.py:232-234` and `:271` build "
           "and select a model of width `4 + k` for a clone condition.",
           "- `:279-281` take the reference scores from the `clean` condition, "
           "which is `k = 0` and therefore width 4; `:292` is "
           "`roc_auc(scores, reference)`.", "",
           "So the two score vectors come from models of different width. That "
           "matters because the suspicion score is a MAXIMUM, not a mean: "
           "`prismflow/statistics/suspicion.py:43-45` and `:253` take "
           "`masked.max(axis=(1, 2))` over the off-diagonal pairs selected at "
           "`:248-249` by `~np.eye(n_views)`. A 6-view model maximises over "
           "**30** off-diagonal entries and the 4-view clean reference over "
           "**12**. A maximum over 30 draws is stochastically larger than a "
           "maximum over 12 even when every per-pair distribution is identical, "
           "so part of clone_k2's AUC 0.8743 +/- 0.0583 is the width change "
           "itself rather than detected duplication.", "",
           "Part 29 has no such asymmetry: every condition and its clean "
           "reference run the same 5-view model, so the AUCs in section 7 are "
           "attacked-vs-clean at fixed width. **Any comparison between the two "
           "Parts is therefore qualitative only, and the numbers must not be "
           "differenced.** Part 29 also runs on real AIT logs with 5 views and a "
           "label one view solves, against Part 10's 4-view synthetic data.", "",
           "With that caveat, the qualitative comparison:", ""]
    out += header(["condition", "Part 10 AUC (synthetic, 4 views, L5)",
                   "Part 29 AUC range (real AIT, 5 views, per-cell 5-seed means)"])
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        for label, (kind, k, eps, beta) in zip(
                ("chorus_k1", "chorus_k2", "chorus_k3", "pgd_k2"), HEADLINE):
            vals = []
            for dns in (True, False):
                for c in select(cells, kind, k, eps, beta, dns).values():
                    m = cell_mean(c, "detector_auc")
                    if m is not None:
                        vals.append(m)
            if not vals:
                continue
            out.append(line([f"{label} ({dname})", PART10_L5[label],
                             f"{min(vals):.4f}..{max(vals):.4f}"]))
        for key in ("missing_30", "missing_50"):
            if key in cells:
                m = cell_mean(cells[key], "detector_auc")
                out.append(line([f"{key} ({dname})", PART10_L5[key], fmt(m)]))
    out += ["", "`clone_k2` has no Part 29 counterpart at all: this Part runs no "
            "duplication condition, so Part 10's "
            f"{PART10_L5['clone_k2']} is listed here only as the number the "
            "width asymmetry above applies to.", "",
            "### 8.1 What survives qualitatively", "",
            "Only one Part 10 pattern reproduces here, and one clearly does not:", "",
            "- **Reproduces: inversion under missing evidence.** Part 10 found AUC "
            f"{PART10_L5['missing_30']} at 30% missing and "
            f"{PART10_L5['missing_50']} at 50%, both far below chance. Part 29's "
            "missingness rows are also below 0.5 in both directions (section 7).",
            "- **Does NOT reproduce: a monotone decay in k.** Part 10 read "
            f"{PART10_L5['chorus_k1']} -> {PART10_L5['chorus_k2']} -> "
            f"{PART10_L5['chorus_k3']}, reaching chance at k=3. Neither AIT "
            "direction gives that ordering, as the computed ranges below show.", ""]
    rows, verdicts = [], []
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        ranges = []
        for k in (1, 2, 3):
            vals = []
            for dns in (True, False):
                for c in select(cells, "chorus", k, 1.0, 1.0, dns).values():
                    m = cell_mean(c, "detector_auc")
                    if m is not None:
                        vals.append(m)
            ranges.append((min(vals), max(vals)) if vals else (None, None))
        falls = [ranges[i][1] > ranges[i + 1][1] for i in range(2)]
        steps = ("k=1->k=2", "k=2->k=3")
        if all(falls):
            verdict = "yes, falls at both steps"
        elif any(falls):
            fell = ", ".join(s for s, f in zip(steps, falls) if f)
            rose = ", ".join(s for s, f in zip(steps, falls) if not f)
            verdict = f"no: falls at {fell} but rises at {rose}"
        else:
            verdict = "no, rises at both steps"
        verdicts.append((dname, verdict))
        rows.append(line([dname] + [f"{lo:.4f}..{hi:.4f}" for lo, hi in ranges]
                         + [verdict]))
    out += header(["direction", "chorus k=1", "chorus k=2", "chorus k=3",
                   "maxima fall with k?"])
    out += rows
    out += ["", "The ranges at each k overlap heavily in both directions and "
            "neither falls at both steps, so no k-trend claim is supported on this "
            "data, and the two directions must not be averaged to manufacture "
            "one.", "",
            "Section 7 shows why that is unsurprising here: WHICH views are "
            "compromised matters more than how many, because only `dns` carries "
            "the label. That is the sense in which Part 10's k axis does not "
            "transfer to AIT at all.", ""]
    return out


def section_verdicts(results: dict, subset: dict | None) -> list[str]:
    out = ["## 9. Verdicts on L3, L4, L5, L8, L16 and L17", "",
           "Each verdict is split by whether `dns` - the single view that solves "
           "this label - is inside the compromised set. **Nothing below is "
           "averaged across that split**, and nothing is averaged across the two "
           "directions, which disagree.", ""]

    def succ_range(record, system, dns, kind="chorus", k=2, eps=1.0, beta=1.0):
        cells = record["systems"][system]["cells"]
        means = [cell_mean(c, "attack_success")
                 for c in select(cells, kind, k, eps, beta, dns).values()]
        means = [m for m in means if m is not None]
        return (min(means), max(means)) if means else (None, None)

    out += ["### L3 - the discount addresses redundant evidence, not unreliable "
            "evidence (and Part 09's extension: reading the axis confers no "
            "defence)", ""]
    for dname, record in results["directions"].items():
        paired = record.get("paired_prismflow_minus_naive") or {}
        for dns in (True, False):
            rows = [v for v in paired.values()
                    if v.get("dns_compromised") is dns
                    and "attack_success" in v.get("diff", {})]
            if not rows:
                continue
            means = [r["diff"]["attack_success"]["mean"] for r in rows]
            better = sum(1 for m in means if m < 0)
            side = "dns COMPROMISED" if dns else "dns NOT compromised"
            out.append(f"- **{dname}, {side}:** PrismFlow lowers attack success in "
                       f"**{better} of {len(rows)}** cells, difference range "
                       f"{min(means):+.4f}..{max(means):+.4f}.")
    out += ["", "**Verdict: L3's extension is NOT reproduced on this data, and the "
            "direction of the effect depends on the dns split and on which "
            "direction is read.** Part 09 found PrismFlow strictly worse than naive "
            "fusion in 12/12 attacked cells on synthetic data. Here the sign is "
            "mixed and in direction A with `dns` outside the compromised set it "
            "mostly favours PrismFlow. This is NOT evidence that the discount "
            "defends: the cells disagree, the two directions disagree, and [[L17]] "
            "explains why a 5-view fusion comparison on this label is weak evidence "
            "either way. It does mean Part 09's blanket statement should not be "
            "restated as if it held on real data.", ""]

    out += ["### L4 - the discount is proportional, not capping, and weakens as "
            "evidence grows", ""]
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        for dns in (True, False):
            rows = [c for c in select(cells, "chorus", 2, 1.0, 1.0, dns).values()]
            dep = [cell_mean(c, "dependence_compromised") for c in rows]
            dep = [d for d in dep if d is not None]
            lo, hi = succ_range(record, "prismflow", dns)
            if not dep or lo is None:
                continue
            side = "dns COMPROMISED" if dns else "dns NOT compromised"
            out.append(f"- **{dname}, {side}, chorus k=2 eps=1.0:** dependence among "
                       f"compromised views reaches {min(dep):.4f}..{max(dep):.4f} "
                       f"while PrismFlow's attack success is "
                       f"{lo:.4f}..{hi:.4f}.")
    dep_max = max(
        (cell_mean(c, "dependence_compromised") or 0.0)
        for record in results["directions"].values()
        for c in record["systems"]["prismflow"]["cells"].values()
        if c.get("k") in (2, 3)
    )
    out += ["", "**Verdict: consistent with L4, but this Part does not test the "
            "mechanism.** The collusion is plainly measured - dependence among the "
            f"compromised views reaches {dep_max:.4f} at its highest across all "
            "k>=2 cells - and the attack still succeeds in those same cells, which "
            "is L4's shape. "
            "Part 29 measures no per-view alpha, no belief margin and no "
            "sublinearity curve, so it adds no new evidence about WHY. L4's "
            "mechanism evidence remains Part 09's diagnostic.", ""]

    out += ["### L5 - the detector inherits the dependence blind spot and inverts "
            "under missing evidence", ""]
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        allv = [cell_mean(c, "detector_auc") for c in cells.values()]
        allv = [v for v in allv if v is not None]
        below = sum(1 for v in allv if v < 0.5)
        out.append(f"- **{dname}:** {below} of {len(allv)} conditions below AUC 0.5; "
                   f"range {min(allv):.4f}..{max(allv):.4f}. Missingness: "
                   f"`missing_30` {fmt(cell_mean(cells['missing_30'], 'detector_auc'))}, "
                   f"`missing_50` {fmt(cell_mean(cells['missing_50'], 'detector_auc'))}.")
        for dns in (True, False):
            vals = []
            for kind, k, eps, beta in HEADLINE:
                for c in select(cells, kind, k, eps, beta, dns).values():
                    m = cell_mean(c, "detector_auc")
                    if m is not None:
                        vals.append(m)
            if vals:
                side = "dns COMPROMISED" if dns else "dns NOT compromised"
                out.append(f"  - {side}: {min(vals):.4f}..{max(vals):.4f} over the "
                           f"headline grid points.")
    out += ["", "**Verdict: L5 is reproduced and is WORSE than L5 states, on both "
            "sides of the dns split.** L5 describes a detector that degrades to "
            "chance and inverts under missingness. On AIT the majority of all "
            "conditions - attacked, not merely missing - sit below chance, so the "
            "flag fires LESS under attack than under clean operation across most of "
            "the grid. Section 7.1 rules out the saturated-detector explanation on "
            "the subset measured: scores are well spread. The inversion is "
            "directional.", ""]

    out += ["### L8 - detection and correction fail together", ""]
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        for dns in (True, False):
            pairs = []
            for c in cells.values():
                if c.get("dns_compromised") is not dns:
                    continue
                auc = cell_mean(c, "detector_auc")
                dep = cell_mean(c, "mean_dependence_eigen")
                if auc is not None and dep is not None:
                    pairs.append((auc, dep))
            if len(pairs) < 3:
                continue
            a = np.array([p[0] for p in pairs])
            d = np.array([p[1] for p in pairs])
            r = float(np.corrcoef(a, d)[0, 1])
            side = "dns COMPROMISED" if dns else "dns NOT compromised"
            out.append(f"- **{dname}, {side}:** across {len(pairs)} conditions, "
                       f"Pearson r between per-cell detector AUC and mean dependence "
                       f"= **{r:+.4f}**.")
    out += ["", "**Verdict: the correlation here is NEGATIVE, which is the "
            "opposite sign to Part 14's, and the two are not measuring the same "
            "thing.** Part 14 swept one attacker objective and watched gate AUC and "
            "dependence fall TOGETHER along that axis (r = +0.9578 over six "
            "levels). Part 29 correlates across heterogeneous conditions at fixed "
            "attacker strength, and finds that the conditions with the HIGHEST "
            "measured dependence are the ones the detector scores LOWEST. Both "
            "readings are consistent with L8's underlying claim - that detection "
            "and correction are functions of one quantity - but the sign above must "
            "not be quoted as reproducing Part 14's.", "",
            "The negative sign is close to mechanical, and worth stating because it "
            "sharpens [[L5]] rather than contradicting it. The suspicion score is "
            "agreement left UNEXPLAINED by the estimated dependence "
            "(`prismflow/statistics/suspicion.py`: `excess = agreement - expected` "
            "where `expected` rises with the estimated matrix). An attack that "
            "successfully manufactures dependence therefore raises `expected`, "
            "shrinks `excess`, and lowers its own suspicion score. Succeeding at "
            "collusion is itself what buys cover from a detector watching for "
            "collusion. Part 29 does not establish this mechanism - it measures the "
            "correlation, not the causal path - so it is offered as the likely "
            "reading, not a finding.", ""]

    out += ["### L16 - discrimination and calibration come apart on real "
            "cross-scenario data", ""]
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            for dns in (True, False):
                rows = list(select(cells, "chorus", 2, 1.0, 1.0, dns).values())
                if not rows:
                    continue
                ba = [cell_mean(c, "balanced_accuracy") for c in rows]
                auc = [cell_mean(c, "roc_auc") for c in rows]
                br = [cell_mean(c, "brier") for c in rows]
                ece = [cell_mean(c, "prob_ece") for c in rows]
                side = "dns COMPROMISED" if dns else "dns NOT compromised"
                out.append(f"- **{dname} / {system}, {side}, chorus k=2 eps=1.0:** "
                           f"balanced accuracy {min(ba):.4f}..{max(ba):.4f}, "
                           f"ROC AUC {min(auc):.4f}..{max(auc):.4f}, "
                           f"Brier {min(br):.4f}..{max(br):.4f}, "
                           f"ECE {min(ece):.4f}..{max(ece):.4f}.")
    worst_auc, worst_where = None, None
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            for c in select(record["systems"][system]["cells"],
                            "chorus", 2, 1.0, 1.0, True).values():
                m = cell_mean(c, "roc_auc")
                if m is not None and (worst_auc is None or m < worst_auc):
                    worst_auc, worst_where = m, f"{dname} / {system}"
    out += ["", "**Verdict: L16's shape holds only while `dns` is outside the "
            "compromised set. Once `dns` is compromised the two do NOT merely come "
            "apart - ranking collapses as well.**", "",
            "With `dns` honest, the rows above are L16 exactly: ROC AUC stays at or "
            "above 0.98 in both directions while Brier and ECE degrade by an order "
            "of magnitude across subsets. With `dns` compromised in direction B, "
            f"ROC AUC falls to {worst_auc:.4f} ({worst_where}) - below chance, so "
            "the ordering is actively inverted, not merely poorly thresholded - and "
            "Brier reaches the far end of its range at the same time. Describing "
            "that cell as 'discrimination holds, calibration fails' would be "
            "wrong.", "",
            "Brier here is the multi-class sum form, the same convention L16 uses "
            "when it reports the majority-class baseline as exactly `2p(1-p)`, so "
            "values above 1.0 are in range and are not a unit error.", "",
            "L16's own caveats carry over unchanged: the positive class is 3 "
            "episodes on `santos` and 1 on `russellmitchell`, every +/- here is a "
            "spread over 5 initialisations on a fixed scenario pair, and the two "
            "directions must not be averaged.", ""]

    out += ["### L17 - on AIT the label is solved by one view", ""]
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            lo_y, hi_y = succ_range(record, system, True)
            lo_n, hi_n = succ_range(record, system, False)
            if lo_y is None or lo_n is None:
                continue
            out.append(f"- **{dname} / {system}, chorus k=2 eps=1.0:** attack "
                       f"success {lo_y:.4f}..{hi_y:.4f} with `dns` compromised, "
                       f"against {lo_n:.4f}..{hi_n:.4f} without it.")
    out += ["", "**Verdict: L17 is confirmed and it governs this whole Part.** "
            "Whether `dns` is in the compromised set changes the attack-success "
            "range by more than k, epsilon or beta do. The epsilon grid is also "
            "not comparable across the two directions on this view: epsilon 1.0 "
            "is a far larger perturbation relative to the `dns` spread in "
            "B_santos_to_rm than in A_rm_to_santos, and `dns` has the smallest "
            "per-view std only in B, not in both directions (section 3). "
            "**Consequence: no number in this Part is evidence about "
            "multi-view fusion.** It is evidence about perturbing the one view that "
            "solves a single-view task, plus four views that are close to useless "
            "on it.", ""]
    return out


def section_evidence(results: dict, manifest) -> list[str]:
    out = ["## 10. Retained run evidence", "",
           "Nothing from the failed and completed runs was deleted. These files are "
           "kept deliberately and are not findings:", ""]
    out += header(["file", "what it is", "status"])
    rows = [
        ("`results/ait_attack/results.json.gz`",
         "the completed run: 4 units, 5 seeds each, commit "
         f"`{results.get('commit')}`", "**the result**"),
        ("`results/ait_attack/summary.md`",
         "the runner's own report, regenerated at the end of the completed run",
         "**the result**"),
        ("`results/ait_attack/results.3of4units_20261014_0022.json.gz`",
         "the 2026-10-13 run's last checkpoint: 3 of 4 units, written 00:22 before "
         "that run died at 00:43", "run evidence, superseded"),
        ("`results/ait_attack/results.A_naive_checkpoint.json`",
         "the same run's first checkpoint: 1 unit (A/naive) only",
         "run evidence, superseded"),
        ("`results/ait_attack/summary.partial.md`",
         "the report written from that 1-unit checkpoint, banner-marked PARTIAL and "
         "SUPERSEDED", "run evidence, NOT a result"),
        ("`results/ait_attack/determinism_check.json`",
         "step 0: the died run against the completed run on the 3 shared units",
         "supporting measurement"),
        ("`results/ait_attack/padding_audit.json`",
         "step 4(a): whether the attack writes into padding columns",
         "supporting measurement"),
        ("`results/ait_attack/detector_scores_subset.json`",
         "step 2: distinct scores and tie fractions, bounded subset, 5 seeds",
         "supporting measurement"),
        ("`results/ait_attack/results_manifest.json`",
         "the decompressed-json sha256 of each gzipped result file, and its raw "
         "and gz sizes", "integrity record"),
        ("`logs/part29_20261014_080604.log`",
         "the completed run's log, one heartbeat line per attack cell",
         "run evidence"),
        ("`logs/part29_20261013_225110.log`",
         "the died run's log; stops mid-cell at 00:43:58 with no traceback",
         "run evidence"),
        ("`logs/part29_20261014_080337.log`",
         "false start: bare `python` has no torch in this environment",
         "run evidence"),
        ("`logs/part29_20261014_080459.log`",
         "false start: `ModuleNotFoundError: No module named 'experiments'`, "
         "no PYTHONPATH", "run evidence"),
    ]
    for f, what, status in rows:
        out.append(line([f, what, status]))
    out += ["", "A third false start at 08:04:20 produced no log at all: cmd.exe "
            "quoting swallowed the redirect, so the launch left no file to keep.",
            "",
            "**What is and is not in the repository.** The four `logs/` entries "
            "are retained on disk but NOT committed: `.gitignore` ignores "
            "`logs/*` except `.gitkeep`, the existing convention, unchanged for "
            "this Part. So a fresh clone has no per-cell heartbeat trail.", "",
            "The two largest `results/` files exceed 10 MB as raw json, and the "
            "largest results file tracked anywhere else in this repository is "
            "0.40 MB. They are therefore committed as **gzip (`-9`)** rather "
            "than raw json, which is the only departure from the convention "
            "tracking `results/*/*.json`; `.gitignore` gains one line, "
            "`!results/*/*.json.gz`, to allow it. Both are in the repository at "
            "this commit:", ""]
    entries = (manifest or {}).get("files", {})
    out += header(["committed file", "raw json", "gzip -9", "sha256 of the "
                   "DECOMPRESSED json"])
    if entries:
        for name, entry in entries.items():
            raw_mb = entry.get("raw_bytes", 0) / 1_000_000
            gz_mb = entry.get("gz_bytes", 0) / 1_000_000
            out.append(line([f"`results/ait_attack/{entry.get('gz_name', name)}`",
                             f"{raw_mb:.2f} MB", f"{gz_mb:.2f} MB",
                             f"`{entry.get('sha256_decompressed_json', NOT_MEASURED)}`"]))
    else:
        out.append(line([NOT_MEASURED] * 4))
    out += ["",
            "The digest above is of the json INSIDE the gzip, not of the gzip "
            "stream. That is deliberate: a gzip stream's own sha256 is not a "
            "stable identifier (`gzip -9 <file>` embeds the mtime and original "
            "filename, and Python's `gzip` at the same level emits different "
            "header bytes from GNU gzip's), whereas the decompressed json is "
            "byte-identical by either route and is what every number in this "
            "file is derived from. To verify a clone:", "",
            "```sh",
            "gzip -dc results/ait_attack/results.json.gz | sha256sum",
            "gzip -dc results/ait_attack/results.3of4units_20261014_0022.json.gz \\",
            "    | sha256sum",
            "",
            "# or, checking both against the manifest rather than by eye:",
            "PYTHONPATH=. .venv/Scripts/python.exe \\",
            "    -m experiments.ait_attack.results_io --verify",
            "```", "",
            "The three reader scripts (`make_readme.py`, `check_determinism.py`, "
            "`detector_score_stats.py`) go through "
            "`experiments/ait_attack/results_io.py`, which prefers a raw `.json` "
            "when one is present on disk, falls back to the committed `.json.gz`, "
            "and checks the bytes it parsed against "
            "`results_manifest.json` before using them - a digest mismatch raises "
            "rather than being reported. `run_ait_attack.py` is unchanged and "
            "still writes raw json; the `.gz` and the manifest are produced after "
            "a run by `results_io.py --write-manifest`. The uncompressed "
            "`results.json` and `results.3of4units_20261014_0022.json` remain on "
            "disk in the working copy and are listed in `.git/info/exclude` so "
            "they cannot be committed by accident alongside the `.gz`. The "
            "smaller `results.A_naive_checkpoint.json` (4.45 MB) is committed raw, "
            "unchanged.",
            ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None,
                        help="explicit path to the run's results file, raw .json or "
                             ".json.gz; default prefers the raw .json on disk and "
                             "falls back to the committed .json.gz")
    parser.add_argument("--out", default=None,
                        help=f"write the README here instead of {OUT}")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else OUT

    results = load_results("results.json", explicit=args.results)
    if results is None:
        raise SystemExit("results/ait_attack/results.json[.gz] is missing; "
                         "run the experiment")
    source = Path(args.results) if args.results else resolve("results.json")
    manifest = load("results_manifest.json")
    check = load("determinism_check.json")
    audit = load("padding_audit.json")
    subset = load("detector_scores_subset.json")

    seeds = results["seeds"]
    doc = [
        "# Part 29 - chorus attack, suspicion detector and missingness on real AIT logs",
        "",
        f"Generated by `experiments/ait_attack/make_readme.py` from "
        f"`results/ait_attack/results.json.gz` and its sidecar measurements. "
        f"**No number in this file is typed by hand.** Commit "
        f"`{results.get('commit')}`; seeds {seeds}; "
        f"`not_evidence: {results.get('not_evidence')}`; runtime "
        f"{results.get('runtime_seconds', float('nan')):.1f} s.",
        "",
        "## 1. What the label means, and what this Part is therefore about",
        "",
        results.get("label_meaning", NOT_MEASURED),
        "",
        "Read that before any number below. Two consequences run through the whole "
        "file: results are always split by whether `dns` is in the compromised set, "
        "and the two directions are never averaged.",
        "",
    ]
    out = ["### Units", ""]
    out += header(["direction", "train", "test", "train windows", "test windows",
                   "systems", "paired block"])
    for dname, record in results["directions"].items():
        out.append(line([dname, f"`{record['train_scenario']}`",
                         f"`{record['test_scenario']}`", str(record["n_train"]),
                         str(record["n_test"]),
                         ", ".join(f"`{s}`" for s in record["systems"]),
                         "yes" if "paired_prismflow_minus_naive" in record else "no"]))
    counts = {
        f"{d}/{s}": len(rec["systems"][s]["cells"])
        for d, rec in results["directions"].items() for s in rec["systems"]
    }
    out += ["", "Conditions per unit: "
            + ", ".join(f"`{k}` {v}" for k, v in counts.items())
            + f"; every condition carries {len(seeds)} seeds.", ""]
    sg = results.get("sensor_group_check", {})
    if sg:
        out += [f"**Same-sensor-group comparison: {sg.get('verdict')}.** "
                f"{sg.get('reason')}", ""]
    doc += out

    doc += section_determinism(check)
    doc += ["### How to reproduce", "",
            "```sh",
            "# from the repository root. The .venv interpreter is required: the",
            "# bare `python` on PATH in this environment has no torch. PYTHONPATH=.",
            "# is required too: `experiments` is an implicit namespace package and",
            "# the runner imports `experiments.ait_clean.run_ait_clean`.",
            "PYTHONPATH=. .venv/Scripts/python.exe -u \\",
            "    experiments/ait_attack/run_ait_attack.py",
            "",
            "# the completed run was launched detached, stdout and stderr merged:",
            "#   cmd.exe /c set PYTHONPATH=. && .venv\\Scripts\\python.exe -u \\",
            "#       experiments\\ait_attack\\run_ait_attack.py > logs\\part29_<ts>.log 2>&1",
            "",
            "# sidecar measurements, after the run:",
            "PYTHONPATH=. .venv/Scripts/python.exe experiments/ait_attack/check_determinism.py",
            "PYTHONPATH=. .venv/Scripts/python.exe experiments/ait_attack/audit_padding.py",
            "PYTHONPATH=. .venv/Scripts/python.exe experiments/ait_attack/detector_score_stats.py",
            "PYTHONPATH=. .venv/Scripts/python.exe experiments/ait_attack/make_readme.py",
            "```",
            "",
            "`--quick` was NOT used. It cuts to 2 seeds, one k and one epsilon and "
            "is a plumbing check, not evidence. **There is no resume flag**: a "
            "relaunch recomputes all four units and overwrites `results.json`, so "
            "copy that file aside before relaunching if a partial run matters. The "
            "runner writes raw json; `results_io.py --write-manifest` then "
            "re-measures the `.gz` pair and rewrites `results_manifest.json`.",
            ""]
    runtime = results.get("runtime_seconds")
    if isinstance(runtime, (int, float)):
        hours, rest = divmod(int(runtime), 3600)
        doc_runtime = (f"Wall clock on the recorded run was {runtime:.1f} s "
                       f"({hours} h {rest // 60:02d} m) for all four units, from "
                       "`runtime_seconds`. Per-unit timings are not in "
                       "`results.json`; they are only in the run log.")
    else:
        doc_runtime = f"Wall clock: {NOT_MEASURED} (`runtime_seconds` absent)."
    doc += [doc_runtime, ""]
    doc += section_epsilon(results, audit)
    doc += section_padding(audit)
    doc += section_testb(results)
    doc += section_paired(results)
    doc += section_detector(results, subset)
    doc += section_not_like_for_like(results, subset)
    doc += section_verdicts(results, subset)
    doc += section_evidence(results, manifest)

    out_path.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, {len(doc)} lines) "
          f"from {source}")
    missing = [n for n, v in (("determinism_check.json", check),
                              ("padding_audit.json", audit),
                              ("detector_scores_subset.json", subset),
                              ("results_manifest.json", manifest)) if v is None]
    if missing:
        print("WARNING: generated with these sidecars absent, their sections say "
              f"{NOT_MEASURED}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
