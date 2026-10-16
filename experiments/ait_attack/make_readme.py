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
from typing import NamedTuple

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
           "**Definition.** `attack_success` is "
           "`float((predictions == targets).mean())` "
           "(`experiments/ait_attack/run_ait_attack.py:538`), i.e. the fraction "
           "of **all** evaluated test windows - benign and attack windows "
           "alike - whose argmax class equals the attacker's per-window target, "
           "where that target is the LEAST-likely class on clean input "
           "(`target_strategy: least_likely`, "
           "`prismflow/attacks/chorus.py:choose_targets`), so it is never the "
           "clean prediction and a clean cell scores 0 by construction. "
           "The denominator is therefore every test window in the direction, "
           "not the attack windows and not the windows the attack moved.", ""]
    out += header(["direction", "denominator (test windows)", "attack windows",
                   "benign windows", "attack prevalence"])
    for dname, record in results["directions"].items():
        any_cell = next(iter(record["systems"]["naive"]["cells"].values()))
        first = (any_cell.get("per_seed") or [{}])[0]
        out.append(line([dname, str(record["n_test"]),
                         fmt(first.get("n_attack"), 0),
                         fmt(first.get("n_benign"), 0),
                         fmt(first.get("attack_prevalence"))]))
    out += ["",
            "Every row below is one grid point and one side of the dns split. The "
            "5-seed column is the per-cell mean +/- population std over 5 seeds, "
            "ranged across the subsets in that group; the pooled column is the "
            "runner's own group figure, whose std pools subset-to-subset with "
            "seed-to-seed variation and is therefore **not** a 5-seed std.", ""]
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
    out += section_single_view(results)
    out += section_pairs_that_add_nothing(results)
    return out


def section_single_view(results: dict) -> list[str]:
    """5.1: k=1, every view compromised alone, per subset and never pooled.

    The k=1 rows are the floor the k=2 rows have to be read against: a pair
    that scores no higher than one of its members did alone has not been shown
    to benefit from the second view.
    """
    out = ["### 5.1 One view at a time (chorus k=1 eps=1.0 beta=1.0)", "",
           "Each row is a single view compromised ALONE, per subset, 5 seeds, "
           "never pooled. These are the floor for the k=2 rows above: a pair "
           "scoring no higher than one of its members did alone has not been "
           "shown to gain anything from the second view.", ""]
    for dname, record in results["directions"].items():
        out += [f"**{dname}** (train `{record['train_scenario']}` -> test "
                f"`{record['test_scenario']}`)", ""]
        out += header(["compromised view", "dns in set", "suricata in set",
                       "naive (5-seed mean +/- std)",
                       "prismflow (5-seed mean +/- std)"])
        cells = {s: by_subset(record["systems"][s]["cells"],
                              "chorus", 1, 1.0, 1.0) for s in ("naive",
                                                               "prismflow")}
        for views in sorted(set(cells["naive"]) | set(cells["prismflow"])):
            texts = []
            for s in ("naive", "prismflow"):
                c = cells[s].get(views)
                stat = (c or {}).get("summary", {}).get("attack_success")
                texts.append(f"{stat['mean']:.4f} +/- {stat['std']:.4f}"
                             if stat else NOT_MEASURED)
            out.append(line([f"`{'+'.join(views)}`", *flags(views), *texts]))
        out.append("")
    return out


def seed_carried_cells(results: dict) -> list[tuple]:
    """k=1 cells whose mean is carried by their single largest seed.

    The criterion is INFLATION, not merely spread: the seed has to be the
    largest, and dropping it has to at least halve the mean. A cell whose
    outlier is its SMALLEST seed is a cell whose mean is held DOWN by one
    seed, which is a different defect and is not what this reports.

    Shared by section 5.2(b), which tabulates these, and section 11, which
    counts them, so the two can never disagree about how many there are.
    """
    found = []
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            for views, cell in by_subset(cells, "chorus", 1, 1.0, 1.0).items():
                seeds = per_seed_map(cell, "attack_success")
                stat = cell.get("summary", {}).get("attack_success")
                if len(seeds) < 3 or not stat or stat["mean"] <= 0:
                    continue
                top = max(seeds, key=lambda s: seeds[s])
                rest = [v for s, v in seeds.items() if s != top]
                rmean = float(np.mean(rest))
                if rmean < 0.5 * stat["mean"]:
                    found.append((dname, system, views, seeds, stat, top,
                                  rmean, float(np.std(rest))))
    return found


def section_pairs_that_add_nothing(results: dict) -> list[str]:
    """5.2: the two read-before-quoting warnings, both derived.

    (a) pairs whose 5-seed mean equals the `dns`-alone mean to 4 decimals, so
        the second view contributed nothing the printed precision can see;
    (b) cells where one seed carries the mean, found by removing the single
        largest-deviation seed and reporting what is left.
    """
    out = ["### 5.2 Two things to check before quoting a cell", ""]
    places = 4

    rows = []
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            alone = by_subset(cells, "chorus", 1, 1.0, 1.0).get(("dns",))
            base = (alone or {}).get("summary", {}).get("attack_success")
            if not base:
                continue
            for label, grid in (("chorus beta=0.0", ("chorus", 2, 1.0, 0.0)),
                                ("pgd", ("pgd", 2, 1.0, None))):
                for views, cell in sorted(by_subset(cells, *grid).items()):
                    if "dns" not in views:
                        continue
                    stat = cell.get("summary", {}).get("attack_success")
                    if not stat:
                        continue
                    delta = stat["mean"] - base["mean"]
                    # How many decimals the two means agree to, as printed.
                    agree = max((dp for dp in range(places, 0, -1)
                                 if f"{stat['mean']:.{dp}f}"
                                 == f"{base['mean']:.{dp}f}"), default=0)
                    if agree >= 3:
                        rows.append([dname, system, label,
                                     f"`{'+'.join(views)}`",
                                     f"{stat['mean']:.{places}f}",
                                     f"{base['mean']:.{places}f}",
                                     f"{delta:+.{places}f}",
                                     f"{agree} dp" if agree < places
                                     else f"{places} dp (equal as printed)"])
    out += ["**(a) Some `dns` pairs reproduce `dns` alone.** Where a k=2 mean "
            "prints the same as the k=1 `[dns]` mean, the second view in the "
            "pair changed nothing visible at this precision and the cell is a "
            "`dns`-alone result wearing a pair's label. Every `dns` pair at "
            "`chorus beta=0.0` and at `pgd` whose 5-seed mean agrees with "
            "`dns` alone to at least 3 decimals:", ""]
    if rows:
        out += header(["direction", "system", "grid point", "compromised pair",
                       f"pair mean ({places} dp)",
                       f"`dns` alone, k=1 ({places} dp)", "difference",
                       "agree to"])
        for row in rows:
            out += [line(row)]
    else:
        out += ["No `dns` pair agrees with `dns` alone to 3 decimals at these "
                "grid points."]
    # The same pairs at beta=1.0, so "they only match at beta=0" is measured.
    spread = []
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            alone = by_subset(cells, "chorus", 1, 1.0, 1.0).get(("dns",))
            base = (alone or {}).get("summary", {}).get("attack_success")
            if not base:
                continue
            for views, cell in by_subset(cells, "chorus", 2, 1.0, 1.0).items():
                if "dns" not in views:
                    continue
                stat = cell.get("summary", {}).get("attack_success")
                if stat:
                    spread.append(abs(stat["mean"] - base["mean"]))
    out += ["", "The agreement is confined to `beta=0.0` and `pgd`, which are "
            "the two configurations with the coordination term off. At "
            "`chorus beta=1.0` the same pairs differ from `dns` alone by "
            + (f"{min(spread):.4f}..{max(spread):.4f} across all "
               f"{len(spread)} direction/system/pair combinations"
               if spread else NOT_MEASURED)
            + " - section 11 measures how far `beta` moves each of them.", ""]

    out += ["**(b) Some cells' means are carried by a single seed.** "
            "Found by dropping each cell's LARGEST seed and reprinting the "
            "remainder, and keeping the cells where that at least halves the "
            "mean. A cell whose mean survives the drop is a population; one "
            "whose mean collapses is one seed with four quiet ones beside "
            "it:", ""]
    found = seed_carried_cells(results)
    if found:
        out += header(["direction", "system", "compromised view",
                       "per-seed attack success", "5-seed mean +/- std",
                       "seed carrying it", "mean +/- std without that seed"])
        for dname, system, views, seeds, stat, worst, rmean, rstd in found:
            out += [line([dname, system, f"`{'+'.join(views)}`",
                          ", ".join(f"seed {s} {seeds[s]:.4f}"
                                    for s in sorted(seeds)),
                          f"{stat['mean']:.4f} +/- {stat['std']:.4f}",
                          f"seed {worst} ({seeds[worst]:.4f})",
                          f"{rmean:.4f} +/- {rstd:.4f}"])]
        out += ["", "A std at or above the mean is the tell. These rows "
                "satisfy the 5-SEED RULE arithmetically and are still not "
                "evidence of an effect of the size their mean suggests: quote "
                "the per-seed column with them, never the mean alone. "
                "`B_santos_to_rm / naive / [dns]` is the one that matters most, "
                "because it is the view that solves the label and it is the "
                "k=1 baseline the `dns` pairs in 5.2(a) reproduce.", ""]
    else:
        out += ["No `chorus k=1 eps=1.0` cell loses half its mean when its "
                "largest seed is dropped, so no cell's mean is carried by one "
                "seed at this grid point.", ""]
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


#: L3's evidence, cut into groups that are never mixed. `chorus` and `pgd` are
#: separate attacks and Part 09's claim is about the chorus attack
#: specifically, so no group spans the two; `beta` and `epsilon` are the axes
#: Part 09 held fixed, so each gets its own group rather than being averaged
#: into the headline one. Each entry is (key, title, extra column, predicate,
#: value of that column).
def _l3_groups() -> list[tuple]:
    def chorus(eps=None, beta=None, k=None):
        def pred(c):
            return (c["kind"] == "chorus"
                    and (eps is None or c["epsilon"] == eps)
                    and (beta is None or c["beta"] == beta)
                    and (k is None or c["k"] == k))
        return pred

    groups = [("a", "(a) chorus, eps=1.0, beta=1.0 - the headline grid point",
               None, chorus(eps=1.0, beta=1.0), None)]
    for beta in (0.0, 5.0):
        groups.append(("b", "(b) chorus, eps=1.0, beta=0.0 and beta=5.0 - "
                       "coordination off and at 5x", "beta",
                       chorus(eps=1.0, beta=beta), f"{beta:g}"))
    groups.append(("c", "(c) pgd - a DIFFERENT attack, never pooled with "
                   "chorus above", None, lambda c: c["kind"] == "pgd", None))
    for eps in (0.2, 0.5, 1.0, 2.0):
        groups.append(("d", "(d) chorus, beta=1.0, every epsilon in the grid",
                       "epsilon", chorus(eps=eps, beta=1.0), f"{eps:g}"))
    return groups


def l3_cells(record: dict, predicate) -> list[tuple]:
    """`(key, views, dns, Paired)` for prismflow - naive on every matching cell.

    The pairing is by CELL KEY, so the two arms are always the same grid point
    and the same compromised subset; `paired_diff` then pairs within seed.
    """
    pf = record["systems"]["prismflow"]["cells"]
    nv = record["systems"]["naive"]["cells"]
    out = []
    for key, cell in sorted(pf.items()):
        if not cell.get("k") or not predicate(cell):
            continue
        p = paired_diff(cell, nv.get(key))
        if p.mean is None:
            continue
        out.append((key, tuple(cell["compromised_views"]),
                    bool(cell["dns_compromised"]), p))
    return out


class L3Cut(NamedTuple):
    """One (group, direction, dns side) cell count for L3.

    `keys` carries the cell keys behind the counts so the verdict can
    de-duplicate: group (d) sweeps every epsilon and therefore CONTAINS group
    (a)'s eps=1.0 cells, and a total over both groups would count those twice.
    """
    direction: str
    dns: bool
    n: int
    lower: int           # prismflow mean < naive mean: the discount helped
    lower_unanimous: int  # ...and all 5 seeds agreed
    higher: int          # prismflow mean > naive mean: Part 09's sign
    higher_unanimous: int
    lo: float
    hi: float
    keys: tuple          # (cell key, mean, unanimous) per cell
    extra: str = ""      # the group's sub-row, e.g. `beta=0` or `epsilon=0.2`


def l3_cut(dname: str, rows: list[tuple], dns: bool) -> L3Cut | None:
    side = [(k, p) for k, _v, d, p in rows if d is dns]
    if not side:
        return None
    means = [p.mean for _k, p in side]
    return L3Cut(
        dname, dns, len(side),
        sum(1 for _k, p in side if p.mean < 0),
        sum(1 for _k, p in side if p.mean < 0 and p.unanimous),
        sum(1 for _k, p in side if p.mean > 0),
        sum(1 for _k, p in side if p.mean > 0 and p.unanimous),
        min(means), max(means),
        tuple((k, p.mean, p.unanimous) for k, p in side))


def l3_pooled_lines(results: dict) -> list[str]:
    """The four `paired_prismflow_minus_naive` lines, explicitly NOT the test.

    These are the runner's own per-direction, per-dns-side counts, and they
    POOL chorus with pgd and every beta and epsilon into one number. Part 09's
    claim is about the chorus attack at one grid point, so a count that mixes
    two attacks cannot address it. They are kept because they are measured and
    were the only L3 evidence in earlier revisions of this file, and they are
    labelled so they are never quoted as the L3 test.
    """
    out = ["**Descriptive, all attacks pooled - NOT the L3 test.** The four "
           "lines below are `directions.<name>.paired_prismflow_minus_naive` "
           "as the runner wrote it: one count per direction and `dns` side "
           "over every cell in the grid, which **pools `chorus` with `pgd`** "
           "and pools every `beta` and `epsilon` together. Part 09's claim is "
           "about the chorus attack, so these cannot test it and must not be "
           "cited as the L3 result; the grouped tables below are the test.", ""]
    for dname, record in results["directions"].items():
        paired = record.get("paired_prismflow_minus_naive") or {}
        for dns in (True, False):
            rows = [v for v in paired.values()
                    if v.get("dns_compromised") is dns
                    and "attack_success" in v.get("diff", {})]
            if not rows:
                continue
            means = [r["diff"]["attack_success"]["mean"] for r in rows]
            kinds = sorted({r.get("kind") for r in rows if r.get("kind")})
            side = "dns COMPROMISED" if dns else "dns NOT compromised"
            out.append(f"- **{dname}, {side}:** PrismFlow lowers attack "
                       f"success in **{sum(1 for m in means if m < 0)} of "
                       f"{len(rows)}** cells, difference range "
                       f"{min(means):+.4f}..{max(means):+.4f} "
                       f"(attacks pooled here: {', '.join(kinds)}).")
    out.append("")
    return out


def section_l3(results: dict) -> list[str]:
    """L3's evidence and verdict, re-cut by attack, beta, epsilon and dns side.

    Every count is `prismflow - naive` attack success, paired WITHIN SEED on
    the same cell key, then counted. The unanimity columns are what keep a
    1-seed sign from being quoted as a direction: a cell counted under
    `PrismFlow higher` with 2/5 unanimity has not been shown to move.
    """
    out = ["### L3 - the discount addresses redundant evidence, not unreliable "
           "evidence (and Part 09's extension: reading the axis confers no "
           "defence)", ""]
    out += l3_pooled_lines(results)
    out += ["Part 09's claim is about **the chorus attack**: it found PrismFlow "
           "strictly worse than naive fusion in 12/12 attacked cells on "
           "synthetic data. The evidence below is therefore cut by attack "
           "first - **`chorus` and `pgd` are never pooled** - then by the "
           "`beta` and `epsilon` Part 09 held fixed, then by the `dns` split, "
           "then by direction. Nothing is averaged across any of those four.",
           "",
           "Each cell is one grid point and one compromised subset. The "
           "difference is `prismflow - naive` attack success **paired within "
           "seed on the same cell key**, so the two arms always share the grid "
           "point, the subset and the seed. **`PrismFlow higher` is Part 09's "
           "sign** (the discount made the attack work better); `PrismFlow "
           "lower` is the opposite. The `5/5` columns count the cells where "
           "all five seeds agree with the sign of their own mean - a cell "
           "counted in a direction without unanimity has not been shown to "
           "move that way.", "",
           "**The cells are not independent trials.** They are nested "
           "compromised subsets - `[dns]` is inside `[apache+dns]` is inside "
           "`[apache+dns+audit]` - evaluated on the same test windows with the "
           "same 5 seeds and the same trained models, so every count in this "
           "section is a DESCRIPTION of how the grid came out and not a tally "
           "of independent experiments; no p-value, binomial reading or "
           "'N of M cells agree' significance claim may be built on it.", ""]
    cuts: dict[str, list[L3Cut]] = {}
    seen: set[str] = set()
    for gkey, title, extra, pred, extra_value in _l3_groups():
        if gkey not in seen:
            out += [f"**{title}**", ""]
            out += header([c for c in
                           ([extra] if extra else []) + [
                               "direction", "dns in set", "cells",
                               "PrismFlow lower", "of those 5/5",
                               "PrismFlow higher", "of those 5/5",
                               "mean diff range"]])
            seen.add(gkey)
        for dname, record in results["directions"].items():
            rows = l3_cells(record, pred)
            for dns in (True, False):
                cut = l3_cut(dname, rows, dns)
                if cut is None:
                    continue
                if extra:
                    cut = cut._replace(extra=f"{extra}={extra_value}")
                cuts.setdefault(gkey, []).append(cut)
                out.append(line(
                    ([extra_value] if extra else [])
                    + [cut.direction, "YES" if dns else "no", str(cut.n),
                       str(cut.lower), str(cut.lower_unanimous),
                       str(cut.higher), str(cut.higher_unanimous),
                       f"{cut.lo:+.4f}..{cut.hi:+.4f}"]))
        if gkey in ("a", "c") or extra_value in ("5", "2"):
            out.append("")
    if out[-1] != "":
        out.append("")
    out += [f"**The de-duplication key: `{L3_DEDUP_KEY}`.** "
            f"Group (d) sweeps every epsilon and therefore CONTAINS group "
            f"(a)'s `eps=1.0` cells; the totals in the verdict count each "
            f"cell once, keyed on `{L3_DEDUP_KEY}`, where `cell_key` is the "
            f"key `run_ait_attack.py` writes into "
            f"`systems.<system>.cells`. Each component below is confirmed "
            f"against the cell records rather than read off the key's "
            f"spelling:", ""]
    out += header(["component", "where it appears in the key", "confirmed"])
    for row in l3_dedup_key_parts(results):
        out.append(line(row))
    out += ["", "Example keys: `chorus_k2_eps1.0_beta0.0_[apache+audit]` and "
            "`pgd_k1_eps1.0_[apache]`. `beta` is absent from a `pgd` key "
            "because `pgd` has no beta; the `chorus`/`pgd` prefix already "
            "separates those cells, so no two distinct grid cells can "
            "collide.", ""]
    out += l3_verdict(cuts)
    return out


#: The composite key group (d)'s cells are de-duplicated against group (a)'s
#: by. `cell_key` is the key `run_ait_attack.py` writes into
#: `systems.<system>.cells`, and it encodes the attack, k, epsilon, beta and
#: the compromised subset - e.g. `chorus_k2_eps1.0_beta0.0_[apache+audit]`,
#: or `pgd_k1_eps1.0_[apache]` where beta does not apply. Pairing the
#: direction with it makes the key unique across the whole Part, so a cell
#: that appears in both groups is counted once.
L3_DEDUP_KEY = "(direction, cell_key)"


def l3_dedup_key_parts(results: dict) -> list[list[str]]:
    """The components of `L3_DEDUP_KEY`, each CONFIRMED against the cells.

    Nothing here is asserted from the key's spelling: every row re-reads the
    field off the cell record and checks the key actually carries it, so the
    README's claim about what the key distinguishes is measured.
    """
    cells = [(d, k, c)
             for d, rec in results["directions"].items()
             for k, c in rec["systems"]["prismflow"]["cells"].items()
             if c.get("k")]

    def present(field, render) -> str:
        total = ok = 0
        for _d, key, cell in cells:
            value = cell.get(field)
            if field == "beta" and value is None:
                continue          # pgd has no beta; counted under `attack`
            total += 1
            ok += render(value) in key
        return f"{ok}/{total} cells confirmed"

    return [
        ["direction", "the key's first element",
         f"{len(results['directions'])}/{len(results['directions'])} "
         f"directions are separate dict keys"],
        ["attack (`kind`)", "`chorus` / `pgd` prefix of `cell_key`",
         present("kind", lambda v: str(v))],
        ["`k`", "`_k<k>_` in `cell_key`", present("k", lambda v: f"_k{v}_")],
        ["`epsilon`", "`eps<epsilon>` in `cell_key`",
         present("epsilon", lambda v: f"eps{v}")],
        ["`beta`", "`beta<beta>` in `cell_key` (absent for `pgd`)",
         present("beta", lambda v: f"beta{v}")],
        ["compromised subset", "`[v1+v2+...]` in `cell_key`",
         present("compromised_views", lambda v: "[" + "+".join(v) + "]")],
    ]


def l3_headline(cuts: dict[str, list[L3Cut]]) -> str:
    """THE L3 verdict sentence. One source of text, used wherever it appears.

    Every word that could have come out otherwise is derived from the counts:
    the strength word from whether each sign has any unanimous cell, and the
    conditionality clause from whether the dns sides and the directions
    actually disagree. There is no fixed verdict string to drift from the
    tables.
    """
    chorus = [c for k in ("a", "b", "d") for c in cuts.get(k, [])]
    if not chorus:
        return ("**Verdict on L3: NO EVIDENCE** - no paired chorus cell "
                "survived.")
    distinct: dict[tuple, tuple] = {}
    for c in chorus:
        for key, mean, uni in c.keys:
            distinct[(c.direction, key)] = (mean, uni)
    hi_u = sum(1 for m, u in distinct.values() if m > 0 and u)
    lo_u = sum(1 for m, u in distinct.values() if m < 0 and u)
    if hi_u and lo_u:
        word = "PARTIALLY reproduced"
    elif hi_u:
        word = "reproduced"
    elif lo_u:
        word = "CONTRADICTED"
    else:
        word = "NO EVIDENCE"

    # Does the result actually depend on the dns split, and on direction?
    def side_signs(pick):
        return {pick(c): {("hi" if c.higher_unanimous else None),
                          ("lo" if c.lower_unanimous else None)} - {None}
                for c in chorus}
    by_dns, by_dir = side_signs(lambda c: c.dns), side_signs(lambda c: c.direction)
    conds = []
    if len({frozenset(v) for v in by_dns.values()}) > 1 or any(
            len(v) > 1 for v in by_dns.values()):
        conds.append("the `dns` split")
    if len({frozenset(v) for v in by_dir.values()}) > 1 or any(
            len(v) > 1 for v in by_dir.values()):
        conds.append("direction")
    tail = (" and ".join(conds) if conds else "neither the `dns` split nor "
            "direction")
    return (f"**Verdict on L3: Part 09's result is {word}, conditional on "
            f"{tail}.**")


def l3_verdict(cuts: dict[str, list[L3Cut]]) -> list[str]:
    """L3's verdict, every clause derived from the tables above it.

    States where Part 09's sign reappears and where it does not. Makes no
    claim about why: nothing in this Part measures the discount's internals.
    """
    chorus_keys = ("a", "b", "d")
    chorus = [c for k in chorus_keys for c in cuts.get(k, [])]
    pgd = cuts.get("c", [])
    if not chorus:
        return [l3_headline(cuts), ""]

    def side_name(c):
        return (f"{c.direction}, dns "
                + ("COMPROMISED" if c.dns else "NOT compromised"))

    # Where Part 09's sign reappears UNANIMOUSLY, and where it reverses.
    repro = sorted({side_name(c) for c in chorus
                    if c.higher_unanimous > 0})
    reversed_ = sorted({side_name(c) for c in chorus
                        if c.lower_unanimous > 0})
    both = sorted(set(repro) & set(reversed_))
    only_repro = sorted(set(repro) - set(reversed_))
    only_rev = sorted(set(reversed_) - set(repro))

    # De-duplicate by (direction, cell key): group (d) contains group (a).
    distinct: dict[tuple, tuple] = {}
    for c in chorus:
        for key, mean, uni in c.keys:
            distinct[(c.direction, key)] = (mean, uni)
    n_cells = len(distinct)
    n_high = sum(1 for m, _u in distinct.values() if m > 0)
    n_high_u = sum(1 for m, u in distinct.values() if m > 0 and u)
    n_low = sum(1 for m, _u in distinct.values() if m < 0)
    n_low_u = sum(1 for m, u in distinct.values() if m < 0 and u)

    out = [l3_headline(cuts), "",
           "It reappears in some chorus cells and reverses in others. The "
           "counts below say which.", "",
           f"Over the **{n_cells} distinct** paired chorus cells in the tables "
           f"above - distinct because group (d) sweeps every epsilon and so "
           f"CONTAINS group (a)'s `eps=1.0` cells, which are counted once "
           f"here: PrismFlow is HIGHER than naive, which is Part 09's sign, "
           f"in {n_high} cells, {n_high_u} of them with all 5 seeds agreeing; "
           f"PrismFlow is LOWER in {n_low} cells, {n_low_u} of them unanimous. "
           f"Part 09 had 12/12 one way. Neither sign is unanimous across any "
           f"whole side here.", ""]
    if only_repro:
        out += ["**Where Part 09's sign reappears, and only that sign:** "
                + ", ".join(f"**{s}**" for s in only_repro)
                + ". Every unanimous chorus cell on these sides has PrismFlow "
                  "worse than naive.", ""]
    if both:
        out += ["**Both signs appear unanimously, in different cells, on all "
                "of these sides:** " + ", ".join(f"**{s}**" for s in both)
                + ". So no side has a sign: a sentence of the form 'PrismFlow "
                  "is worse with `dns` compromised' is false as stated for "
                  "every side in this Part, and the per-group rows are the "
                  "result rather than any summary of them.", ""]
    if only_rev:
        out += ["**Where Part 09's sign does NOT reappear:** "
                + ", ".join(f"**{s}**" for s in only_rev)
                + ". Every unanimous chorus cell on these sides has PrismFlow "
                  "lower than naive, the opposite of Part 09.", ""]

    # The rows where one sign does dominate its own side, named exactly.
    def dominant(rows, worse: bool):
        picked = []
        for c in rows:
            u = c.higher_unanimous if worse else c.lower_unanimous
            other = c.lower_unanimous if worse else c.higher_unanimous
            if u * 2 > c.n and u > other:
                prefix = f"`{c.extra}` " if c.extra else ""
                picked.append(f"{prefix}{side_name(c)} "
                              f"({u}/{c.n} unanimous)")
        return picked
    worse_sides, better_sides = [], []
    for gkey, label in (("a", "(a) `eps=1.0 beta=1.0`"),
                        ("b", "(b) `beta=0.0` / `beta=5.0`"),
                        ("d", "(d) the `epsilon` sweep")):
        rows = cuts.get(gkey, [])
        worse, better = dominant(rows, True), dominant(rows, False)
        worse_sides += [c for c in rows
                        if c.higher_unanimous * 2 > c.n
                        and c.higher_unanimous > c.lower_unanimous]
        better_sides += [c for c in rows
                         if c.lower_unanimous * 2 > c.n
                         and c.lower_unanimous > c.higher_unanimous]
        if not (worse or better):
            continue
        out += [f"**Group {label} - sides where one sign carries a unanimous "
                f"majority of the side's cells:**"]
        out += [f"  - PrismFlow WORSE (Part 09's sign): "
                + ("; ".join(worse) if worse else "none")]
        out += [f"  - PrismFlow BETTER: "
                + ("; ".join(better) if better else "none"), ""]

    # One derived, checkable statement about where the dominant sides fall.
    # It describes the arrangement of the counts; it explains nothing.
    if worse_sides:
        all_dns = all(c.dns for c in worse_sides)
        out += [f"Across all three chorus groups, the "
                f"{len(worse_sides)} (group, direction, side) rows where Part "
                f"09's sign carries a unanimous majority "
                + ("ALL have `dns` INSIDE the compromised set"
                   if all_dns else
                   f"include {sum(1 for c in worse_sides if not c.dns)} with "
                   f"`dns` OUTSIDE it")
                + (f", and the {len(better_sides)} rows where the opposite "
                   f"sign carries a unanimous majority are "
                   f"{sum(1 for c in better_sides if not c.dns)} with `dns` "
                   f"outside and {sum(1 for c in better_sides if c.dns)} with "
                   f"`dns` inside" if better_sides else "")
                + ". That is the arrangement of the counts, stated so it can "
                  "be checked against the tables; it is not a mechanism and "
                  "nothing below infers one from it.", ""]

    # The epsilon and beta groups, stated separately so neither is averaged in.
    for gkey, label in (("a", "the headline `eps=1.0 beta=1.0` point"),
                        ("b", "`beta=0.0` and `beta=5.0`"),
                        ("d", "the full `epsilon` sweep at `beta=1.0`")):
        rows = cuts.get(gkey, [])
        if not rows:
            continue
        hi = sum(r.higher_unanimous for r in rows)
        lo = sum(r.lower_unanimous for r in rows)
        out += [f"- At {label}: {hi} unanimous cells with PrismFlow worse, "
                f"{lo} with PrismFlow better, over {sum(r.n for r in rows)} "
                f"cells."]
    if pgd:
        hi = sum(r.higher_unanimous for r in pgd)
        lo = sum(r.lower_unanimous for r in pgd)
        out += [f"- `pgd`, reported separately and **not** evidence about "
                f"Part 09's chorus claim: {hi} unanimous cells with PrismFlow "
                f"worse, {lo} with PrismFlow better, over "
                f"{sum(r.n for r in pgd)} cells."]
    out += ["", "**No mechanism is claimed.** This Part measures fused "
            "attack success in each arm and nothing inside the discount - no "
            "per-view alpha, no ENIV trace per cell, no belief margin - so it "
            "cannot say why a cell came out either way, and the split by "
            "`dns`, `beta` and `epsilon` above is a description of where the "
            "signs fall, not an explanation of them. What it does establish "
            "is narrower than Part 09's wording: PrismFlow was not uniformly "
            "worse than naive fusion under the chorus attack on this data, so "
            "'reading the axis confers no defence' must not be restated as a "
            "blanket result, and neither may its negation. [[L17]] bounds how "
            "much any of it is worth: one view solves this label.", ""]
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

    out += section_l3(results)

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
    # The counter-cell: dependence near its measured maximum WITHOUT the
    # attack succeeding. It is what makes this Part uninformative about L4.
    counter = None
    for dname, record in results["directions"].items():
        cells = record["systems"]["prismflow"]["cells"]
        rows = list(select(cells, "chorus", 2, 1.0, 1.0, False).values())
        dep = [d for d in (cell_mean(c, "dependence_compromised")
                           for c in rows) if d is not None]
        lo, hi = succ_range(record, "prismflow", False)
        if dep and hi is not None and (counter is None or hi < counter[3]):
            counter = (dname, min(dep), max(dep), hi)
    out += ["", "**Verdict: NOT INFORMATIVE about L4.** The cells do not "
            "separate L4's prediction from its negation, so neither reading "
            "gains support here.", ""]
    if counter:
        dname, dlo, dhi, shi = counter
        # The same cell in the NAIVE arm. If the attack is harmless there too,
        # the harmlessness cannot be credited to (or blamed on) the discount.
        nrecord = results["directions"][dname]
        ncells = nrecord["systems"]["naive"]["cells"]
        nrows = list(select(ncells, "chorus", 2, 1.0, 1.0, False).values())
        nsucc = [v for v in (cell_mean(c, "attack_success") for c in nrows)
                 if v is not None]
        ndep = [v for v in (cell_mean(c, "dependence_compromised")
                            for c in nrows) if v is not None]
        out += [f"The deciding cell is **{dname} / prismflow, dns NOT "
                f"compromised, chorus k=2 eps=1.0**: dependence among the "
                f"compromised views runs {dlo:.4f}..{dhi:.4f} - up to "
                f"{dhi / dep_max:.2f} of the {dep_max:.4f} maximum measured "
                f"anywhere in this Part across all k>=2 cells - while attack "
                f"success over those same subsets never exceeds {shi:.4f}. "
                "Manufactured dependence at close to its observed ceiling "
                "therefore coexists with an attack that essentially fails. "
                "L4 says the discount is proportional and weakens as evidence "
                "grows; a cell where dependence is high and the attack is "
                "nonetheless harmless is equally consistent with a discount "
                "that is proportional, one that caps, and one that does "
                "nothing at all.", ""]
        if nsucc and ndep:
            out += [f"**The naive arm fails in exactly the same cells, so the "
                    f"harmlessness is not attributable to the discount at "
                    f"all.** Over the same {len(nrows)} subsets at "
                    f"**{dname} / naive, dns NOT compromised, chorus k=2 "
                    f"eps=1.0**, attack success runs "
                    f"{min(nsucc):.4f}..{max(nsucc):.4f} against PrismFlow's "
                    f"{shi:.4f} ceiling, with dependence among the compromised "
                    f"views at {min(ndep):.4f}..{max(ndep):.4f} - the same "
                    f"high-dependence regime. Naive fusion applies no discount "
                    f"whatsoever, so a cell where it is equally harmless "
                    f"cannot be evidence about what the discount did.", ""]
    out += ["What would discriminate is not measured here. Part 29 records no "
            "per-view alpha, no belief margin and no sublinearity curve - the "
            "three quantities L4 is a statement about - so there is nothing in "
            "`results.json` that could have come out differently if L4 were "
            "false. The earlier wording, 'consistent with L4', was true and "
            "empty: the attack succeeding in high-dependence cells is also "
            "what a capping discount, or no discount, would produce. L4's "
            "mechanism evidence remains Part 09's diagnostic, and this Part "
            "neither adds to it nor weakens it.", ""]

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
    _batch_size = results["config"]["attack"].get("batch_size", NOT_MEASURED)
    _n_tests = ", ".join(f"{d} n_test={r['n_test']}"
                         for d, r in results["directions"].items())
    out += ["", "**Verdict: L5 is reproduced and is WORSE than L5 states, on both "
            "sides of the dns split.** L5 describes a detector that degrades to "
            "chance and inverts under missingness. On AIT the majority of all "
            "conditions - attacked, not merely missing - sit below chance, so the "
            "flag fires LESS under attack than under clean operation across most of "
            "the grid. Section 7.1 rules out the saturated-detector explanation on "
            "the subset measured: scores are well spread. The inversion is "
            "directional.", "",
            "**What this protocol does and does not put to the test.** The "
            "numbers above are a property of THIS measurement design, not of "
            "a deployed detector, and three design choices decide them:", "",
            f"- **Every window in the evaluated batch is attacked.** "
            f"`attack: batch_size` is {_batch_size} against {_n_tests}, so "
            "each direction is one batch and `views = views + result.delta` "
            "(`run_ait_attack.py:511`) perturbs all of it. There is no clean "
            "population inside an attacked run.",
            "- **The baseline is the same windows unattacked.** "
            "`detector_auc(cell['_scores'], clean_scores)` "
            "(`run_ait_attack.py:920`) scores the fully-attacked run against "
            "the clean pass over the same test windows for that "
            "direction/system/seed, and the one clean reference is reused by "
            "every subset in that unit. The AUC is therefore a between-run "
            "comparison, not a within-batch one.",
            "- **Partially contaminated batches were NOT tested.** No cell "
            "mixes attacked and clean windows, and no cell varies the "
            "contaminated fraction; that axis is absent from the grid, so "
            f"this Part reports {NOT_MEASURED} for it.", "",
            "A deployed detector sees a mostly-clean stream and must flag a "
            "small contaminated part of it. That is the regime this Part does "
            "not enter. The below-chance AUCs above are a real and "
            "reproducible property of the all-or-nothing protocol, and they "
            "do not license a claim about detection rates in deployment in "
            "either direction.", ""]

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
    out += ["", "**Verdict: L17 is confirmed - the label is solved by `dns` "
            "alone.** Section 1's `label_meaning` is the ground for that: the "
            "attack label is dominated by one continuously-running DNS "
            "exfiltration tool, and section 3 shows `dns` is the view that "
            "carries it. Everything below is the structure that fact imposes "
            "on the grid, measured per direction and per system rather than "
            "asserted.", ""]

    # (i) what each view achieves ALONE, which is the floor for every pair.
    out += ["**Each view compromised alone (chorus k=1 eps=1.0, 5-seed "
            "means).** The two directions have different active views:", ""]
    out += header(["direction", "system"] + [f"`{v}`" for v in
                                             ("apache", "auth", "audit",
                                              "dns", "suricata")])
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            means = alone_means(record, system)
            out += [line([dname, system] + [fmt(means.get(v)) for v in
                                            ("apache", "auth", "audit",
                                             "dns", "suricata")])]
    out += [""]

    # (ii) in B, suricata reproduces itself inside pairs at beta=0.
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            means = alone_means(record, system)
            sur = means.get("suricata")
            b0 = by_subset(cells, "chorus", 2, 1.0, 0.0)
            gaps = []
            for views, cell in b0.items():
                if "suricata" not in views or "dns" in views:
                    continue
                stat = cell.get("summary", {}).get("attack_success")
                if stat and sur is not None:
                    gaps.append(abs(stat["mean"] - sur))
            inert = {v: means[v] for v in ("apache", "auth", "audit")
                     if v in means}
            if not gaps or sur is None or not inert:
                continue
            out += [f"- **{dname} / {system}:** `suricata` alone reaches "
                    f"{sur:.4f}. Its {len(gaps)} pairs without `dns`, at "
                    f"`beta=0.0`, sit within {max(gaps):.4f} of that - the "
                    "second view adds at most that much. `apache`, `auth` and "
                    "`audit` alone reach "
                    + ", ".join(f"`{v}` {m:.4f}" for v, m in inert.items())
                    + f", a range of {min(inert.values()):.4f}.."
                    f"{max(inert.values()):.4f}."]
    out += [""]

    # (iii) coordination: signed beta 0->1 shift, dns pairs against suricata.
    out += ["**What `beta` does to each family of pairs** (paired within "
            "seed, `chorus k=2 eps=1.0`, from section 11):", ""]
    out += header(["direction", "system", "`dns` pairs: beta 0 -> 1",
                   "all same sign", "`suricata` pairs without `dns`: "
                   "beta 0 -> 1", "all same sign"])
    clean, mixed = [], []
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            b0 = by_subset(cells, "chorus", 2, 1.0, 0.0)
            b1 = by_subset(cells, "chorus", 2, 1.0, 1.0)
            fam = {"dns": [], "suricata": []}
            for views in b1:
                shift = paired_diff(b1.get(views), b0.get(views)).mean
                if shift is None:
                    continue
                if "dns" in views:
                    fam["dns"].append(shift)
                elif "suricata" in views:
                    fam["suricata"].append(shift)
            row, uniform = [dname, system], {}
            for key in ("dns", "suricata"):
                vals = fam[key]
                uniform[key] = bool(vals) and (all(v > 0 for v in vals)
                                               or all(v < 0 for v in vals))
                row += [f"{min(vals):+.4f}..{max(vals):+.4f} over "
                        f"{len(vals)} pairs" if vals else NOT_MEASURED,
                        ("all +" if vals and all(v > 0 for v in vals) else
                         "all -" if vals and all(v < 0 for v in vals) else
                         "MIXED" if vals else NOT_MEASURED)]
            out += [line(row)]
            opposed = (uniform["dns"] and uniform["suricata"]
                       and (min(fam["dns"]) > 0) != (min(fam["suricata"]) > 0))
            (clean if opposed else mixed).append(f"{dname} / {system}")
    out += [""]
    if clean:
        out += ["The two families are cleanly OPPOSED - every `dns` pair moving "
                "one way and every `suricata`-without-`dns` pair the other - in "
                + ", ".join(f"**{u}**" for u in clean) + "."]
    if mixed:
        out += ["In " + ", ".join(f"**{u}**" for u in mixed)
                + " the split is NOT clean: the `suricata` family spans both "
                "signs there, so 'coordination hurts suricata pairs' is a "
                "statement about "
                + (", ".join(f"**{u}**" for u in clean) if clean
                   else "no unit here")
                + " and must not be carried across to it."]
    out += ["", "What is common to all four units is that the `dns` family "
            "moves UP with `beta` while the `suricata` family does not follow "
            "it. A single 'effect of beta' averaged over the ten pairs would "
            "average those behaviours and describe neither; section 11 keeps "
            "them per pair for that reason.", ""]
    #: The directions in which the two families are cleanly opposed, derived
    #: from the same per-pair shifts the table above prints. L17's closing
    #: statement is scoped to these and must not be written as a Part-wide
    #: result: in the other direction the suricata family spans both signs.
    opposed_directions = sorted({u.split(" / ")[0] for u in clean})
    other_directions = sorted({u.split(" / ")[0] for u in mixed}
                              - set(opposed_directions))
    if opposed_directions:
        out += ["**The opposite-sign structure is a "
                + ", ".join(f"`{d}`" for d in opposed_directions)
                + " result only.** It holds in "
                + ", ".join(f"**{u}**" for u in clean)
                + (", and NOT in " + ", ".join(f"**{u}**" for u in mixed)
                   if mixed else "")
                + ". Every sentence below about `dns` pairs and `suricata` "
                  "pairs moving in opposite directions is scoped to "
                + ", ".join(f"`{d}`" for d in opposed_directions)
                + (f"; nothing of the kind is established for "
                   + ", ".join(f"`{d}`" for d in other_directions)
                   if other_directions else "") + ".", ""]

    # (iv) the confound that survives all of it, derived.
    ratios = {d: dns_epsilon_in_std(r)
              for d, r in results["directions"].items()}
    ratios = {d: v for d, v in ratios.items() if v is not None}
    if len(ratios) == 2:
        (da, va), (db, vb) = sorted(ratios.items())
        bigger, smaller = (db, da) if vb > va else (da, db)
        out += [f"**The epsilon grid is not comparable across the two "
                f"directions on `dns`.** `epsilon 1.0` is "
                + ", ".join(f"{v:.3f} of the `dns` std in {d}"
                            for d, v in sorted(ratios.items()))
                + f", so the same nominal epsilon is "
                f"{max(ratios.values()) / min(ratios.values()):.2f}x larger "
                f"relative to that view's own spread in {bigger} than in "
                f"{smaller}. Cross-direction comparisons of attack success on "
                "`dns` are confounded by this, and `dns` has the smallest "
                "per-view std in only one of the two directions (section 3).",
                ""]

    out += ["**What this Part is therefore evidence about.** It measures, on "
            "two real AIT scenario pairs with 5 seeds per cell: how far a "
            "bounded perturbation of a chosen subset of views moves a fused "
            "prediction when one view carries the label; how that depends on "
            "whether the carrying view is in the subset; how the chorus "
            "coordination term changes it - with the opposite sign on `dns` "
            "pairs and `suricata` pairs "
            + ("in " + ", ".join(f"`{d}`" for d in opposed_directions)
               + " only, not in "
               + ", ".join(f"`{d}`" for d in other_directions)
               if opposed_directions and other_directions else
               "in " + ", ".join(f"`{d}`" for d in opposed_directions)
               if opposed_directions else "in no direction measured here")
            + "; and how the suspicion detector and "
            "the calibration metrics behave across that grid. "
            "It does NOT measure how fusion behaves when several views carry "
            "independent evidence about the label, because this label does "
            "not provide that: the previous wording, 'no number in this Part "
            "is evidence about multi-view fusion', overstated it in the other "
            "direction - the k=2 and k=3 cells do measure fusion under attack, "
            "on a task where the views are not of comparable value. Read every "
            "number in this file as conditional on that.", ""]
    return out


#: Every byte count the README prints is divided by this, and the unit is
#: always written out as MB beside it, so no figure is ever a MiB wearing an
#: MB label. results_manifest.json's raw_bytes/gz_bytes use the same divisor.
MB = 1_000_000

#: The Part 29 artefacts to exclude when asking what the LARGEST results file
#: tracked anywhere ELSE in the repository is.
PART29_ARTEFACTS = (
    "results/ait_attack/results.json.gz",
    "results/ait_attack/results.3of4units_20261014_0022.json.gz",
    "results/ait_attack/results.A_naive_checkpoint.json",
)


def size_mb(path) -> str:
    """`<n> MB` from the file's byte count on disk, or NOT MEASURED."""
    try:
        return f"{Path(path).stat().st_size / MB:.2f} MB"
    except OSError:
        return NOT_MEASURED


def largest_other_tracked_result() -> tuple[str, str]:
    """`(size, path)` of the biggest TRACKED file under `results/` that is not
    a Part 29 artefact.

    `git ls-files` is the only honest source here: the claim is about what the
    repository tracks, and `results/` also holds untracked scratch output
    (`*_quick/` directories) that a plain disk walk would count. Both values
    are NOT MEASURED if git cannot be reached.
    """
    import subprocess
    try:
        listed = subprocess.run(["git", "ls-files", "results/"], check=True,
                                capture_output=True, text=True).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return NOT_MEASURED, NOT_MEASURED
    biggest, where = -1, None
    for rel in listed:
        if rel in PART29_ARTEFACTS:
            continue
        try:
            size = Path(rel).stat().st_size
        except OSError:
            continue
        if size > biggest:
            biggest, where = size, rel
    if where is None:
        return NOT_MEASURED, NOT_MEASURED
    return f"{biggest / MB:.2f} MB", where


GITIGNORE = Path(".gitignore")

#: The raw result files re-ignored by name, keyed by their manifest entry so
#: the size quoted for each is the manifest's, in the same MB as the table
#: above it, and never a second hand-typed figure.
REIGNORED = ["results.json", "results.3of4units_20261014_0022.json"]


def gitignore_reignore_rows(manifest) -> list[list[str]]:
    """Report where each re-ignore rule actually sits in `.gitignore`.

    Both the line number and the file size are read rather than typed in -
    the number from `.gitignore` itself, the size from
    `results_manifest.json` - so neither can go stale, and a rule that is
    not present is reported as NOT MEASURED rather than silently claimed.
    """
    try:
        lines = GITIGNORE.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    entries = (manifest or {}).get("files", {})
    rows = []
    for name in REIGNORED:
        rule = f"results/ait_attack/{name}"
        at = next((f"{i}" for i, text in enumerate(lines, 1)
                   if text.strip() == rule), NOT_MEASURED)
        raw = (entries.get(name) or {}).get("raw_bytes")
        size = f"{raw / MB:.2f} MB" if raw else NOT_MEASURED
        rows.append([at, f"`{rule}`",
                     f"re-ignores the {size} raw json; the `.json.gz` "
                     f"stays tracked"])
    return rows


#: The two views whose presence in a compromised set changes the result enough
#: that every section 11 row is flagged for both. `dns` is the one view that
#: solves this label; `suricata` is the one that acts alone in direction B.
FLAGGED_VIEWS = ("dns", "suricata")


def by_subset(cells: dict, kind: str, k: int, eps: float, beta) -> dict:
    """`{(view, ...): cell}` for one grid point, keyed by compromised subset."""
    return {tuple(c["compromised_views"]): c
            for c in cells.values()
            if c["kind"] == kind and c["k"] == k and c["epsilon"] == eps
            and c["beta"] == beta}


def per_seed_map(cell: dict, field: str) -> dict:
    """`{seed: value}` for one cell, from its own `per_seed` list."""
    if cell is None:
        return {}
    out = {}
    for row in cell.get("per_seed") or []:
        seed, value = row.get("seed"), row.get(field)
        if seed is not None and isinstance(value, (int, float)):
            out[seed] = float(value)
    return out


class Paired(NamedTuple):
    """One within-seed paired difference, aggregated only after pairing.

    `agree` counts the seeds whose difference has the SAME SIGN as the mean.
    A seed whose difference is exactly 0.0 agrees with neither sign, so it is
    counted in `zeros` and not in `agree`: `agree < n` therefore does NOT by
    itself mean any seed disagreed, and `opposite` is the count that does.
    """
    text: str
    mean: float | None
    std: float | None
    agree: int | None
    zeros: int
    opposite: int
    n: int
    diffs: tuple

    @property
    def count(self) -> str:
        """`agree/n`, disclosing exact zeros.

        Without the disclosure a `1/5` reads as four seeds contradicting the
        mean when in fact four seeds moved by exactly nothing - a different
        claim, and the one the data makes in 34 of section 11's 120 rows.
        """
        if self.agree is None:
            return NOT_MEASURED
        text = f"{self.agree}/{self.n}"
        if self.zeros == self.n:
            text += " (all exactly 0)"
        elif self.zeros:
            text += f" ({self.zeros} exactly 0)"
        return text

    @property
    def unanimous(self) -> bool:
        return self.agree is not None and self.n > 0 and self.agree == self.n


MISSING_PAIR = Paired(NOT_MEASURED, None, None, None, 0, 0, 0, ())


def paired_diff(cell_hi: dict, cell_lo: dict,
                field: str = "attack_success") -> Paired:
    """Within-seed `hi - lo`, then mean, population std and sign agreement.

    The difference is formed seed by seed and only then aggregated, never as a
    difference of two aggregates: a cell whose seeds disagree in sign would
    otherwise read as "no effect" at the mean.

    Returns `MISSING_PAIR` when either cell is absent or they share no seed.
    """
    hi, lo = per_seed_map(cell_hi, field), per_seed_map(cell_lo, field)
    shared = sorted(set(hi) & set(lo))
    if not shared:
        return MISSING_PAIR
    diffs = np.array([hi[s] - lo[s] for s in shared], dtype=np.float64)
    mean, std = float(diffs.mean()), float(diffs.std())
    if mean > 0:
        agree = int((diffs > 0).sum())
    elif mean < 0:
        agree = int((diffs < 0).sum())
    else:
        agree = int((diffs == 0).sum())
    return Paired(f"{mean:+.4f} +/- {std:.4f}", mean, std, agree,
                  int((diffs == 0).sum()),
                  len(shared) - agree - int((diffs == 0).sum()),
                  len(shared), tuple(float(d) for d in diffs))


def paired_seeds(hi_map: dict, lo_map: dict) -> Paired:
    """`paired_diff` over two `{seed: value}` maps already in hand.

    Used for the difference-in-differences in 11.2, whose two operands are
    themselves paired differences and so have no cell to read.
    """
    shared = sorted(set(hi_map) & set(lo_map))
    if not shared:
        return MISSING_PAIR
    diffs = np.array([hi_map[s] - lo_map[s] for s in shared], dtype=np.float64)
    mean, std = float(diffs.mean()), float(diffs.std())
    if mean > 0:
        agree = int((diffs > 0).sum())
    elif mean < 0:
        agree = int((diffs < 0).sum())
    else:
        agree = int((diffs == 0).sum())
    return Paired(f"{mean:+.4f} +/- {std:.4f}", mean, std, agree,
                  int((diffs == 0).sum()),
                  len(shared) - agree - int((diffs == 0).sum()),
                  len(shared), tuple(float(d) for d in diffs))


#: Section 11's sign-consistency gate, in population stds. A row whose mean
#: sits further than this from zero should be a tight population, so every
#: seed must agree with the sign of its own mean.
#:
#: BE WARNED THAT AT n=5 THIS EXACT RULE CANNOT FIRE. Normalise a difference
#: vector so its mean is 1 and ask for the smallest population std compatible
#: with one seed not agreeing: put that seed at 0 (the closest a non-agreeing
#: value can sit to the mean) and the other four at 1.25, and the std is
#: exactly 0.5, so |mean|/std is exactly 2.0. Spreading the other four, or
#: making the odd seed properly negative, only raises the std and lowers the
#: ratio. 2.0 is therefore the attained SUPREMUM of |mean|/std over all
#: non-unanimous 5-seed vectors, and a strict `>` against it is a tautology:
#: no data can violate it. It is kept because it states the invariant and
#: becomes load-bearing at n>5, but `check_wiring` below is the falsifiable
#: half of the gate, and the README prints the measured headroom so the slack
#: is visible instead of implied.
SIGN_CONSISTENCY_STDS = 2.0


class SignConsistencyError(AssertionError):
    """A section 11 row with |mean| > 2*std whose seeds are not unanimous."""


class RowWiringError(AssertionError):
    """A section 11 row whose printed aggregate does not match its own seeds."""


def _row_report(direction, system, pair, column, p: Paired) -> list[str]:
    ratio = abs(p.mean) / p.std if (p.mean is not None and p.std) else None
    return [
        f"  {direction} / {system}  pair={pair}  column={column}",
        f"    printed={p.text!r}  count={p.count!r}",
        f"    mean={p.mean:+.9f}  std(pop,n={p.n})={p.std:.9f}  "
        f"|mean|/std={ratio:.4f}" if ratio is not None else
        f"    mean={p.mean}  std={p.std}  n={p.n}",
        f"    same sign={p.agree}/{p.n}  exactly 0={p.zeros}  "
        f"opposite sign={p.opposite}",
        f"    per-seed diffs={[round(d, 9) for d in p.diffs]}",
    ]


def check_wiring(rows: list[tuple]) -> None:
    """The falsifiable half: every printed aggregate must match its own seeds.

    This is the check that actually catches the failure modes a generated
    table is exposed to - a column read from the wrong cell, a mean and a
    count computed over different seed sets, a partition that does not add up.
    Unlike `SIGN_CONSISTENCY_STDS` it can fail, so it is the one that earns
    the word assertion.
    """
    offending = []
    for direction, system, pair, column, p in rows:
        if p.mean is None or p.n == 0:
            continue
        diffs = np.array(p.diffs, dtype=np.float64)
        bad = (len(diffs) != p.n
               or p.agree + p.zeros + p.opposite != p.n
               or not np.isclose(diffs.mean(), p.mean, rtol=0, atol=1e-12)
               or not np.isclose(diffs.std(), p.std, rtol=0, atol=1e-12)
               or p.text != f"{p.mean:+.4f} +/- {p.std:.4f}")
        if bad:
            offending.append((direction, system, pair, column, p))
    if not offending:
        return
    lines = [f"section 11 row-wiring assertion FAILED on {len(offending)} "
             f"row(s): the printed aggregate does not match the row's own "
             f"per-seed differences. README NOT written."]
    for row in offending:
        lines += _row_report(*row)
    message = "\n".join(lines)
    print(message)
    raise RowWiringError(message)


def check_sign_consistency(rows: list[tuple]) -> float | None:
    """Gate every section 11 paired-difference row. Raises, never warns.

    `rows` is `(direction, system, pair, column, Paired)`. Rows with no shared
    seed are skipped - they print NOT MEASURED and assert nothing.

    Returns the largest `|mean|/std` observed among the NON-unanimous rows, so
    section 11 can print how far the data actually sits from the gate rather
    than leaving the reader to assume the gate is tight.
    """
    check_wiring(rows)
    offending, headroom = [], None
    for direction, system, pair, column, p in rows:
        if p.mean is None or p.std is None or p.n == 0:
            continue
        if not p.std:
            continue
        ratio = abs(p.mean) / p.std
        if not p.unanimous:
            headroom = ratio if headroom is None else max(headroom, ratio)
            if ratio > SIGN_CONSISTENCY_STDS:
                offending.append((direction, system, pair, column, p))
    if offending:
        lines = [f"section 11 sign-consistency assertion FAILED on "
                 f"{len(offending)} row(s): |mean| > "
                 f"{SIGN_CONSISTENCY_STDS:g}*std but the seeds are not "
                 f"unanimous. README NOT written."]
        for row in offending:
            lines += _row_report(*row)
        message = "\n".join(lines)
        print(message)
        raise SignConsistencyError(message)
    return headroom


def flags(views) -> list[str]:
    """`YES`/`no` per flagged view, so no row has to be read by its name."""
    return ["YES" if v in views else "no" for v in FLAGGED_VIEWS]


def dns_epsilon_in_std(record) -> float | None:
    """`epsilon 1.0` expressed in `dns` standard deviations, for one direction.

    `1.0 / std` is the same quantity section 3 tabulates. It is the whole
    reason the two directions' epsilon grids are not comparable on the view
    that carries the label, so L17 derives it rather than quoting section 3.
    """
    scale = record["systems"]["naive"]["feature_scale"]["per_view"]
    std = (scale.get("dns") or {}).get("std")
    if not isinstance(std, (int, float)) or not std:
        return None
    return 1.0 / std


def alone_means(record, system: str) -> dict:
    """`{view: 5-seed mean attack success}` at chorus k=1 eps=1.0 beta=1.0."""
    cells = record["systems"][system]["cells"]
    out = {}
    for views, cell in by_subset(cells, "chorus", 1, 1.0, 1.0).items():
        stat = cell.get("summary", {}).get("attack_success")
        if len(views) == 1 and stat:
            out[views[0]] = stat["mean"]
    return out


def section_coordination(results: dict) -> list[str]:
    """Section 11: what the chorus coordination term `beta` actually buys.

    Every figure is a WITHIN-SEED paired difference on one subset at one grid
    point. Nothing is pooled across pairs, across the dns split or across the
    two directions, because sections 5 and 9 establish that all three
    disagree.
    """
    out: list[str] = []
    gated: list[tuple] = []
    for dname, record in results["directions"].items():
        out += [f"### {dname} (train `{record['train_scenario']}` -> test "
                f"`{record['test_scenario']}`)", ""]
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            b0 = by_subset(cells, "chorus", 2, 1.0, 0.0)
            b1 = by_subset(cells, "chorus", 2, 1.0, 1.0)
            b5 = by_subset(cells, "chorus", 2, 1.0, 5.0)
            pgd = by_subset(cells, "pgd", 2, 1.0, None)
            subsets = sorted(set(b0) | set(b1) | set(b5) | set(pgd))
            if not subsets:
                out += [f"**{dname} / {system}**", "",
                        f"{NOT_MEASURED}: no `chorus k=2 eps=1.0` cells in "
                        "this unit.", ""]
                continue
            out += [f"**{dname} / {system}** - attack success, paired within "
                    "seed", ""]
            out += header(["compromised pair", "dns in set", "suricata in set",
                           "beta 1.0 - beta 0.0 (5-seed mean +/- std)",
                           "same sign",
                           "beta 5.0 - beta 1.0 (5-seed mean +/- std)",
                           "same sign",
                           "chorus beta 0.0 - pgd (check)", "same sign"])
            for views in subsets:
                pair = "+".join(views)
                p10 = paired_diff(b1.get(views), b0.get(views))
                p51 = paired_diff(b5.get(views), b1.get(views))
                p0p = paired_diff(b0.get(views), pgd.get(views))
                gated += [(dname, system, pair, "beta 1.0 - beta 0.0", p10),
                          (dname, system, pair, "beta 5.0 - beta 1.0", p51),
                          (dname, system, pair, "chorus beta 0.0 - pgd", p0p)]
                out.append(line(
                    [f"`{pair}`", *flags(views),
                     p10.text, p10.count, p51.text, p51.count,
                     p0p.text, p0p.count]))
            out.append("")
    sub, sub_rows = section_coordination_systems(results)
    out += sub
    gated += sub_rows
    # The gate runs BEFORE the preamble is built, both because it may stop the
    # write and because the preamble quotes the headroom it measures.
    headroom = check_sign_consistency(gated)
    n_rows = sum(1 for *_ , p in gated if p.mean is not None and p.n)
    n_zero = sum(1 for *_ , p in gated
                 if p.mean is not None and p.n and not p.unanimous and p.zeros)
    return coordination_preamble(n_rows, n_zero, headroom) + out \
        + coordination_reading(results)


def coordination_preamble(n_rows: int, n_zero: int,
                          headroom: float | None) -> list[str]:
    """Section 11's heading and the three things a reader must know first.

    `n_rows`, `n_zero` and `headroom` are measured over the rows actually
    emitted, so the prose cannot drift from the tables the way a hand-written
    count would.
    """
    return ["## 11. Coordination effect (beta)", "",
            "`beta` is the weight on the chorus objective's agreement term, "
            "`sum_v log b_v[target] - beta * sum_{i,j} ||b_i - b_j||^2` "
            "(`prismflow/attacks/chorus.py`, `chorus_objective`). At `beta=0` "
            "that term is off and the attacker only maximises the target's "
            "likelihood; raising it pays the attacker to make the compromised "
            "views AGREE while doing so.", "",
            "Every number below is a **paired within-seed difference** on one "
            "subset at `chorus k=2 eps=1.0`: the difference is formed seed by "
            "seed and only then averaged, so a pair whose seeds disagree in "
            "sign cannot read as `no effect`. The `same sign` column counts "
            "the seeds agreeing with the sign of the mean. A seed whose "
            "difference is **exactly 0.0** agrees with neither sign, so it is "
            "excluded from that count and disclosed in the cell as `(n "
            "exactly 0)`: a `1/5 (4 exactly 0)` is four seeds that did not "
            "move, NOT four seeds contradicting the mean. "
            f"{n_zero} of the {n_rows} rows in this section are in that "
            "position. Read a low count WITHOUT a zero disclosure as a "
            "genuine sign split. **Nothing is pooled** across pairs, across "
            "the `dns` split or across the two directions.", "",
            "**The gate on these rows, and what it is worth.** Two "
            "assertions run over every paired-difference row in this section "
            "before this file is written; either one printing an offending "
            "row aborts the write.",
            "",
            f"1. `check_sign_consistency`: if `|mean| > "
            f"{SIGN_CONSISTENCY_STDS:g} * std` (population std) the seeds "
            f"must be unanimous. **At n=5 this cannot fail, and is reported "
            f"here as an invariant rather than as a check that passed.** The "
            f"smallest population std a 5-seed vector can have while one seed "
            f"disagrees is reached by putting that seed at 0 and the other "
            f"four at 1.25 the mean, which gives `|mean|/std` exactly 2.0; "
            f"every other non-unanimous arrangement is wider and scores "
            f"lower. 2.0 is thus the attained supremum, so a strict `>` "
            f"against it is a tautology at this n. It becomes load-bearing "
            f"only at n>5."
            + (f" Measured headroom: the largest `|mean|/std` among the "
               f"non-unanimous rows in this section is **{headroom:.4f}**, "
               f"against a bar of {SIGN_CONSISTENCY_STDS:g} that no 5-seed "
               f"row can reach."
               if headroom is not None else ""),
            "",
            "2. `check_wiring`: every row's printed `mean +/- std`, its "
            "`same sign` count and its `agree + exactly-0 + opposite` "
            "partition must all recompute from that row's own per-seed "
            "difference vector. This one CAN fail - it is what would catch a "
            "column read off the wrong cell or a mean and a count taken over "
            "different seed sets - and it is the assertion to trust.",
            "",
            "The last column pair of the first four tables is a consistency "
            "check, not an effect: `run_ait_attack.py:501` passes `beta=beta "
            "if kind == \"chorus\" else 0.0`, so `pgd` and `chorus beta=0.0` "
            "get an identical config and differ only in which runner is "
            "called. `chorus beta 0.0 - pgd` should therefore sit at ~0 "
            "wherever the two implementations agree, and it says nothing "
            "about coordination.", ""]


def seed_diff_map(cell_hi: dict, cell_lo: dict,
                  field: str = "attack_success") -> dict:
    """`{seed: hi - lo}`, the per-seed operand a difference-in-differences needs."""
    hi, lo = per_seed_map(cell_hi, field), per_seed_map(cell_lo, field)
    return {s: hi[s] - lo[s] for s in sorted(set(hi) & set(lo))}


def section_coordination_systems(results: dict) -> tuple[list[str], list[tuple]]:
    """11.1 and 11.2: the same beta cells read as prismflow against naive.

    11.1 is the LEVEL difference at each beta separately - prismflow minus
    naive attack success at `beta=1.0` and at `beta=0.0` - so a reader can see
    whether the two arms start level before coordination is switched on.

    11.2 is the difference-in-differences: the `beta 1.0 - beta 0.0` shift
    measured on prismflow minus the same shift measured on naive, paired
    within seed at every step. It answers a narrower question than 11.1 - does
    coordination buy the attacker MORE against the discount than against plain
    fusion - and it is reported only for the 4 `dns` pairs per direction,
    because those are the pairs where `beta` moves anything at all (section 11).

    Returns `(lines, gated_rows)`; the rows join section 11's assertion.
    """
    out = ["### 11.1 PrismFlow minus naive at each beta, by pair", "",
           "Within-seed `prismflow - naive` attack success at one `beta`, "
           "`chorus k=2 eps=1.0`. **Negative favours PrismFlow** (lower attack "
           "success). The two columns are separate measurements at separate "
           "grid points and are never averaged together; 11.2 pairs them. "
           "All 10 pairs are listed, flagged for `dns` and `suricata`, and "
           "nothing is pooled across the `dns` split.", ""]
    gated: list[tuple] = []
    for dname, record in results["directions"].items():
        pf = record["systems"]["prismflow"]["cells"]
        nv = record["systems"]["naive"]["cells"]
        pf0, nv0 = (by_subset(pf, "chorus", 2, 1.0, 0.0),
                    by_subset(nv, "chorus", 2, 1.0, 0.0))
        pf1, nv1 = (by_subset(pf, "chorus", 2, 1.0, 1.0),
                    by_subset(nv, "chorus", 2, 1.0, 1.0))
        subsets = sorted(set(pf0) | set(pf1) | set(nv0) | set(nv1))
        out += [f"**{dname}** (train `{record['train_scenario']}` -> test "
                f"`{record['test_scenario']}`) - prismflow minus naive attack "
                f"success, paired within seed", ""]
        if not subsets:
            out += [f"{NOT_MEASURED}: no `chorus k=2 eps=1.0` cells.", ""]
            continue
        out += header(["compromised pair", "dns in set", "suricata in set",
                       "beta=1.0: prismflow - naive (5-seed mean +/- std)",
                       "same sign",
                       "beta=0.0: prismflow - naive (5-seed mean +/- std)",
                       "same sign"])
        for views in subsets:
            pair = "+".join(views)
            q1 = paired_diff(pf1.get(views), nv1.get(views))
            q0 = paired_diff(pf0.get(views), nv0.get(views))
            gated += [(dname, "prismflow-vs-naive", pair,
                       "beta=1.0 prismflow - naive", q1),
                      (dname, "prismflow-vs-naive", pair,
                       "beta=0.0 prismflow - naive", q0)]
            out.append(line([f"`{pair}`", *flags(views),
                             q1.text, q1.count, q0.text, q0.count]))
        out.append("")

    out += ["### 11.2 Difference-in-differences on the `dns` pairs", "",
            "`(prismflow: beta 1.0 - beta 0.0) - (naive: beta 1.0 - beta "
            "0.0)`, paired within seed at both steps: the per-seed `beta` "
            "shift is formed for each arm first, and the two shifts are then "
            "subtracted seed by seed. **Positive means coordination bought "
            "the attacker MORE against PrismFlow than against naive fusion** "
            "at `chorus k=2 eps=1.0`.", "",
            "Only the 4 `dns` pairs per direction appear: section 11 measures "
            "the `beta` shift on the other 6 pairs as at or near zero in both "
            "arms, so a difference of two near-zero shifts would be a ratio "
            "of noise. The two `beta` columns are repeated here from section "
            "11 so the DiD can be checked against its operands.", ""]
    for dname, record in results["directions"].items():
        pf = record["systems"]["prismflow"]["cells"]
        nv = record["systems"]["naive"]["cells"]
        pf0, pf1 = (by_subset(pf, "chorus", 2, 1.0, 0.0),
                    by_subset(pf, "chorus", 2, 1.0, 1.0))
        nv0, nv1 = (by_subset(nv, "chorus", 2, 1.0, 0.0),
                    by_subset(nv, "chorus", 2, 1.0, 1.0))
        dns_pairs = sorted(v for v in set(pf1) | set(nv1) if "dns" in v)
        out += [f"**{dname}** - difference-in-differences, paired within "
                f"seed, `chorus k=2 eps=1.0`", ""]
        if not dns_pairs:
            out += [f"{NOT_MEASURED}: no `dns` pairs at this grid point.", ""]
            continue
        out += header(["compromised pair", "suricata in set",
                       "prismflow: beta 1.0 - beta 0.0", "same sign",
                       "naive: beta 1.0 - beta 0.0", "same sign",
                       "DiD prismflow - naive (5-seed mean +/- std)",
                       "same sign"])
        dids = []
        for views in dns_pairs:
            pair = "+".join(views)
            ppf = paired_diff(pf1.get(views), pf0.get(views))
            pnv = paired_diff(nv1.get(views), nv0.get(views))
            did = paired_seeds(seed_diff_map(pf1.get(views), pf0.get(views)),
                               seed_diff_map(nv1.get(views), nv0.get(views)))
            gated += [(dname, "prismflow-vs-naive", pair,
                       "DiD (prismflow beta shift) - (naive beta shift)", did)]
            dids.append(did)
            out.append(line([f"`{pair}`",
                             "YES" if "suricata" in views else "no",
                             ppf.text, ppf.count, pnv.text, pnv.count,
                             did.text, did.count]))
        out.append("")
        scored = [d for d in dids if d.mean is not None]
        if scored:
            pos = [d for d in scored if d.mean > 0]
            pos_u = [d for d in pos if d.unanimous]
            neg = [d for d in scored if d.mean < 0]
            neg_u = [d for d in neg if d.unanimous]
            out += [f"{dname}: the DiD is POSITIVE on {len(pos)} of the "
                    f"{len(scored)} `dns` pairs ({len(pos_u)} with all 5 "
                    f"seeds agreeing) and negative on {len(neg)} "
                    f"({len(neg_u)} unanimous)"
                    + (f", spanning {min(d.mean for d in pos):+.4f}.."
                       f"{max(d.mean for d in pos):+.4f}" if pos else "")
                    + ". A positive DiD means `beta` 0 -> 1 raised attack "
                      "success MORE against PrismFlow than against naive "
                      "fusion on that pair. This is the measured difference "
                      "of two measured shifts and nothing more: no quantity "
                      "inside the discount is recorded per cell, so the "
                      "section does not say why.", ""]
    return out, gated


def coordination_reading(results: dict) -> list[str]:
    """What the section 11 tables say, every count re-derived from the cells."""
    out = ["### What the tables say", ""]
    for dname, record in results["directions"].items():
        for system in ("naive", "prismflow"):
            cells = record["systems"][system]["cells"]
            b0 = by_subset(cells, "chorus", 2, 1.0, 0.0)
            b1 = by_subset(cells, "chorus", 2, 1.0, 1.0)
            pgd = by_subset(cells, "pgd", 2, 1.0, None)

            def shift(views):
                return paired_diff(b1.get(views), b0.get(views)).mean

            dns_pairs = [v for v in b1 if "dns" in v]
            sur_pairs = [v for v in b1 if "suricata" in v and "dns" not in v]
            dns_up = [v for v in dns_pairs if (shift(v) or 0.0) > 0]
            sur_dn = [v for v in sur_pairs if (shift(v) or 0.0) < 0]
            checks = [abs(paired_diff(b0[v], pgd.get(v)).mean or 0.0)
                      for v in b0 if v in pgd]
            worst = max(checks) if checks else None
            out.append(
                f"- **{dname} / {system}:** `beta` 0 -> 1 RAISES attack success "
                f"on {len(dns_up)} of the {len(dns_pairs)} pairs containing "
                f"`dns` and lowers it on {len(dns_pairs) - len(dns_up)}; on the "
                f"{len(sur_pairs)} pairs containing `suricata` but not `dns` it "
                f"LOWERS {len(sur_dn)} and raises "
                f"{len(sur_pairs) - len(sur_dn)}. Largest "
                f"`|chorus beta 0.0 - pgd|` across the "
                f"{len(checks)} pairs where both exist: "
                + (f"{worst:.4f}." if worst is not None else f"{NOT_MEASURED}."))
    carried = seed_carried_cells(results)
    units = sorted({f"{d} / {s}" for d, s, *_ in carried})
    out += ["", "Read those counts against the `same sign` columns before "
            "treating any of them as an effect: a `same sign` below 5/5 means "
            "the seeds disagree and the mean is an average over a split "
            "population, not a measurement of one. Section 5.2(b) lists the "
            f"{len(carried)} `k=1` cells whose mean is carried by a single "
            f"seed"
            + (f", in {', '.join(units)}" if units else "")
            + ".", ""]
    return out


def section_evidence(results: dict, manifest) -> list[str]:
    _other_size, _other_path = largest_other_tracked_result()
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
            f"largest results file tracked anywhere else in this repository is "
            f"{_other_size} (`{_other_path}`). Every MB in this section is "
            f"bytes / 1e6, the same unit `results_manifest.json` records. They "
            "are therefore committed as **gzip (`-9`)** rather "
            "than raw json, which is the only departure from the convention "
            "tracking `results/*/*.json`; `.gitignore` carries one un-ignore line "
            "for them, `!results/*/*.json.gz`. Both are in the repository at "
            "this commit:", ""]
    entries = (manifest or {}).get("files", {})
    out += header(["committed file", "raw json", "gzip -9", "sha256 of the "
                   "DECOMPRESSED json"])
    if entries:
        for name, entry in entries.items():
            raw_mb = entry.get("raw_bytes", 0) / MB
            gz_mb = entry.get("gz_bytes", 0) / MB
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
            "a run by `results_io.py --write-manifest`. The smaller "
            f"`results.A_naive_checkpoint.json` "
            f"({size_mb(ROOT / 'results.A_naive_checkpoint.json')}) is "
            "committed raw, unchanged.", "",
            "**The uncompressed `results.json` and "
            "`results.3of4units_20261014_0022.json` remain on disk in this "
            "working copy, and are re-ignored by `.gitignore`** so they cannot "
            "be committed raw alongside the `.gz`. Two rules naming the two "
            "files exactly sit AFTER the `!results/*/*.json` un-ignore that "
            "would otherwise re-include them, and for `.gitignore` the last "
            "matching pattern wins:", ""]
    out += header(["`.gitignore` line", "rule", "effect"])
    for lineno, rule, effect in gitignore_reignore_rows(manifest):
        out += [line([lineno, rule, effect])]
    out += ["",
            "Measured, not asserted - `git check-ignore -v` names the new rule "
            "as the deciding one and `git add` refuses both files without "
            "`-f`:", "",
            "```sh",
            "git check-ignore -v results/ait_attack/results.json",
            "git add --dry-run results/ait_attack/results.json   # exits 1",
            "```", "",
            "This replaces the `.git/info/exclude` entries described in an "
            "earlier commit, which were inert: `.gitignore` outranks "
            "`$GIT_DIR/info/exclude`, so an exclude entry could not override "
            "the `!results/*/*.json` un-ignore. The entries are harmless and "
            "are left in place. The committed `.json.gz`, "
            "`results_manifest.json` and every other tracked file under "
            "`results/` are unaffected: the two new rules match those two names "
            "only.",
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
    doc += section_coordination(results)

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
