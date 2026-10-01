---
name: prismflow-honest-reporting
description: >
  Enforces PrismFlow's evidentiary rules for anything that touches a number,
  chart, dataset, or claim of measurement: never fabricate a figure, source,
  prediction, or failure (not even as a placeholder or "for now" example);
  show absent data as "not measured" rather than 0, blank, or a guess;
  require 5+ seeds with mean +/- std before calling something a finding, and
  label single-pass/single-seed results plainly as not one; cite the exact
  committed file (and field, where relevant) behind every number; report
  suspected defects in frozen modules instead of silently patching them; never
  put a secret or API key in chat, a commit, or a web-facing file; commit each
  logical step of a task separately. Use this proactively whenever asked to
  fill in a missing metric, populate a dashboard or chart with data, make a
  result "look better" or "more complete," add sample/mock/placeholder data,
  summarize an experiment's results, or touch anything under app.py,
  prismflow/, results/, data/, or web/ in this repo — even when the request
  doesn't mention these rules by name.
---

# PrismFlow honest reporting

PrismFlow's thesis is about evidence: agreement should count in proportion to
independence, not in proportion to volume. That standard has to apply to the
project's own reporting about itself, or every downstream number is
untrustworthy for the same reason the product exists to catch. This skill
exists because the shortcut is always available and always looks harmless in
the moment — one placeholder number, one demo-friendly mock row, one "just
this once" polish pass — and each one quietly breaks the thing the project is
about.

## The rules, and why they're not optional

**Never fabricate.** Not a number, not a source, not a predicted outcome, not
a failure case. This includes placeholders you intend to flag as fake ("I'll
put 0.15 here as an example ECE") — once a plausible-looking number is in a
file or a chart, it gets copied, screenshotted, and cited long after the
caveat is forgotten. If you don't have the real value, don't write a value.

**Absent data is "not measured," never 0, blank, or estimated.** A 0 reads as
"we measured this and it was zero" — a completely different claim from "we
never ran this." Interpolating a plausible-looking number from nearby data is
still fabrication; it just wears a disguise. If a chart or table needs a cell
for a metric that has no committed source, the cell says `not measured`.

**5-seed rule.** Nothing is a finding on fewer than 5 seeds with a reported
mean and standard deviation. A single pass can be useful for a quick sanity
check, but it gets labelled exactly that — e.g. "single-pass, NOT A FINDING"
— every time it's shown, not just the first time. Watch in particular for a
population where the effect cancels in the mean: check the sign of each seed
before trusting an aggregate that reads as "no effect."

**Every number cites its file.** When you report a figure, name the committed
file (and field or key, if the file has more than one candidate number) it
came from. If you can't point to where a number came from, you don't have
permission to write it down. This is what makes "not measured" and "5-seed
rule" enforceable from the outside — a reviewer can check the citation instead
of taking the claim on faith.

**Frozen modules are reported, not fixed.** If a module was previously
verified and marked frozen (check CLAUDE.md and PROGRESS.md/docs for which —
in this repo that's `app.py` and `prismflow/`), and you find something that
looks like a defect while working on something else, stop and report it. Do
not fix it as a side effect of the task you're actually doing, even if the
fix is trivial and even if leaving it feels wrong. A silent fix to a frozen
module means nobody can trust that "frozen" still means "verified."

**No secrets in chat, commits, or web-facing files.** Never paste an API key,
token, or credential into a conversation or a file, even to explain why
something failed. Before a commit that touches anything served to a browser
(anything under `web/`), scan the diff for key-like strings and confirm
nothing beyond the intended app logic reaches the client.

**Commit each logical step separately.** A task that has multiple distinct
pieces (a data layer, then a UI layer, then a style pass) gets a commit per
piece, each one independently reviewable and revertable, not one commit that
bundles everything.

## What to actually do when a shortcut is tempting

1. **Locate the real source first.** Before writing any number, find the
   committed file it should come from (`data/`, `results/`, `docs/`, or
   wherever the project keeps its measured output). If it exists, cite it.
2. **If it doesn't exist, say so and stop there** — don't fill the gap with
   your best guess "just so the section isn't empty." Write `not measured`
   (or the UI equivalent — a greyed cell, an explicit label) and, if it's
   useful, name what would need to run to produce the real number.
3. **If someone asks you to run the real experiment**, check whether it can
   run 5+ seeds. If it can't right now (time, cost, API budget), say that
   plainly rather than quietly reporting fewer.
4. **If a fix opportunity appears in frozen code**, write down what you found
   and where, and ask before touching it — don't fold it into the current
   diff.
5. **If asked to "make it look better"** with no new data behind the ask,
   separate what's cosmetic (labels, layout, color, rounding for display)
   from what would change the reported values — cosmetic changes are fine,
   value changes need a real source.

## Worked examples

**"Fill the missing ECE value."**
Wrong: compute a plausible ECE from a related metric, or interpolate between
neighboring cells, and write it in.
Right: check whether any committed results file actually contains a
calibration run producing ECE. If none exists, the cell stays `not measured`,
and say directly that no committed run produced that number — offer to run
the calibration experiment (with 5+ seeds) if that's in scope, rather than
inventing a number to close the gap.

**"Add sample data to the dashboard."**
Wrong: generate mock rows so the dashboard has something to render and looks
finished, planning to swap in real data "later."
Right: point out that sample/placeholder data in a dashboard is exactly what
this project's rules prohibit, because it's easy for a fabricated row to get
mistaken for a real one once it's rendered next to real ones. Ask which
committed results file should back the dashboard, or ship it empty with a
"not measured" state until real data exists.

**"Just make the chart look better."**
Wrong: treat this as license to smooth a jagged line, round away outliers, or
adjust axis scales in a way that changes what the chart claims — anything
that makes the underlying numbers look different than they are.
Right: separate presentation from data. Improve typography, color, layout,
labeling, and legend clarity freely — none of that touches what's being
claimed. Leave every plotted value exactly as computed. If smoothing or
outlier handling is genuinely warranted statistically, say so explicitly and
show both versions rather than silently replacing the real one.

## Verification checklist before finishing a task this skill applies to

- [ ] Every number in the output has a named source file behind it.
- [ ] No cell, chart point, or metric was filled with 0/blank/an estimate
      where "not measured" was the honest answer.
- [ ] Anything claimed as a "finding" has >=5 seeds and reports mean +/- std;
      anything with fewer is labelled as not a finding, every time it's shown.
- [ ] Nothing in a frozen module changed; any suspected defect there was
      reported, not patched.
- [ ] No key, token, or credential appears in the diff, especially under
      `web/`.
- [ ] If the task had multiple logical steps, each is (or will be) its own
      commit.
