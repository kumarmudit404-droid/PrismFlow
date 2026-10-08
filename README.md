# PrismFlow

**Dependence-Discounted Evidential Fusion under Collusive View Compromise.**

When several sources agree, confidence usually goes up. That is only sound if
the sources are independent. If they are copies of one another, the agreement
is an echo and the confidence built on it is counterfeit. PrismFlow tests one
hypothesis about that: agreement between views should contribute evidence in
proportion to the *effective independence* of those views, not in proportion to
how many views agree — and discounting fused confidence by estimated inter-view
dependence (ENIV, the effective number of independent views) should yield
fusion that holds up when some of those views are colluding or compromised.

It is reported as a research project rather than as a product. Half the
hypothesis is supported and half is refuted, and both halves are stated
plainly: the estimator **detects** collusion reliably, and the correction
driven from that signal **did not defend** against it. A second result fell out
of the work and may be the most portable part of it — on the canonical 6-view
benchmark, the six views behave like about three and a quarter. Every figure
anywhere in this repository names the committed file it came from, and anything
that was not measured is written as `not measured` rather than as a zero, a
blank or an estimate.

[EDIT ME — a paragraph in your own words: why you started this, and what you
want a reader to take away from it.]

## Quickstart

```bash
pip install -r requirements.txt

python web/serve.py      # the site: http://127.0.0.1:825
pytest                   # the test suite: 960 tests
```

`web/serve.py` serves `web/` and nothing above it, so `.env` is never one URL
away from a browser. The site is static, reads only committed files, and makes
no network request of any kind.

## Project layout

```
prismflow/       the library
  data/          multi-view dataset generation and loading
  models/        encoders and evidence heads
  statistics/    dependence estimation, calibration, suspicion detection
  eniv/          the ENIV estimator and the discount
  attacks/       colluding / compromised view generators
  evaluation/    multi-seed experiment harness
  v2/            the semantic multi-angle path: connectors, angles, fusion
  utils/         config, seed, logging, device, checkpoint
app.py           the V1 Streamlit demo, frozen
experiments/     experiment source; one directory per experiment
results/         committed findings -- every number on the site comes from here
data/v2/         the 48-row V2 evaluation dataset and its sources
configs/         YAML experiment configs
parts/           the brief for each Part, as written before it was built
docs/            the contract, the limitations, the protocols
tests/           960 tests, unit and integration
web/             the static site: both V1 and V2, as a guided tour
tools/           repository tooling (secret scanning)
```

Start with `docs/CONTRACT.md`. It is the project contract — data flow,
component classification, tensor shape conventions, reproducibility rules — and
it carries the dated annotations that record what the evidence actually
supports.

## What the results show, and what they do not

`docs/PROJECT_STATUS.md` is the full picture with a committed file behind every
figure. In short:

**Shown.** On Handwritten (UCI Multiple Features), the canonical 6-view
benchmark, measured ENIV is 3.28 ± 0.07 of 6 — and a permutation control that
makes the views independent by construction moves it to 5.51 ± 0.08, so most of
the shortfall is real redundancy rather than estimator floor (5 seeds,
`results/real/README.md`). Under a coordinated attack the estimator detects
collusion reliably: ENIV falls 3.36 → 2.70 and dependence among the compromised
views rises +0.7351 ± 0.0914 (5 seeds, `results/chorus/`).

**Not shown.** The defence half of the hypothesis is refuted as tested.
PrismFlow's attack success rate was *higher* than the undiscounted baseline's
in 12 of 12 attacked cells (+0.0267 ± 0.0071 at ε = 1.0, 0/5 seeds better;
Part 09, `experiments/chorus/`). Detection of redundancy and robustness to it
are separate capabilities, and only the first is evidenced.

**Never measured.** On real retrieved evidence, no calibration metric exists.
Four of five angles retrieve almost nothing — regulatory has no connector at
all — so of 48 rows only one ever reached two angles producing claims. The
harness refused to compute ECE or Brier on that rather than dress a single cell
as a curve (`docs/v2-known-limitations.md` V2-L6). That refusal is the honest
summary of V2's state.

The limitations are not an appendix to this. Read them:

- `docs/KNOWN_LIMITATIONS.md` — V1, L1 through L15
- `docs/v2-known-limitations.md` — V2, V2-L1 through V2-L7

## Reproducing a figure

Every number has a file. `results/<experiment>/` holds findings,
`experiments/<name>/` the source that produced them, and `parts/` the brief
each was built against. The rules the project holds itself to are in
`CLAUDE.md` and `docs/CONTRACT.md`: previously-passing modules are frozen,
every experiment runs at least 5 seeds and reports mean and standard deviation,
and no result is ever fabricated — not even as a placeholder.

## Checking for secrets before a commit

No credential is committed to this repository; see `SECURITY.md`.
`tools/check_secrets.py` looks for two things: strings shaped like a provider
key, and the **actual values** in your `.env`, which is what catches a key with
no recognisable prefix. It never prints a value — a hit names the variable and
where it turned up.

```bash
python tools/check_secrets.py          # staged changes
python tools/check_secrets.py --all    # every tracked file
```

To run it on every commit, install it as a local hook. This is deliberately not
installed for you and is not committed — `.git/hooks/` is not tracked, and a
hook that appears in a working copy without the owner choosing it is a
surprise:

```bash
printf '#!/bin/sh\nexec python tools/check_secrets.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

A non-zero exit refuses the commit. If a finding is a false positive, add the
exact `(path, text)` pair to `ACKNOWLEDGED` in the script rather than loosening
a pattern, so the same string elsewhere is still reported.

## How this was built

This project was developed with [Claude Code](https://claude.com/claude-code)
as an AI coding assistant, across the whole of V1, V2 and the site. That is
stated here plainly because it is true, and because a reader deciding how much
to trust a result is entitled to know how it was produced.

It was built under rules the author set and enforced, which are the reason the
assistance is visible in the output rather than hidden by it:

- **Frozen modules.** Once a module passed, it was frozen. A suspected defect
  in an earlier Part had to be *reported and left alone*, never fixed as a side
  effect of later work — so "frozen" keeps meaning "verified".
- **A 5-seed minimum.** Nothing is a finding on fewer than 5 seeds with a
  reported mean and standard deviation. Single-pass results are labelled as not
  findings every time they are shown, not just the first time.
- **No fabricated numbers.** Never a figure, a source, a predicted outcome or a
  failure case — not even as a placeholder. Absent data is written
  `not measured`, never 0, blank or interpolated.
- **An audit before every push.** Including a scan for credentials over
  everything the site can serve.

Those rules were applied against the assistant as much as with it. The record
of that is in the repository: `docs/CONTRACT.md` carries dated annotations
where evidence contradicted the project's own stated hypothesis and the claim
was narrowed rather than defended; `docs/v2-known-limitations.md` records work
that was reported and deliberately *not* fixed because it fell outside its
authorisation; and `web/README.md` records a feature that was built without
authorisation, verified on request, and then removed entirely.

**The design decisions, the review and the verdicts were the author's.** What
an experiment should test, whether a result counted as evidence, what was
allowed to be called a finding, and what had to be reported as a failure —
those were decided by a person, and the commit history records who decided
what.

[EDIT ME — your own words on working this way: what you'd keep, what you'd do
differently, and what you want a reader to understand about the division of
labour.]

## Author

Mudit Kumar. See `CITATION.cff`.

## Licence

Apache-2.0. See `LICENSE`.
