# V2 known limitations -- retrieval coverage blocks the evaluation

Scope: PrismFlow V2 (Parts 17-24), the semantic multi-angle path
`views -> encoders -> evidence -> dependence -> ENIV -> discount -> fusion`.
V1's limitations are in `KNOWN_LIMITATIONS.md` and are not repeated here.

Each entry names its evidence and what would change our understanding of it,
following the convention of the V1 file.

## Status in one paragraph

Part 24's dataset and harness are **built and verified**. The evaluation they
exist to run is **blocked**, and the blocker is upstream of everything Part 24
measures: on 48 real rows only one angle retrieves usefully, so the components
under test -- dependence, ENIV, the discount and fusion -- are almost never
reached. **No calibration metric (ECE, Brier) has been computed on real data,
and none can be computed meaningfully today.** This is a coverage problem in the
connectors, not a defect in the fusion mathematics.

## Provenance of the numbers below

| | |
|---|---|
| run | `results/v2/part24_pipeline_verification.json`, commit `1bb901e` |
| harness | `experiments/v2/part24_pipeline_verification.py` at `17f4123` |
| rows | 48 of the eventual 50 (`data/v2/evaluation_queries.json`; row 040 absent) |
| settings | query derivation ON; 4 of 5 angles wired live |
| reasoner | Groq `openai/gpt-oss-120b` via the OpenAIReasoner code path -- NOT Claude, NOT OpenAI |
| fusion | offline `LexicalAdjudicator`; Part 23's Claude path NOT exercised |
| passes | ONE. No seeds, no mean, no std |

**None of this is a Part 24 finding under `docs/CONTRACT.md` section 5.** A
single pass is not evidence, and the 5-SEED RULE is not satisfied by any number
in this document. They are retrieval-coverage observations, which is a different
kind of claim: what a connector returns is a property of the source and the
query, not a measurement of the method.

Two passes were run on 2026-09-27. Retrieval reproduced **exactly** across both;
claim counts did not, because the reasoner sends no temperature and no seed, so
the provider samples at its own default. Where a claim count is quoted below it
is flagged, because the second pass exhausted a daily quota (V2-L5).

**Where each pass lives.** Both passes are committed, so every figure below can
be re-derived:

* second pass -- `results/v2/part24_pipeline_verification.json`
* first pass -- `results/v2/part24_pipeline_verification_pass1.json`

The first pass wrote to the second pass's path and was overwritten there; the
copy recovered under the `_pass1` name is byte-identical to what the harness
produced. The two figures taken from it -- its 38/48 and 1/48 in V2-L6, and row
032's ENIV and confidence -- are therefore checkable at that path. This mattered
enough to recover because a later pass cannot reproduce them: they need a day's
unspent quota (V2-L5).

**The first-pass file predates `17f4123`,** the commit that split provider
errors from genuine zeros, so it carries no `zero_claim_cause` and no
`zero_claim_rows_*` fields and its 10 zero-claim rows remain unsplit. That is
correct and expected for a pass that ran before the fix existed, not a gap in
the file. The breakdown for that pass -- 5 provider failures (005, 018 on TPM
429; 020, 042, 046 on `json_validate_failed`) and 5 genuine zeros (014, 021,
031, 047, 048) -- was recovered from that run's log and is recorded in
`17f4123`'s commit message.

One figure remains an inference rather than a measurement: the "roughly 100k
tokens per pass" in V2-L5 comes from the provider's own TPD counter across the
day, not from per-call accounting, because the harness records evidence tokens
and not reasoner tokens.

## V2-L1. Retrieval coverage: one angle of five contributes

**Evidence:** the 48-row run, per angle.

| angle | rows with >=1 record | records | source |
|---|---|---|---|
| tech | 48/48 | 480 | arXiv 480, GitHub 0 |
| market | 6/48 | 10 | NewsAPI |
| financial | 0/48 | 0 | yfinance |
| sentiment | 1/48 | 3 | StackExchange |
| regulatory | 0/48 | 0 | no connector exists |

Identical in both passes, to the record.

**What this shows.** The pipeline is fed by one angle on 42 of 48 rows. Since
`aggregate_dependence` excludes angles that produced no claims, a one-angle row
yields a 1x1 dependence matrix and a degenerate ENIV, and fusion has nothing to
adjudicate. The thesis under test -- that agreement should count in proportion to
the effective independence of the views -- needs at least two views to be a
statement about anything.

**What would change it:** a connector whose queries match the dataset's
language, for any of the four thin angles.

## V2-L2. GitHub returns 0 records on every row (pre-existing)

**Evidence:** tech-angle provenance is `{'github': 0, 'arxiv': N}` on all 48
rows. The same holds in the earlier 17-row run (`github=0, arxiv=170`), which
predates the Part 19 query-derivation addendum.

**What this shows.** Derivation fixed the *shape* problem it was written for --
GitHub's documented 422 on over-long queries, recorded in
`docs/part24-query-mismatch.md` -- but GitHub still contributes nothing, so the
tech angle is arXiv alone with no working fallback chain. Because the zero is
present on both sides of the derivation change, **it is not a regression from
that change.** `GITHUB_TOKEN` in `.env` is 7 characters and is very likely a
placeholder, which is the first thing to check.

**Frozen-module note:** reported, not fixed, per the FROZEN MODULES rule in
`CLAUDE.md`. Connector code belongs to Part 17/19.

**What would change it:** a valid credential, or the request and response of one
live GitHub search captured and read.

## V2-L3. The arXiv fallback is query-independent in membership

**Evidence, verified live on 2026-09-27 (retrieval only, no reasoner):** the tech
angle was asked for three unrelated rows and returned **the same ten records
every time** -- identical set membership, differing only in BM25 order.

| row | domain | pitch, opening words |
|---|---|---|
| 020 | Tech-OSS | "a standard way to write built-in generic collections such as `list[str]`" |
| 046 | Financial-Product | "a regulated crypto exchange and financial platform" |
| 049 | Financial-Product | "a low-cost international money-transfer product" |

Pairwise overlap 10/10 on all three pairs. The ten are astrophysics and
particle-physics papers: GWTC-4.0 and GWTC-5.0 gravitational-wave catalogues,
ATLAS detector performance, KAGRA ultralight dark matter, IceCube neutrino
follow-ups. **A query about Python generic-collection syntax retrieves the
LIGO-Virgo-KAGRA gravitational-wave transient catalogue.**

Corroborated across the whole run by an independent signal: **18 of 48 rows share
the byte-identical tech token total of 1076** (28 distinct totals across 48
rows), and those 18 span all three domains -- 003, 020, 026, 027, 031, 032, 034,
036, 038, 039, 042, 044, 045, 046, 047, 048, 049, 050.

**What this shows.** On a large subset of rows, "tech retrieves on 48/48" states
reachability, not relevance: the angle returns a fixed corpus unrelated to the
query. Any downstream number computed over those rows -- dependence between tech
and another angle, ENIV, a discount -- would be computed over evidence that does
not answer the question asked. The reasoner behaved correctly and said so, e.g.
"None of the retrieved records discuss regulated cryptocurrency exchanges", which
is why this surfaced as low claim counts rather than as confident wrong claims.

**Mechanism not identified.** Ten unrelated hits is not the signature of a query
that matched nothing -- that returns zero. It is the signature of a query whose
terms are dropped or ignored, leaving a generic listing. Whether that happens in
the derived query, in the arXiv request construction, or in arXiv's own handling
is not established here, and this document does not guess.

**Frozen-module note:** reported, not fixed. Part 17/19 own it.

**What would change it:** the exact URL the arXiv connector sends for row 020,
compared against the same search issued by hand.

## V2-L4. Financial is structurally incompatible with the dataset, by design conflict

**Evidence:** 0 records on 48/48 rows, including all 10 Financial-Product rows.
Provenance is `{'yfinance': 0}` on every one, so the connector *was* called.
Derivation is working as specified -- row 049's pitch reduced to 55 characters
under yfinance's 60-character ceiling. One transport error in 48 rows (row 049,
curl 16); the other 47 were silent zeros.

**What this shows.** This is **not a bug in either component.** Both behave as
designed, and the designs conflict:

* `YFinanceConnector` resolves a query to instruments through Yahoo's ticker
  search. It needs a company name or a symbol.
* The dataset's pitches are deliberately **anonymised** -- "Build a mobile-first
  auto insurer that uses telematics", "Build a consumer buy-now-pay-later
  product" -- so the company is exactly what has been removed. Nothing is left
  for a ticker search to resolve.

No amount of keyword extraction bridges this: a keyword is not a ticker, and the
derivation module says so in its own docstring ("yfinance wants a ticker and no
keyword extraction produces one ... This closes the shape mismatch, not the
corpus mismatch"). The mismatch was predicted before it was measured; the 48-row
run confirms it at n=48.

**What would change it:** either a ticker or company annotation on the financial
rows -- which changes the dataset's anonymisation property and must therefore be
a deliberate decision, not a patch -- or a financial connector that takes prose
(SEC EDGAR full-text, a news-backed fundamentals source) instead of a symbol.

## V2-L5. The free Groq tier cannot fund a 5-seed evaluation

**Evidence:** the second pass exhausted Groq's free-tier **daily** token
allowance mid-run. The 429 bodies report `tokens per day (TPD): Limit 200000,
Used 198062`, with retry hints up to 12m18s. 31 of that run's 32 reasoner
failures are TPD 429s; 1 is `json_validate_failed`.

**What this shows.** One 48-row pass costs roughly 100k tokens, so this key
affords about **two passes per day**. The 5-SEED RULE requires five. The free
tier therefore cannot fund a Part 24 evaluation at all, independently of every
other limitation in this document.

A second, smaller provider defect: `openai/gpt-oss-120b` fails JSON-mode
generation on a minority of calls by emitting `"confidence":0. nine` -- the word
where a number belongs -- which the API rejects as `json_validate_failed`. Three
such failures in the first pass, one in the second.

**What would change it:** a funded key. The Claude and OpenAI accounts are still
unfunded (see `docs/part20-groq-verification.md`), so Part 23's Claude
adjudicator and the gpt-4o path remain unverified.

## V2-L6. Consequence: ECE and Brier cannot yet be computed

**Evidence:** rows reaching fusion, i.e. rows with >=2 angles producing claims:

| pass | rows with >=1 angle | rows fused (>=2 angles) | rows scored |
|---|---|---|---|
| first, 2026-09-27 | 38/48 | **1/48** (row 032) | 1 |
| second, 2026-09-27 (quota-limited) | 20/48 | **0/48** | 0 |

The harness refused to compute metrics in both passes, printing "Too few rows
completed for metrics. Reporting no ECE, no Brier and no conflict figures rather
than computing them on a handful of rows."

**What this shows.** Calibration is a property of a distribution of confidences
against outcomes. With one fused row there is no distribution: an ECE over n=1 is
the error of a single point, and a Brier score over n=1 is a squared difference.
Reporting either would dress one cell as a curve. The refusal is the correct
behaviour, and it is the honest summary of V2's current state.

The single fused row is **not** a validated result either: at n=1, one pass, with
a lexical adjudicator standing in for Claude, row 032's ENIV of 1.648 and
confidence of 0.6731 demonstrate that the code path executes end to end. Nothing
more.

**What would change it:** V2-L1 through V2-L4. Fixing the reasoner quota alone
(V2-L5) would raise claim counts on rows that are already covered, but it cannot
create a second angle on a row where nothing was retrieved. **Coverage first,
then seeds, then metrics** -- in that order, because each is a precondition for
the next.

## How the two failure modes are told apart

Until `17f4123` the harness recorded only `len(claimset.claims)`, so a provider
error and a genuine zero were the same number. `reasoners/base.py` catches
`ReasonerError`, logs it, and returns an empty claimset carrying `.error`, so the
call looked successful. The harness now records `claimset.error` and splits the
outcome three ways. On the committed run:

| outcome | rows |
|---|---|
| failed (provider error) | 26 |
| genuinely returned zero claims | 2 (017, 020) |
| mixed fate: produced claims, still lost an angle | 2 (016, 025) |
| clean | 18 |

26 + 2 + 2 + 18 = 48. Without this split the run reads as "20 of 48 rows produced
claims, 0 fusible", which would have implied a pipeline regression that did not
occur -- the difference from the first pass is a daily quota, not a change in the
pipeline. Any future V2 run should be read through these three counts and never
through their sum.
