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

**Updated 2026-09-28.** Two of the coverage blockers are now fixed at the root.
**V2-L2** (GitHub returned 0 on every row) and **V2-L3** (arXiv returned the same
ten records for unrelated queries) were both caused by a connector accepting a
bag of words into a field that takes a query language; both are fixed in the
connectors and re-verified at N=6, where fusible rows went 0 of 6 to 1 of 6 and
row 032 produced this pipeline's first ENIV from live retrieval (1.7634,
confidence 0.5580). **That is one row, one pass, no seeds, and it is not a
calibration result.**

What stays open, and still blocks the evaluation:

* **V2-L4** -- financial is structurally incompatible with the dataset. Unchanged
  and not fixable in a connector: the pitches are deliberately anonymised and
  yfinance needs a ticker. It is a design conflict requiring a decision.
* **V2-L5** -- the free Groq tier funds about two passes a day against the
  5-SEED RULE's five. Unchanged. Row 046 lost its claims to a TPD 429 in the
  N=6 run.
* **V2-L6** -- ECE and Brier still cannot be computed. One fused row out of six
  is not a distribution. **Coverage first, then seeds, then metrics** still
  holds; this work advanced coverage only.
* **V2-L7** -- new, and created by the V2-L2 fix: some GitHub records exceed the
  entire per-angle token budget and are dropped whole.

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

## V2-L2. RESOLVED 2026-09-28. GitHub returned 0 records because the query was an 8-12 term conjunction

**Status: fixed** in `prismflow/v2/connectors/github.py` (commit `d25d0f5`),
re-verified at N=6 (commit `185beb5`). The original entry's evidence and its
reasoning are kept below, because the zero it recorded was real.

**Original evidence (unchanged):** tech-angle provenance was `{'github': 0,
'arxiv': N}` on all 48 rows. The same held in the earlier 17-row run
(`github=0, arxiv=170`), which predates the Part 19 query-derivation addendum.
Because the zero was present on both sides of the derivation change, it was not
a regression from it -- that inference was correct.

**Root cause.** GitHub's search ANDs space-separated terms across repository
name, description and README. The connector forwarded whatever it was handed,
which after derivation was 8-12 terms, so no repository matched all of them.
Every call returned HTTP **200** with `total_count: 0` -- not a 422, not a 403,
not a 401 -- which is why it was silent for two runs. Measured live on
2026-09-28, row 002:

| terms ANDed | `total_count` |
|---|---|
| 12 / 8 / 6 / 4 / 3 / 2 | 0 |
| 1 (`Bell Media`) | 87 |
| 2, joined with explicit `OR` | 69,906 |

Two ANDed terms already return zero on some rows. Eight never had a chance.

**The GITHUB_TOKEN hypothesis was tested and is DISPROVEN.** This entry
previously said the 7-character `GITHUB_TOKEN` was "very likely a placeholder,
which is the first thing to check". It was checked. It is not the cause:
unauthenticated search answers **200** with `x-ratelimit-limit: 10` and returns
real results for a one-term query. The placeholder still costs three-fold
throughput and `_resolve_token` still rejects it, but it never had anything to do
with the zeros. Recorded plainly because the guess was wrong and the wrong guess
is in the git history.

**The fix.** `GitHubConnector.build_q()` reduces an incoming query to
`AND_TERMS = 2` quoted terms. 2 is measured, not chosen -- six rows, two per
domain, k=10, phrases quoted; "overlap" is the mean pairwise count of shared
repositories between unrelated rows, and has to be 0:

| strategy | rows with records | median `total_count` | overlap |
|---|---|---|---|
| as-built, 8-12 ANDed | 0 of 6 | 0 | 0.00 |
| all terms ORed | 0 of 6 | HTTP 422, >5 operators | n/a |
| top-1 | 6 of 6 | 1,541,034 | 2.00 |
| **top-2 ANDed** | **4 of 6** | **100** | **0.00** |
| top-3 ANDed | 2 of 6 | 0 | 0.00 |
| top-3 ORed | 6 of 6 | 1,618,457 | 3.33 |
| top-3 ORed `in:name,description` | 6 of 6 | 983,735 | 3.33 |

Top-1 and every OR variant retrieve plenty and retrieve the wrong thing: totals
in the millions, dominated by whichever mega-repository matches any one term, and
a non-zero overlap between unrelated rows. GitHub also rejects more than five
boolean operators outright ("More than five AND / OR / NOT operators were used",
HTTP 422), which rules OR-joining out on its own.

**Before / after, N=6:** GitHub tech provenance went **0 records on 6 of 6 rows
-> 10 records on 4 of 6** (032, 020, 046, 049). Rows 002 and 003 still return 0;
those are honest zeros for one specific term pair, not the systematic zero.

**What remains true.** Relevance is still limited by what leads the derived term
list. Commit `81c9550` records that all 32 newly sourced pitches follow one of
two templates -- 12 open "Pitch a", 20 open "Build a" -- so `build` or `pitch`
often takes one of the two AND slots. Dropping those verbs was measured (same
4-of-6 coverage, same 0.00 overlap, median `total_count` 100 -> 60, and visibly
more on-topic descriptions on 3 of the 4 rows returning records). It is a
`STOPWORDS` change, which rewrites every derived query and therefore every Part
18 cache key, and it is not required to fix this limitation, so it is **not
applied**. It is the cheapest remaining retrieval-quality improvement.

## V2-L3. RESOLVED 2026-09-28. The arXiv query was query-independent because the connector was sent raw prose

**Status: fixed** in `prismflow/v2/connectors/arxiv.py` (commit `d25d0f5`),
re-verified at N=6 (commit `185beb5`). The original entry said "mechanism not
identified"; it is identified now. Its evidence is kept below unchanged.

**Original evidence (unchanged), verified live on 2026-09-27:** the tech angle
was asked for three unrelated rows and returned **the same ten records every
time** -- identical set membership, differing only in BM25 order.

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

*Both of those reproduce exactly from the committed run and from live calls on
2026-09-28. The corroborating signal was the thread that led to the mechanism.*

**Root cause.** `ArxivConnector.fetch` sent `all:{query}` verbatim. `BaseAngle`
derives a query only when it EXCEEDS that connector's limit
(`prismflow/v2/angles/base.py`, `if len(query) <= limit: return query`), and
arXiv's limit of 220 is the most permissive in `CONNECTOR_QUERY_LIMITS`. So any
pitch of 220 characters or fewer reached arXiv as **prose**. arXiv binds nothing
in a prose sentence and falls back to its own default top-relevance listing --
the big-collaboration physics papers above, the same ones for every query.

The set of rows this affects is exact: **the 36 rows whose pitch is <= 220
characters are precisely the 36 rows that recorded no `query derived for arxiv`
warning.** Commit `81c9550` had already recorded why so many pitches are short:
29 of the 32 newly sourced ones fall under `MIN_PITCH_CHARS = 120`, mean ~108
characters against ~290 for the original 17.

Measured live on 2026-09-28, rows 020 / 046 / 049, k=10:

| what was sent | mean n | pairwise overlap | term-hit | generic physics |
|---|---|---|---|---|
| raw pitch (as-built) | 10.0 | **10/10 identical** | 0.07 | **1.00** |
| derived terms | 10.0 | **0** | 0.97 | 0.00 |

"generic physics" is the fraction of returned titles that are big-collaboration
physics; "term-hit" the fraction whose title contains at least one of that row's
own query terms. An empty or bare `all:` query was ruled out separately: it
returns HTTP 400 with no records, so this was never "the query matched nothing".

**The fix.** `ArxivConnector.build_search_query()` reduces whatever arrives to
terms and ORs them with an explicit `all:` on each, quoting multi-word phrases.
It never pastes prose into `search_query`, **regardless of length**. Strategies
measured:

| strategy | mean n | overlap | term-hit | generic |
|---|---|---|---|---|
| as-built, raw prose | 10.0 | 10.00 | 0.07 | 1.00 |
| terms, space separated | 10.0 | 0.00 | 0.97 | 0.00 |
| **terms, each `all:`-prefixed, OR** | **10.0** | **0.00** | **0.97** | **0.00** |
| terms, each `all:`-prefixed, AND | 0.0 | 0.00 | 0.00 | 0.00 |
| top-4 ANDed | 0.3 | 0.00 | 1.00 | 0.00 |
| top-6 ORed | 10.0 | 0.00 | 0.93 | 0.00 |

ANDing is unusable: four terms already reduce most rows to zero. Space-separated
and explicitly-ORed score identically, so arXiv's implicit operator here is OR;
the explicit form is used anyway, because it does not rely on undocumented
default behaviour and it is legible in a request log.

**`base.py` was NOT changed.** The length gate still behaves as documented. The
fix belongs in the connector because a connector should not accept prose in a
field that takes a query language, however short the prose is.

**Before / after, N=6:** distinct tech token totals went **2 of 6 -> 6 of 6**,
and rows sitting on the collapsed 1076 total went **5 -> 0**:

| row | tokens before | after |
|---|---|---|
| 002 | 860 | 889 |
| 032 | 1076 | 314 |
| 003 | 1076 | 882 |
| 020 | 1076 | 637 |
| 046 | 1076 | 659 |
| 049 | 1076 | 496 |

**A caveat on what the N=6 run proves.** On the 4 rows where GitHub now returns
records, arXiv provenance is absent: GitHub occupies the primary slot, so the
fallback chain stops before arXiv. The tech angle swapped one source for the
other rather than gaining both. The arXiv fix is therefore exercised at N=6 only
on rows 002 and 003 -- where it is visible regardless (003: 1076 -> 882 with
arXiv still serving 10 records).

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

## V2-L7. NEW, created by the V2-L2 fix: some GitHub records exceed the whole token budget

**Evidence:** the N=6 re-verification
(`results/v2/part24_reverify_n6.json`, commit `185beb5`). Row 020's tech records
dropped from 10 to 7, with the warning "3 record(s) exceed the entire 2000-token
budget on their own and can never be included (largest 5255)".

**What this shows.** A GitHub README is far longer than an arXiv abstract, so now
that GitHub returns records, some individual records cannot fit the angle's
entire evidence budget and are dropped whole. This limitation did not exist while
GitHub returned nothing -- it is a direct consequence of fixing V2-L2, and it
costs real coverage on at least one of six rows.

**What this is not.** It is not a defect in either connector: they returned the
records they were asked for. It is a budgeting and truncation question in Part
19's evidence assembly, which is outside the narrow authorisation under which
V2-L2 and V2-L3 were fixed, so it is reported and not fixed.

**What would change it:** per-record truncation instead of whole-record
rejection, or a per-record cap set as a fraction of the angle budget.

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
