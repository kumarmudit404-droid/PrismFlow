# Part 19 retrieval and Part 24's dataset expect different query shapes

**Status: finding.** Recorded 2026-09-26, from a single live verification run
over the 17 labeled rows in `data/v2/evaluation_queries.json`.

This is a structural gap between two Parts that were never reconciled with each
other. It is **not a bug in either one**: Part 19's retrieval behaves as
specified, Part 24's dataset field is as documented, and the connectors are
enforcing limits their upstream APIs genuinely impose. Nothing here needs
fixing in place; something needs *deciding* between them.

## The mismatch in one sentence

`EvaluationQuery.query_text` returns `idea_pitch`, which is long-form prose by
design, and the four connectors that currently work expect short keyword
queries, two of them with hard length limits.

`docs/part24-dataset-schema.md` defines `idea_pitch` as "the pitch as it would
have read BEFORE the outcome was known", and the workbook's Instructions tab
asks for "a short pitch/description, roughly the way you'd phrase a query to
PrismFlow". Both are satisfied by the current rows. The rows average a few
hundred characters; the longest measured in the run was **585 characters**.

Nothing sits between the two. There is no query-shortening, keyword-extraction
or entity-selection step anywhere between `load_evaluation_dataset()` and
`BaseAngle.retrieve()`. The pitch text is handed to the connectors verbatim.

## What the run showed

One pass, 17 rows, four angles wired live, Groq (`openai/gpt-oss-120b`) as the
reasoner via the `OpenAIReasoner` code path. **Not Part 24 evidence** -- see
"Status of these numbers" below.

**1 of 17 rows produced a fusible result.** Dependence and ENIV are undefined
below two contributing angles, so fusion never ran on the other 16.

| angles that produced claims | rows | ids |
|---|---|---|
| 0 | 7 | 005, 007, 012, 013, 014, 015, 016 |
| 1 | 9 | 002, 003, 004, 006, 008, 009, 010, 011, 018 |
| 2 | 1 | 017 |

No ECE, Brier or conflict figures were computed. The harness refused rather
than reporting metrics derived from a single row.

## Evidence, per connector

**GitHub -- HTTP 422 on every row.**

```
[tech] github (primary) failed: QueryError: query rejected as unprocessable
       (source=github, status=422)
```

GitHub's search API rejected the pitch text outright on all 17 rows. The tech
angle's fallback chain worked exactly as designed and degraded to arXiv, which
returned 10 records per row -- so `tech` was the only angle retrieving anything
on a typical row, and it was doing so from its *secondary* source throughout.

**NewsAPI -- an explicit, documented length limit.**

```
newsapi: queryTooLong: Your query is too long (585 chars). Please reduce your
         query to 500 chars, or split it into multiple smaller requests.
         (source=newsapi, status=400) -- backing off 1.00s then retrying (attempt 2/4)
...
newsapi: giving up after 4 attempts
[market] newsapi (primary) failed: UpstreamError: queryTooLong ...
```

This is the clearest statement of the gap available: the upstream API names the
limit (500 characters) and the actual size (585). The retry/backoff logic
behaved correctly and could not help, because the request is not transiently
failing -- it is too long, and it will be too long every time. Four attempts
with 1s/2s/4s backoff were spent on a deterministic rejection.

**StackExchange and yfinance -- zero records, no error.**

```
[sentiment]  no records retrieved from any of 1 configured source(s)
[financial]  no records retrieved from any of 1 configured source(s)
```

Both connected, queried and returned nothing on 16 of 17 rows. No auth failure,
no rate limit, no exception -- a long prose query simply matches nothing in a
Q&A corpus or a ticker lookup. This is the quieter and more dangerous half of
the finding: a connector returning zero records looks identical to a topic with
no coverage.

**Regulatory -- no connector exists.**

```
[regulatory] no connector is configured for the regulatory angle
             (slots tried: primary, secondary, tertiary); returning empty evidence
```

Separate from the query-shape problem and already recorded in
`prismflow/v2/angles/regulatory_angle.py`: "No connector exists yet; inject one
when it is built." data.gov, EUR-Lex and RBI are all unimplemented. Only four
of the five angles can retrieve at all today.

## Row 017: a technically successful run with a meaningless result

The single row that reached fusion is the one that most clearly shows why row
counts are not a measure of health.

Row 017 (PEP 679, Rejected) reached two angles: `tech`, via arXiv as usual, and
`financial`, via a **single** yfinance record. The reranker flagged that record
itself:

```
[financial] records=1  claims=1
            no bm25 term overlap between the query and any record;
            retrieval order preserved
```

Zero term overlap between the query and the only record retrieved. The angle
then produced a claim from it, dependence was estimated across the two angles,
and the pipeline reported **ENIV 1.987, discount 0.994, confidence 0.7998** --
numbers that are arithmetically correct and mean nothing, because one of the
two "independent views" they rest on was reasoning over a document unrelated to
the query.

Nothing in the run failed. No exception was raised, no warning escalated, and
had the other 16 rows behaved this way the harness would have produced a
complete set of calibration metrics. Those metrics would have been worthless,
and nothing in the output would have said so. **The reranker's own warning is
the only signal that anything was wrong**, and it is a per-angle note that no
gate currently reads.

That is the strongest argument for fixing the query shape deliberately rather
than by loosening a threshold until rows start passing: the failure mode of
this gap is not an error, it is a plausible number.

## Status of these numbers

Not Part 24 evidence under `docs/CONTRACT.md` section 5, and not partial
evidence pending more rows:

- **n = 17** of the eventual 50, 16 of them Tech-OSS. No domain balance.
- **Groq, not Claude and not OpenAI**, both unfunded. Groq is
  OpenAI-*compatible*, not OpenAI; this says nothing about `gpt-4o` and nothing
  about Claude.
- **Fusion adjudication was the offline `LexicalAdjudicator`.** Part 23's
  Claude path was not exercised.
- **One pass, no seeds** -- see below.

### Why there is no seed loop

Nothing on this path varies as a function of a seed, so a seed loop would
select nothing, which is how the 5-SEED RULE gets faked.

Connector results come from Part 18's cache, whose stated invariant is that a
cached result is indistinguishable from a fresh one. The reranker sorts stably
and preserves retrieval order on ties, deliberately, so dependence is not a
tie-breaking artefact. The sentence-transformer encoder is deterministic. Part
21's dependence and Part 22's ENIV contain no RNG. Fusion's candidate
generation is encoder-based and the offline adjudicator is deterministic.

That leaves the reasoner, and `prismflow/v2/reasoners/openai_reasoner.py` sends
`model`, `max_tokens`, `messages` and `response_format` -- **no `temperature`,
no `top_p`, no `seed`**. The provider therefore samples at its own default and
repeated runs do differ, but a seed index neither selects that variation nor
reproduces it. Reporting such runs as "5 seeds" would dress uncontrolled
sampler noise as a reproducible measurement -- the inverse of the `std = 0.0`
defect caught in Parts 21 and 22, and harder to spot, because an absent
measurement at least looks suspiciously stable while noise looks like data.

Where seeds in this repo *do* select something -- `experiments/v2`'s fusion
e2e and ENIV redundancy runs -- they draw synthetic fixtures: banked queries,
planted contradictions, confidence bands. Over 17 fixed real rows there is no
fixture left to draw.

## What this does and does not settle

It settles that the retrieval layer and the evaluation dataset are currently
incompatible, and that no Part 24 gate can be computed until that is resolved.

It does not settle how to resolve it. The options differ in what they make
Part 24 *measure*, which is why this document records the finding and stops:

- derive a short keyword query from each pitch, and accept that Part 24 then
  measures the pipeline *plus* that derivation step;
- add a second authored field to the dataset (a retrieval query beside the
  pitch), and accept the hand-authoring cost and the annotator variance;
- narrow the angles to the ones whose corpora suit this dataset, and accept a
  smaller ENIV ceiling;
- source rows whose pitches are naturally short, and accept the selection
  effect on what kinds of ideas are in the benchmark.

Each changes the claim Part 24 ends up making. None should be folded into an
unrelated commit.

## Reproducing

```
.venv\Scripts\python.exe experiments/v2/part24_pipeline_verification.py
.venv\Scripts\python.exe experiments/v2/part24_pipeline_verification.py --limit 2
```

Spends live GitHub, arXiv, NewsAPI, StackExchange and yfinance requests, and
one Groq completion per angle that retrieved evidence. Re-running produces a
*different* run rather than a reproduction of this one, for the sampling reason
given above.
