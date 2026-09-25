# Part 20 verification status: the OpenAIReasoner code path, via Groq

Part 20 status (2026-09-25): OpenAIReasoner's code path -- request shape,
response_format={'type':'json_object'} handling, usage field names,
finish_reason mapping, parsing.py's JSON extraction, citation grounding --
verified against a live server via injection (Groq, model openai/gpt-oss-120b,
OpenAI-compatible API, no frozen file under prismflow/v2/reasoners/ modified).

NOT verified: claude_reasoner.py -- never called, nothing learned. gpt-4o -- a
different model on a different vendor's server. Groq is OpenAI-compatible, not
OpenAI, and this narrows but does not close Part 20's UNVERIFIED flag.

Never record this as a blanket 'Part 20 verified.'

The run described here was performed in a prior session. This document brings
that already-completed verification into the repo record; it does not report a
new run. The script is `experiments/v2/test_groq_reasoner_probe.py`.

## The run

One query ("semiconductor export controls"), one pass, over live NewsAPI Market
evidence. Injection only: `OpenAIReasoner.__init__` already accepts a `client`
and only constructs `AsyncOpenAI()` when none is given, so pointing a client at
Groq's OpenAI-compatible endpoint required no change to any frozen file.

| what the run recorded | value |
|---|---|
| model | `openai/gpt-oss-120b` (Groq) |
| claim citations grounded in the evidence | 4/4 |
| ungrounded citations | 0 |
| `response_format={"type": "json_object"}` | honoured |
| `finish_reason` | mapped to `stop_reason=stop` |
| `parsing.py` strategy that succeeded | `direct` |

Token usage and wall-clock latency for this run are **not recorded**. The
prior-session record preserves the citation-grounding result and the mapping
outcomes above and nothing further, so no token count or timing is stated here
rather than reconstructed. If the original console output is recovered, add the
figures here and cite it.

## What it did NOT establish

- **nothing about `claude_reasoner.py`.** It was never called.
- **nothing about `gpt-4o`.** That is a different model on a different
  vendor's server.

## Why the flag stays open

Closing it needs one live call per reasoner against its own provider and its
own model. Under the NEVER FABRICATE rule a reasoner result must not be
stubbed, and a third provider's success must not be reported as the flag
closing.

The two real keys are in `.env` and both AUTHENTICATE; the accounts have a zero
balance. Anthropic returns HTTP 400 `invalid_request_error` "credit balance is
too low" (a bad key would be 401 `authentication_error`); OpenAI returns HTTP
429 `insufficient_quota` / `credit_balance_exhausted` with no `retry-after`,
while `GET /v1/models` returns 200 with 126 models. Check the billing balance
before re-diagnosing the keys as malformed: `experiments/v2/test_api_keys_manual.py`
separates a format failure from an auth failure from a billing failure.

`GROQ_API_KEY` lives in `.env`. Nothing in `prismflow/` reads it, and it is
deliberately absent from `.env.template` for that reason; only the probe script
above reads it.

## This is not a finding

Per `docs/CONTRACT.md` section 5, a finding needs five seeds and mean/sd. This
was one query on one pass with no seeds, and claim text produced by a model is
not a measurement. Nothing here is citable as a Part 24 result, and the script
writes no file under `results/` so that it cannot become one by accident.

## Re-running it

The committed script defaults to a **static fixture** for the evidence, so a
re-run spends one Groq completion and nothing else -- in particular it spends no
NewsAPI request and reads no rotating credential. That default exercises the
code path, not the original run's inputs: the fixture text is written for the
script and is not a transcript of the articles the original run retrieved, so
claim text or citation counts obtained from it do not reproduce the 4/4 result
above. `--live-newsapi` restores the original live Market retrieval and spends
NewsAPI requests against `NEWSAPI_KEY` (free tier: 100 per day, with no
remaining-quota field in any response).

The Market/NewsAPI connector is not blocked by any of this.
