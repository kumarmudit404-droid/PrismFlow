# Part 25 — V2 Demo Integration (same localhost, same theme)

## Objective
Add PrismFlow V2 (semantic multi-angle analysis) to the EXISTING Streamlit
demo as new tabs, using the same dark theme, the same sidebar pattern, and the
same "nothing on this screen is evidence" honesty banner as V1. One app, one
localhost, no second process. This is the LAST V2 part — run only after Parts
17-24 are complete and sealed.

## Prerequisites
- Parts 17-24 complete; results exist under results/v2/
- V1 app.py runs (streamlit run app.py --server.fileWatcherType none)
- V1 is tagged v1-final

## Scope
CREATE only:
- prismflow/app/v2_panels.py    -- V2 rendering panels (DUMB layer, like panels.py)
- tests/v2/test_v2_app.py       -- asserts the V2 demo layer contains no model/API logic

MODIFY (surgically, additively -- do NOT rewrite):
- app.py -- add a top-level mode switch (V1 / V2) OR add V2 tabs alongside the
  existing "Independence budget / Run and results / Saved figures" tabs.
  The existing V1 tabs and their behaviour MUST remain byte-for-byte unchanged
  in what they render. Add, never replace.

DO NOT MODIFY:
- Any V1 code under prismflow/models, prismflow/statistics, prismflow/eniv,
  prismflow/evaluation
- prismflow/app/panels.py (V1's demo panels)
- Any V1 experiment or result

## Requirements

### Theme and pattern consistency (this is the point of the part)
- Reuse the exact same page config, colours, and layout idioms already in
  app.py. Do not introduce a new colour scheme or a second st.set_page_config.
- Reuse the existing honesty-banner idiom: V2's panels must carry an equivalent
  of "PrismFlow is a research prototype... see docs before quoting any number",
  pointing at docs/V2_KNOWN_LIMITATIONS.md.
- The sidebar keeps its existing V1 controls; V2 gets its own clearly-labelled
  section or a mode toggle at the very top -- your choice, but the V1 path must
  still be reachable and unchanged.

### V2 demo flow (read-only over committed results by default)
1. A "Semantic analysis" area with:
   - a query input box (text)
   - an angle selector (Tech / Market / Financial / Regulatory / Sentiment)
   - a "Run analysis" button
2. Two run modes, clearly labelled:
   - REPLAY (default, no API cost): load a pre-computed example from
     results/v2/end_to_end/ or results/v2/calibration/ and render it. This is
     what runs with no keys and no spend.
   - LIVE (explicit opt-in, requires API keys): calls the real V2 pipeline
     (prismflow/v2/...). Must show an estimated-cost warning BEFORE running and
     require a second confirm click, because each live query costs real money.
3. Render, reusing the same visual grammar as V1:
   - the semantic dependence matrix as a heatmap (like V1's dependence heatmap)
   - semantic ENIV vs nominal angle count (like V1's independence budget panel)
   - per-angle claims with confidence, and the DISCOUNTED fused confidence
   - flagged conflicts, shown prominently (never hidden)
   - the audit trail (raw confidences, ENIV, discount factor) in an expander

### The demo layer stays a demo layer (same rule as V1's test_app.py)
- v2_panels.py takes already-computed values and renders them. It must not
  call an LLM, embed text, estimate dependence, or compute ENIV itself.
- app.py's V2 path may ORCHESTRATE the existing V2 engine (call
  prismflow/v2/... functions) but must not reimplement any of their logic.
- test_v2_app.py asserts (AST-based, like V1's test): v2_panels.py imports no
  anthropic/openai/torch.nn and defines no fusion/ENIV/dependence math.

## Tests to Write
1. test_v2_panels_has_no_engine_logic -- AST check: no LLM/embedding/ENIV calls
   inside v2_panels.py
2. test_app_v1_path_unchanged -- the V1 tab-rendering functions app.py exposes
   still exist with the same names and signatures (guards against V1 regression)
3. test_replay_mode_needs_no_api_keys -- REPLAY renders from results/v2/ with
   all API-key env vars unset
4. test_live_mode_requires_confirmation -- LIVE path is gated behind an explicit
   confirm flag (no accidental spend)

## Success Criteria
- streamlit run app.py --server.fileWatcherType none launches ONE app with BOTH
  V1 and V2 reachable, same theme, same port.
- With no API keys set, REPLAY mode renders a full V2 example end to end.
- V1's three tabs render exactly as before (no visual or behavioural change).
- All V1 tests still pass (full suite green), plus the 4 new V2 app tests.

## Deliverable
Update docs/v2-progress.md:
- Screenshot description of the integrated app (V1 tabs + V2 section)
- Confirmation V1 path is unchanged (test_app_v1_path_unchanged passing)
- The REPLAY-vs-LIVE cost-safety design, stated explicitly
- Note: "This concludes V2. The demo now serves V1 and V2 from one localhost."

## Standing rules (same as all V2 parts)
1. V1 is frozen; add, never rewrite.
2. All V2 code under prismflow/v2/ or prismflow/app/v2_panels.py only.
3. Never fabricate results; REPLAY shows real committed numbers only.
4. Every part writes tests.
5. LIVE mode must never run without an explicit, cost-aware confirmation.
