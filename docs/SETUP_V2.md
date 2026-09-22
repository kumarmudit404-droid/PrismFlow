# PrismFlow V2 Commands — Corrected Setup

## What changed from the original SETUP.md
1. Removed `sqlite3` from the pip install line — it is part of the Python
   standard library and has NO pip package; the original line errors out.
2. Made extraction location explicit and gated on the V1 seal.
3. Added Part 25 (demo integration) so V2 appears as new tabs in the SAME
   localhost Streamlit app, matching the existing theme — the original pack
   was backend-only and never touched app.py.
4. Requirements are APPENDED to the existing requirements.txt, never replaced.

## Step 0 — Seal V1 FIRST (do not skip)
    cd "E:\PERSONAL\project\PrIsM FlOw"
    git status                 # must be clean, all Part 16 work committed
    git tag v1-final
    git push origin v1-final   # your clean rollback point

## Step 1 — Extract into the V1 repo root (same-repo, on purpose)
Extract the zip so its .claude/commands/*.md files land in the project's
EXISTING .claude/commands/ alongside part01..part16. Result:

    PrismFlow/
    ├── .claude/commands/
    │   ├── part01.md ... part16.md     (V1, already there)
    │   ├── part17.md ... part24.md     (V2 backend, from this pack)
    │   ├── part25.md                   (V2 demo integration, added here)
    │   └── v2-index.md
    ├── prismflow/          (V1 code — FROZEN)
    ├── prismflow/v2/       (all V2 code lives here — created by parts 17+)
    ├── app.py              (existing demo — Part 25 ADDS tabs, never rewrites)
    └── docs/

## Step 2 — Install V2 dependencies (append, don't replace)
    .venv\Scripts\pip.exe install requests aiohttp sentence-transformers scikit-learn openai anthropic rank-bm25 tiktoken scipy python-dotenv praw yfinance
    # NOTE: no sqlite3 — it ships with Python.

Then append to requirements.txt (do not overwrite V1's lines):
    requests>=2.31
    aiohttp>=3.9
    sentence-transformers>=2.6
    scikit-learn>=1.4
    openai>=1.30
    anthropic>=0.30
    rank-bm25>=0.2
    tiktoken>=0.6
    scipy>=1.12
    python-dotenv>=1.0
    praw>=7.7.0
    yfinance>=0.2.32

## Step 3 — API keys in a .env file (never commit it)
Create .env in the repo root:
    GITHUB_TOKEN=...
    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...
    NEWSAPI_KEY=...
    REDDIT_CLIENT_ID=...
    REDDIT_CLIENT_SECRET=...
Confirm .env is in .gitignore (add it if not — it holds secrets).

## Step 4 — Restart Claude Code and confirm commands loaded
    claude
    /help        # should list /part17 ... /part25 and /v2-index

## Step 5 — Run part-wise, sealing between each
    /v2-index    # read the roadmap
    /part17      # then, after review + tests + commit:
    /part18
    ... through /part24
    /part25      # LAST — adds V2 tabs to the existing localhost demo

## Between every part (same discipline as V1)
1. Review the generated code
2. pytest tests/v2/ -v
3. git commit -m "Part N: <title>"
4. Update docs/v2-progress.md with any deviation
5. Only then move to the next part

## The one localhost, both versions
After Part 25, the SAME command runs everything:
    .venv\Scripts\streamlit.exe run app.py --server.fileWatcherType none
V1 tabs (Independence budget / Run and results / Saved figures) stay exactly
as they are; V2 adds a "Semantic analysis" section using the same dark theme,
same sidebar pattern, same "nothing here is evidence" honesty banner.
No second app, no second port, no data-file clash — V2 writes only under
results/v2/ and reads only under prismflow/v2/.
