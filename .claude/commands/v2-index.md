# PrismFlow V2 — Semantic Multi-Angle Analysis (corrected index)

## Usage
Type /part17, /part18, ... /part24, then /part25 (demo integration) LAST.
Each command loads full context and asks Claude Code to implement that Part.

## Prerequisites (before running any V2 command)
- V1 parts 1-16 sealed and tagged v1-final (see docs/v1-parts-1-16-consolidated.md)
- Working directory: PrismFlow project root
- Dependencies (NOTE: no sqlite3 -- it is in the Python standard library):
  pip install requests aiohttp sentence-transformers scikit-learn openai anthropic rank-bm25 tiktoken scipy python-dotenv praw yfinance

## Known blockers (open)
- docs/v1-parts-1-16-consolidated.md does not exist, though this file lists
  it as a prerequisite. Record currently lives in project memory only.

## Roadmap Overview
| Command  | Title                          | Depends On | Est. Effort |
|----------|--------------------------------|------------|-------------|
| /part17  | Connector Interface            | V1 sealed  | 4 hours     |
| /part18  | Cache Layer                    | Part 17    | 3 hours     |
| /part19  | Per-Angle Retrieval Pipeline   | 17-18      | 6 hours     |
| /part20  | Angle Reasoners                | Part 19    | 6 hours     |
| /part21  | Semantic Dependence Estimation | Part 20    | 8 hours     |
| /part22  | ENIV Extension                 | Part 21    | 4 hours     |
| /part23  | Fusion Layer (Claude API)      | 20-22      | 5 hours     |
| /part24  | Calibration & Evaluation       | all above  | 10 hours    |
| /part25  | Demo Integration (same app)    | 17-24      | 3 hours     |

## Pipeline Gates (hard stops)
- Part 22: baseline ENIV >= 3.5 AND discount monotonic in correlation,
  else STOP before Part 23.
- Part 24: ECE < 0.15, redundancy Pearson r > 0.7, conflict recall > 0.7,
  cost p95 < $0.20 — all four, else STOP. No fabrication.

## Standing Rules for All V2 Parts
1. Do NOT modify V1 code (parts 1-16 frozen)
2. All V2 code under prismflow/v2/ (demo panels: prismflow/app/v2_panels.py)
3. All V2 experiments under experiments/v2/
4. All V2 results under results/v2/
5. Minimum 5 seeds for any evidence claim
6. Every part writes tests to tests/v2/
7. Reference docs/CONTRACT.md before starting any Part
8. Part 25 adds V2 to the SAME localhost demo; it must not alter V1's tabs
