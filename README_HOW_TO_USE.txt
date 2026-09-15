================================================================================
PRISMFLOW - 15-DAY EXECUTION PACK
16 part prompts, a verification gate, and two automation routes.
================================================================================

CONTENTS
  parts/       PART_01 .. PART_16 prompt files, one per part
  scripts/     install_commands.sh, gate.sh, run_headless.sh
  README_HOW_TO_USE.txt   this file


--------------------------------------------------------------------------------
THE CORRECTION THAT MATTERS MOST
--------------------------------------------------------------------------------
Pasting a prompt takes three seconds. That is not your bottleneck.
What needs automating is the VERIFICATION GATE between parts: tests passing, no
out-of-scope file touched, no frozen module rewritten, no fabricated number left
in the repo. scripts/gate.sh is the real automation in this pack.

Automate the scaffolding. Supervise the science.


--------------------------------------------------------------------------------
SETUP (5 minutes, once)
--------------------------------------------------------------------------------
  mkdir PrismFlow && cd PrismFlow && git init
  cp -r /path/to/this/pack/parts   .
  cp -r /path/to/this/pack/scripts .
  chmod +x scripts/*.sh
  ./scripts/install_commands.sh parts

Restart Claude Code. Type /part01.


--------------------------------------------------------------------------------
ROUTE A - SLASH COMMANDS  (recommended for Parts 01-16)
--------------------------------------------------------------------------------
install_commands.sh writes each prompt into .claude/commands/partNN.md.
Project-scoped commands live in .claude/commands/ in the repo root and are
committed to version control, so the whole plan travels with the project.

Daily loop:

    /part05                 <- run the part inside Claude Code
    [read the output]       <- YOU look at what it did
    ./scripts/gate.sh       <- verify before moving on
    git add -A && git commit -m "part 05"
    /part06

Note: Anthropic has unified custom commands with skills, and
.claude/skills/<name>/SKILL.md is the newer recommended format. The
.claude/commands/ format still works and is simpler for single-file prompts,
which is why this pack uses it.


--------------------------------------------------------------------------------
ROUTE B - HEADLESS RUNNER  (scaffolding parts and sweeps only)
--------------------------------------------------------------------------------
Adding -p (or --print) to any claude command makes it non-interactive: it reads
the prompt, runs to completion, prints the result, and exits with a code a script
can check. --output-format json returns a structured envelope with is_error,
result, session_id and cost, so the script checks a field rather than parsing
prose.

    ./scripts/run_headless.sh 01 02 03

The runner stops at the first gate failure and commits after each passing part.

DO NOT chain the experiment parts unattended: 06 (clone), 09 (Chorus),
13 (tail dependence), 14 (adaptive adversary). Their entire value is that a human
looks at the resulting curve. A pipeline that generates the clone plot and never
shows it to you has defeated the point of the project.

On permissions: --permission-mode bypassPermissions clears every approval prompt,
but current docs recommend it only for a sandbox with no real user data. The
runner uses an --allowedTools allowlist instead, which is the safer default.


--------------------------------------------------------------------------------
ROUTE C - MANUAL PASTE
--------------------------------------------------------------------------------
Open parts/PART_NN_*.txt, paste into Claude Code, run gate.sh afterwards.
Zero setup, maximum control. Perfectly respectable.


--------------------------------------------------------------------------------
15-DAY SCHEDULE
--------------------------------------------------------------------------------
 Day  Part  Content                                          Research value
 ---  ----  -----------------------------------------------  --------------
  1   01    Contract, skeleton, config                       -
  1-2 02    Synthetic generator with known rho               CALIBRATION STANDARD
  2   03    View encoders                                    -
  3   04    Evidential head, Dempster fusion, baseline       -
  4   05    Dependence, ENIV, discount, ESTIMATOR VALIDATION HIGH
  5   06    CLONE EXPERIMENT                                 DECISIVE
  6   07    Calibration suite                                HIGH
  6-7 08    Missing and noisy views                          MEDIUM
  7-8 09    CHORUS ATTACK                                    DECISIVE
  8-9 10    Defended model, suspicion detector, matrix       HIGH
  9-10 11   Shared/private + HSIC                            MEDIUM
 10-11 12   Per-sample ENIV, two-timescale training          HIGH
 11-12 13   Tail dependence / copula                         HEADLINE FIGURE
 12   14    ADAPTIVE ADVERSARY                               CRITICAL
 13   15    Real dataset, full matrix                        MEDIUM
 14-15 16   Demo, docs, audit, paper outline                 -

HARD GATES - do not pass without them
  Part 05: the estimator-vs-truth plot must show ENIV tracking analytic n_eff.
  Part 06: the clone curve must exist. If both curves are flat, STOP and diagnose.

IF YOU FALL BEHIND, cut in this order:
  1st  Part 15 (real dataset)   - synthetic results still stand
  2nd  Part 11 (HSIC)           - not on the critical path
  3rd  Part 12 (per-sample)     - V1 global ENIV is a valid fallback
  NEVER cut Part 05, 06, 09, or 14.


--------------------------------------------------------------------------------
THE 12 DEFECTS THIS PACK FIXES IN THE ORIGINAL GPT PLAN
--------------------------------------------------------------------------------
 1  Research at the end of the queue (clone was Part 13 of 21)   -> now Part 06
 2  No dataset named anywhere in 21 parts                        -> Parts 02, 15
 3  No ground-truth validation of ENIV                           -> Parts 02, 05
 4  No adaptive-adversary evaluation                             -> Part 14
 5  PySide6 + PyInstaller demo (1-2 days, zero research value)   -> Streamlit
 6  No mention of the estimator-gaming problem                   -> Parts 05, 12
 7  No seed / variance protocol                                  -> 5 seeds, all parts
 8  Four contract docs read by every session                     -> one CONTRACT.md
 9  Parts 08/09/10 = 200 lines across three sessions             -> merged
10  No numerical-stability rule for Dempster's 1-C denominator   -> Part 04
11  No warning about EDL KL annealing collapse                   -> Part 04
12  No isolation rule for the suspicion detector                 -> Part 10 + gate.sh


--------------------------------------------------------------------------------
CONTEXT DISCIPLINE (why the prompts are written the way they are)
--------------------------------------------------------------------------------
  - one fresh session per Part; never carry context between parts
  - every prompt names the exact files that may be created or modified
  - previously-passing modules are FROZEN; suspected defects are REPORTED, not fixed
  - no Part both implements and refactors (refactoring reads the whole repo)
  - CLAUDE.md stays under 60 lines; it is prepended to every session
  - run the full test suite at the START of each day, not only at the end

CHARACTERISTIC FAILURE MODE
  The agent decides an earlier module is "inconsistent" and rewrites it, silently
  breaking three downstream modules. The FROZEN rule and gate.sh exist to catch
  this. If you see a Part editing files outside its declared scope, stop it.


--------------------------------------------------------------------------------
THE ONE RULE ABOVE ALL OTHERS
--------------------------------------------------------------------------------
Never fabricate a number. Not in a plot, not in a table, not in a README, not as
a placeholder you intend to replace later.

A project with three honest results and four items marked NOT VERIFIED is
credible. A project with sixteen impressive results, one of which is invented, is
worthless - and the invented one is always the one someone asks about.
================================================================================
