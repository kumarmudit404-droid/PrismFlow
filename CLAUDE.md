# PrismFlow - Project Memory

Thesis: Agreement between views should contribute evidence in proportion to
the effective independence of those views, not in proportion to how many
views agree.

Data flow:
`views -> encoders -> evidence -> dependence -> ENIV -> discount -> fusion -> prediction + confidence + diagnostics`

## Rules

- FROZEN MODULES: previously-passing modules are frozen. If you suspect a
  defect in an earlier module, REPORT it and stop — do not fix it as a side
  effect of a later Part.
- 5-SEED RULE: every experiment runs a minimum of 5 seeds and reports mean
  and standard deviation. Single-seed results are not evidence and must
  never be reported as a finding.
- NEVER FABRICATE: never fabricate experimental results, not even as
  placeholders.

See `docs/CONTRACT.md` for the full contract (component classification,
tensor shape conventions, reproducibility rules).

## Environment

- GREP CRASHES ON `-i` + MULTIPLE `-e`: this box's GNU grep 3.0 (Git for
  Windows / MSYS2) aborts with SIGABRT (exit 134) when case-insensitive
  matching is combined with two or more `-e` patterns — it returns no output,
  so piping through `cat` or discarding stderr makes a crash look like a clean
  "no matches". Use a single `-e`, or `-E "a|b"` alternation, or ripgrep.
