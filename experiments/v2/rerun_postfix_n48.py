r"""Re-run the 48-row pipeline on CURRENT connector code -- post V2-L2/L3 fix.

  ONE PASS, N=48, GROQ, LEXICAL FUSION, NOT A PART 24 FINDING

WHY THIS EXISTS
---------------
results/v2/part24_pipeline_verification.json and its ``_pass1`` sibling were
both produced BEFORE the Part 19 commit that fixed V2-L2 (GitHub returned 0 on
every row because the query was an 8-12 term conjunction). The only run that
exercises the fix is results/v2/part24_reverify_n6.json, which covers 6 rows
and shows GitHub returning 10 records on each. Six rows cannot say whether the
48-row retrieval wall moved. This re-runs all 48 on current code so the
before/after is measured rather than extrapolated.

WHY A WRAPPER AND NOT A FLAG ON THE HARNESS
-------------------------------------------
Identical reasoning to experiments/v2/reverify_n6.py, which established this
pattern: the harness writes to results/v2/part24_pipeline_verification.json,
the committed pre-fix run that docs/v2-known-limitations.md and the Part 25
website both cite. It is a frozen Part 24 module and is NOT modified here --
not even to add an ``--out`` argument. Its output file is moved aside
afterwards and the committed one restored from git, verified by sha256 rather
than assumed.

WHAT IT IS NOT
--------------
Not Part 24 evidence, and nothing here changes that: one pass, ``seeds: null``,
Groq ``openai/gpt-oss-120b`` through the OpenAIReasoner code path, and the
offline LexicalAdjudicator standing in for Claude. The harness stamps
``is_part24_evidence: false`` into the output itself. This run exists to show
whether the retrieval wall described in V2-L1 and V2-L6 has moved, not to
produce a calibration result.

  .venv\Scripts\python.exe experiments/v2/rerun_postfix_n48.py
  .venv\Scripts\python.exe experiments/v2/rerun_postfix_n48.py --pick 002,032 \
      --dest results/v2/_smoke.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COMMITTED = ROOT / "results/v2/part24_pipeline_verification.json"
DEFAULT_DEST = ROOT / "results/v2/part24_pipeline_verification_postfix.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pick", default=None,
                    help="comma-separated row ids, for a smoke check; "
                         "omitted means all 48")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="where the run's output is moved to")
    args = ap.parse_args()
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = ROOT / dest

    import experiments.v2.part24_pipeline_verification as H

    if not COMMITTED.exists():
        print("the committed pre-fix run is missing; refusing to run a job "
              "whose restore step cannot be verified")
        return 1
    before_sha = sha(COMMITTED)
    print("committed pre-fix 48-row run sha256 %s  (must be identical afterwards)"
          % before_sha[:16])

    full = H.load_evaluation_dataset(str(ROOT / "data/v2/evaluation_queries.json"))
    if args.pick:
        want = [r.strip() for r in args.pick.split(",") if r.strip()]
        order = {r: i for i, r in enumerate(want)}
        chosen = sorted((q for q in full if q.id in want), key=lambda q: order[q.id])
        if len(chosen) != len(want):
            missing = sorted(set(want) - {q.id for q in chosen})
            print("rows not in the dataset: %s" % missing)
            return 1
        H.load_evaluation_dataset = lambda *_a, **_k: chosen
        print("SMOKE CHECK: %s" % [(q.id, q.domain) for q in chosen])
    else:
        print("all %d committed rows" % len(full))

    rc = 1
    try:
        rc = asyncio.run(H.main_async(None, False))
    finally:
        if COMMITTED.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(COMMITTED), str(dest))
        # git is the restore guarantee: the committed file is tracked, so this
        # returns the exact bytes rather than a reconstruction.
        subprocess.run(["git", "checkout", "--",
                        "results/v2/part24_pipeline_verification.json"],
                       cwd=str(ROOT), check=False)
        if COMMITTED.exists():
            after = sha(COMMITTED)
            print("\ncommitted pre-fix run sha256 %s  %s"
                  % (after[:16],
                     "RESTORED, identical" if after == before_sha
                     else "!!! CHANGED !!!"))
            if after != before_sha:
                raise SystemExit("the committed run was modified; refusing to continue")
        print("this run written to %s" % dest)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
