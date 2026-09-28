"""Part A step 4 -- re-verify the V2-L2/L3 fix on 6 rows, 2 per domain.

  ONE PASS, N=6, GROQ, LEXICAL FUSION, NOT A PART 24 FINDING

WHY A WRAPPER AND NOT --limit 6
--------------------------------
Two reasons, both about not damaging committed evidence:

  * ``--limit N`` takes the FIRST N rows, which is 1 Startup and 5 Tech-OSS.
    The brief asks for 2 per domain.
  * the harness writes to results/v2/part24_pipeline_verification.json -- the
    committed 48-row run that docs/v2-known-limitations.md and the whole Part 25
    website cite. A --limit run would overwrite it.

So the harness runs UNMODIFIED (it is a frozen Part 24 module) and only its
dataset loader is patched, to return the six chosen rows. The output file is
moved aside afterwards and the committed one restored from git, which is checked
by sha256 rather than assumed.

The six rows are the ones the diagnosis measured on, so the strategy numbers and
this run describe the same rows.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("E:/PERSONAL/project/PrIsM FlOw")
sys.path.insert(0, str(ROOT))

COMMITTED = ROOT / "results/v2/part24_pipeline_verification.json"
DEST = ROOT / "results/v2/part24_reverify_n6.json"
PICK = ["002", "032", "003", "020", "046", "049"]   # 2 Startup, 2 Tech-OSS, 2 Financial


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


import experiments.v2.part24_pipeline_verification as H  # noqa: E402

before_sha = sha(COMMITTED)
print("committed 48-row run sha256 %s  (must be identical afterwards)" % before_sha[:16])

real_loader = H.load_evaluation_dataset
full = real_loader(str(ROOT / "data/v2/evaluation_queries.json"))
chosen = [q for q in full if q.id in PICK]
order = {r: i for i, r in enumerate(PICK)}
chosen.sort(key=lambda q: order[q.id])
if len(chosen) != len(PICK):
    raise SystemExit("expected %d rows, selected %d" % (len(PICK), len(chosen)))
print("selected rows: %s" % [(q.id, q.domain) for q in chosen])

H.load_evaluation_dataset = lambda *_a, **_k: chosen

rc = 1
try:
    rc = asyncio.run(H.main_async(None, False))
finally:
    if COMMITTED.exists():
        shutil.move(str(COMMITTED), str(DEST))
    # git is the restore guarantee: the committed file is tracked, so this
    # returns the exact bytes rather than a reconstruction.
    subprocess.run(["git", "checkout", "--", "results/v2/part24_pipeline_verification.json"],
                   cwd=str(ROOT), check=False)
    if COMMITTED.exists():
        after = sha(COMMITTED)
        print("\ncommitted 48-row run sha256 %s  %s"
              % (after[:16], "RESTORED, identical" if after == before_sha else "!!! CHANGED !!!"))
        if after != before_sha:
            raise SystemExit("the committed run was modified; refusing to continue")
    print("N=6 run written to %s" % DEST)

sys.exit(rc)
