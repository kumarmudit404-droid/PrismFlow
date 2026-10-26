"""Scan every blob that a push would carry, at the path git records for it.

`check_secrets.py --all` reads the working tree, which is the wrong set for a
push: a credential committed and then deleted is gone from the tree and still
in the history being pushed. This walks the objects reachable from the branch
but not from the remote, which is exactly what `git push` would send.

WHY THE BLOBS ARE NOT WRITTEN TO A TEMP TREE
--------------------------------------------
`check_secrets.ACKNOWLEDGED` and `SELF` are keyed by repo-relative path, so a
mirror under some temp root exempts nothing: the acknowledged fakes in
tests/v2/ all report, and check_secrets' own pattern definitions report too, so
the scan reads "dirty" for the wrong reason -- and a flat `git log -p` dump
reads "clean" for the wrong reason, because no line of it carries a path at
all. Each blob is therefore scanned in memory against the path git names it
with, via `check_secrets.scan_text`. Nothing is materialised on disk, which
also means a scan of history that does contain a key does not leave a copy of
it lying in a temp directory.

    python tools/push_scan.py                     # origin/main..HEAD
    python tools/push_scan.py main..my-branch     # any rev range

Exit 0 clean, 1 on a finding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import check_secrets as cs

DEFAULT_RANGE = "origin/main..HEAD"


def _suffix(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return ("." + name.rsplit(".", 1)[1].lower()) if "." in name else ""


def named_objects(root: Path, rev_range: str) -> list[tuple[str, str]]:
    """(sha, path) for every named object in the range, deduplicated.

    `rev-list --objects` names trees as well as blobs, and a directory with no
    dot in its name passes the suffix filter, so the type is checked in
    read_blobs rather than guessed from the path.
    """
    out = subprocess.run(["git", "-C", str(root), "rev-list", "--objects",
                          *rev_range.split()],
                         capture_output=True, text=True, check=True)
    seen, pairs = set(), []
    for line in out.stdout.splitlines():
        sha, _, path = line.partition(" ")
        if not path or (sha, path) in seen:
            continue
        if _suffix(path) not in cs.TEXT_SUFFIXES:
            continue
        seen.add((sha, path))
        pairs.append((sha, path))
    return pairs


def read_blobs(root: Path, pairs: list[tuple[str, str]]):
    """Yield (path, text) for the blobs among `pairs`, one git process for all.

    `cat-file --batch` answers each sha with a `<sha> <type> <size>` header,
    then exactly <size> bytes, then a newline. Non-blobs and non-UTF-8 blobs
    are skipped; a blob that is not text cannot hold a key we would recognise.
    """
    if not pairs:
        return
    proc = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                          input="\n".join(sha for sha, _ in pairs).encode(),
                          capture_output=True)
    buf, at = proc.stdout, 0
    for _, path in pairs:
        nl = buf.find(b"\n", at)
        if nl == -1:
            return
        header = buf[at:nl].split()
        at = nl + 1
        if len(header) < 3:          # "<sha> missing"
            continue
        kind, size = header[1].decode(), int(header[2])
        body, at = buf[at:at + size], at + size + 1
        if kind != "blob":
            continue
        try:
            yield path, body.decode("utf-8")
        except UnicodeDecodeError:
            continue


def scan_range(root: Path, rev_range: str,
               secrets: dict[str, str] | None = None) -> tuple[int, list[str]]:
    secrets = cs.env_values() if secrets is None else secrets
    pairs = named_objects(root, rev_range)
    scanned, findings = 0, []
    for path, text in read_blobs(root, pairs):
        scanned += 1
        findings.extend(cs.scan_text(path, text, secrets))
    return scanned, findings


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 2
    rev_range = argv[1] if len(argv) > 1 else DEFAULT_RANGE
    secrets = cs.env_values()
    scanned, findings = scan_range(cs.ROOT, rev_range, secrets)
    print("push_scan: %s" % rev_range)
    print("  %d blob version(s) scanned, %d value(s) loaded from .env"
          % (scanned, len(secrets)))
    if not findings:
        print("  clean")
        return 0
    # Deduplicated: the same blob is named by every commit that carries it, so
    # one planted key would otherwise print once per commit in the range.
    for line in sorted(set(findings)):
        print("  " + line)
    print("  %d finding(s) -- push refused" % len(set(findings)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
