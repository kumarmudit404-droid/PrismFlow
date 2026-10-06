"""Copy the runtime site into web/dist/ and nothing else.

The rule is an allowlist, not a blocklist: a file is copied only if some entry
below names it. Anything new in web/ is therefore excluded until someone adds
it here, which is the behaviour we want -- a developer file that appears in
web/ must never reach a deploy just because nobody remembered to exclude it.

    python web/build_dist.py            # build, then scan
    python web/build_dist.py --scan     # scan an existing dist/ only

The scan is the second half of the job. It greps everything in dist/ for the
things that should never be served: absolute developer paths, the account
name, email addresses, loopback hosts with ports, bare IP addresses, .env
variable names, and provider key shapes. It exits non-zero if it finds any.

dist/ is ignored by git; this writes it, it is never committed.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

# --------------------------------------------------------------------------
# The allowlist
# --------------------------------------------------------------------------

# Pages served at the site root. theme_preview.html is deliberately absent: it
# is a palette-comparison page for whoever is editing the theme, not part of
# the site, and it says so in its own first line.
PAGES = ["index.html", "scores.html", "404.html"]

# (directory relative to web/, allowed suffixes). A suffix of "" means every
# file in that directory is allowed.
TREES = [
    ("assets/css", {".css"}),
    ("assets/js", {".js"}),
    ("assets/img", {".png", ".svg", ".jpg", ".jpeg", ".webp", ".ico"}),
    # Vendored libraries ship with their licence files, which is the condition
    # we hold them under. README.md is included here and only here: it records
    # each library's version and licence, and that attribution travels with the
    # code it describes.
    ("assets/vendor", {".js", ".md", ""}),
    # Every chart on the site reads from here. .xlsx is the Part 24 workbook,
    # which the dataset section offers as a download.
    ("data", {".json", ".png", ".xlsx"}),
]

# Refused even where a tree above would otherwise allow them. Source maps leak
# original sources; .py is tooling; the rest are editor and OS droppings.
DENY_SUFFIXES = {".map", ".py", ".pyc", ".pyo", ".log", ".bak", ".orig", ".swp"}
DENY_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
DENY_DIR_NAMES = {"__pycache__", "shots", "dist", ".git", ".idea", ".vscode"}


def allowed(path: Path) -> bool:
    rel = path.relative_to(HERE)
    if any(part in DENY_DIR_NAMES for part in rel.parts):
        return False
    if path.name in DENY_NAMES or path.suffix.lower() in DENY_SUFFIXES:
        return False
    if len(rel.parts) == 1:
        return rel.name in PAGES
    for tree, suffixes in TREES:
        tree_parts = Path(tree).parts
        if rel.parts[:len(tree_parts)] == tree_parts:
            return "" in suffixes or path.suffix.lower() in suffixes
    return False


def build() -> list[Path]:
    if DIST.exists():
        shutil.rmtree(DIST)
    copied = []
    for src in sorted(HERE.rglob("*")):
        if not src.is_file() or not allowed(src):
            continue
        dst = DIST / src.relative_to(HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


# --------------------------------------------------------------------------
# The leak scan
# --------------------------------------------------------------------------

SCAN_SUFFIXES = {".html", ".js", ".css", ".json", ".md", ".txt", ".svg"}

PATTERNS = [
    ("absolute path", re.compile(r"(?:[A-Za-z]:[\\/]{1,2}(?:Users|PERSONAL|Program)|/Users/|/home/)[^\s\"'<>)]*")),
    ("account name", re.compile(r"\bMUDIT\b", re.I)),
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("loopback with port", re.compile(r"\b(?:localhost|127\.0\.0\.1)(?::\d+)\b")),
    ("ip address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("env variable name", re.compile(r"\b[A-Z][A-Z0-9]{2,}_(?:API_)?(?:KEY|SECRET|TOKEN|PASSWORD|CLIENT_ID|CLIENT_SECRET)\b")),
    ("provider key", re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")),
]

# Reviewed, and allowed to stay. Each entry is an exact (file, text) pair, so
# the same string appearing anywhere else is still reported.
#
# REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are variable NAMES with no values,
# inside a quotation from prismflow/v2/angles/sentiment_angle.py explaining
# why the Reddit connector cannot work from this network. That quotation is
# limitation text and is not edited. Naming a variable is not disclosing it.
ACKNOWLEDGED = {
    ("index.html", "REDDIT_CLIENT_ID"),
    ("index.html", "REDDIT_CLIENT_SECRET"),
}


def _texts(path: Path):
    """Yield (label, text) for a file, looking inside .xlsx rather than past it.

    A workbook is a zip. Scanning only its name would let anything inside it
    ship unread, so its text parts are unpacked and scanned too.
    """
    if path.suffix.lower() == ".xlsx":
        import zipfile
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith((".xml", ".rels", ".txt")):
                        yield name, zf.read(name).decode("utf-8", "replace")
        except zipfile.BadZipFile:
            return
        return
    if path.suffix.lower() not in SCAN_SUFFIXES:
        return
    try:
        yield None, path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return


def scan(root: Path) -> tuple[list, list]:
    hits, known = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        for part, text in _texts(path):
            where = rel if part is None else "%s!%s" % (rel, part)
            for lineno, line in enumerate(text.splitlines(), 1):
                for label, pattern in PATTERNS:
                    for match in pattern.finditer(line):
                        value = match.group(0)
                        if label == "ip address" and _looks_like_version(line, match):
                            continue
                        row = (label, where, lineno, value)
                        if (rel, value) in ACKNOWLEDGED:
                            known.append(row)
                        else:
                            hits.append(row)
    return hits, known


def _looks_like_version(line: str, match: re.Match) -> bool:
    """A dotted quad inside a version string is not a host.

    Only treat it as a version when the surrounding text says so, so a real
    address written next to the word "version" is still reported.
    """
    context = line[max(0, match.start() - 24):match.end() + 8].lower()
    return any(word in context for word in ("version", "v=", "sha", "revision"))


def main() -> int:
    scan_only = "--scan" in sys.argv
    if not scan_only:
        copied = build()
        total = sum(p.stat().st_size for p in copied)
        print("build_dist -> %s" % DIST)
        print("  %d files, %.1f MB" % (len(copied), total / 1e6))
        for page in PAGES:
            mark = "ok" if (DIST / page).exists() else "MISSING"
            print("  %-16s %s" % (page, mark))
    if not DIST.exists():
        print("no dist/ to scan; run without --scan first")
        return 2

    hits, known = scan(DIST)
    print("\nleak scan over dist/")
    for label, rel, lineno, value in known:
        print("  acknowledged: %-18s %s:%d  %s" % (label, rel, lineno, value))
    if not hits:
        print("  none" + (" beyond the acknowledged entries above" if known else ""))
        return 0
    for label, rel, lineno, value in hits:
        print("  %-20s %s:%d  %s" % (label, rel, lineno, value))
    print("  %d match(es)" % len(hits))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
