"""Refuse to let a credential reach a commit.

Two checks, and the second is the one that matters:

1. **Provider shapes.** The same patterns web/build_data.py aborts on, so a
   key with a recognisable prefix is caught wherever it appears.

2. **The actual values in .env.** Shapes only catch providers we thought of.
   This reads the real .env, takes each value, and looks for that exact string
   in the files being checked. A NewsAPI key that looks like a hex blob, or a
   cookie with no prefix at all, is caught by this and by nothing else.

The .env values are never printed. A hit reports the VARIABLE NAME and where
the value turned up, which is all anyone needs to fix it and is the most that
can be said without leaking the thing being protected.

    python tools/check_secrets.py              # staged changes (pre-commit)
    python tools/check_secrets.py --all        # every tracked file
    python tools/check_secrets.py path [path]  # named paths

Exit 0 clean, 1 on a finding, 2 on a usage problem.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"

# Kept in step with web/build_data.py's SECRET_PATTERNS. If one list grows, so
# should the other.
PROVIDER_PATTERNS = [
    # The leading \b is load-bearing and is NOT in build_data.py's copy: without
    # it, `sk-` matches inside "risk-coverage" and reports prismflow/evaluation/
    # metrics.py as a leaking key.
    ("openai / anthropic", re.compile(r"\bsk-[A-Za-z0-9]{8,}")),
    ("groq", re.compile(r"\bgsk_[A-Za-z0-9]{8,}")),
    ("google", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    ("newsapi-shaped", re.compile(r"\b[0-9a-f]{32}\b")),
    ("github", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]

# A placeholder is not a secret. .env.template is built out of these.
PLACEHOLDERS = {"", "...", "changeme", "your-key-here", "none", "null"}

# Only values whose variable name says "credential" are searched for. .env also
# holds ordinary configuration -- CACHE_DIR=./cache/v2, CACHE_DB=sqlite:///...,
# LOG_LEVEL=INFO -- and those strings appear all over the source legitimately.
# Searching for them would bury a real finding under dozens of false ones,
# which is how a hook ends up disabled.
CREDENTIAL_NAME = re.compile(
    r"(?:API_)?(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|PASS|COOKIE|CREDENTIAL|AUTH|CLIENT_ID)$")

# Reviewed, and allowed to stay: exact (path, text) pairs. Deliberately exact,
# so the same shape in any other file, or a different value in these files, is
# still a finding. Every entry here is a value someone has confirmed is fake.
ACKNOWLEDGED = {
    ("tests/v2/test_connectors.py", "ghp_realdeadbeefrealdeadbeef"),
    ("tests/v2/test_newsapi_connector.py", "0123456789abcdef0123456789abcdef"),
    ("tests/v2/test_newsapi_connector.py", "abcdef0123456789abcdef0123456789"),
}

# Scanning our own definitions would report every pattern as a finding.
SELF = {"tools/check_secrets.py", "web/build_data.py", "web/build_dist.py",
        ".env.template"}

TEXT_SUFFIXES = {".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt",
                 ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".csv", ""}


def env_values() -> dict[str, str]:
    """Map value -> variable name, for every non-placeholder value in .env."""
    if not ENV.exists():
        return {}
    values = {}
    for line in ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, raw = line.partition("=")
        name = name.strip()
        if not CREDENTIAL_NAME.search(name):
            continue
        # Trailing `# comment` on a value line is a comment, not part of it.
        raw = raw.split("#")[0].strip().strip("'\"")
        if raw.lower() in PLACEHOLDERS or raw.endswith("...") or len(raw) < 8:
            continue
        values[raw] = name
    return values


def git(*args) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), *args],
                         capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l.strip()]


def targets(argv: list[str]) -> list[Path]:
    if "--all" in argv:
        rels = git("ls-files")
    elif len(argv) > 1:
        rels = argv[1:]
    else:
        rels = git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    paths = []
    for rel in rels:
        p = (ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            paths.append(p)
    return paths


def scan_text(rel: str, text: str, secrets: dict[str, str]) -> list[str]:
    """Apply both checks to one file's text, at the path it is recorded under.

    Split out of check() so tools/push_scan.py can run these rules over a blob
    it never writes to disk. ACKNOWLEDGED and SELF are keyed by repo-relative
    path, so `rel` has to be that path and not wherever the bytes came from.
    """
    if rel in SELF or rel == ".env":
        return []
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, pattern in PROVIDER_PATTERNS:
            for match in pattern.finditer(line):
                hit = match.group(0)
                if (rel, hit) in ACKNOWLEDGED:
                    continue
                if label == "newsapi-shaped" and _is_content_hash(line, match):
                    continue
                findings.append(
                    "%s:%d  matches a %s key pattern" % (rel, lineno, label))
        for value, name in secrets.items():
            if value in line:
                findings.append(
                    "%s:%d  contains the value of %s from .env" % (rel, lineno, name))
    return findings


def check(paths: list[Path]) -> list[str]:
    secrets = env_values()
    findings = []
    for path in paths:
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        findings.extend(scan_text(rel, text, secrets))
    return findings


def _is_content_hash(line: str, match: re.Match) -> bool:
    """A 32-hex run is a NewsAPI key shape, but it is also every MD5 ever.

    Recorded API fixtures are full of them: gravatar avatars, etags, and paper
    filenames. Both tests below are structural -- they key off what surrounds
    the hex, never off the hex itself -- so a bare key on its own line still
    reports.
    """
    low = line.lower()
    if any(token in low for token in
           ("gravatar.com", "md5", "etag", '"sha', "sha1", "sha256", "checksum")):
        return True
    # Inside a URL: take the delimiter-bounded token holding the match.
    start = max((line.rfind(c, 0, match.start()) for c in ' "\'<>()[],'), default=-1)
    end = min((p for p in (line.find(c, match.end()) for c in ' "\'<>()[],')
               if p != -1), default=len(line))
    token = line[start + 1:end].lower()
    return token.startswith(("http://", "https://")) or "://" in token


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 2
    paths = targets(argv)
    secrets = env_values()
    if not paths:
        print("check_secrets: nothing to check")
        return 0
    findings = check(paths)
    print("check_secrets: %d file(s), %d value(s) loaded from .env"
          % (len(paths), len(secrets)))
    if not ENV.exists():
        print("  note: no .env present, so only provider shapes were checked")
    if not findings:
        print("  clean")
        return 0
    for line in findings:
        print("  " + line)
    print("  %d finding(s) -- commit refused" % len(findings))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
