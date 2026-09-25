"""Validate the three API keys in .env: format, then one minimal live call each.

Connectivity and credential verification only. Writes NO results JSON and
nothing it prints is a Part 24 finding -- a single unseeded pass is not
evidence under CONTRACT.md section 5.

NEVER prints a key, any part of a key, or any model/API response content.
Lengths and pass/fail are all that leave this script.

Live-call cost, if all three formats pass: 1 NewsAPI request (of 100/day on the
free tier), 1 Claude call at max_tokens=1, 1 OpenAI call at max_tokens=1.

    python experiments/v2/test_api_keys_manual.py            # format + live
    python experiments/v2/test_api_keys_manual.py --format-only
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The live calls use the models Part 20 will actually call, read from the
# reasoners themselves -- validating a model this project never uses would
# prove nothing about Part 20.
from prismflow.v2.reasoners.claude_reasoner import DEFAULT_CLAUDE_MODEL
from prismflow.v2.reasoners.openai_reasoner import DEFAULT_OPENAI_MODEL

NEWSAPI_RE = re.compile(r"^[0-9a-f]{32}$")


def read_env_keys(path: Path) -> dict:
    """Read KEY=VALUE lines from .env. No os.environ side effects."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ---------------------------------------------------------------- format


def check_newsapi_format(v: str):
    if not v:
        return False, "absent or empty"
    if not NEWSAPI_RE.match(v):
        kind = ("wrong length" if len(v) != 32
                else "contains non-lowercase-hex characters")
        return False, f"expected 32 lowercase hex chars, got {len(v)} ({kind})"
    return True, "32 lowercase hex chars"


def check_anthropic_format(v: str):
    if not v:
        return False, "absent or empty"
    problems = []
    if not v.startswith("sk-ant-"):
        problems.append("does not start with sk-ant-")
    if len(v) < 100:
        problems.append(f"length {len(v)} is under the ~100 char minimum")
    if problems:
        return False, "; ".join(problems)
    return True, f"sk-ant- prefix, length {len(v)}"


def check_openai_format(v: str):
    if not v:
        return False, "absent or empty"
    problems = []
    if not v.startswith("sk-"):
        problems.append("does not start with sk-")
    if len(v) < 48:
        problems.append(f"length {len(v)} is under the ~48 char minimum")
    if problems:
        return False, "; ".join(problems)
    return True, f"sk- prefix, length {len(v)}"


# ------------------------------------------------------------- live calls


def live_newsapi(key: str):
    """One headline, pageSize=1. Reports counts and status only."""
    import requests

    r = requests.get(
        "https://newsapi.org/v2/top-headlines",
        params={"country": "us", "pageSize": 1},
        headers={"X-Api-Key": key},
        timeout=30,
    )
    if r.status_code != 200:
        body = r.json() if "json" in r.headers.get("content-type", "") else {}
        return False, (f"HTTP {r.status_code} code="
                       f"{body.get('code', '?')} (message not printed)")
    payload = r.json()
    if payload.get("status") != "ok":
        return False, f"status={payload.get('status')!r}"
    return True, (f"HTTP 200, status=ok, "
                  f"{len(payload.get('articles', []))} article returned "
                  f"(content not printed)")


def live_anthropic(key: str):
    """One max_tokens=1 call on the model Part 20's Claude reasoner uses."""
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=DEFAULT_CLAUDE_MODEL,
        max_tokens=1,
        # Sonnet 5 runs adaptive thinking by default, which has no room under a
        # 1-token ceiling. Disabled so this is a genuine 1-token completion.
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": "hi"}],
    )
    # stop_reason will be max_tokens at this ceiling -- that is a success for a
    # credential check: the request authenticated and the model replied.
    return True, (f"HTTP 200, model={resp.model}, "
                  f"stop_reason={resp.stop_reason}, "
                  f"output_tokens={resp.usage.output_tokens} "
                  f"(content not printed)")


def live_openai(key: str):
    """One max_tokens=1 call on the model Part 20's OpenAI reasoner uses."""
    import openai

    client = openai.OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    return True, (f"HTTP 200, model={resp.model}, "
                  f"finish_reason={resp.choices[0].finish_reason}, "
                  f"completion_tokens={resp.usage.completion_tokens} "
                  f"(content not printed)")


def describe_api_error(exc: Exception) -> str:
    """Render an API failure without printing a key.

    The provider error envelope (status, type, code, message) is what separates
    "this key is wrong" from "this account has no credits" -- collapsing both
    into a class name makes the script useless for the only question it exists
    to answer. Envelopes never contain the credential.
    """
    parts = [type(exc).__name__]
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"HTTP {status}")
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        # The OpenAI SDK leaves .body None on some 429s; the envelope is still
        # in the raw response, and it is the part that names the cause.
        resp = getattr(exc, "response", None)
        text = getattr(resp, "text", None)
        if text:
            try:
                body = json.loads(text)
            except ValueError:
                body = None
    if isinstance(body, dict):
        err = body.get("error") or {}
        for field in ("type", "code"):
            if err.get(field):
                parts.append(f"{field}={err[field]}")
        if err.get("message"):
            parts.append(f"-- {err['message']}")
    return " ".join(parts)


CHECKS = [
    ("NEWSAPI_KEY", check_newsapi_format, live_newsapi),
    ("ANTHROPIC_API_KEY", check_anthropic_format, live_anthropic),
    ("OPENAI_API_KEY", check_openai_format, live_openai),
]


def main(argv) -> int:
    format_only = "--format-only" in argv

    print(f"SOURCE: {ENV_PATH}")
    print(f"models: anthropic={DEFAULT_CLAUDE_MODEL} openai={DEFAULT_OPENAI_MODEL}")
    if not ENV_PATH.is_file():
        print("FAIL: .env does not exist", file=sys.stderr)
        return 2
    env = read_env_keys(ENV_PATH)

    print("\n--- STEP 1: FORMAT ---")
    fmt = {}
    for name, fmt_fn, _ in CHECKS:
        ok, detail = fmt_fn(env.get(name, ""))
        fmt[name] = ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    failed_fmt = [n for n, ok in fmt.items() if not ok]
    if failed_fmt:
        print(f"\nFORMAT FAILED: {', '.join(failed_fmt)}")
        print("No live call attempted for a key that fails format.")

    print("\n--- STEP 2: ONE LIVE CALL PER FORMAT-PASSING KEY ---")
    if format_only:
        print("  skipped (--format-only)")
        return 1 if failed_fmt else 0

    live = {}
    for name, _, live_fn in CHECKS:
        if not fmt[name]:
            print(f"  SKIP  {name}: format failed")
            continue
        try:
            ok, detail = live_fn(env[name])
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, describe_api_error(exc)
        live[name] = ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    print("\n--- SUMMARY ---")
    all_ok = True
    for name, _, _ in CHECKS:
        if not fmt[name]:
            state, all_ok = "FAIL (format)", False
        elif not live.get(name):
            state, all_ok = "FAIL (live call)", False
        else:
            state = "PASS"
        print(f"  {name}: {state}")
    print(f"\nALL THREE PASS: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
