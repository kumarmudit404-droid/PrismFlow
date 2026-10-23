"""Headless page checks for the Part 25 site. Python stdlib only, no npm.

    .venv\\Scripts\\python.exe web/verify_page.py shot  out.png [--path /] [--width 375]
    .venv\\Scripts\\python.exe web/verify_page.py reveal
    .venv\\Scripts\\python.exe web/verify_page.py real
    .venv\\Scripts\\python.exe web/verify_page.py eval  "expression"

WHY THIS EXISTS RATHER THAN PLAYWRIGHT OR PUPPETEER
---------------------------------------------------
The project takes no browser-automation dependency: the reveal guarantee and the
accessibility work in phases (c) and (d) were measured with headless Edge driven
over the DevTools Protocol, and reproducing those measurements must not require
`npm install`. So this speaks CDP directly, over a WebSocket implemented on
`socket` and `struct` from the standard library. It is deliberately minimal --
one frame per message, no continuation frames, no compression -- because the only
traffic is small JSON and one base64 screenshot.

Edge is launched with a throwaway profile so it never touches the user's own,
and it is always terminated, including on error.

WHAT THE REVEAL MATRIX CHECKS
-----------------------------
`web/assets/css/app.css` hides every reveal target while
`:root[data-reveal="armed"]` is set, and `reveal-failsafe.js` owns both the
attribute and the deadline that clears it. The pass condition is that NO target
is left at computed opacity 0 once the deadline has passed, under each fault
injected below. A page that ends with hidden content has failed, whatever else
it does.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
PORT = 9411
# The site under test. 825 is web/serve.py's own default, and these must
# agree or every check silently measures a chrome-error page instead of the
# site -- which is what a stale 8251 here did. Override with PRISMFLOW_WEB_PORT.
# 127.0.0.1 and never localhost: localhost resolves to ::1 first on this box
# and serve.py binds IPv4 only.
BASE = "http://127.0.0.1:%s" % os.environ.get("PRISMFLOW_WEB_PORT", "825")


def browser_path() -> str:
    for candidate in EDGE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise SystemExit("no Edge or Chrome binary found; looked in:\n  " +
                     "\n  ".join(EDGE_CANDIDATES))


# --------------------------------------------------------------------------
# A minimal WebSocket client. Enough for CDP and nothing more.
# --------------------------------------------------------------------------

class WS:
    def __init__(self, url: str) -> None:
        match = re.match(r"ws://([^:/]+):(\d+)(/.*)", url)
        if not match:
            raise ValueError("unexpected debugger url: %r" % url)
        host, port, path = match.group(1), int(match.group(2)), match.group(3)
        self.sock = socket.create_connection((host, port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n" % (path, host, port, key)
        ).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise IOError("debugger closed during handshake")
            buf += chunk
        if b"101" not in buf.split(b"\r\n")[0]:
            raise IOError("websocket upgrade refused: %r" % buf[:120])
        self.rest = buf.split(b"\r\n\r\n", 1)[1]
        self.next_id = 0

    def _recv(self, n: int) -> bytes:
        out, self.rest = self.rest[:n], self.rest[n:]
        while len(out) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise IOError("debugger closed")
            need = n - len(out)
            out += chunk[:need]
            self.rest += chunk[need:]
        return out

    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        header = b"\x81"
        mask = os.urandom(4)
        length = len(data)
        if length < 126:
            header += struct.pack("!B", 0x80 | length)
        elif length < (1 << 16):
            header += struct.pack("!BH", 0x80 | 126, length)
        else:
            header += struct.pack("!BQ", 0x80 | 127, length)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(header + mask + masked)

    def recv(self) -> dict:
        while True:
            first, second = struct.unpack("!BB", self._recv(2))
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv(8))[0]
            if second & 0x80:
                self._recv(4)          # server frames are unmasked in practice
            body = self._recv(length)
            if opcode == 0x8:
                raise IOError("debugger sent close")
            if opcode == 0x9:          # ping -> pong
                self.sock.sendall(b"\x8a\x80" + os.urandom(4))
                continue
            if opcode in (0x1, 0x2):
                return json.loads(body.decode("utf-8", "replace"))

    def call(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self.next_id += 1
        mine = self.next_id
        self.send({"id": mine, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv()
            if msg.get("id") == mine:
                if "error" in msg:
                    raise IOError("%s -> %s" % (method, msg["error"]))
                return msg.get("result", {})
        raise TimeoutError(method)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class Browser:
    """Headless Edge over CDP.

    ``gpu=True`` drops ``--disable-gpu`` and forces the D3D11 ANGLE backend, so
    WebGL runs on the real adapter instead of SwiftShader. That distinction is
    the whole point of the phase (f) step 2 check: a screenshot of a WebGL
    scene taken on a software rasteriser proves nothing about whether it draws
    correctly on this laptop, and the phase (c) prism was reported as broken
    "on a real GPU as well as in software" without that ever being separated.
    ``verify_page.py gpu`` prints the unmasked renderer string so which one ran
    is never in doubt.
    """

    def __init__(self, extra_args=(), gpu: bool = False) -> None:
        self.profile = tempfile.mkdtemp(prefix="prismflow_verify_")
        gpu_args = ([
            "--use-angle=d3d11",
            "--enable-gpu-rasterization",
            # Headless Chromium refuses the GPU for WebGL on some Windows
            # configurations without this. It lifts the blocklist; it does not
            # force software, and the renderer string proves which one ran.
            "--ignore-gpu-blocklist",
        ] if gpu else ["--disable-gpu"])
        self.proc = subprocess.Popen(
            [browser_path(), "--headless=new", *gpu_args,
             "--remote-debugging-port=%d" % PORT,
             "--user-data-dir=%s" % self.profile,
             "--no-first-run", "--no-default-browser-check",
             "--window-size=1440,2200", "about:blank", *extra_args],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws = None
        target = None
        for _ in range(60):
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/json/list" % PORT, timeout=2) as r:
                    pages = [t for t in json.load(r) if t.get("type") == "page"]
                if pages:
                    target = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.4)
        if not target:
            self.kill()
            raise SystemExit("headless browser did not expose a debugger target")
        self.ws = WS(target)
        self.ws.call("Page.enable")
        self.ws.call("Runtime.enable")

    def goto(self, url: str, settle: float = 1.5) -> None:
        self.ws.call("Page.navigate", {"url": url})
        time.sleep(settle)

    def js(self, expression: str, timeout: float = 60.0):
        res = self.ws.call("Runtime.evaluate", {
            "expression": expression, "returnByValue": True,
            "awaitPromise": True}, timeout=timeout)
        if res.get("exceptionDetails"):
            return {"__error__": str(res["exceptionDetails"])[:300]}
        return res.get("result", {}).get("value")

    def shot(self, out: Path, full: bool = True) -> Path:
        if full:
            metrics = self.js(
                "JSON.stringify({w:Math.max(document.documentElement.scrollWidth,1440),"
                "h:document.documentElement.scrollHeight})")
            dim = json.loads(metrics)
            self.ws.call("Emulation.setDeviceMetricsOverride", {
                "width": int(dim["w"]), "height": min(int(dim["h"]), 20000),
                "deviceScaleFactor": 1, "mobile": False})
        data = self.ws.call("Page.captureScreenshot", {"format": "png"})["data"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data))
        return out

    def kill(self) -> None:
        if self.ws:
            self.ws.close()
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        shutil.rmtree(self.profile, ignore_errors=True)


# --------------------------------------------------------------------------
# The reveal matrix
# --------------------------------------------------------------------------

COUNT_HIDDEN = """
(() => {
  const sel = '.chapter__lede, .panel, .limit';
  const all = Array.from(document.querySelectorAll(sel));
  const hidden = all.filter(e => parseFloat(getComputedStyle(e).opacity) < 0.01);
  return JSON.stringify({
    targets: all.length,
    hidden: hidden.length,
    armed: document.documentElement.getAttribute('data-reveal'),
    chars: (document.body.innerText || '').length
  });
})()
"""

# Each condition is a fault injected BEFORE the page's own scripts run, using
# CDP's addScriptToEvaluateOnNewDocument, so no file in web/ is modified.
CONDITIONS = [
    ("baseline", ""),
    ("throw right after data-motion is set", """
     new MutationObserver((m, o) => {
       if (document.documentElement.getAttribute('data-motion') === 'live') {
         o.disconnect();
         window.__boom = setTimeout(() => { throw new Error('injected'); }, 0);
       }
     }).observe(document.documentElement, {attributes:true, attributeFilter:['data-motion']});
     """),
    ("anime.js import fails", """
     const realFetch = window.fetch;
     Object.defineProperty(window, 'fetch', {value: realFetch, writable: true});
     document.addEventListener('DOMContentLoaded', () => {}, {once:true});
     """),
    ("motion.js throws on load", """
     window.addEventListener('error', e => e.preventDefault());
     const o = new MutationObserver(() => {
       const s = document.querySelector('script[src*="motion.js"]');
       if (s) { o.disconnect(); s.setAttribute('src', '/assets/js/__missing_motion.js'); }
     });
     o.observe(document.documentElement, {childList:true, subtree:true});
     """),
    ("requestAnimationFrame never fires", "window.requestAnimationFrame = function(){ return 0; };"),
    ("IntersectionObserver is missing", "delete window.IntersectionObserver;"),
    ("reveal-failsafe.js is missing", """
     const o = new MutationObserver(() => {
       const s = document.querySelector('script[src*="reveal-failsafe.js"]');
       if (s) { o.disconnect(); s.setAttribute('src', '/assets/js/__missing_failsafe.js'); }
     });
     o.observe(document.documentElement, {childList:true, subtree:true});
     """),
    ("prefers-reduced-motion: reduce", "__EMULATE_REDUCED_MOTION__"),
]


def reveal_matrix(path: str = "/") -> int:
    print("REVEAL MATRIX -- pass condition: hidden == 0 after the deadline")
    print("  sampled 9s after load; app.css hides only while data-reveal=\"armed\"\n")
    print("  %-42s %-9s %-8s %-8s %s" % ("condition", "targets", "hidden", "armed", "innerText"))
    failures = 0
    for name, fault in CONDITIONS:
        b = Browser()
        try:
            if fault == "__EMULATE_REDUCED_MOTION__":
                b.ws.call("Emulation.setEmulatedMedia", {"features": [
                    {"name": "prefers-reduced-motion", "value": "reduce"}]})
            elif fault.strip():
                b.ws.call("Page.addScriptToEvaluateOnNewDocument", {"source": fault})
            b.goto(BASE + path, settle=0.5)
            time.sleep(9.0)
            raw = b.js(COUNT_HIDDEN)
            info = json.loads(raw) if isinstance(raw, str) else {"targets": "?", "hidden": "?"}
            bad = not isinstance(info.get("hidden"), int) or info["hidden"] != 0
            failures += 1 if bad else 0
            print("  %-42s %-9s %-8s %-8s %-8s %s"
                  % (name, info.get("targets"), info.get("hidden"),
                     info.get("armed"), info.get("chars"), "FAIL" if bad else ""))
        finally:
            b.kill()
    print("\n  %s" % ("all conditions pass: no content is left hidden"
                      if not failures else "%d CONDITION(S) FAILED" % failures))
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Part 31: the real-data section
# --------------------------------------------------------------------------
#
# Four things a passing reveal matrix would not catch, because they are about
# CONTENT rather than visibility:
#
#   1. every mount filled -- a placeholder still reading "loading committed
#      data" is a fetch that failed, and looks identical to a slow one;
#   2. no console error and no uncaught exception anywhere on the page;
#   3. no horizontal scroll at 375px, the brief's width;
#   4. every claim cell carries a verdict word from the allowed vocabulary
#      AND the rule and the source file beside it. A verdict shown on its own
#      is exactly the failure this section exists to prevent, so it is gated
#      rather than trusted. Both negative words are accepted, because they
#      say different things: NO EVIDENCE is measured-but-inconclusive, and
#      "needs more data" is not-run-or-not-measurable. Which one a given cell
#      should carry is the builder's decision and is argued in its rule; this
#      check only enforces that the word is one of the four and that the rule
#      is there to be read.
#
# The error collector is installed through addScriptToEvaluateOnNewDocument,
# so it is in place before the page's own scripts run. A hook the page itself
# installs cannot see an error thrown while the page is loading, which is the
# case that matters here.

REAL_MOUNTS = ["real-ait", "real-lanl", "real-claims", "limits-v1-realdata"]

COLLECT_ERRORS = """
window.__pfErrors = [];
window.addEventListener('error', function (e) {
  window.__pfErrors.push('error: ' + (e.message || e.type));
});
window.addEventListener('unhandledrejection', function (e) {
  window.__pfErrors.push('unhandled rejection: ' + String(e.reason));
});
(function () {
  var real = console.error;
  console.error = function () {
    window.__pfErrors.push('console.error: ' +
      Array.prototype.join.call(arguments, ' '));
    return real.apply(console, arguments);
  };
})();
"""

REAL_CHECKS = """
(() => {
  const out = {mounts: {}, claims: {ok: 0, bad: []}};
  for (const id of %s) {
    const n = document.getElementById(id);
    out.mounts[id] = !n ? "MISSING"
      : /placeholder/.test(n.className) ? "PLACEHOLDER"
      : n.innerHTML.length < 200 ? "EMPTY"
      : "ok";
  }
  const words = ["CONFIRMS", "CONTRADICTS", "NO EVIDENCE", "needs more data"];
  const cells = document.querySelectorAll(".claims tbody td");
  for (const td of cells) {
    const pill = td.querySelector(".pill");
    const rule = td.querySelector(".claims__rule");
    const src = td.querySelector(".src code");
    const word = pill ? pill.textContent.trim() : null;
    if (!word || words.indexOf(word) < 0) { out.claims.bad.push("no verdict word"); }
    else if (!rule || rule.textContent.length < 40) { out.claims.bad.push(word + ": no rule"); }
    else if (!src) { out.claims.bad.push(word + ": no source file"); }
    else { out.claims.ok += 1; }
  }
  out.claims.total = cells.length;
  // Both negative words must survive on the page. They were one word once,
  // and collapsing them again would report an experiment that was never run
  // as a negative finding -- which is the thing this vocabulary exists to
  // stop, so it is gated rather than left to review.
  out.claims.words = Array.from(new Set(
    Array.from(cells).map(function (td) {
      var p = td.querySelector(".pill");
      return p ? p.textContent.trim() : "";
    }))).sort();
  out.errors = (window.__pfErrors || []).slice(0, 10);
  // Per SECTION, not per document. The page as a whole already scrolls
  // sideways at phone width -- scores.html does it too, with nothing from
  // this Part on it -- so a document-level figure would gate the new section
  // on behaviour it did not cause and cannot fix without touching sections
  // this Part is not allowed to change. What IS this Part's business is
  // whether #real is worse than the sections already shipped, so the check
  // measures every section and compares.
  out.sections = {};
  document.querySelectorAll("section.chapter").forEach(function (s) {
    out.sections[s.id] = s.scrollWidth - s.clientWidth;
  });
  return JSON.stringify(out);
})()
""" % json.dumps(REAL_MOUNTS)


def real_section(path: str = "/") -> int:
    print("REAL-DATA SECTION -- mounts filled, claim cells ruled, no overflow")
    print("  sampled 7s after load, at both widths")
    print("")
    failures = []
    for label, width in (("desktop 1440px", 1440), ("phone 375px", 375)):
        b = Browser()
        try:
            b.ws.call("Page.addScriptToEvaluateOnNewDocument",
                      {"source": COLLECT_ERRORS})
            b.ws.call("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": 900, "deviceScaleFactor": 1,
                "mobile": width < 700})
            b.goto(BASE + path, settle=2.0)
            time.sleep(7.0)
            raw = b.js(REAL_CHECKS)
            info = json.loads(raw) if isinstance(raw, str) else {}

            for mount_id, state in sorted(info.get("mounts", {}).items()):
                bad = state != "ok"
                if bad:
                    failures.append("%s %s %s" % (label, mount_id, state))
                print("  %-14s %-22s %-12s %s"
                      % (label, mount_id, state, "FAIL" if bad else ""))

            claims = info.get("claims", {})
            bad = bool(claims.get("bad")) or claims.get("total", 0) < 9
            if bad:
                failures.append("%s claim cells: %r" % (label, claims.get("bad")))
            print("  %-14s %-22s %d of %d with verdict+rule+source %s"
                  % (label, "claim cells", claims.get("ok", 0),
                     claims.get("total", 0), "FAIL" if bad else ""))

            words = claims.get("words") or []
            missing = [w for w in ("NO EVIDENCE", "needs more data")
                       if w not in words]
            if missing:
                failures.append("%s verdict vocabulary collapsed: %s absent"
                                % (label, ", ".join(missing)))
            print("  %-14s %-22s %s %s"
                  % (label, "verdict vocabulary", ", ".join(words),
                     "FAIL" if missing else ""))

            sections = info.get("sections") or {}
            mine = sections.get("real")
            others = {k: v for k, v in sections.items() if k != "real"}
            baseline = min(others.values()) if others else 0
            bad = mine is None or mine > max(baseline, 1)
            if bad:
                failures.append(
                    "%s #real overflows by %spx against a %dpx floor on the "
                    "existing sections" % (label, mine, baseline))
            print("  %-14s %-22s #real %spx, existing sections %s %s"
                  % (label, "horizontal overflow", mine,
                     ", ".join("%s %dpx" % kv for kv in sorted(others.items())),
                     "FAIL" if bad else ""))

            errors = info.get("errors") or []
            failures += ["%s console: %s" % (label, e) for e in errors]
            print("  %-14s %-22s %d %s"
                  % (label, "console errors", len(errors), "FAIL" if errors else ""))
        finally:
            b.kill()
    print("")
    if failures:
        print("  %d FAILURE(S):" % len(failures))
        for f in failures:
            print("    " + f)
    else:
        print("  all real-data checks pass")
    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "reveal":
        return reveal_matrix(sys.argv[2] if len(sys.argv) > 2 else "/")
    if mode == "real":
        return real_section(sys.argv[2] if len(sys.argv) > 2 else "/")
    if mode == "shot":
        # shot OUT [--path /] [--at SELECTOR] [--tokens /assets/css/x.css]
        out = Path(sys.argv[2])
        args = sys.argv[3:]
        path = "/"
        selector = None
        tokens = None
        # The brief's phone width is a reviewable artefact, so it is a flag
        # rather than a one-off script: a screenshot nobody can regenerate is
        # not evidence about the layout.
        width = 1440
        for i, a in enumerate(args):
            if a == "--path" and i + 1 < len(args):
                path = args[i + 1]
            if a == "--at" and i + 1 < len(args):
                selector = args[i + 1]
            if a == "--tokens" and i + 1 < len(args):
                tokens = args[i + 1]
            if a == "--width" and i + 1 < len(args):
                width = int(args[i + 1])
        b = Browser()
        try:
            if width != 1440:
                b.ws.call("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": 900, "deviceScaleFactor": 1,
                    "mobile": width < 700})
            if tokens:
                # A candidate palette photographed on the REAL site, without
                # editing a single file in web/.
                #
                # It has to go in BEFORE the page's own scripts run, not after.
                # motion.js reads the angle tokens once at init to paint the
                # ambient canvas; a sheet appended after load re-colours the CSS
                # and leaves the canvas painted in the OLD palette, which looks
                # like the theme half-failed. Measured: the first attempt did
                # exactly that -- dark canvas over a light base.
                b.ws.call("Page.addScriptToEvaluateOnNewDocument", {"source":
                    "document.addEventListener('readystatechange', () => {}, {once:true});"
                    "(function add() {"
                    "  if (!document.head) { return requestAnimationFrame(add); }"
                    "  const l = document.createElement('link');"
                    "  l.rel = 'stylesheet'; l.href = " + json.dumps(tokens) + ";"
                    "  document.head.appendChild(l);"
                    "})();"})
            b.goto(BASE + path, settle=2.0)
            time.sleep(7.0)          # past the reveal deadline, so nothing is mid-fade
            if selector:
                # Clip to one element's own box. A full-page shot of this site is
                # 20000px tall, which is a file rather than something anyone can
                # read, so each view is captured where it lives.
                box = b.js(
                    "(() => { const e = document.querySelector(" +
                    json.dumps(selector) + "); if (!e) return null;"
                    " const r = e.getBoundingClientRect(), pad = 2;"
                    " return JSON.stringify({x: Math.max(0, r.left + scrollX - pad),"
                    " y: r.top + scrollY, w: Math.ceil(r.width) + pad * 2,"
                    " h: Math.ceil(r.height)}); })()")
                if not box:
                    raise SystemExit("selector not found: %s" % selector)
                rect = json.loads(box)
                b.ws.call("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": min(int(rect["h"]) + 40, 16000),
                    "deviceScaleFactor": 1, "mobile": width < 700})
                time.sleep(0.6)
                # The resize re-flowed the page: the rect above describes the
                # OLD layout. Measuring again is the only way the clip lands
                # on the element the caller asked for.
                box = b.js(
                    "(() => { const e = document.querySelector(" +
                    json.dumps(selector) + "); if (!e) return null;"
                    " const r = e.getBoundingClientRect(), pad = 2;"
                    " return JSON.stringify({x: Math.max(0, r.left + scrollX - pad),"
                    " y: r.top + scrollY, w: Math.ceil(r.width) + pad * 2,"
                    " h: Math.ceil(r.height)}); })()")
                rect = json.loads(box)
                data = b.ws.call("Page.captureScreenshot", {
                    "format": "png", "captureBeyondViewport": True,
                    "clip": {"x": rect["x"], "y": rect["y"], "width": rect["w"],
                             "height": min(rect["h"], 16000), "scale": 1}})["data"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(data))
            else:
                b.shot(out)
            print("wrote %s (%d bytes)" % (out, out.stat().st_size))
        finally:
            b.kill()
        return 0
    if mode == "gpu":
        # gpu EXPR [PATH] -- same as `eval`, but on the real adapter.
        b = Browser(gpu=True)
        try:
            b.goto(BASE + (sys.argv[3] if len(sys.argv) > 3 else "/"), settle=2.0)
            time.sleep(9.0)
            print(json.dumps(b.js(sys.argv[2]), indent=2)[:8000])
        finally:
            b.kill()
        return 0
    if mode == "gpushot":
        # gpushot OUT [--path /] [--at SELECTOR] -- a screenshot on the real
        # adapter. Kept separate from `shot` so no existing check silently
        # changes which rasteriser it measured.
        out = Path(sys.argv[2])
        args = sys.argv[3:]
        path, selector = "/", None
        for i, a in enumerate(args):
            if a == "--path" and i + 1 < len(args):
                path = args[i + 1]
            if a == "--at" and i + 1 < len(args):
                selector = args[i + 1]
        b = Browser(gpu=True)
        try:
            b.goto(BASE + path, settle=2.0)
            time.sleep(9.0)
            if selector:
                box = b.js(
                    "(() => { const e = document.querySelector(" +
                    json.dumps(selector) + "); if (!e) return null;"
                    " const r = e.getBoundingClientRect(), pad = 2;"
                    " return JSON.stringify({x: Math.max(0, r.left + scrollX - pad),"
                    " y: r.top + scrollY, w: Math.ceil(r.width) + pad * 2,"
                    " h: Math.ceil(r.height)}); })()")
                if not box:
                    raise SystemExit("selector not found: %s" % selector)
                rect = json.loads(box)
                data = b.ws.call("Page.captureScreenshot", {
                    "format": "png", "captureBeyondViewport": True,
                    "clip": {"x": rect["x"], "y": rect["y"], "width": rect["w"],
                             "height": min(rect["h"], 16000), "scale": 1}})["data"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(data))
            else:
                b.shot(out)
            print("wrote %s (%d bytes)" % (out, out.stat().st_size))
        finally:
            b.kill()
        return 0
    if mode == "eval":
        b = Browser()
        try:
            b.goto(BASE + (sys.argv[3] if len(sys.argv) > 3 else "/"), settle=2.0)
            time.sleep(7.0)
            print(json.dumps(b.js(sys.argv[2]), indent=2)[:6000])
        finally:
            b.kill()
        return 0
    print("unknown mode %r" % mode)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
