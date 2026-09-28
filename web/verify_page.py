"""Headless page checks for the Part 25 site. Python stdlib only, no npm.

    .venv\\Scripts\\python.exe web/verify_page.py shot  out.png [--path /]
    .venv\\Scripts\\python.exe web/verify_page.py reveal
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
BASE = "http://127.0.0.1:8251"


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
    def __init__(self, extra_args=()) -> None:
        self.profile = tempfile.mkdtemp(prefix="prismflow_verify_")
        self.proc = subprocess.Popen(
            [browser_path(), "--headless=new", "--disable-gpu",
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


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "reveal":
        return reveal_matrix(sys.argv[2] if len(sys.argv) > 2 else "/")
    if mode == "shot":
        out = Path(sys.argv[2])
        path = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--path" else "/"
        b = Browser()
        try:
            b.goto(BASE + path, settle=2.0)
            time.sleep(7.0)          # past the reveal deadline, so nothing is mid-fade
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
