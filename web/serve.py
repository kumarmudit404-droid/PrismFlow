"""Local static server for the Part 25 site.

    python web/serve.py            # http://127.0.0.1:825
    python web/serve.py --port 9000

WHY THE ROOT IS ``web/`` AND NOT THE REPOSITORY ROOT
-----------------------------------------------------
Serving the repository root would put ``.env`` one URL away from a browser on
this machine. It holds real API keys. A local server is still a server, so the
document root is this directory and nothing above it is reachable.

The consequence is deliberate: the site cannot read ``results/`` directly.
Phase (b) adds a build step that copies the specific committed JSON it needs
into ``web/data/``, which keeps the site static, keeps the served surface small,
and makes "which files does this site read" answerable by listing one folder.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler with no caching, so an edit shows up on reload."""

    def send_head(self):
        """Reject any path carrying a separator that Windows would honour.

        Tested with 19 traversal payloads against this root: none returned the
        bytes of ``.env``, ``CLAUDE.md`` or ``app.py``. But ``..%5c.env`` and
        friends answered **200**, because the decoded backslash made the handler
        discard the component and fall back to the directory index. Nothing
        escaped -- yet a 200 on a traversal attempt is exactly the signal a
        future regression would hide behind, and it makes an honest scan read
        like a finding. Such paths are now refused outright, so 404 means 404.
        """
        raw = self.path.lower()
        if "\\" in self.path or "%5c" in raw or "%255c" in raw or "%2e%2e" in raw:
            self.send_error(404, "Not Found")
            return None
        return super().send_head()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        # This site never needs to reach the network. Saying so in a header
        # means a stray <script src="https://..."> fails loudly in the console
        # instead of silently working and adding an undeclared dependency.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'",
        )
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # One line per request, without the noisy default timestamp block.
        print("  %s" % (fmt % args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=825)
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default; this is not meant to be exposed")
    args = parser.parse_args()

    os.chdir(HERE)
    handler = functools.partial(Handler, directory=str(HERE))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print("PrismFlow Part 25 -- serving %s" % HERE)
        print("  http://%s:%d" % (args.host, args.port))
        print("  document root is web/ only; the repository root is NOT served")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
