"""Local static server for the Part 25 site.

    python web/serve.py            # http://127.0.0.1:825
    python web/serve.py --port 9000

Use 127.0.0.1 rather than localhost: on a dual-stack Windows box localhost
resolves to ::1 first and this server binds IPv4 only, so localhost costs a
failed connection before the client falls back.

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
import socket
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


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded, and deliberately *not* address-reusing on Windows.

    Both properties are bug fixes, not preferences.

    THREADED: the single-threaded ``TCPServer`` this used to be deadlocked on
    any browser. Chrome and Edge open speculative "preconnect" sockets and hold
    them open without sending a request line. A single-threaded server accepts
    one, blocks in ``readline()`` waiting for bytes that never arrive, and stops
    answering everything else. ``curl`` sends its request immediately and so
    never triggers it -- which is why this looked like a browser problem and not
    a server problem. Measured: one idle socket, and every later request timed
    out until that socket closed.

    NO ADDRESS REUSE ON WINDOWS: ``SO_REUSEADDR`` does not mean on Windows what
    it means on POSIX. There it permits rebinding a port stuck in TIME_WAIT;
    here it lets a second process bind a port another process is *actively
    listening on*. The second one then prints a correct-looking banner and
    receives no traffic at all, because the kernel keeps delivering to the
    first. Measured: a second instance bound 825 and stayed up while
    ``Get-NetTCPConnection`` still showed only the original owner.
    """

    daemon_threads = True
    allow_reuse_address = os.name != "nt"


def port_is_taken(host: str, port: int) -> bool:
    """True if something is already listening, so the bind can refuse loudly.

    ``allow_reuse_address = False`` already makes a duplicate bind raise on
    Windows, but this probe is what turns that into a message naming the real
    problem instead of a bare OSError 10048.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=825)
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default; this is not meant to be exposed")
    args = parser.parse_args()

    os.chdir(HERE)
    handler = functools.partial(Handler, directory=str(HERE))

    if port_is_taken(args.host, args.port):
        print("refusing to start: %s:%d is already serving something."
              % (args.host, args.port))
        print("  a stale server would otherwise keep the port and this one")
        print("  would print a working URL while answering nothing.")
        print("  stop it, or pass --port for a different one.")
        return 1

    with Server((args.host, args.port), handler) as httpd:
        # Report the port the socket actually got, never the one that was
        # asked for. With --port 0 they differ, and a banner that cannot be
        # wrong is worth more here than one that is usually right.
        bound_host, bound_port = httpd.server_address[:2]
        print("PrismFlow Part 25 -- serving %s" % HERE)
        print("  http://%s:%d" % (bound_host, bound_port))
        print("  use 127.0.0.1, not localhost: localhost resolves to ::1 first")
        print("  on this box and this server binds IPv4 only")
        print("  document root is web/ only; the repository root is NOT served")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
