#!/usr/bin/env python3
"""Tiny HTTP stub used to verify the Swift ServerProcess spawn path.

The real omlx server arrives via the bundled venvstacks runtime that
`apps/omlx-mac/Scripts/build.sh` embeds into the .app. When iterating on
the Swift code without a full venvstacks build, point ServerProcess at
this script via:

    OMLX_PYTHON_OVERRIDE=/usr/bin/python3 \\
    OMLX_DEV_SERVER_SCRIPT=$(pwd)/apps/omlx-mac/Scripts/dev_server.py \\
    apps/omlx-mac/build/Build/Products/Debug/oMLX.app/Contents/MacOS/oMLX

Then `curl http://127.0.0.1:8080/health` should return 200 with
`{"status":"ok","stub":true}`.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server convention
        if self.path == "/health":
            payload = json.dumps({"status": "ok", "stub": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Pipe access logs to stderr so the parent's tail captures them.
        import sys

        sys.stderr.write("dev_server: " + (format % args) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), StubHandler)
    print(f"dev_server: listening on http://{args.host}:{args.port}/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
