#!/usr/bin/env python3
"""Static file server with cross-origin isolation headers.

Adds Cross-Origin-Opener-Policy: same-origin and
Cross-Origin-Embedder-Policy: require-corp on every response so
SharedArrayBuffer is available to pages — required for testing the
multi-threaded WASM backend.
"""
import http.server
import sys


class COIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Same-origin resources don't need CORP, but transformers.js
        # fetches its own assets from CDN; explicit Resource-Policy
        # keeps the embedder happy if the CDN response lacks one.
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), COIHandler)
    print(f"serving with COOP/COEP on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
