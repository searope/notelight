"""Serve web-demo/ on localhost, with the right types for ES modules and wasm.

    python scripts/serve_demo.py            # http://127.0.0.1:8000
    python scripts/serve_demo.py --port 9000
    python scripts/serve_demo.py --isolated # lets onnxruntime-web use threads

`python -m http.server` takes MIME types from the operating system; on
Windows the registry often maps `.mjs` to `text/plain`, and browsers refuse to
run a module served that way (onnxruntime-web ships as `.mjs`). This server
sets the types the demo needs and otherwise serves files as `http.server`
does. It binds to 127.0.0.1 only.

`--isolated` also sends the COOP and COEP headers that make the page
cross-origin isolated, which WebAssembly threads (SharedArrayBuffer) need;
the demo then runs the embedder on 4 threads instead of 1.
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent / "web-demo"


class DemoHandler(SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
    }
    isolated = False

    def end_headers(self) -> None:
        if self.isolated:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--directory", type=Path, default=ROOT, help="folder to serve")
    parser.add_argument(
        "--isolated", action="store_true", help="send COOP/COEP so wasm threads are allowed"
    )
    args = parser.parse_args()
    DemoHandler.isolated = args.isolated
    handler = functools.partial(DemoHandler, directory=str(args.directory))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"serving {args.directory} at http://127.0.0.1:{args.port}/ (Ctrl+C to stop)")
        server.serve_forever()


if __name__ == "__main__":
    main()
