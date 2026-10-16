"""A localhost web UI: one page plus a small read-only JSON API.

`notelight serve` runs it (install with `pip install notelight[web]`). The
API is a thin layer over the MCP server's `NotesService`, so the browser
gets exactly what an agent gets, with the same checks:

- `GET /api/search?q=...&k=10&folder=...`: hybrid search across folders.
- `GET /api/context?location=path:12-30&folder=...&context=5`: lines of an
  indexed file. Only paths in an index's file list are read.
- `GET /api/sources`: the served folders.
- `GET /api/status?folder=...`: generation, size and staleness of one index.

The server binds to 127.0.0.1 and rejects requests whose `Host` header is not
a loopback name, so a web page on another site cannot reach it through DNS
rebinding. It never writes an index and never takes the writer lock.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from importlib import resources
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from mcp.server.mcpserver.exceptions import ToolError

import notelight
from notelight.mcp_server import (
    MAX_K,
    FileContext,
    IndexStatus,
    NotesService,
    SearchResponse,
    Source,
    default_embedder_factory,
)

__all__ = ["ALLOWED_HOSTS", "DEFAULT_PORT", "create_app", "run"]

#: Port `notelight serve` listens on unless `--port` says otherwise.
DEFAULT_PORT = 8765
#: Host names (without port) the server answers to.
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})

#: The page and its assets, served from package data. Nothing else is served.
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

_SECURITY_HEADERS = {
    # Scripts, styles and fetches only from this server; no framing.
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}


def host_allowed(host: str | None) -> bool:
    """True if a `Host` header names this machine's loopback interface."""
    if not host:
        return False
    host = host.strip().lower()
    if host.startswith("["):  # [::1] or [::1]:8765
        name, _, port = host.partition("]")
        name += "]"
        port = port.removeprefix(":")
    else:
        name, _, port = host.partition(":")
    return name in ALLOWED_HOSTS and (port == "" or port.isdigit())


def create_app(service: NotesService) -> FastAPI:
    """The web app serving `service` and the page."""
    app = FastAPI(
        title="notelight",
        version=notelight.__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def check_host(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # A rebinding attack points the attacker's own domain at 127.0.0.1,
        # so the browser still sends that domain as Host: refuse it.
        if not host_allowed(request.headers.get("host")):
            return PlainTextResponse("invalid Host header", status_code=400)
        response = await call_next(request)
        response.headers.update(_SECURITY_HEADERS)
        return response

    @app.exception_handler(ToolError)
    async def tool_error(request: Request, exc: ToolError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    for route, (name, media_type) in _ASSETS.items():
        app.add_api_route(
            route, _asset_endpoint(name, media_type), methods=["GET"], include_in_schema=False
        )

    @app.get("/api/search")
    def search(
        q: Annotated[str, Query(max_length=1000)],
        k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
        folder: str | None = None,
    ) -> SearchResponse:
        return service.search(q, k=k, folder=folder or None)

    @app.get("/api/context")
    def context(
        location: Annotated[str, Query(max_length=4096)],
        folder: str | None = None,
        context: Annotated[int, Query(ge=0, le=100)] = 5,
    ) -> FileContext:
        return service.read_context(location, context=context, folder=folder or None)

    @app.get("/api/sources")
    def sources() -> list[Source]:
        return [
            Source(
                folder=str(i.folder),
                files=i.files,
                chunks=i.chunks,
                embedder=i.embedder,
                error=i.error,
            )
            for i in service.sources()
        ]

    @app.get("/api/status")
    def status(folder: str | None = None) -> IndexStatus:
        return service.index_status(folder or None)

    return app


def _asset_endpoint(name: str, media_type: str) -> Callable[[], Response]:
    body = (resources.files("notelight.web") / "static" / name).read_bytes()

    def endpoint() -> Response:
        return Response(body, media_type=media_type)

    return endpoint


def run(
    folders: Sequence[str | Path] | None,
    *,
    port: int = DEFAULT_PORT,
    offline: bool = False,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Serve the UI on http://127.0.0.1:`port` until interrupted."""
    service = NotesService(folders, embedder_factory=default_embedder_factory(offline))
    url = f"http://127.0.0.1:{port}/"
    config = uvicorn.Config(
        create_app(service), host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    if on_ready is not None:
        on_ready(url)
    server.run()
