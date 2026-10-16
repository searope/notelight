"""The web UI's JSON API and page, driven through FastAPI's TestClient.

Every test indexes the `notes` fixture with the fake HashingEmbedder, so
nothing is downloaded. Skipped when the `web` extra is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from notelight import cli
from notelight.embed import HashingEmbedder
from notelight.indexer import sync
from notelight.mcp_server import NotesService
from notelight.web import create_app, host_allowed

BASE = "http://127.0.0.1:8765"


@pytest.fixture
def indexed(notes: Path) -> Path:
    sync(notes, HashingEmbedder())
    return notes


@pytest.fixture
def client(indexed: Path) -> TestClient:
    return TestClient(create_app(NotesService()), base_url=BASE)


# -- the page ------------------------------------------------------------------
def test_page_and_assets_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="q"' in page.text and "/app.js" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    for asset, kind in [("/app.js", "text/javascript"), ("/style.css", "text/css")]:
        response = client.get(asset)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(kind)


def test_page_has_no_external_resources(client: TestClient) -> None:
    for asset in ["/", "/app.js", "/style.css"]:
        text = client.get(asset).text
        assert "http://" not in text and "https://" not in text, asset


@pytest.mark.parametrize("path", ["/static/app.js", "/index.html", "/../pyproject.toml", "/docs"])
def test_nothing_else_is_served(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


# -- search --------------------------------------------------------------------
def test_search(client: TestClient, indexed: Path) -> None:
    response = client.get("/api/search", params={"q": "sourdough", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "sourdough"
    top = body["results"][0]
    assert top["path"] == "recipes/bread.txt"
    assert top["location"].startswith("recipes/bread.txt:")
    assert top["folder"] == str(indexed)
    assert "Sourdough" in top["snippet"]
    assert body["skipped"] == []


def test_search_one_folder_with_headings(client: TestClient, indexed: Path) -> None:
    response = client.get("/api/search", params={"q": "lifetimes", "folder": str(indexed)})
    assert response.status_code == 200
    top = response.json()["results"][0]
    assert top["path"] == "rust/ownership.md"
    assert top["heading_path"][0] == "Rust"


@pytest.mark.parametrize(
    ("params", "status"),
    [({"q": "  "}, 400), ({"q": "x", "k": 0}, 422), ({"q": "x", "k": 51}, 422), ({}, 422)],
)
def test_search_rejects_bad_input(
    client: TestClient, params: dict[str, object], status: int
) -> None:
    response = client.get("/api/search", params=params)
    assert response.status_code == status
    assert response.json()["detail"]


def test_not_indexed_folder(client: TestClient, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    response = client.get("/api/search", params={"q": "bread", "folder": str(other)})
    assert response.status_code == 400
    assert "not indexed" in response.json()["detail"]
    status = client.get("/api/status", params={"folder": str(other)})
    assert status.status_code == 400
    assert "not indexed" in status.json()["detail"]


def test_folder_not_served(indexed: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.md").write_text("bread\n", encoding="utf-8")
    sync(other, HashingEmbedder())
    client = TestClient(create_app(NotesService([indexed])), base_url=BASE)
    response = client.get("/api/search", params={"q": "bread", "folder": str(other)})
    assert response.status_code == 400
    assert "not served" in response.json()["detail"]
    assert [s["folder"] for s in client.get("/api/sources").json()] == [str(indexed)]


# -- context -------------------------------------------------------------------
def test_context(client: TestClient, indexed: Path) -> None:
    hit = client.get("/api/search", params={"q": "lifetimes"}).json()["results"][0]
    response = client.get(
        "/api/context", params={"location": hit["location"], "folder": hit["folder"], "context": 1}
    )
    assert response.status_code == 200
    ctx = response.json()
    assert ctx["path"] == "rust/ownership.md"
    assert ctx["line_start"] == max(1, hit["line_start"] - 1)
    assert "Lifetimes describe" in ctx["text"]
    assert ctx["total_lines"] == 10


@pytest.mark.parametrize(
    "location",
    [
        "../secret.txt",
        "rust/../../secret.txt",
        "..\\secret.txt",
        "rust/../recipes/../../secret.txt:1-3",
        ".env",
        "rust/missing.md:1",
    ],
)
def test_context_refuses_paths_outside_the_index(
    client: TestClient, indexed: Path, location: str
) -> None:
    (indexed.parent / "secret.txt").write_text("password\n", encoding="utf-8")
    (indexed / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    response = client.get("/api/context", params={"location": location})
    assert response.status_code == 400
    assert "not an indexed file" in response.json()["detail"]
    assert "password" not in response.text and "TOKEN" not in response.text


def test_context_refuses_absolute_path_outside(client: TestClient, indexed: Path) -> None:
    secret = indexed.parent / "secret.txt"
    secret.write_text("password\n", encoding="utf-8")
    response = client.get("/api/context", params={"location": str(secret)})
    assert response.status_code == 400
    assert "password" not in response.text


def test_context_accepts_absolute_path_inside(client: TestClient, indexed: Path) -> None:
    response = client.get("/api/context", params={"location": str(indexed / "recipes/bread.txt")})
    assert response.status_code == 200
    assert response.json()["path"] == "recipes/bread.txt"


# -- sources and status --------------------------------------------------------
def test_sources_and_status(client: TestClient, indexed: Path) -> None:
    sources = client.get("/api/sources").json()
    assert sources == [
        {
            "folder": str(indexed),
            "files": 3,
            "chunks": sources[0]["chunks"],
            "embedder": HashingEmbedder().model_id,
            "error": None,
        }
    ]
    status = client.get("/api/status", params={"folder": str(indexed)}).json()
    assert status["stale"] is False and status["files"] == 3

    (indexed / "new.md").write_text("# New\n\nfresh note\n", encoding="utf-8")
    status = client.get("/api/status", params={"folder": str(indexed)}).json()
    assert status["stale"] is True and status["added"] == 1
    assert "notelight index" in status["hint"]


# -- Host header ---------------------------------------------------------------
@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1.evil.example"])
def test_bad_host_header_is_rejected(client: TestClient, host: str) -> None:
    for path in ["/", "/api/sources", "/api/search?q=bread"]:
        response = client.get(path, headers={"Host": host})
        assert response.status_code == 400, path
        assert "Host" in response.text


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("127.0.0.1", True),
        ("127.0.0.1:8765", True),
        ("localhost:8765", True),
        ("LOCALHOST", True),
        ("[::1]:8765", True),
        ("[::1]", True),
        ("", False),
        (None, False),
        ("localhost:abc", False),
        ("localhost.example", False),
        ("[::1].example", False),
        ("0.0.0.0", False),
        ("192.168.1.10:8765", False),
    ],
)
def test_host_allowed(host: str | None, ok: bool) -> None:
    assert host_allowed(host) is ok


# -- CLI -----------------------------------------------------------------------
def test_serve_refuses_unindexed_folder(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["serve", "--folder", str(tmp_path)])
    assert result.exit_code == 2
    assert "not indexed" in result.output


def test_serve_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastapi" or name.startswith("fastapi."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "notelight.web", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = CliRunner().invoke(cli.app, ["serve"])
    assert result.exit_code == 1
    assert "notelight[web]" in result.output
