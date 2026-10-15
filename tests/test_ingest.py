"""Folder walking, ignore rules and decoding."""

from __future__ import annotations

from pathlib import Path

from notelight.ingest import IGNORE_FILE, MAX_FILE_BYTES, decode, walk


def _touch(root: Path, rel: str, text: str = "x") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _walked(root: Path) -> list[str]:
    return [f.rel_path for f in walk(root)]


def test_walk_finds_text_files_with_posix_paths(tmp_path: Path) -> None:
    for rel in ["a.md", "sub/b.txt", "sub/deeper/c.markdown", "image.png", "code.py"]:
        _touch(tmp_path, rel)
    # Forward slashes on every OS, including Windows.
    assert _walked(tmp_path) == ["a.md", "sub/b.txt", "sub/deeper/c.markdown"]


def test_walk_skips_hidden_vcs_and_build_dirs(tmp_path: Path) -> None:
    for rel in [
        "keep.md",
        ".hidden.md",
        ".git/notes.md",
        ".obsidian/x.md",
        "node_modules/pkg/readme.md",
        "target/doc.md",
        "__pycache__/x.md",
        "venv/x.md",
    ]:
        _touch(tmp_path, rel)
    assert _walked(tmp_path) == ["keep.md"]


def test_notelightignore_uses_gitignore_syntax(tmp_path: Path) -> None:
    for rel in ["a.md", "drafts/b.md", "notes/secret.md", "notes/ok.md", "log.txt"]:
        _touch(tmp_path, rel)
    _touch(tmp_path, IGNORE_FILE, "# comment\ndrafts/\nsecret.md\n*.txt\n")
    assert _walked(tmp_path) == ["a.md", "notes/ok.md"]


def test_walk_skips_huge_files(tmp_path: Path) -> None:
    _touch(tmp_path, "small.md")
    (tmp_path / "huge.md").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
    assert _walked(tmp_path) == ["small.md"]


def test_markdown_detection(tmp_path: Path) -> None:
    _touch(tmp_path, "a.MD")
    _touch(tmp_path, "b.txt")
    flags = {f.rel_path: f.is_markdown for f in walk(tmp_path)}
    assert flags == {"a.MD": True, "b.txt": False}


def test_decode_utf8_with_bom_and_crlf() -> None:
    bom = chr(0xFEFF)
    d = decode((bom + "line one\r\nline two\r\n").encode())
    assert d.text == "line one\nline two\n"
    assert d.encoding == "utf-8-sig"


def test_decode_falls_back_to_cp1252() -> None:
    text = "café " + chr(0x2013) + " résumé"  # é and an en dash: not ASCII
    d = decode(text.encode("cp1252"))
    assert d.text == text
    assert d.encoding == "cp1252"


def test_decode_skips_binary() -> None:
    d = decode(b"PK\x03\x04\x00\x00binary")
    assert d.text is None and d.skip_reason == "binary"


def test_decode_skips_undecodable() -> None:
    # 0x81 is undefined in Windows-1252 and invalid as a UTF-8 start byte.
    d = decode(b"\x81\x81\x81")
    assert d.text is None and d.skip_reason is not None
