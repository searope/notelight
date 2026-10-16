"""Render docs/assets/mcp-demo.gif from a real MCP session.

The script indexes this repository's `docs/` folder with bge-small (in a
temporary NOTELIGHT_HOME, reusing your model cache), starts `notelight mcp`
as a subprocess over stdio, calls the tools the way an agent would, and draws
the exact requests and responses as an animated terminal transcript. Nothing
in the GIF is typed by hand: re-run the script after changing search code and
the GIF follows.

    pip install pillow
    python scripts/make_demo_gif.py            # writes docs/assets/mcp-demo.gif
    python scripts/make_demo_gif.py --query "how are indexes updated safely"

Needs the bge-small model in the cache (run `notelight index` once on
anything) because the server is started with --offline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "docs" / "assets" / "mcp-demo.gif"
DEFAULT_QUERY = "what happens if I re-index while the agent is searching"

# Canvas and colours (a dark terminal theme).
WIDTH = 1120
PAD = 24
FONT_SIZE = 17
LINE_H = 24
BG = (24, 24, 32)
BAR = (40, 40, 52)
FG = (220, 222, 228)
DIM = (130, 134, 150)
CYAN = (110, 200, 230)
YELLOW = (235, 200, 110)
GREEN = (140, 210, 140)
MAGENTA = (200, 150, 230)

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Span:
    text: str
    color: Color = FG


Line = list[Span]


# ---------------------------------------------------------------------------
# 1. A real session
# ---------------------------------------------------------------------------
def run_session(query: str) -> dict[str, Any]:
    """Index docs/ and call the MCP tools over stdio; return their structured results."""
    from notelight import paths

    home = Path(tempfile.mkdtemp(prefix="notelight-demo-"))
    env = {
        **os.environ,
        paths.HOME_ENV: str(home),
        paths.MODEL_CACHE_ENV: os.environ.get(paths.MODEL_CACHE_ENV)
        or str(paths.model_cache_dir()),
    }
    cli = [sys.executable, "-m", "notelight.cli"]
    subprocess.run([*cli, "index", str(DOCS), "--offline", "--quiet"], env=env, check=True)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "notelight.cli", "mcp", "--offline", "--folder", str(DOCS)],
        env={k: env[k] for k in (paths.HOME_ENV, paths.MODEL_CACHE_ENV)},
    )

    async def session() -> dict[str, Any]:
        async with Client(params, read_timeout_seconds=120) as client:
            out: dict[str, Any] = {}
            for name, args in [
                ("list_sources", {}),
                ("search", {"query": query, "k": 3}),
            ]:
                result = await client.call_tool(name, args)
                if result.is_error:
                    raise SystemExit(f"{name} failed: {result.content}")
                out[name] = result.structured_content
            top = out["search"]["results"][0]["location"]
            result = await client.call_tool("read_context", {"location": top, "context": 1})
            if result.is_error:
                raise SystemExit(f"read_context failed: {result.content}")
            out["read_context"] = result.structured_content
            return out

    return anyio.run(session)


# ---------------------------------------------------------------------------
# 2. The transcript, as coloured lines
# ---------------------------------------------------------------------------
def _call(name: str, args: dict[str, Any]) -> Line:
    rendered = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return [Span("● ", MAGENTA), Span(name, YELLOW), Span(f"({rendered})", FG)]


def _clip(text: str, cols: int) -> str:
    return text if len(text) <= cols else text[: cols - 1] + "…"


def build_scenes(query: str, data: dict[str, Any], cols: int) -> list[tuple[Line, list[Line]]]:
    """(call line, response lines) for each tool call, in order."""
    sources = data["list_sources"]["result"]
    scenes: list[tuple[Line, list[Line]]] = []
    scenes.append(
        (
            _call("list_sources", {}),
            [
                [
                    Span("  └ ", DIM),
                    Span(Path(s["folder"]).name + "/", CYAN),
                    Span(f"  {s['files']} files, {s['chunks']} chunks, {s['embedder']}", DIM),
                ]
                for s in sources
            ],
        )
    )

    hits: list[Line] = []
    for n, hit in enumerate(data["search"]["results"], 1):
        heading = " > ".join(hit["heading_path"][-2:])
        hits.append(
            [
                Span(f"  └ {n}. " if n == 1 else f"    {n}. ", DIM),
                Span(hit["location"], CYAN),
                Span(_clip(f"  {heading}", cols - 8 - len(hit["location"])), DIM),
            ]
        )
        hits.append([Span("       " + _clip(hit["snippet"], cols - 8), FG)])
    scenes.append((_call("search", {"query": query, "k": 3}), hits))

    ctx = data["read_context"]
    body: list[Line] = []
    width = len(str(ctx["line_end"]))
    for offset, raw in enumerate(ctx["text"].split("\n")):
        number = ctx["line_start"] + offset
        wrapped = textwrap.wrap(raw, cols - width - 9) or [""]
        for i, part in enumerate(wrapped):
            gutter = f"{number:>{width}} │ " if i == 0 else " " * width + " │ "
            color = GREEN if raw.startswith("#") else FG
            body.append([Span("    " + gutter, DIM), Span(part, color)])
    first = [Span("  └ ", DIM), Span(ctx["location"], CYAN), Span(" (from disk)", DIM)]
    requested = data["search"]["results"][0]["location"]
    scenes.append((_call("read_context", {"location": requested, "context": 1}), [first]))
    scenes[-1][1].extend(body)
    return scenes


# ---------------------------------------------------------------------------
# 3. Frames
# ---------------------------------------------------------------------------
def _font(bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
        if not bold
        else ["CascadiaMono.ttf", "consolab.ttf", "DejaVuSansMono-Bold.ttf", "Menlo.ttc"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _font()


def _char_width() -> int:
    box = FONT.getbbox("M" * 10)
    return max(1, round((box[2] - box[0]) / 10))


def draw(lines: list[Line], title: str, height: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, WIDTH, 34], fill=BAR)
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([16 + i * 22, 11, 28 + i * 22, 23], fill=c)
    d.text((96, 8), title, font=FONT, fill=DIM)
    y = 34 + PAD
    cw = _char_width()
    for line in lines:
        x = PAD
        for span in line:
            d.text((x, y), span.text, font=FONT, fill=span.color)
            x += cw * len(span.text)
        y += LINE_H
    return img


def render(scenes: list[tuple[Line, list[Line]]], query: str, title: str) -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []
    # Tall enough for the whole transcript, so nothing scrolls away.
    total = 2 + sum(2 + len(response) for _, response in scenes) + 1
    height = 34 + 2 * PAD + total * LINE_H

    def snap(lines: list[Line], ms: int) -> None:
        frames.append(draw(lines, title, height))
        durations.append(ms)

    header: list[Line] = [
        [Span("You: ", MAGENTA), Span(f"Search my notes: {query}. Cite the files.", FG)],
        [],
    ]
    lines: list[Line] = []
    # The question, typed.
    text = header[0][1].text
    for end in range(0, len(text) + 1, 3):
        snap([[header[0][0], Span(text[:end], FG)]], 40)
    lines = list(header)
    snap(lines, 700)

    for call, response in scenes:
        # The tool call appears, then its response line by line.
        lines.append(call)
        snap(lines, 600)
        for row in response:
            lines.append(row)
            snap(lines, 90)
        lines.append([])
        snap(lines, 1400)
    lines.append(
        [
            Span("Tool calls and results are real ", DIM),
            Span("notelight mcp", CYAN),
            Span(" output over stdio (bge-small, this repo's docs/).", DIM),
        ]
    )
    snap(lines, 4000)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    palette = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=64) for f in frames]
    palette[0].save(
        OUT,
        save_all=True,
        append_images=palette[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    seconds = sum(durations) / 1000
    print(f"wrote {OUT} ({len(frames)} frames, {seconds:.1f}s, {OUT.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--dump", action="store_true", help="Print the session as JSON too.")
    args = parser.parse_args()
    data = run_session(args.query)
    if args.dump:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    cols = (WIDTH - 2 * PAD) // _char_width()
    render(build_scenes(args.query, data, cols), args.query, "notelight mcp  ·  docs/")


if __name__ == "__main__":
    main()
