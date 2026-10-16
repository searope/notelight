"""Token ids from Hugging Face `tokenizers`, to test the browser tokenizer against.

The browser demo tokenizes queries with its own WordPiece port
(web-demo/tokenizer.js). This script records what the Python library, which
the indexer uses, produces for bge-small's pinned tokenizer.json:

    python scripts/make_tokenizer_fixture.py           # writes the committed fixture
    python scripts/make_tokenizer_fixture.py --sweep   # compares far more, commits nothing

The committed fixture (web-demo/tests/fixtures/tokenizer-cases.json) is a few
dozen hand-picked cases (accents, CJK, punctuation, control characters,
special tokens, truncation), checked by web-demo/tests/tokenizer.test.mjs.
`--sweep` writes a large comparison file to a temporary folder and runs
web-demo/tests/tokenizer-sweep.mjs on it: every
Unicode code point in context, and every line and chunk-sized paragraph of
the Markdown in this repository. It needs Node on PATH.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

from notelight.embed import BGE_SMALL_EN_V15, ensure_model_files

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "web-demo" / "tests" / "fixtures" / "tokenizer-cases.json"
SWEEP = ROOT / "web-demo" / "tests" / "tokenizer-sweep.mjs"

CASES = (
    "",
    "Hello, world!",
    BGE_SMALL_EN_V15.query_prefix + "memory mapped files",
    "naïve café résumé Ångström ÀÉÎÕÜ",
    "camelCaseIdentifier snake_case_name kebab-case SCREAMING_CASE",
    "C:\\Users\\serge\\notes\\todo.md and ./relative/path.txt",
    "https://example.com/path?q=1&x=2#frag",
    "中文分词测试 and 日本語のテキスト",
    "Ελληνικά ΣΟΦΟΣ σοφός",
    "İstanbul straße ﬁne ligature",
    "emoji 🙂👍🏽 test 👨‍👩‍👧",
    "tabs\tand\nnewlines\r\nand\u00a0nbsp\u2003em\u3000ideographic",
    "zero\u200bwidth\u200djoiner\ufeffbom\u00adsoft hyphen",
    "control\u0000chars\u0007\u001b[0m\u0085nel\u000bvt\u000cff",
    "replacement \ufffd char",
    "[CLS] literal [MASK] tokens [SEP][PAD][UNK] inside",
    "[mask] lowercase is not special, nor [ CLS ]",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "x" * 100,
    "x" * 101,
    "1,234.56 €100 $5 50% #hashtag @mention ~tilde^caret |pipe| <angle>",
    "don't won't can't it's",
    "e.g. i.e. etc. U.S.A.",
    "`code` **bold** _italics_ # Heading > quote - [link](url)",
    "Привет, мир! Ёлка",
    "한국어 텍스트",
    "العربية نص",
    "हिन्दी पाठ",
    "  leading and trailing spaces  ",
    "Combining: e\u0301 a\u0308 n\u0303 vs precomposed é ä ñ",
    "Fullwidth "
    + "".join(map(chr, (0xFF21, 0xFF22, 0xFF23, 0xFF11, 0xFF12, 0xFF13)))
    + " and "
    + "".join(map(chr, (0xFF76, 0xFF80, 0xFF76, 0xFF85))),
    "math ∑ ∫ √ ≤ ≥ ≠ ∞ and arrows → ⇒",
    f"U+166D {chr(0x166D)} and U+1734 x{chr(0x1734)}y (categories that changed)",
    " ".join(f"word{i}" for i in range(600)),  # truncated at 512 tokens
)


def tokenizer() -> Tokenizer:
    _, path = ensure_model_files(BGE_SMALL_EN_V15)
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=BGE_SMALL_EN_V15.max_tokens)
    return tok


def records(tok: Tokenizer, texts: list[str]) -> list[dict[str, object]]:
    out = []
    for start in range(0, len(texts), 10_000):
        batch = texts[start : start + 10_000]
        out += [
            {"text": t, "ids": e.ids} for t, e in zip(batch, tok.encode_batch(batch), strict=True)
        ]
    return out


def sweep_texts() -> list[str]:
    texts = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        texts.append(f"x{ch}y {ch}{ch}A Ab{ch}")
    for md in sorted(ROOT.glob("**/*.md")):
        if any(part in {"target", "node_modules", ".venv"} for part in md.parts):
            continue
        content = md.read_text(encoding="utf-8")
        texts += [line for line in content.splitlines() if line.strip()]
        texts += [p for p in content.split("\n\n") if p.strip()]
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", action="store_true", help="compare far more text with Node")
    args = parser.parse_args()
    tok = tokenizer()

    if not args.sweep:
        fixture = {
            "comment": "Written by scripts/make_tokenizer_fixture.py from HF tokenizers.",
            "cases": records(tok, list(CASES)),
        }
        text = json.dumps(fixture, ensure_ascii=False, indent=1) + "\n"
        FIXTURE.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {FIXTURE.relative_to(ROOT)} ({len(CASES)} cases)")
        return 0

    _, tokenizer_path = ensure_model_files(BGE_SMALL_EN_V15)
    with tempfile.TemporaryDirectory() as tmp:
        expected = Path(tmp) / "expected.jsonl"
        with expected.open("w", encoding="utf-8", newline="\n") as f:
            for rec in records(tok, sweep_texts()):
                # ASCII escapes: Node's readline also splits lines on U+2028.
                f.write(json.dumps(rec) + "\n")
        done = subprocess.run(["node", str(SWEEP), str(tokenizer_path), str(expected)])
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
