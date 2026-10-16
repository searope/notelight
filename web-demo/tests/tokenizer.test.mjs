// The browser tokenizer gives the same ids as Hugging Face `tokenizers`.
//
//   node --test web-demo/tests/
//
// Expected ids come from scripts/make_tokenizer_fixture.py; its --sweep mode
// compares every Unicode code point and all Markdown in the repository.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { test } from "node:test";

import { BGE_SMALL_EN_V15 as SPEC } from "../embed.js";
import { normalize, preTokenize, WordPieceTokenizer } from "../tokenizer.js";
import { FIXTURES, modelFile, requireOrSkip } from "./helpers.mjs";

const bytes = await modelFile(SPEC.tokenizer);
const json = bytes && JSON.parse(new TextDecoder().decode(bytes));
const missing = "tokenizer.json is not in the model cache (set NOTELIGHT_MODEL_DOWNLOAD=1)";

test("ids match HF tokenizers on the fixture cases", async (t) => {
  if (!requireOrSkip(t, json, missing)) return;
  const tokenizer = new WordPieceTokenizer(json, SPEC.maxTokens);
  const { cases } = JSON.parse(await readFile(join(FIXTURES, "tokenizer-cases.json"), "utf8"));
  assert.ok(cases.length >= 30);
  for (const { text, ids } of cases) {
    assert.deepEqual(tokenizer.encode(text).ids, ids, JSON.stringify(text.slice(0, 60)));
  }
});

test("long input is truncated to 512 tokens with [SEP] last", (t) => {
  if (!requireOrSkip(t, json, missing)) return;
  const tokenizer = new WordPieceTokenizer(json, SPEC.maxTokens);
  const enc = tokenizer.encode("word ".repeat(1000));
  assert.equal(enc.ids.length, 512);
  assert.equal(enc.ids[0], 101);
  assert.equal(enc.ids.at(-1), 102);
  assert.deepEqual(new Set(enc.typeIds), new Set([0]));
  assert.deepEqual(new Set(enc.attentionMask), new Set([1]));
});

test("an unsupported tokenizer.json is refused, not approximated", (t) => {
  if (!requireOrSkip(t, json, missing)) return;
  const other = { ...json, normalizer: { ...json.normalizer, lowercase: false } };
  assert.throws(() => new WordPieceTokenizer(other), /unsupported tokenizer.json: normalizer/);
  const bpe = { ...json, model: { ...json.model, type: "BPE" } };
  assert.throws(() => new WordPieceTokenizer(bpe), /unsupported tokenizer.json: model BPE/);
});

test("normalization: accents, case, controls, CJK", () => {
  assert.equal(normalize("Café ÅNGSTRÖM"), "cafe angstrom");
  assert.equal(normalize("a\u0000b\u0007c\tD e"), "abc d e");
  assert.equal(normalize("ΣΟΦΟΣ"), "σοφοσ"); // per code point: no final sigma
  assert.equal(normalize("x中y"), "x 中 y");
});

test("pre-tokenization isolates punctuation, including ASCII symbols", () => {
  assert.deepEqual(preTokenize("don't $5+x"), ["don", "'", "t", "$", "5", "+", "x"]);
  assert.deepEqual(preTokenize("  a　b  "), ["a", "b"]);
});
