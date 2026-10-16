// Compare web-demo/tokenizer.js with Hugging Face `tokenizers` on a large
// file of expected ids. Run by `python scripts/make_tokenizer_fixture.py
// --sweep`, which writes the file; not part of `node --test`.
//
//   node web-demo/tests/tokenizer-sweep.mjs <tokenizer.json> <expected.jsonl>
import { createReadStream, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { argv, exit } from "node:process";

import { WordPieceTokenizer } from "../tokenizer.js";

const [tokenizerPath, expectedPath] = argv.slice(2);
if (!tokenizerPath || !expectedPath) {
  console.error("usage: node tokenizer-sweep.mjs <tokenizer.json> <expected.jsonl>");
  exit(2);
}
const tokenizer = new WordPieceTokenizer(JSON.parse(readFileSync(tokenizerPath, "utf8")));

let total = 0;
let differ = 0;
const lines = createInterface({ input: createReadStream(expectedPath, "utf8"), crlfDelay: Infinity });
for await (const line of lines) {
  if (!line) continue;
  const { text, ids } = JSON.parse(line);
  total += 1;
  const got = tokenizer.encode(text).ids;
  if (got.length !== ids.length || got.some((id, i) => id !== ids[i])) {
    differ += 1;
    if (differ <= 20) {
      console.log(`differs: ${JSON.stringify(text.slice(0, 80))}`);
      console.log(`  tokenizers: ${ids.join(" ")}`);
      console.log(`  tokenizer.js: ${got.join(" ")}`);
    }
  }
}
console.log(`${total.toLocaleString("en")} texts, ${differ} differ`);
exit(differ ? 1 : 0);
