// A BERT WordPiece tokenizer: the part of Hugging Face `tokenizers` that
// bge-small's tokenizer.json uses, ported so the browser produces the same
// token ids as the Python embedder (python/notelight/embed.py).
//
// The pipeline, in the order `tokenizers` runs it:
//   1. added tokens ([CLS], [SEP], [MASK], ...) are split out of the raw text;
//   2. BertNormalizer: drop control characters, map whitespace to " ", put
//      spaces around CJK ideographs, strip accents (NFD, drop marks), lowercase;
//   3. BertPreTokenizer: split on whitespace, and isolate each punctuation mark;
//   4. WordPiece: greedy longest-match-first over the vocabulary, "##" for
//      word-internal pieces, [UNK] for a word with any unknown piece;
//   5. truncate to max length, then wrap in [CLS] ... [SEP].
//
// Anything in tokenizer.json outside this subset is refused in the
// constructor rather than silently tokenized differently. Parity with
// `tokenizers` is tested in tests/tokenizer.test.mjs against ids the Python
// library produced (scripts/make_tokenizer_fixture.py).

const WHITESPACE = /\p{White_Space}/u;
// What `tokenizers` calls control: Cc, Cf, Co, Cs, but not unassigned (Cn).
const OTHER = /[\p{Cc}\p{Cf}\p{Co}\p{Cs}]/u;
const PUNCTUATION = /[!-/:-@[-`{-~]|\p{P}/u; // ASCII punctuation, or Unicode P*
const MARK = /\p{Mn}/u;

// Characters assigned in newer Unicode versions than the category tables in
// `tokenizers`, which therefore treats them as unassigned: not control, not
// punctuation, not an accent to strip. The JavaScript engine knows them, so
// they are exempted here. Found by sweeping every code point through both
// tokenizers (scripts/make_tokenizer_fixture.py --sweep); hex ranges.
const NEWER_UNICODE = (
  "61d 7fd 890-891 897-89f 8ca-8e2 9fd-9fe a76 afa-aff b55 c04 c3c c77 c84 " +
  "d00 d3b-d3c d81 eba ece 180f 1885-1886 1abf-1add 1ae0-1aeb " +
  "1b4e-1b4f 1b7d-1b7f 1df6-1dfb 2e43-2e4f 2e52-2e5d a82c a8c5 a8ff a9bd " +
  "10d24-10d27 10d69-10d6e 10eab-10ead 10ed0 10efa-10eff 10f46-10f50 " +
  "10f55-10f59 10f82-10f89 11070 11073-11074 110c2 110cd 111cf 1123e " +
  "11241 1133b 113bb-113c0 113ce 113d0 113d2 113d4-113d5 113d7-113d8 " +
  "113e1-113e2 11438-1143f 11442-11444 11446 1144b-1144f 1145a-1145b " +
  "1145d-1145e 11660-1166c 116b9 1182f-11837 11839-1183b 1193b-1193c " +
  "1193e 11943-11946 119d4-119d7 119da-119db 119e0 119e2 11a01-11a0a " +
  "11a33-11a38 11a3b-11a47 11a51-11a56 11a59-11a5b 11a8a-11a96 11a98-11a9c " +
  "11a9e-11aa2 11b00-11b09 11b60 11b62-11b64 11b66 11be1 11c30-11c36 " +
  "11c38-11c3d 11c3f 11c41-11c45 11c70-11c71 11c92-11ca7 11caa-11cb0 " +
  "11cb2-11cb3 11cb5-11cb6 11d31-11d36 11d3a 11d3c-11d3d 11d3f-11d45 11d47 " +
  "11d90-11d91 11d95 11d97 11ef3-11ef4 11ef7-11ef8 11f00-11f01 11f36-11f3a " +
  "11f40 11f42-11f4f 11f5a 11fff 12ff1-12ff2 13430-13440 13447-13455 " +
  "1611e-16129 1612d-1612f 16d6d-16d6f 16e97-16e9a 16f4f 16fe2 16fe4 " +
  "1cf00-1cf2d 1cf30-1cf46 1e000-1e006 1e008-1e018 1e01b-1e021 1e023-1e024 " +
  "1e026-1e02a 1e08f 1e130-1e136 1e2ae 1e2ec-1e2ef 1e4ec-1e4ef 1e5ee-1e5ef " +
  "1e5ff 1e6e3 1e6e6 1e6ee-1e6ef 1e6f5 1e944-1e94a 1e95e-1e95f"
)
  .split(" ")
  .map((r) => r.split("-").map((h) => parseInt(h, 16)))
  .map(([lo, hi = lo]) => [lo, hi]);

/** Whether `cp` lies in one of the sorted, disjoint `ranges` (binary search). */
function inRanges(ranges, cp) {
  let lo = 0;
  let hi = ranges.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cp < ranges[mid][0]) hi = mid - 1;
    else if (cp > ranges[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

const isNewer = (ch) => inRanges(NEWER_UNICODE, ch.codePointAt(0));

// Characters whose category changed after those tables: `tokenizers` still
// sees U+166D and U+111C9 as punctuation (Po), and U+1734 and U+1171E as
// accents to strip (Mn).
const OLD_PUNCTUATION = new Set(["\u{166d}", "\u{111c9}"]);
const OLD_MARK = new Set(["\u{1734}", "\u{1171e}"]);

/** Code point ranges `tokenizers` treats as CJK ideographs. */
const CJK = [
  [0x4e00, 0x9fff],
  [0x3400, 0x4dbf],
  [0x20000, 0x2a6df],
  [0x2a700, 0x2b73f],
  [0x2b740, 0x2b81f],
  [0x2b920, 0x2ceaf],
  [0xf900, 0xfaff],
  [0x2f800, 0x2fa1f],
];

const isCjk = (cp) => CJK.some(([lo, hi]) => cp >= lo && cp <= hi);
const isWhitespace = (ch) => ch === "\t" || ch === "\n" || ch === "\r" || WHITESPACE.test(ch);
const isControl = (ch) =>
  ch !== "\t" && ch !== "\n" && ch !== "\r" && OTHER.test(ch) && !isNewer(ch);
const isPunctuation = (ch) =>
  OLD_PUNCTUATION.has(ch) || (PUNCTUATION.test(ch) && !isNewer(ch) && !OLD_MARK.has(ch));
const isMark = (ch) =>
  OLD_MARK.has(ch) || (MARK.test(ch) && !isNewer(ch) && !OLD_PUNCTUATION.has(ch));

function unsupported(what) {
  return new Error(`unsupported tokenizer.json: ${what}`);
}

/** BertNormalizer with clean_text, handle_chinese_chars, strip_accents and lowercase. */
export function normalize(text) {
  let cleaned = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp === 0 || cp === 0xfffd || isControl(ch)) continue;
    if (isWhitespace(ch)) cleaned += " ";
    else if (isCjk(cp)) cleaned += ` ${ch} `;
    else cleaned += ch;
  }
  // Strip accents, then lowercase per code point as `tokenizers` does (so
  // no final-sigma rule).
  let lower = "";
  for (const ch of cleaned.normalize("NFD")) {
    if (!isMark(ch)) lower += ch.toLowerCase();
  }
  return lower;
}

/** BertPreTokenizer: words split on whitespace, each punctuation mark on its own. */
export function preTokenize(text) {
  const words = [];
  let word = "";
  for (const ch of text) {
    if (WHITESPACE.test(ch)) {
      if (word) words.push(word);
      word = "";
    } else if (isPunctuation(ch)) {
      if (word) words.push(word);
      words.push(ch);
      word = "";
    } else {
      word += ch;
    }
  }
  if (word) words.push(word);
  return words;
}

/** Escape a string for use inside a regular expression. */
const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Token ids for one query, as `tokenizers.Tokenizer.encode` returns them. */
export class WordPieceTokenizer {
  /**
   * @param {object} json  the parsed tokenizer.json
   * @param {number} maxLength  truncation length including [CLS] and [SEP]
   */
  constructor(json, maxLength = 512) {
    const { normalizer: n, pre_tokenizer: pre, model, post_processor: post } = json;
    if (n?.type !== "BertNormalizer" || !n.clean_text || !n.handle_chinese_chars || !n.lowercase) {
      throw unsupported(`normalizer ${JSON.stringify(n)}`);
    }
    if (n.strip_accents === false) throw unsupported("strip_accents: false");
    if (pre?.type !== "BertPreTokenizer") throw unsupported(`pre_tokenizer ${pre?.type}`);
    if (model?.type !== "WordPiece") throw unsupported(`model ${model?.type}`);
    if (post?.type !== "TemplateProcessing") throw unsupported(`post_processor ${post?.type}`);
    const single = post.single.map((p) => p.SpecialToken?.id ?? p.Sequence?.id);
    if (single.join(" ") !== "[CLS] A [SEP]") throw unsupported(`template ${single.join(" ")}`);

    this.vocab = new Map(Object.entries(model.vocab));
    this.unk = this.#id(model.unk_token);
    this.prefix = model.continuing_subword_prefix;
    this.maxChars = model.max_input_chars_per_word;
    this.cls = post.special_tokens["[CLS]"].ids[0];
    this.sep = post.special_tokens["[SEP]"].ids[0];
    this.maxLength = maxLength;

    // Added tokens match the raw text before normalization. Longest first, so
    // the alternation picks the longest of tokens that start at one position.
    this.added = new Map();
    for (const t of json.added_tokens ?? []) {
      if (t.normalized || t.single_word || t.lstrip || t.rstrip) {
        throw unsupported(`added token options for ${t.content}`);
      }
      this.added.set(t.content, t.id);
    }
    const alternatives = [...this.added.keys()].sort((a, b) => b.length - a.length);
    this.addedPattern = alternatives.length
      ? new RegExp(alternatives.map(escapeRegExp).join("|"), "g")
      : null;
  }

  #id(token) {
    const id = this.vocab.get(token);
    if (id === undefined) throw unsupported(`${token} is not in the vocabulary`);
    return id;
  }

  /** Append the WordPiece ids of one pre-tokenized word to `ids`. */
  #wordPiece(word, ids) {
    const chars = [...word];
    if (chars.length > this.maxChars) {
      ids.push(this.unk);
      return;
    }
    const pieces = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let id;
      while (start < end) {
        const sub = (start > 0 ? this.prefix : "") + chars.slice(start, end).join("");
        id = this.vocab.get(sub);
        if (id !== undefined) break;
        end -= 1;
      }
      if (id === undefined) {
        ids.push(this.unk);
        return;
      }
      pieces.push(id);
      start = end;
    }
    ids.push(...pieces);
  }

  /** Ids of the text between added tokens. */
  #segment(text, ids) {
    for (const word of preTokenize(normalize(text))) this.#wordPiece(word, ids);
  }

  /**
   * Encode `text` as one sequence.
   * @returns {{ids: number[], typeIds: number[], attentionMask: number[]}}
   */
  encode(text) {
    const ids = [];
    let last = 0;
    if (this.addedPattern) {
      for (const m of text.matchAll(this.addedPattern)) {
        this.#segment(text.slice(last, m.index), ids);
        ids.push(this.added.get(m[0]));
        last = m.index + m[0].length;
      }
    }
    this.#segment(text.slice(last), ids);
    const body = ids.slice(0, Math.max(0, this.maxLength - 2));
    const all = [this.cls, ...body, this.sep];
    return { ids: all, typeIds: all.map(() => 0), attentionMask: all.map(() => 1) };
  }
}
