// Find and read the text files in a folder the user picked, with the same
// rules as python/notelight/ingest.py, so a folder indexed here and by the CLI
// gives the same files in the same order:
//
// - hidden files and directories (names starting with `.`) are skipped, and so
//   are common build and dependency directories (SKIP_DIRS);
// - a `.notelightignore` file at the folder root uses .gitignore syntax, with
//   the semantics of pathspec's `GitIgnoreSpec` (ported below);
// - only .md, .markdown and .txt are read, up to 10 MiB each; PDF and DOCX are
//   found (up to 50 MiB) so the indexer can say why it skips them;
// - files with a NUL byte in the first 8 KiB are binary; text is UTF-8 (with
//   or without a BOM), else Windows-1252, with line endings normalized to \n.
//
// Paths are relative to the folder root with `/` separators, and the walk is
// in Python's order: each folder's files sorted by code point, then its
// subfolders, depth first.
//
// A folder is a tree of plain objects, so the same walk reads a
// FileSystemDirectoryHandle (`fromDirectoryHandle`), the files of an
// `<input webkitdirectory>` (`fromFiles`), or, in the Node tests, the disk:
//   directory: { kind: "directory", name, entries(): AsyncIterable<node> }
//   file:      { kind: "file", name, file(): Promise<{size, lastModified, arrayBuffer()}> }

export const IGNORE_FILE = ".notelightignore";
export const MARKDOWN_SUFFIXES = new Set([".md", ".markdown"]);
export const TEXT_SUFFIXES = new Set([...MARKDOWN_SUFFIXES, ".txt"]);
export const DOCUMENT_SUFFIXES = new Set([".pdf", ".docx"]);
export const MAX_FILE_BYTES = 10 * 1024 * 1024;
export const MAX_DOCUMENT_BYTES = 50 * 1024 * 1024;
export const SKIP_DIRS = new Set([
  "node_modules",
  "target",
  "build",
  "dist",
  "__pycache__",
  "venv",
  "site-packages",
]);
const BINARY_SNIFF_BYTES = 8192;

/** `Path(name).suffix.lower()`: the last `.ext`, or "" (none, or a trailing dot). */
export function suffixOf(name) {
  const i = name.lastIndexOf(".");
  return i > 0 && i < name.length - 1 ? name.slice(i).toLowerCase() : "";
}

/** Largest file of type `suffix` that is indexed (`ingest.max_bytes`). */
export const maxBytes = (suffix) => (DOCUMENT_SUFFIXES.has(suffix) ? MAX_DOCUMENT_BYTES : MAX_FILE_BYTES);

/** Whether to chunk with Markdown structure (`SourceFile.is_markdown`, text files only). */
export const isMarkdown = (path) => MARKDOWN_SUFFIXES.has(suffixOf(path));

/** Python's `sorted` order for strings: by code point, not by UTF-16 unit. */
export function compareCodePoints(a, b) {
  const x = a[Symbol.iterator]();
  const y = b[Symbol.iterator]();
  for (;;) {
    const p = x.next();
    const q = y.next();
    if (p.done || q.done) return p.done === q.done ? 0 : p.done ? -1 : 1;
    const d = p.value.codePointAt(0) - q.value.codePointAt(0);
    if (d !== 0) return d;
  }
}

// --------------------------------------------------------------------------
// Decoding (ingest.decode)
// --------------------------------------------------------------------------

/** The five bytes Python's cp1252 codec rejects (WHATWG maps them to C1 controls). */
const CP1252_UNDEFINED = new Set([0x81, 0x8d, 0x8f, 0x90, 0x9d]);

/**
 * Decode file bytes as text: UTF-8 (a BOM is dropped), then Windows-1252.
 * @returns {{text: string|null, encoding?: string, skipReason?: string}}
 */
export function decode(bytes) {
  if (bytes.subarray(0, BINARY_SNIFF_BYTES).includes(0)) return { text: null, skipReason: "binary" };
  let text;
  let encoding;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    encoding = "utf-8-sig";
  } catch {
    if (bytes.some((b) => CP1252_UNDEFINED.has(b))) {
      return { text: null, skipReason: "not UTF-8 or Windows-1252 text" };
    }
    text = new TextDecoder("windows-1252").decode(bytes);
    encoding = "cp1252";
  }
  return { text: text.replace(/\r\n/g, "\n").replace(/\r/g, "\n"), encoding };
}

// --------------------------------------------------------------------------
// .notelightignore: pathspec 1.1's GitIgnoreSpec, ported
// --------------------------------------------------------------------------

/** `str.splitlines()`: every line boundary Python knows, no trailing empty line. */
export function splitLines(text) {
  const lines = text.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/** Python's `str.rstrip()` whitespace (`str.isspace`), which differs from JS `\s`. */
const TRAILING_SPACE = /[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/u;
const DIR_MARK = "ps_d";
const DIR_MARK_CG = `(?<${DIR_MARK}>/)`;
const DIR_MARK_OPT = `(?:${DIR_MARK_CG}|$)`;

/** Thrown for a pattern pathspec rejects (Python's sync fails the same way). */
export class IgnorePatternError extends Error {
  name = "IgnorePatternError";
}

class RangeNotationError extends Error {}

/** `re.escape` for one character, limited to what a `u`-flag RegExp accepts escaped. */
const escapeChar = (c) => (/[\\^$.*+?()[\]{}|/]/.test(c) ? `\\${c}` : c);

/** `_GitIgnoreBasePattern._translate_segment_glob(pattern, "raise")`. */
function translateGlob(pattern) {
  const chars = Array.from(pattern); // code points, as Python indexes str
  let regex = "";
  let escape = false;
  let i = 0;
  const end = chars.length;
  while (i < end) {
    const char = chars[i++];
    if (escape) {
      escape = false;
      regex += escapeChar(char);
    } else if (char === "\\") {
      escape = true;
    } else if (char === "*") {
      regex += "[^/]*";
    } else if (char === "?") {
      regex += "[^/]";
    } else if (char === "[") {
      let j = i;
      if (j < end && (chars[j] === "!" || chars[j] === "^")) j++;
      if (j < end && chars[j] === "]") j++;
      while (j < end && chars[j] !== "]") j++;
      if (j >= end) throw new RangeNotationError(pattern);
      j++;
      let expr = "[";
      if (chars[i] === "!" || chars[i] === "^") {
        expr += "^";
        i++;
      }
      let body = chars.slice(i, j).join("").replace(/\\/g, "\\\\");
      // A `]` first in a class is literal in Python's re; JS reads `[]` as empty.
      if (body.startsWith("]")) body = `\\${body}`;
      regex += expr + body;
      i = j;
    } else {
      regex += escapeChar(char);
    }
  }
  if (escape) throw new IgnorePatternError(`Escape character found with no next character to escape: ${pattern}`);
  return regex;
}

/**
 * `GitIgnoreSpecPattern.pattern_to_regex`: the regex source and whether a
 * match excludes (`include: true`) or re-includes (`false`), or nulls for a
 * line that is not a pattern.
 */
export function patternToRegex(line) {
  const original = line;
  let pattern = line.endsWith("\\ ") ? line : line.replace(TRAILING_SPACE, "");
  if (!pattern || pattern.startsWith("#") || pattern === "/") return { regex: null, include: null };
  let include = true;
  if (pattern.startsWith("!")) {
    include = false;
    pattern = pattern.slice(1);
  }
  const segs = pattern.split("/");
  const isDir = segs[segs.length - 1] === "";
  // __normalize_segments
  if (segs[0] === "") {
    segs.shift();
  } else if (segs.length === 1 || (segs.length === 2 && segs[1] === "")) {
    if (segs[0] !== "**") segs.unshift("**");
  }
  if (segs.length === 0) throw new IgnorePatternError(`Invalid git pattern: ${original}`);
  if (segs[segs.length - 1] === "") segs[segs.length - 1] = "**";
  for (let i = segs.length - 1; i > 0; i--) {
    if (segs[i - 1] === "**" && segs[i] === "**") segs.splice(i, 1);
  }
  const n = segs.length;
  if (n === 1 && segs[0] === "**") return { regex: isDir ? DIR_MARK_CG : ".", include };
  if (n === 2 && segs[0] === "**" && segs[1] === "*") return { regex: ".", include };
  if (n === 3 && segs[0] === "**" && segs[1] === "*" && segs[2] === "**") {
    return { regex: isDir ? DIR_MARK_CG : "/", include };
  }
  // __translate_segments
  const out = [];
  let needSlash = false;
  const end = n - 1;
  try {
    segs.forEach((seg, i) => {
      if (seg === "**") {
        if (i === 0) {
          out.push("^(?:.+/)?");
        } else if (i < end) {
          out.push("(?:/.+)?");
          needSlash = true;
        } else {
          out.push(isDir ? DIR_MARK_CG : "/");
        }
      } else {
        if (i === 0) out.push("^");
        if (needSlash) out.push("/");
        out.push(seg === "*" ? "[^/]+" : translateGlob(seg));
        if (i === end) out.push(DIR_MARK_OPT);
        needSlash = true;
      }
    });
  } catch (e) {
    if (e instanceof RangeNotationError) return { regex: null, include: null }; // git drops it
    if (e instanceof IgnorePatternError) throw new IgnorePatternError(`Invalid git pattern: ${original}`);
    throw e;
  }
  return { regex: out.join(""), include };
}

/**
 * Compile `.notelightignore` text into `{ match(path) }`, where `path` is
 * relative and `/`-separated, with a trailing `/` for a directory.
 */
export function compileIgnore(text) {
  const patterns = [];
  for (const line of splitLines(text)) {
    const { regex, include } = patternToRegex(line);
    if (include === null) continue;
    let re;
    try {
      re = new RegExp(regex, "u");
    } catch {
      throw new IgnorePatternError(`Invalid git pattern: ${line}`);
    }
    patterns.push({ re, include });
  }
  patterns.reverse(); // the simple backend scans from the last pattern
  return {
    patterns: patterns.length,
    /** `GitIgnoreSpec.match_file`: whether `path` is ignored. */
    match(path) {
      let include = null;
      let priority = 0;
      for (const p of patterns) {
        const m = p.re.exec(path);
        if (!m) continue;
        // A match through the directory marker ("foo/" matching "foo/bar")
        // ranks below a direct match, so "!foo/bar" can re-include it.
        const level = m.groups?.[DIR_MARK] ? 1 : 2;
        if (level > priority) {
          include = p.include;
          priority = level;
        }
        if (level === 2) break;
      }
      return include === true;
    },
  };
}

/**
 * Read `.notelightignore` as Python does: UTF-8 with replacement characters
 * (a BOM is kept, as `read_text(encoding="utf-8")` keeps it), universal newlines.
 */
export function ignoreText(bytes) {
  return new TextDecoder("utf-8", { ignoreBOM: true }).decode(bytes);
}

// --------------------------------------------------------------------------
// Folder sources
// --------------------------------------------------------------------------

/** A FileSystemDirectoryHandle (File System Access API) as a folder tree. */
export function fromDirectoryHandle(handle) {
  return {
    kind: "directory",
    name: handle.name,
    async *entries() {
      for await (const h of handle.values()) {
        yield h.kind === "directory"
          ? fromDirectoryHandle(h)
          : { kind: "file", name: h.name, file: () => h.getFile() };
      }
    },
  };
}

/**
 * Files from `<input webkitdirectory>` as a folder tree. `items` are
 * `{path, file}` with `path` as in `webkitRelativePath` ("root/sub/a.md");
 * the first segment, the folder that was picked, becomes the root.
 */
export function fromFiles(items) {
  const root = { kind: "directory", name: "", children: new Map() };
  for (const { path, file } of items) {
    const parts = path.split("/");
    if (!root.name) root.name = parts[0];
    let dir = root;
    for (const part of parts.slice(1, -1)) {
      let next = dir.children.get(part);
      if (!next) {
        next = { kind: "directory", name: part, children: new Map() };
        dir.children.set(part, next);
      }
      dir = next;
    }
    const name = parts[parts.length - 1];
    dir.children.set(name, { kind: "file", name, file: async () => file });
  }
  const wrap = (d) => ({
    kind: "directory",
    name: d.name,
    async *entries() {
      for (const c of d.children.values()) yield c.kind === "directory" ? wrap(c) : c;
    },
  });
  return wrap(root);
}

// --------------------------------------------------------------------------
// The walk (ingest.walk)
// --------------------------------------------------------------------------

async function listEntries(dir) {
  const out = [];
  for await (const e of dir.entries()) out.push(e);
  return out;
}

/**
 * Every indexable file under `root`, in Python's order.
 * @returns {Promise<{files: Array<{path: string, suffix: string, size: number,
 *   lastModified: number, read: () => Promise<Uint8Array>}>, ignorePatterns: number}>}
 */
export async function walk(root) {
  const rootEntries = await listEntries(root);
  const ignoreEntry = rootEntries.find((e) => e.kind === "file" && e.name === IGNORE_FILE);
  let spec = null;
  if (ignoreEntry) {
    const f = await ignoreEntry.file();
    spec = compileIgnore(ignoreText(new Uint8Array(await f.arrayBuffer())));
  }
  const files = [];
  const byName = (a, b) => compareCodePoints(a.name, b.name);

  async function visit(entries, prefix) {
    const dirs = entries
      .filter(
        (e) =>
          e.kind === "directory" &&
          !e.name.startsWith(".") &&
          !SKIP_DIRS.has(e.name) &&
          !(spec && spec.match(`${prefix}${e.name}/`)),
      )
      .sort(byName);
    for (const e of entries.filter((x) => x.kind === "file").sort(byName)) {
      const suffix = suffixOf(e.name);
      if (e.name.startsWith(".") || !(TEXT_SUFFIXES.has(suffix) || DOCUMENT_SUFFIXES.has(suffix))) continue;
      const path = prefix + e.name;
      if (spec && spec.match(path)) continue;
      let file;
      try {
        file = await e.file();
      } catch {
        continue; // vanished or unreadable; the next sync will see it
      }
      if (file.size > maxBytes(suffix)) continue;
      files.push({
        path,
        suffix,
        size: file.size,
        lastModified: file.lastModified,
        read: async () => new Uint8Array(await file.arrayBuffer()),
      });
    }
    for (const d of dirs) await visit(await listEntries(d), `${prefix}${d.name}/`);
  }

  await visit(rootEntries, "");
  return { files, ignorePatterns: spec?.patterns ?? 0 };
}
