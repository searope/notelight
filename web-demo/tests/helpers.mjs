// Shared by the Node tests: where the model files, the vendored runtime and
// the wasm package live, and how to skip (locally) or fail (in CI) when one is
// missing.
import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { BGE_SMALL_EN_V15 as SPEC, fileUrl, loadVerified } from "../embed.js";

export const DEMO = join(dirname(fileURLToPath(import.meta.url)), "..");
export const FIXTURES = join(DEMO, "tests", "fixtures");
export const ORT = join(DEMO, "vendor", "onnxruntime-web", "ort.wasm.min.mjs");
export const PKG = join(DEMO, "pkg", "notelight_wasm.js");

/**
 * The Python model cache (`notelight.paths.model_cache_dir`), so the tests
 * reuse a model the CLI already downloaded.
 */
export function modelCacheDir(env = process.env) {
  if (env.NOTELIGHT_MODEL_CACHE) return env.NOTELIGHT_MODEL_CACHE;
  if (env.NOTELIGHT_HOME) return join(env.NOTELIGHT_HOME, "models");
  // platformdirs.user_cache_path("notelight", appauthor=False)
  const cache =
    process.platform === "win32"
      ? join(env.LOCALAPPDATA ?? join(homedir(), "AppData", "Local"), "notelight", "Cache")
      : process.platform === "darwin"
        ? join(homedir(), "Library", "Caches", "notelight")
        : join(env.XDG_CACHE_HOME || join(homedir(), ".cache"), "notelight");
  return join(cache, "models");
}

/** Path of one pinned file in the model cache, laid out as `embed.py` lays it out. */
export const modelPath = (file) =>
  join(modelCacheDir(), SPEC.repo.replace("/", "--"), SPEC.revision, ...file.path.split("/"));

/**
 * A `loadVerified` cache over the Python model cache folder, so a download in
 * the tests lands where the CLI would look for it.
 */
export const fileCache = {
  async get(url) {
    const file = [SPEC.onnx, SPEC.tokenizer].find((f) => url === fileUrl(SPEC, f));
    return file && existsSync(modelPath(file)) ? new Uint8Array(await readFile(modelPath(file))) : null;
  },
  async put(url, bytes) {
    const file = [SPEC.onnx, SPEC.tokenizer].find((f) => url === fileUrl(SPEC, f));
    await mkdir(dirname(modelPath(file)), { recursive: true });
    await writeFile(modelPath(file), bytes);
  },
  async delete() {},
};

/**
 * The verified bytes of one model file. Downloads it from the Hub only when
 * NOTELIGHT_MODEL_DOWNLOAD=1 (CI sets it); otherwise returns null if it is
 * not cached, like `pytest`, which never downloads.
 */
export async function modelFile(file) {
  if (!existsSync(modelPath(file)) && process.env.NOTELIGHT_MODEL_DOWNLOAD !== "1") return null;
  const { bytes } = await loadVerified(fileUrl(SPEC, file), file, { cache: fileCache });
  return bytes;
}

/**
 * Skip test `t` with `reason` when `ok` is false, or fail in CI, where a
 * silently skipped parity test would prove nothing.
 */
export function requireOrSkip(t, ok, reason) {
  if (ok) return true;
  if (process.env.CI) throw new Error(`required in CI: ${reason}`);
  t.skip(reason);
  return false;
}

/** Decode base64 little-endian float32 (the fixture's vector format). */
export function float32FromBase64(b64) {
  const bytes = Buffer.from(b64, "base64");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
}

/** Cosine similarity of two vectors. */
export function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

/**
 * A folder on disk as an ingest.js folder tree, the way the browser sees a
 * FileSystemDirectoryHandle: names, kinds, and files with size and mtime.
 */
export function fromDisk(dir, name = basename(dir)) {
  return {
    kind: "directory",
    name,
    async *entries() {
      for (const e of await readdir(dir, { withFileTypes: true })) {
        const path = join(dir, e.name);
        if (e.isDirectory()) {
          yield fromDisk(path, e.name);
        } else if (e.isFile()) {
          yield {
            kind: "file",
            name: e.name,
            async file() {
              const st = await stat(path);
              return {
                size: st.size,
                lastModified: Math.floor(st.mtimeMs),
                arrayBuffer: async () => {
                  const b = await readFile(path);
                  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
                },
              };
            },
          };
        }
      }
    },
  };
}
