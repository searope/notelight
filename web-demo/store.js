// Where the browser keeps indexes: a few files per index, under paths like
// "indexes/<id>/gen-000002/docs.bin". The same small interface sits over the
// Origin Private File System (OPFS), over IndexedDB where OPFS is missing or
// cannot write, and over a Map in the Node tests:
//
//   read(path)         -> Uint8Array | null
//   write(path, bytes) -> replaces the file; on OPFS the new bytes appear only
//                         once they are all written (createWritable swaps
//                         them in on close)
//   list(dir)          -> names of the files and folders directly in `dir`
//   remove(path)       -> deletes a file or a folder with everything in it
//
// Both browser stores are private to this origin and stay on this device.

const DB_NAME = "notelight";
const FILES = "files";

/** A store over a Map, for tests. */
export function memoryStore() {
  const files = new Map();
  return {
    kind: "memory",
    files,
    async read(path) {
      const b = files.get(path);
      return b ? b.slice() : null;
    },
    async write(path, bytes) {
      files.set(path, Uint8Array.from(bytes));
    },
    async list(dir) {
      return childNames([...files.keys()], dir);
    },
    async remove(path) {
      for (const k of [...files.keys()]) if (k === path || k.startsWith(`${path}/`)) files.delete(k);
    },
  };
}

/** Names directly inside `dir` among a flat list of file paths. */
function childNames(paths, dir) {
  const prefix = dir ? `${dir}/` : "";
  const names = new Set();
  for (const p of paths) if (p.startsWith(prefix)) names.add(p.slice(prefix.length).split("/")[0]);
  return [...names].sort();
}

// --------------------------------------------------------------------------
// OPFS
// --------------------------------------------------------------------------

/** A store in the Origin Private File System, under a `notelight` folder. */
export async function opfsStore() {
  const top = await (await navigator.storage.getDirectory()).getDirectoryHandle(DB_NAME, { create: true });

  async function dirOf(parts, create) {
    let d = top;
    for (const p of parts) d = await d.getDirectoryHandle(p, { create });
    return d;
  }
  const split = (path) => {
    const parts = path.split("/").filter(Boolean);
    return { dirs: parts.slice(0, -1), name: parts[parts.length - 1] };
  };
  const missing = (e) => e?.name === "NotFoundError" || e?.name === "TypeMismatchError";

  return {
    kind: "opfs",
    async read(path) {
      const { dirs, name } = split(path);
      try {
        const file = await (await (await dirOf(dirs, false)).getFileHandle(name)).getFile();
        return new Uint8Array(await file.arrayBuffer());
      } catch (e) {
        if (missing(e)) return null;
        throw e;
      }
    },
    async write(path, bytes) {
      const { dirs, name } = split(path);
      const handle = await (await dirOf(dirs, true)).getFileHandle(name, { create: true });
      if (handle.createWritable) {
        const w = await handle.createWritable();
        await w.write(bytes);
        await w.close();
      } else {
        // Safari before 26 has no createWritable; workers get a sync handle.
        const h = await handle.createSyncAccessHandle();
        try {
          h.truncate(0);
          h.write(bytes, { at: 0 });
          h.flush();
        } finally {
          h.close();
        }
      }
    },
    async list(dir) {
      try {
        const d = await dirOf(dir.split("/").filter(Boolean), false);
        const names = [];
        for await (const name of d.keys()) names.push(name);
        return names.sort();
      } catch (e) {
        if (missing(e)) return [];
        throw e;
      }
    },
    async remove(path) {
      const { dirs, name } = split(path);
      try {
        await (await dirOf(dirs, false)).removeEntry(name, { recursive: true });
      } catch (e) {
        if (!missing(e)) throw e;
      }
    },
  };
}

// --------------------------------------------------------------------------
// IndexedDB
// --------------------------------------------------------------------------

const done = (req) =>
  new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

/** Open (creating if needed) the notelight IndexedDB database. */
export function openDatabase() {
  const req = indexedDB.open(DB_NAME, 1);
  req.onupgradeneeded = () => {
    const db = req.result;
    if (!db.objectStoreNames.contains(FILES)) db.createObjectStore(FILES);
    if (!db.objectStoreNames.contains("handles")) db.createObjectStore("handles");
  };
  return done(req);
}

/** A store in IndexedDB: one record per file, keyed by its path. */
export async function indexedDbStore() {
  const db = await openDatabase();
  const tx = (mode) => db.transaction(FILES, mode).objectStore(FILES);
  const under = (path) => IDBKeyRange.bound(`${path}/`, `${path}/\uffff`);
  return {
    kind: "indexeddb",
    async read(path) {
      const v = await done(tx("readonly").get(path));
      return v ? new Uint8Array(v) : null;
    },
    async write(path, bytes) {
      // Store a standalone ArrayBuffer, not a view into a larger one.
      await done(tx("readwrite").put(bytes.slice().buffer, path));
    },
    async list(dir) {
      const keys = await done(tx("readonly").getAllKeys(dir ? under(dir) : undefined));
      return childNames(keys, dir);
    },
    async remove(path) {
      const store = tx("readwrite");
      await done(store.delete(path));
      await done(store.delete(under(path)));
    },
  };
}

/**
 * The best store this context can write to: OPFS if it works here (a probe
 * file is written, read back and removed), else IndexedDB. Private browsing
 * modes and older browsers refuse OPFS, or offer it without a way to write.
 */
export async function openStore() {
  if (globalThis.navigator?.storage?.getDirectory) {
    try {
      const store = await opfsStore();
      const probe = new Uint8Array([110, 108]);
      await store.write("probe", probe);
      const back = await store.read("probe");
      await store.remove("probe");
      if (back?.length === probe.length) return store;
    } catch {
      // fall through to IndexedDB
    }
  }
  return indexedDbStore();
}

// --------------------------------------------------------------------------
// Folder handles, so "Update" can re-read a folder without picking it again
// --------------------------------------------------------------------------

/** Remember a FileSystemDirectoryHandle for index `id` (IndexedDB clones it). */
export async function saveHandle(id, handle) {
  const db = await openDatabase();
  await done(db.transaction("handles", "readwrite").objectStore("handles").put(handle, id));
}

/** The handle saved for `id`, or null. */
export async function loadHandle(id) {
  const db = await openDatabase();
  return (await done(db.transaction("handles", "readonly").objectStore("handles").get(id))) ?? null;
}

/** Forget the handle for `id`. */
export async function deleteHandle(id) {
  const db = await openDatabase();
  await done(db.transaction("handles", "readwrite").objectStore("handles").delete(id));
}
