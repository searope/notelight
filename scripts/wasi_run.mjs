// Run a wasm32-wasip1 binary (a `cargo test` test harness) under Node's
// built-in WASI, so the core's unit tests run as 32-bit WebAssembly without
// installing wasmtime. Cargo uses it as the runner for that target (see
// .cargo/config.toml):
//
//   cargo test --target wasm32-wasip1 -p notelight-core --lib
//
// The tests need no filesystem, so nothing is preopened.
import { readFile } from "node:fs/promises";
import { argv, exit } from "node:process";
import { WASI } from "node:wasi";

const [wasmPath, ...args] = argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [wasmPath, ...args], returnOnExit: true });
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
