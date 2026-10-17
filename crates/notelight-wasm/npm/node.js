// The package's entry point under Node (the "node" export condition).
//
// The wasm-bindgen `--target web` glue loads the .wasm with `fetch`, which
// Node cannot do for a file: URL, so this entry reads the module from disk
// and instantiates it synchronously on import. Calling the default export
// (`await init()`) afterwards is a no-op, so the same code runs in browsers,
// bundlers and Node.
import { readFileSync } from "node:fs";

import { initSync } from "./notelight_wasm.js";

initSync({ module: readFileSync(new URL("./notelight_wasm_bg.wasm", import.meta.url)) });

export * from "./notelight_wasm.js";
export { default } from "./notelight_wasm.js";
