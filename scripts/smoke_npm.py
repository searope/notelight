"""Install a packed notelight-wasm tarball the way a user would and use it.

    python scripts/build_npm.py --pack
    python scripts/smoke_npm.py dist/npm/notelight-wasm-<version>.tgz

In a fresh scratch project (outside the repository, so nothing resolves to
the sources), it runs `npm install <tarball>`, then
`crates/notelight-wasm/npm/smoke.mjs`, which searches the committed sample
index (BM25, and hybrid against the Python CLI's top 10) and builds an index
in memory, then type-checks `smoke-types.ts` against the package's
declarations with a pinned TypeScript.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "crates" / "notelight-wasm" / "npm"
SAMPLE = ROOT / "web-demo" / "sample-index"
QUERIES = ROOT / "web-demo" / "tests" / "fixtures" / "queries.json"
TYPESCRIPT = "typescript@5.9.3"

TSCONFIG = {
    "compilerOptions": {
        "module": "nodenext",
        "moduleResolution": "nodenext",
        "target": "es2022",
        # wasm-bindgen declares `[Symbol.dispose]()` on every class.
        "lib": ["es2022", "dom", "esnext.disposable"],
        "strict": True,
        "noEmit": True,
        "skipLibCheck": False,
    },
    "files": ["smoke-types.ts"],
}


def run(*args: str, cwd: Path) -> None:
    exe = shutil.which(args[0])
    if exe is None:
        raise SystemExit(f"{args[0]} not found; install Node.js")
    subprocess.run([exe, *args[1:]], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tarball", type=Path, help="the .tgz from build_npm.py --pack")
    args = parser.parse_args()
    tarball = args.tarball.resolve(strict=True)

    work = Path(tempfile.mkdtemp(prefix="notelight-npm-smoke-"))
    try:
        project = {"name": "smoke", "private": True, "type": "module"}
        (work / "package.json").write_text(json.dumps(project), encoding="utf-8")
        (work / "tsconfig.json").write_text(json.dumps(TSCONFIG), encoding="utf-8")
        for name in ("smoke.mjs", "smoke-types.ts"):
            shutil.copyfile(SOURCES / name, work / name)
        npm_install = ["npm", "install", "--no-audit", "--no-fund", "--ignore-scripts"]
        run(*npm_install, str(tarball), TYPESCRIPT, cwd=work)
        run("node", "smoke.mjs", str(SAMPLE), str(QUERIES), cwd=work)
        run("node", str(work / "node_modules" / "typescript" / "bin" / "tsc"), "-p", ".", cwd=work)
        print("types ok")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
