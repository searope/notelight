# Contributing to notelight

Thanks for helping. Bug reports and small, focused pull requests are the most
useful. For anything larger, open an issue first so we can agree on the
approach. Security problems go through [SECURITY.md](SECURITY.md), not issues.

## Set up

You need [Rust](https://rustup.rs) (stable) and Python 3.10 or newer. On
Windows, building the extension also needs the MSVC C++ build tools
("Desktop development with C++" in the Visual Studio Installer).

PowerShell (Windows):

```powershell
git clone https://github.com/searope/notelight
cd notelight
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,documents,web]"   # builds the Rust extension via maturin
notelight version
```

bash (Linux, macOS):

```bash
git clone https://github.com/searope/notelight
cd notelight
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,documents,web]"
notelight version
```

After changing Rust code, rebuild the extension with `maturin develop --release`.

## Checks

CI runs these on Linux, macOS and Windows; run them before you push:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings   # pedantic lints, pinned toolchain in CI
cargo test -p notelight-core                            # unit, integration and property tests
ruff check .
ruff format --check .
mypy                                                    # strict
pytest                                                  # no network; uses a fake embedder
```

CI also builds the Rust docs with `RUSTDOCFLAGS=-D warnings` and checks the
minimum supported Rust version (1.88).

`pytest` never downloads anything. The tests that run the real bge-small
model are opt-in; they download about 130 MB once:

```powershell
$env:NOTELIGHT_MODEL_TESTS = "1"
pytest tests/test_embed.py tests/test_model_check.py
```

```bash
NOTELIGHT_MODEL_TESTS=1 pytest tests/test_embed.py tests/test_model_check.py
```

Set `NOTELIGHT_MODEL_CACHE` to a folder that already holds the model to skip
the download.

## Conventions

- New logic gets unit tests; invariants get proptests against a naive
  reference implementation.
- Rust stays plain: no `unsafe` (the memory map in `index/storage.rs` is the
  one exception), binary formats decoded with `from_le_bytes` at validated
  offsets. Public items need docs, with `# Errors` and `# Panics` sections.
- Anything exported from Rust gets a type stub in `python/notelight/_core.pyi`.
- Changing an on-disk layout means bumping `SCHEMA_VERSION` and updating
  [docs/index-format.md](docs/index-format.md) in the same commit.
- README claims must be backed by tests or benchmarks. If a number changes,
  update the README (and `docs/results.md` for eval numbers).
- User-visible changes get a line under "Unreleased" in
  [CHANGELOG.md](CHANGELOG.md).

## Commit messages

A short summary line in the imperative, then a bullet body saying what
changed and why:

```text
Reject index files with overlapping sections

- The reader now checks that section ranges do not overlap on open, so a
  crafted manifest cannot alias two arrays.
- New proptest: random section tables against a naive overlap check.
```

## License

By contributing, you agree that your contributions are dual-licensed under
MIT and Apache-2.0, like the rest of the project.
