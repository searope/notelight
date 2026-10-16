## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] New logic has unit tests; invariants have proptests against a naive reference (Rust)
- [ ] CI checks pass locally: `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p notelight-core`, `pytest`, `ruff check .`, `ruff format --check .`, `mypy`
- [ ] Public Rust items have docs (with `# Errors` / `# Panics` where they apply); new `_core` exports have stubs in `python/notelight/_core.pyi`
- [ ] On-disk layout unchanged, or `SCHEMA_VERSION` is bumped and `docs/index-format.md` updated in the same commit
- [ ] README claims are backed by tests or benchmarks; changed numbers are updated in the README (and `docs/results.md` for eval numbers)
- [ ] `CHANGELOG.md` has an entry under Unreleased for user-visible changes
- [ ] Commit messages: a summary line plus a bullet body saying what changed and why
