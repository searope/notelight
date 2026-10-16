# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/searope/notelight/security/advisories/new)
("Report a vulnerability" on the Security tab), not in a public issue. Include
the notelight version (`notelight version`), your OS, and the steps or files
that show the problem. A fix ships as a patch release with a published
advisory that credits you, unless you prefer otherwise.

Only the latest release on PyPI gets security fixes.

## Threat model

notelight indexes files you point it at and serves them back to you and to
your AI agents, on your own machine. It is built so that:

- **Nothing leaves the machine.** No telemetry, no analytics, no accounts.
  The only network access is a one-time download of the embedding model
  (bge-small from the Hugging Face Hub) and, for `notelight eval`, of the BEIR
  datasets. Both are pinned and checked against a SHA-256 before use, and
  `--offline` turns downloads off entirely.
- **The web UI is localhost-only.** `notelight serve` binds to `127.0.0.1`
  and rejects requests whose `Host` header is not `127.0.0.1`, `localhost` or
  `[::1]`, which blocks DNS-rebinding attacks from web pages. It is read-only:
  it never writes or changes an index.
- **The MCP server is read-only and limited to indexed files.** `notelight mcp`
  speaks MCP over stdio to the client that started it, never takes the index
  writer lock, and `read_context` serves only paths listed in the index's own
  file table. A prompt-injected agent cannot use it to read other files.
- **Index files are untrusted input.** The Rust core validates every index
  file fully on open (bounds-checked, overflow-safe offsets) before searching
  it, and contains no `unsafe` code apart from the memory map in
  `index/storage.rs`.

Out of scope: an attacker who already runs code as your user (they can read
your notes and indexes directly), and what an MCP client or agent does with
the text notelight returns to it. Indexes are stored unencrypted in your user
data directory, with the same protection as the notes they were built from.

Examples of what we do want to hear about: the web UI reachable from another
machine or another origin, `read_context` returning a file that is not
indexed, a crafted index file crashing the reader or reading out of bounds, or a
download that skips verification.
