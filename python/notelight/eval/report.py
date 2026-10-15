"""Markdown tables for evaluation results (the README and `docs/results.md`
are pasted from this output)."""

from __future__ import annotations

from collections.abc import Sequence

from notelight.eval.runner import DatasetResult

__all__ = ["format_markdown"]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], right_from: int = 1) -> str:
    align = ["---" if i < right_from else "---:" for i in range(len(header))]
    lines = [header, align, *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def _speed(ms: float) -> str:
    return f"{ms:.2f}" if ms < 10 else f"{ms:.1f}"


def format_markdown(results: Sequence[DatasetResult]) -> str:
    """Quality, latency and Rust-vs-NumPy tables for one or more datasets."""
    if not results:
        return "No results."
    k = results[0].k
    methods = [m.method for m in results[0].methods]
    parts: list[str] = []

    if methods:
        header = ["Method", *(f"{r.dataset} nDCG@{k}" for r in results)]
        header += [f"{r.dataset} R@{k}" for r in results]
        rows = [
            [
                f"`{name}`",
                *(f"{r.method(name).ndcg:.3f}" for r in results),
                *(f"{r.method(name).recall:.3f}" for r in results),
            ]
            for name in methods
        ]
        parts.append(_table(header, rows))

        header = ["Method", *(f"{r.dataset} ms/query" for r in results)]
        rows = [[f"`{n}`", *(_speed(r.method(n).latency_ms) for r in results)] for n in methods]
        parts.append(_table(header, rows))

    baseline_rows = [
        [
            r.dataset,
            f"`{b.search}`",
            _speed(b.rust_ms),
            _speed(b.numpy_ms),
            f"{b.numpy_ms / b.rust_ms:.1f}x" if b.rust_ms > 0 else "n/a",
            f"{b.agreement:.3f}",
        ]
        for r in results
        for b in r.baseline
    ]
    if baseline_rows:
        header = ["Dataset", "Search", "Rust ms", "NumPy ms", "NumPy / Rust", f"Top-{k} agreement"]
        parts.append(_table(header, baseline_rows, right_from=2))

    sizes = ", ".join(f"{r.dataset} {r.docs:,} docs / {r.queries} queries" for r in results)
    parts.append(
        f"Embedder `{results[0].embedder}`; {sizes}; {results[0].candidates} candidates re-scored."
    )
    return "\n\n".join(parts) + "\n"
