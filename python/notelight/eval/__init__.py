"""Retrieval-quality evaluation on BEIR datasets and a notes benchmark (`notelight eval`).

- `datasets`: pinned BEIR downloads and a loader for the BEIR folder layout
- `notes`: short, note-style queries over this repository's docs, labelled by line range
- `metrics`: recall@k, nDCG@k and MRR@k, as in the BEIR reference evaluation
- `baseline`: the vector search re-implemented in NumPy, for comparison
- `runner`: embeds a corpus once, then scores every method
- `report`: Markdown tables of the results
"""

from notelight.eval.baseline import NumpyVectors
from notelight.eval.datasets import (
    BEIR_DATASETS,
    Dataset,
    DatasetSpec,
    ensure_dataset,
    load_beir,
    load_dataset,
)
from notelight.eval.metrics import mrr_at_k, ndcg_at_k, overlap_at_k, recall_at_k
from notelight.eval.notes import NOTES_SPLITS, NOTES_STYLES, load_notes
from notelight.eval.report import format_markdown
from notelight.eval.runner import (
    BASELINE_SEARCHES,
    METHODS,
    BaselineRow,
    DatasetResult,
    MethodResult,
    StyleResult,
    evaluate,
)

__all__ = [
    "BASELINE_SEARCHES",
    "BEIR_DATASETS",
    "METHODS",
    "NOTES_SPLITS",
    "NOTES_STYLES",
    "BaselineRow",
    "Dataset",
    "DatasetResult",
    "DatasetSpec",
    "MethodResult",
    "NumpyVectors",
    "StyleResult",
    "ensure_dataset",
    "evaluate",
    "format_markdown",
    "load_beir",
    "load_dataset",
    "load_notes",
    "mrr_at_k",
    "ndcg_at_k",
    "overlap_at_k",
    "recall_at_k",
]
