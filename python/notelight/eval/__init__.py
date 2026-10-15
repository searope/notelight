"""Retrieval-quality evaluation on BEIR datasets (`notelight eval`).

- `datasets`: pinned BEIR downloads and a loader for the BEIR folder layout
- `metrics`: recall@k and nDCG@k, as in the BEIR reference evaluation
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
from notelight.eval.metrics import ndcg_at_k, overlap_at_k, recall_at_k
from notelight.eval.report import format_markdown
from notelight.eval.runner import (
    BASELINE_SEARCHES,
    METHODS,
    BaselineRow,
    DatasetResult,
    MethodResult,
    evaluate,
)

__all__ = [
    "BASELINE_SEARCHES",
    "BEIR_DATASETS",
    "METHODS",
    "BaselineRow",
    "Dataset",
    "DatasetResult",
    "DatasetSpec",
    "MethodResult",
    "NumpyVectors",
    "ensure_dataset",
    "evaluate",
    "format_markdown",
    "load_beir",
    "load_dataset",
    "ndcg_at_k",
    "overlap_at_k",
    "recall_at_k",
]
