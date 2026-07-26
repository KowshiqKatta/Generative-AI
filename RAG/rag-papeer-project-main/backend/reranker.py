"""Cross-encoder reranking for retrieved chunks.

Why this exists
---------------
Bi-encoder similarity search optimises *recall*, not *precision*: it reliably
pulls the right passage into the top ~20, but rarely into the top 4. The eval
run in `eval_results.json` shows exactly this signature — Contextual Recall of
1.00 alongside Contextual Relevancy of 0.09, i.e. the right content was
retrieved and then buried in noise.

A cross-encoder scores the (query, passage) pair jointly rather than comparing
two independently-computed vectors, which is far more accurate but too slow to
run over a whole collection. The standard fix is retrieve-then-rerank: overfetch
cheaply with the bi-encoder, then reorder the small candidate pool.

Runs locally via FlashRank — no API key, no per-call cost, ~34 MB model.

Failure policy
--------------
Reranking is an optimisation, never a dependency. If the model is missing,
fails to load, or throws at inference time, we log and fall back to the original
similarity order. A reranker problem must never take the app down.
"""

import logging
import os
from functools import lru_cache

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


# ── Config (all overridable via environment) ──────────────────────────────────

RERANK_ENABLED = _env_flag("RERANK_ENABLED", "true")
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANK_CACHE_DIR = os.getenv("RERANK_CACHE_DIR", "./rerank_cache")

# How many candidates to pull per final chunk, and the hard ceiling on the pool.
# 5x balances reranker quality against Qdrant latency and embedding-cache misses.
OVERFETCH_MULTIPLIER = int(os.getenv("RERANK_OVERFETCH", "5"))
OVERFETCH_MAX = int(os.getenv("RERANK_OVERFETCH_MAX", "40"))


@lru_cache(maxsize=1)
def _get_ranker():
    """Load the cross-encoder once per process. Cached across Streamlit reruns."""
    from flashrank import Ranker

    logger.info("Loading reranker model %s", RERANK_MODEL)
    return Ranker(model_name=RERANK_MODEL, cache_dir=RERANK_CACHE_DIR)


def candidate_pool_size(k: int) -> int:
    """How many chunks to fetch from the vector store to later trim down to `k`."""
    if not RERANK_ENABLED:
        return k
    return min(max(k * OVERFETCH_MULTIPLIER, k), OVERFETCH_MAX)


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Reorder `docs` by cross-encoder relevance and return the best `top_n`."""
    if not RERANK_ENABLED or not docs:
        return docs[:top_n]

    try:
        from flashrank import RerankRequest

        ranker = _get_ranker()
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    except Exception as exc:
        logger.warning(
            "Reranking unavailable (%s: %s) — falling back to similarity order.",
            type(exc).__name__,
            exc,
        )
        return docs[:top_n]

    out: list[Document] = []
    for result in ranked[:top_n]:
        doc = docs[int(result["id"])]
        # Surface the score in the graph-state inspector for debugging.
        doc.metadata = {
            **(doc.metadata or {}),
            "rerank_score": round(float(result["score"]), 4),
        }
        out.append(doc)
    return out
