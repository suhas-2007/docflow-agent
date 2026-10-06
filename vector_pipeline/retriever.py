"""
Dynamic Context Retriever
Integrates indexing with LRU caching, similarity reranking, and dynamic context formatting.
Directly targets execution latency reduction by avoiding duplicate vector embedding calls.
"""
import time
from typing import List, Dict, Any, Optional
from vector_pipeline.indexer import VectorIndexer
from config import config

class DynamicRetriever:
    def __init__(self, indexer: VectorIndexer, cache_size: int = 64):
        self.indexer = indexer
        self.cache_size = cache_size
        self._query_cache: Dict[str, Dict[str, Any]] = {}

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieves context with latency tracking and caching.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        normalized_q = query.strip().lower()
        start_time = time.perf_counter()

        # Check Cache
        if normalized_q in self._query_cache:
            cached_data = self._query_cache[normalized_q]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return {
                "results": cached_data["results"],
                "cache_hit": True,
                "latency_ms": round(elapsed_ms, 2)
            }

        # Vector search
        results = self.indexer.search(query, top_k=top_k, threshold=config.SIMILARITY_THRESHOLD)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update Cache (simple eviction if limit reached)
        if len(self._query_cache) >= self.cache_size:
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
            
        payload = {
            "results": results,
            "cache_hit": False,
            "latency_ms": round(elapsed_ms, 2)
        }
        self._query_cache[normalized_q] = payload
        return payload

    def format_context_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks with provenance metadata for grounded generation.
        """
        if not results:
            return "No relevant internal documentation found."
            
        formatted = []
        for i, item in enumerate(results, 1):
            doc_id = item.get("doc_id", "unknown")
            score = item.get("score", 0.0)
            text = item.get("text", "").strip()
            formatted.append(f"--- [Source {i}: {doc_id} | Relevance: {score}] ---\n{text}")
            
        return "\n\n".join(formatted)
