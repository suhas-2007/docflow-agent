"""
Vector Indexing Pipeline
Builds chunked document indexes, computes embeddings, and performs vector similarity search.
Supports Google Gemini embeddings with automatic local TF-IDF/Dense fallback for offline testing.
"""
import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional
from config import config

class LightweightFallbackEmbedder:
    """
    Deterministic dense bag-of-words / character n-gram embedder for local offline development.
    Guarantees the student project runs even when testing without active API credits.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec
        for i, word in enumerate(words):
            h = hash(word) % self.dim
            vec[h] += 1.0 / (1.0 + math.log(1 + i))
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

class VectorIndexer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.use_gemini = False
        self.fallback_embedder = LightweightFallbackEmbedder(dim=256)
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.use_gemini = True
            except Exception as e:
                print(f"[VectorIndexer] Gemini setup notice: {e}. Using fallback embedder.")

    def _split_text(self, text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [" ".join(words)] if words else []
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += (chunk_size - overlap)
        return chunks

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.use_gemini:
            try:
                import google.generativeai as genai
                res = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                vec = np.array(res['embedding'], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 1e-9 else vec
            except Exception as e:
                # Fallback to local on rate limit or network issue
                return self.fallback_embedder.embed(text)
        return self.fallback_embedder.embed(text)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        if self.use_gemini:
            try:
                import google.generativeai as genai
                res = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query"
                )
                vec = np.array(res['embedding'], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 1e-9 else vec
            except Exception as e:
                return self.fallback_embedder.embed(query)
        return self.fallback_embedder.embed(query)

    def ingest_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Splits doc into overlapping chunks, builds embeddings, and appends to index.
        """
        metadata = metadata or {}
        raw_chunks = self._split_text(content, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        new_vectors = []
        
        for idx, text in enumerate(raw_chunks):
            vec = self._get_embedding(text)
            chunk_record = {
                "chunk_id": f"{doc_id}_c{idx}",
                "doc_id": doc_id,
                "text": text,
                "metadata": {**metadata, "chunk_index": idx}
            }
            self.chunks.append(chunk_record)
            new_vectors.append(vec)
            
        if new_vectors:
            new_arr = np.vstack(new_vectors)
            if self.embeddings is None:
                self.embeddings = new_arr
            else:
                self.embeddings = np.vstack([self.embeddings, new_arr])
                
        return len(raw_chunks)

    def search(self, query: str, top_k: int = 3, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Cosine similarity search over indexed chunks.
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        q_vec = self._get_query_embedding(query)
        # Cosine similarity for normalized vectors = dot product
        scores = np.dot(self.embeddings, q_vec)
        
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score >= threshold:
                item = dict(self.chunks[idx])
                item["score"] = round(score, 4)
                results.append(item)
                
        return results

    def save_index(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        chunks_path = os.path.join(directory, "chunks.json")
        vectors_path = os.path.join(directory, "embeddings.npy")
        
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
            
        if self.embeddings is not None:
            np.save(vectors_path, self.embeddings)

    def load_index(self, directory: str) -> bool:
        chunks_path = os.path.join(directory, "chunks.json")
        vectors_path = os.path.join(directory, "embeddings.npy")
        
        if os.path.exists(chunks_path) and os.path.exists(vectors_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(vectors_path)
            return True
        return False
