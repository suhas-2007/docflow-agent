# RAG and Memory Management Specification

## Vector Indexing Pipeline
The vector indexing pipeline converts unstructured text into dense representations:
- **Chunking**: Text is split into sliding windows of 400-500 words with 80-100 word overlap to preserve contextual continuity across chunk boundaries.
- **Embedding Generation**: Uses Google's `text-embedding-004` (768-dimensional embeddings) to capture semantic relationships.
- **Index Persistence**: Chunk metadata (source document ID, chunk index, section headers) are serialized alongside vector arrays for instant zero-latency reloading.

## Dynamic Retrieval and Ranking
- Incoming user queries are normalized and transformed into query vectors.
- A cosine similarity threshold of 0.45+ filters out low-confidence hallucinations.
- Top-K relevant chunks (K=3) are retrieved and injected into the LLM synthesis context.

## Custom Memory Architecture
Standard conversation buffers dump entire dialogue histories into the prompt, resulting in quadratic token growth and increasing time-to-first-token (TTFT).
The custom memory structure resolves this via a dual-tier approach:
1. **Tier 1 (Working Memory)**: Keeps the last 5 dialogue turns verbatim.
2. **Tier 2 (Entity & Semantic Memory)**: Extracts key entities, user constraints, and active document IDs into a structured key-value state.
3. **Auto-Compaction**: Older dialogue turns are summarized into a compact session note rather than dropped entirely.
