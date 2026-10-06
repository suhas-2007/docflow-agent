# DocFlow-Agent: Low-Latency Agentic Microservice

> **Built by Suhas Raavi** | *Freelance AI Engineering Project (2025)*  
> **Tech Stack:** Python • LangGraph • Google Gemini API • Flask • NumPy • Vector Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Google Gemini API](https://img.shields.io/badge/LLM-Gemini_1.5_Flash-4285F4.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Background & Freelance Scope

During a freelance contract in 2025, the client needed an internal technical assistant to query documentation, calculate metrics, and verify technical claims across their engineering specifications. 

Their initial proof-of-concept had two glaring issues:
1. **Unacceptable Latency (3–5 seconds per turn)**: Traditional ReAct loops called tools one-by-one in serial roundtrips (Model $\rightarrow$ Tool 1 $\rightarrow$ Model $\rightarrow$ Tool 2 $\rightarrow$ Model $\rightarrow$ Tool 3).
2. **Context Window Inflation & Token Costs**: Appending unpruned conversation logs bloated prompt tokens quadratically, slowing down Time-To-First-Token (TTFT) and driving up API bills.

As a student developer balancing budget and speed, I re-architected their workflow into a **LangGraph-driven microservice** with **parallel tool execution** and a **custom dual-tier memory system**.

---

## 📊 Benchmark Results: Proving the Latency Reduction

To verify my optimizations before handing off the project, I wrote an automated benchmark suite (`benchmarks/latency_benchmark.py`) comparing the baseline ReAct approach against this architecture.

| Workflow Component | Baseline (Sequential) | Optimized (LangGraph + Cache) | Improvement |
| :--- | :--- | :--- | :--- |
| **Multi-Step Tools Execution** *(Doc Retrieval + Metric Calc + Fact Check)* | `93.4 ms` | `58.3 ms` | **37.5% – 42% faster** |
| **Repeated Semantic Retrieval** *(Warm LRU Query Cache)* | `0.121 ms` | `0.005 ms` | **95.7% speedup** |
| **Inference Model** | Heavy LLMs | Google Gemini 1.5 Flash | **~60% faster TTFT** |

> Run the benchmark yourself: `python benchmarks/latency_benchmark.py`

---

## 🏗️ How It Works (Architecture)

```
                            [ User Query ]
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   LangGraph Router      │
                     │  (Intent Classification)│
                     └────────────┬────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       ┌─────────────────────┐         ┌─────────────────────┐
       │   Tool 1: Retriever │         │ Tool 2 & 3: Metrics │
       │  + LRU Vector Cache │ ◄─┬─►   │  & Fact Verification│
       └─────────────────────┘   │     └─────────────────────┘
                  │       (Parallel ThreadPool)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │    Custom Dual-Tier     │
                     │    Working Memory       │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Google Gemini Synthesis │
                     │  (Grounded Citations)   │
                     └─────────────────────────┘
```

### 1. Parallel Multi-Step Tool Dispatching
Instead of waiting on each tool consecutively, independent tools (document chunk retrieval, numerical extraction, grounding verification) are dispatched concurrently using Python's `ThreadPoolExecutor` inside the LangGraph tool node.

### 2. Vector Indexing Pipeline
- **Sliding-Window Chunking**: 400-word chunks with an 80-word overlap to ensure technical definitions aren't cleaved at chunk boundaries.
- **Dual Embedding Support**: Uses Google's `text-embedding-004` (768-dim) in production, paired with an automatic deterministic fallback embedder so local tests and CI pass without needing API credits.
- **Cosine Thresholding**: Enforces a `0.45` similarity floor to filter out irrelevant noise.
- **LRU Query Caching**: Caches normalized search queries to eliminate duplicate embedding roundtrips on follow-ups.

### 3. Custom Memory Structure
Instead of dumping raw conversational history into prompts:
- **Tier 1 (Working Memory)**: Keeps a tight 5-turn sliding window of recent dialogue.
- **Tier 2 (Entity & Topic State)**: Stores active document references and extracted user parameters.
- **Auto-Compaction**: Summarizes older conversation steps if the turn count exceeds the threshold.

---

## 📂 Repository Structure

```
├── agent/
│   ├── graph.py               # LangGraph StateGraph (Router -> Parallel Tools -> Synthesizer)
│   ├── state.py               # TypedDict state contract
│   └── tools.py               # Concurrent (ThreadPoolExecutor) & sequential tool executors
├── vector_pipeline/
│   ├── indexer.py             # Sliding-window chunker, Gemini embeddings & fallback vectorizer
│   └── retriever.py           # Similarity search + LRU retrieval cache
├── memory/
│   └── custom_memory.py       # Dual-tier working memory, entity tracker, compaction
├── microservice/
│   └── app.py                 # Flask REST API microservice & Interactive Dashboard
├── sample_docs/               # Sample technical markdown files for ingestion
├── benchmarks/
│   └── latency_benchmark.py   # Latency benchmark suite
├── cli.py                     # Terminal-based interactive agent shell
├── config.py                  # System & model configuration
├── requirements.txt           # Dependencies
└── INTERVIEW_TALK_TRACK.md    # Interview prep sheet with answers to tough questions
```

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/suhas-2007/docflow-agent.git
cd docflow-agent
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Add your free [Google AI Studio Gemini API Key](https://aistudio.google.com/):
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
```
*(The system automatically falls back to offline embeddings and mock synthesis if no key is supplied, so you can test immediately).*

### 3. Run Benchmark
```bash
python benchmarks/latency_benchmark.py
```

### 4. Start the Microservice & Web UI
```bash
python microservice/app.py
```
Visit **`http://127.0.0.1:5000`** in your browser to interact with the assistant, trigger benchmarks, and inspect memory live.

### 5. Or Use the Terminal CLI
```bash
python cli.py
```

---

## 💡 Lessons Learned (Student Engineer Takeaways)

1. **AsyncIO vs ThreadPoolExecutor in LangGraph**: While AsyncIO is great, many third-party retrieval and SDK tools have blocking network calls. Wrapping them in a managed `ThreadPoolExecutor` inside the tool node proved to be the most resilient way to achieve true parallel execution without thread deadlocks.
2. **Chunk Overlap Matters**: In technical documentation, code snippets or architectural specifications often span across 300+ words. An 80-word overlap was the sweet spot to maintain context without duplicating data.
3. **Memory Pruning Saves Real Dollars**: For the client, switching from naive history concatenation to a 5-turn sliding window + entity store reduced average prompt token counts by ~40%, directly reducing API billing.

---

## 📄 License
MIT License. Free to use, fork, and study for educational or commercial purposes.
