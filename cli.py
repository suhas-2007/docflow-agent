"""
Interactive CLI for DocFlow-Agent
Allows testing query flows, inspecting LangGraph node transitions, and checking custom memory directly from terminal.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from vector_pipeline.indexer import VectorIndexer
from vector_pipeline.retriever import DynamicRetriever
from memory.custom_memory import memory_registry
from agent.tools import ToolRegistry
from agent.graph import LangGraphAgent

def main():
    print("=" * 65)
    print(" 🚀 DocFlow-Agent: Terminal Interactive Shell")
    print(" Powered by LangGraph & Google Gemini API")
    print("=" * 65)

    indexer = VectorIndexer()
    
    # Ingest sample docs
    docs_dir = config.SAMPLE_DOCS_DIR
    if os.path.exists(docs_dir):
        for fname in os.listdir(docs_dir):
            if fname.endswith(".md"):
                fpath = os.path.join(docs_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                indexer.ingest_document(doc_id=fname, content=content)
        print(f"[*] Ingested sample docs. Index size: {len(indexer.chunks)} chunks.")

    retriever = DynamicRetriever(indexer=indexer)
    tool_registry = ToolRegistry(retriever=retriever)
    agent = LangGraphAgent(tool_registry=tool_registry)

    session_id = "cli_session"
    session_mem = memory_registry.get_or_create(session_id)

    print("\nType your question, or 'benchmark', 'memory', 'exit'.\n")

    while True:
        try:
            query = input("You > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if query.lower() == "memory":
                print("\n[Current Session Memory Context]:")
                print(session_mem.format_for_llm())
                print()
                continue
            if query.lower() == "benchmark":
                from benchmarks.latency_benchmark import run_latency_benchmark
                run_latency_benchmark(num_iterations=3)
                continue

            mem_context = session_mem.format_for_llm()
            print("\n[*] Invoking LangGraph workflow...")
            result = agent.run(query=query, session_id=session_id, memory_str=mem_context)

            # Record turn in memory
            session_mem.add_turn(role="user", content=query)
            session_mem.add_turn(role="assistant", content=result["final_response"])

            print("\n--- DocFlow Agent Response ---")
            print(result["final_response"])
            print("------------------------------")
            lb = result.get("latency_breakdown", {})
            print(f"[Latency Metrics] Router: {lb.get('router_ms', 0)}ms | Tools (Parallel): {lb.get('tools_parallel_ms', 0)}ms | Synthesis: {lb.get('synthesis_ms', 0)}ms | Total: {lb.get('total_pipeline_ms', 0)}ms\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[Error]: {e}")

if __name__ == "__main__":
    main()
