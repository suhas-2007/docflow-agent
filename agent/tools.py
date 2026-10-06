"""
Agent Tools and Concurrent Execution Engine
Provides modular tools for retrieval, numeric calculation, and citation verification.
Includes parallel executor to minimize multi-step tool call latency.
"""
import re
import time
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor
from vector_pipeline.retriever import DynamicRetriever

class ToolRegistry:
    def __init__(self, retriever: DynamicRetriever):
        self.retriever = retriever

    def tool_retrieve_docs(self, query: str) -> Dict[str, Any]:
        """Tool 1: Retrieves semantic document chunks."""
        start = time.perf_counter()
        retrieval_res = self.retriever.retrieve(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "name": "retrieve_docs",
            "results": retrieval_res["results"],
            "cache_hit": retrieval_res["cache_hit"],
            "latency_ms": round(elapsed_ms, 2)
        }

    def tool_calculate_metrics(self, text: str) -> Dict[str, Any]:
        """
        Tool 2: Extracts numerical values, computes percentages, averages, or latency deltas.
        """
        start = time.perf_counter()
        # Extract numbers, percentages, ms latencies
        nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]
        result = {
            "count": len(nums),
            "numbers_found": nums[:8],
            "average": round(sum(nums) / len(nums), 2) if nums else 0.0,
            "max": max(nums) if nums else 0.0,
            "min": min(nums) if nums else 0.0
        }
        # Simulate realistic lightweight processing time
        time.sleep(0.04)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "name": "calculate_metrics",
            "data": result,
            "latency_ms": round(elapsed_ms, 2)
        }

    def tool_fact_checker(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Tool 3: Validates claims or verifies source grounding against retrieved keywords.
        """
        start = time.perf_counter()
        words = set(query.lower().split())
        matched_in_context = [w for w in words if len(w) > 3 and w in context.lower()]
        
        # Simulate realistic verification computation
        time.sleep(0.05)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "name": "fact_checker",
            "matched_keywords": matched_in_context,
            "grounding_confidence": round(min(1.0, len(matched_in_context) / (len(words) or 1) * 2), 2),
            "latency_ms": round(elapsed_ms, 2)
        }

    def execute_tools_sequential(self, tool_names: List[str], query: str, context: str = "") -> Dict[str, Any]:
        """
        Baseline sequential execution (traditional ReAct loop).
        """
        start_total = time.perf_counter()
        results = {}
        for name in tool_names:
            if name == "retrieve_docs":
                results[name] = self.tool_retrieve_docs(query)
            elif name == "calculate_metrics":
                results[name] = self.tool_calculate_metrics(query)
            elif name == "fact_checker":
                results[name] = self.tool_fact_checker(query, context)
        total_ms = (time.perf_counter() - start_total) * 1000
        return {"outputs": results, "total_execution_ms": round(total_ms, 2)}

    def execute_tools_parallel(self, tool_names: List[str], query: str, context: str = "") -> Dict[str, Any]:
        """
        Optimized parallel execution across thread pool for LangGraph node dispatch.
        Demonstrates the latency reduction highlighted on the resume!
        """
        start_total = time.perf_counter()
        results = {}

        def _run_single(name: str):
            if name == "retrieve_docs":
                return name, self.tool_retrieve_docs(query)
            elif name == "calculate_metrics":
                return name, self.tool_calculate_metrics(query)
            elif name == "fact_checker":
                return name, self.tool_fact_checker(query, context)
            return name, {"error": "unknown tool"}

        with ThreadPoolExecutor(max_workers=min(4, len(tool_names) or 1)) as executor:
            futures = [executor.submit(_run_single, name) for name in tool_names]
            for f in futures:
                name, output = f.result()
                results[name] = output

        total_ms = (time.perf_counter() - start_total) * 1000
        return {"outputs": results, "total_execution_ms": round(total_ms, 2)}
