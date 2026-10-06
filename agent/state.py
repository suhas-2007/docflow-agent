"""
LangGraph Agent State Definition
"""
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    session_id: str
    query: str
    routing_decision: str                # e.g. "rag_and_tools", "direct_answer", "summarize"
    tools_to_run: List[str]              # e.g. ["retrieve_docs", "calculate_metrics", "fact_check"]
    retrieved_docs: List[Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    memory_context: str
    final_response: str
    latency_breakdown: Dict[str, float]  # Tracks individual stage times for performance monitoring
