"""
LangGraph Workflow Orchestrator
Constructs state machine for intent routing, concurrent multi-tool execution, and Gemini synthesis.
"""
import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.tools import ToolRegistry
from config import config

class LangGraphAgent:
    def __init__(self, tool_registry: ToolRegistry, api_key: Optional[str] = None):
        self.tool_registry = tool_registry
        self.api_key = api_key or config.GEMINI_API_KEY
        self.gemini_client = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.gemini_client = genai.GenerativeModel(config.DEFAULT_MODEL)
            except Exception as e:
                print(f"[LangGraphAgent] Gemini client note: {e}")

        # Build State Graph
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)

        # 1. Routing / Planning Node
        def classify_and_plan(state: AgentState) -> Dict[str, Any]:
            start = time.perf_counter()
            query = state["query"].lower()
            
            tools = []
            decision = "rag_and_tools"
            
            # Smart student heuristic router
            if any(w in query for w in ["what", "how", "explain", "why", "where", "docs", "architecture", "find"]):
                tools.append("retrieve_docs")
            if any(w in query for w in ["calculate", "latency", "percent", "average", "ratio", "benchmark", "numbers", "score", "metric"]):
                tools.append("calculate_metrics")
            if any(w in query for w in ["verify", "fact", "check", "confirm", "true", "accurate"]):
                tools.append("fact_checker")
                
            # Default to doc retrieval if no specific tool triggered
            if not tools:
                tools = ["retrieve_docs"]

            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Initialize latency breakdown
            breakdown = dict(state.get("latency_breakdown") or {})
            breakdown["router_ms"] = round(elapsed_ms, 2)
            
            return {
                "routing_decision": decision,
                "tools_to_run": tools,
                "latency_breakdown": breakdown
            }

        # 2. Parallel Tool Execution Node (Key for resume bullet!)
        def execute_tools(state: AgentState) -> Dict[str, Any]:
            tools = state.get("tools_to_run", ["retrieve_docs"])
            query = state["query"]
            
            # Execute selected tools concurrently
            exec_res = self.tool_registry.execute_tools_parallel(tools, query)
            tool_outputs = exec_res["outputs"]
            
            retrieved = []
            if "retrieve_docs" in tool_outputs:
                retrieved = tool_outputs["retrieve_docs"].get("results", [])

            breakdown = dict(state.get("latency_breakdown") or {})
            breakdown["tools_parallel_ms"] = exec_res["total_execution_ms"]

            return {
                "retrieved_docs": retrieved,
                "tool_outputs": tool_outputs,
                "latency_breakdown": breakdown
            }

        # 3. Gemini Synthesis Node
        def synthesize_response(state: AgentState) -> Dict[str, Any]:
            start = time.perf_counter()
            query = state["query"]
            docs = state.get("retrieved_docs", [])
            tools_out = state.get("tool_outputs", {})
            memory_str = state.get("memory_context", "")
            
            # Format context block
            ctx_text = self.tool_registry.retriever.format_context_for_prompt(docs)
            
            tool_insights = []
            if "calculate_metrics" in tools_out:
                tool_insights.append(f"Metric calculation: {tools_out['calculate_metrics'].get('data')}")
            if "fact_checker" in tools_out:
                tool_insights.append(f"Fact check: confidence={tools_out['fact_checker'].get('grounding_confidence')}, matched={tools_out['fact_checker'].get('matched_keywords')}")
            
            tools_summary = " | ".join(tool_insights) if tool_insights else "None"

            final_text = ""
            
            if self.gemini_client:
                prompt = f"""You are DocFlow-Agent, an AI microservice assistant designed with LangGraph and RAG.
User Query: {query}

[Session Memory Context]:
{memory_str}

[Retrieved Documents]:
{ctx_text}

[Tool Results]:
{tools_summary}

Instructions:
1. Provide a direct, highly grounded answer based strictly on the retrieved documents and tool results.
2. If citing documents, mention the Source title or doc_id.
3. Be concise, professional, and actionable.
"""
                try:
                    resp = self.gemini_client.generate_content(prompt)
                    final_text = resp.text
                except Exception as e:
                    final_text = f"[Gemini API fallback - Error: {e}]\nBased on retrieved data: {docs[0]['text'][:300] if docs else 'No data found'}"
            else:
                # Built-in deterministic generation for offline student demos
                doc_part = f"From source '{docs[0].get('doc_id', 'KnowledgeBase')}': {docs[0].get('text', '')[:250]}..." if docs else "No specific documents matched."
                tool_part = f" Computed stats: {tools_summary}." if tool_insights else ""
                final_text = (
                    f"**Analysis Summary**:\n"
                    f"{doc_part}\n\n"
                    f"{tool_part}\n"
                    f"*Synthesized by DocFlow LangGraph Agent.*"
                )

            elapsed_ms = (time.perf_counter() - start) * 1000
            breakdown = dict(state.get("latency_breakdown") or {})
            breakdown["synthesis_ms"] = round(elapsed_ms, 2)
            breakdown["total_pipeline_ms"] = round(
                breakdown.get("router_ms", 0) + breakdown.get("tools_parallel_ms", 0) + elapsed_ms, 2
            )

            return {
                "final_response": final_text,
                "latency_breakdown": breakdown
            }

        # Add Nodes
        builder.add_node("classify_and_plan", classify_and_plan)
        builder.add_node("execute_tools", execute_tools)
        builder.add_node("synthesize_response", synthesize_response)

        # Connect Graph
        builder.add_edge(START, "classify_and_plan")
        builder.add_edge("classify_and_plan", "execute_tools")
        builder.add_edge("execute_tools", "synthesize_response")
        builder.add_edge("synthesize_response", END)

        return builder

    def run(self, query: str, session_id: str = "default_session", memory_str: str = "") -> Dict[str, Any]:
        initial_state: AgentState = {
            "session_id": session_id,
            "query": query,
            "routing_decision": "",
            "tools_to_run": [],
            "retrieved_docs": [],
            "tool_outputs": {},
            "memory_context": memory_str,
            "final_response": "",
            "latency_breakdown": {}
        }
        return self.app.invoke(initial_state)
