"""
Agentic Microservice Application (Flask + LangGraph + Google Gemini API)
Exposes REST endpoints for query execution, document ingestion, memory inspection, and latency benchmarks.
"""
import sys
import os
import glob
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from config import config
from vector_pipeline.indexer import VectorIndexer
from vector_pipeline.retriever import DynamicRetriever
from memory.custom_memory import memory_registry
from agent.tools import ToolRegistry
from agent.graph import LangGraphAgent

app = Flask(__name__)
CORS(app)

# Initialize Core Services
print("[Microservice] Initializing Vector Indexer and Retriever...")
indexer = VectorIndexer()
retriever = DynamicRetriever(indexer=indexer)
tool_registry = ToolRegistry(retriever=retriever)
agent = LangGraphAgent(tool_registry=tool_registry)

# Pre-load sample docs if index empty
def preload_sample_documents():
    docs_dir = config.SAMPLE_DOCS_DIR
    if os.path.exists(docs_dir):
        files = glob.glob(os.path.join(docs_dir, "*.md"))
        for filepath in files:
            doc_id = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks_count = indexer.ingest_document(doc_id=doc_id, content=content, metadata={"source": doc_id})
            print(f"[Microservice] Indexed {doc_id} -> {chunks_count} chunks.")

preload_sample_documents()

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocFlow Agent | LangGraph & Gemini Microservice</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0B0F19;
            --surface: #111827;
            --surface-border: #1F2937;
            --primary: #3B82F6;
            --primary-glow: rgba(59, 130, 246, 0.25);
            --accent: #10B981;
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
            --code-bg: #030712;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: rgba(17, 24, 39, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--surface-border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .brand-badge {
            background: linear-gradient(135deg, #2563EB, #7C3AED);
            color: white;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            letter-spacing: 0.05em;
        }
        .brand h1 {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .header-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.1);
            color: #34D399;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }
        .status-dot {
            width: 7px;
            height: 7px;
            background: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10B981;
        }
        main {
            flex: 1;
            max-width: 1300px;
            width: 100%;
            margin: 0 auto;
            padding: 1.5rem 2rem;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 1.5rem;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .card-header {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--surface-border);
            font-weight: 600;
            font-size: 0.95rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-area {
            flex: 1;
            padding: 1.25rem;
            overflow-y: auto;
            max-height: 520px;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message {
            max-width: 85%;
            padding: 0.85rem 1.1rem;
            border-radius: 10px;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .msg-user {
            align-self: flex-end;
            background: #2563EB;
            color: #fff;
            border-bottom-right-radius: 2px;
        }
        .msg-agent {
            align-self: flex-start;
            background: #1F2937;
            border: 1px solid #374151;
            border-bottom-left-radius: 2px;
        }
        .msg-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            padding-top: 0.4rem;
            display: flex;
            gap: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .input-bar {
            padding: 1rem 1.25rem;
            border-top: 1px solid var(--surface-border);
            display: flex;
            gap: 0.75rem;
            background: #0D1322;
        }
        input[type="text"] {
            flex: 1;
            background: #161F30;
            border: 1px solid #2D3748;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: white;
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }
        button.btn-primary {
            background: #2563EB;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0 1.25rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        button.btn-primary:hover {
            background: #1D4ED8;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        .metric-badge {
            font-family: 'JetBrains Mono', monospace;
            background: #030712;
            border: 1px solid #1F2937;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            color: #60A5FA;
            margin-bottom: 0.5rem;
        }
        .quick-btn {
            background: #1F2937;
            border: 1px solid #374151;
            color: #D1D5DB;
            padding: 0.4rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            text-align: left;
            margin-bottom: 0.4rem;
            width: 100%;
        }
        .quick-btn:hover {
            background: #2D3748;
            color: white;
        }
        pre {
            background: var(--code-bg);
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            overflow-x: auto;
            color: #A7F3D0;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="brand-badge">STUDENT PORTFOLIO</span>
            <h1>DocFlow-Agent Microservice</h1>
        </div>
        <div class="header-meta">
            <span>LangGraph Engine</span>
            <span>Gemini API</span>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>Active Microservice</span>
            </div>
        </div>
    </header>

    <main>
        <div class="card">
            <div class="card-header">
                <span>Interactive Agent Workflow Dialogue</span>
                <span style="font-size: 0.8rem; color: var(--text-muted);">Session: <code id="session-tag">session_demo</code></span>
            </div>
            <div class="chat-area" id="chat-messages">
                <div class="message msg-agent">
                    <strong>DocFlow Assistant:</strong> Hello! I am the LangGraph-powered agent microservice. I dynamically retrieve indexed technical docs, execute multi-step tools in parallel, and maintain a custom dual-tier memory buffer. Ask me anything!
                </div>
            </div>
            <form class="input-bar" id="chat-form" onsubmit="handleSend(event)">
                <input type="text" id="query-input" placeholder="e.g. How does parallel tool invocation reduce execution latency?" autocomplete="off" />
                <button type="submit" class="btn-primary" id="send-btn">Send</button>
            </form>
        </div>

        <div class="sidebar">
            <div class="card" style="padding: 1.25rem;">
                <h3 style="font-size: 0.95rem; margin-bottom: 0.75rem;">Sample Inquiries (Demo)</h3>
                <button class="quick-btn" onclick="sendPrompt('How does parallel tool invocation reduce execution latency?')">⚡ Latency Optimization</button>
                <button class="quick-btn" onclick="sendPrompt('Explain the vector indexing chunking and embedding specs.')">📚 Vector Indexing & RAG</button>
                <button class="quick-btn" onclick="sendPrompt('Calculate metrics for latencies: 450, 320, 210, 180 ms.')">🧮 Multi-Tool: Metric Calculator</button>
                <button class="quick-btn" onclick="sendPrompt('What are the tiers in custom memory architecture?')">🧠 Custom Memory Structure</button>
            </div>

            <div class="card" style="padding: 1.25rem;">
                <h3 style="font-size: 0.95rem; margin-bottom: 0.75rem;">Real-Time Execution Latency</h3>
                <div class="metric-badge" id="latency-display">
                    Pipeline Latency: Awaiting call...
                </div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                    Parallel LangGraph execution vs Serial calls
                </div>
                <button class="quick-btn" style="background:#2563EB; color:white; text-align:center;" onclick="runBenchmark()">Run Live Benchmark</button>
                <pre id="benchmark-output" style="margin-top:0.5rem; display:none;"></pre>
            </div>

            <div class="card" style="padding: 1.25rem;">
                <h3 style="font-size: 0.95rem; margin-bottom: 0.5rem;">Custom Memory Inspector</h3>
                <pre id="memory-state">Active session initialized.</pre>
            </div>
        </div>
    </main>

    <script>
        const sessionId = "session_" + Math.random().toString(36).substring(2, 7);
        document.getElementById('session-tag').innerText = sessionId;

        async function handleSend(e) {
            e.preventDefault();
            const input = document.getElementById('query-input');
            const query = input.value.trim();
            if (!query) return;

            input.value = "";
            appendMessage("user", query);

            const sendBtn = document.getElementById('send-btn');
            sendBtn.disabled = true;
            sendBtn.innerText = "Thinking...";

            try {
                const res = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: query, session_id: sessionId })
                });
                const data = await res.json();
                
                const metaText = `Routing: ${data.routing_decision} | Parallel Tools: ${data.latency_breakdown.tools_parallel_ms || 0}ms | Total: ${data.latency_breakdown.total_pipeline_ms || 0}ms`;
                appendMessage("agent", data.response, metaText);

                document.getElementById('latency-display').innerText = 
                    `Total Pipeline: ${data.latency_breakdown.total_pipeline_ms}ms\\nRouter: ${data.latency_breakdown.router_ms}ms | Tools (Parallel): ${data.latency_breakdown.tools_parallel_ms}ms | Synthesis: ${data.latency_breakdown.synthesis_ms}ms`;

                // Update memory display
                updateMemory();
            } catch (err) {
                appendMessage("agent", "Error communicating with microservice: " + err);
            } finally {
                sendBtn.disabled = false;
                sendBtn.innerText = "Send";
            }
        }

        function sendPrompt(text) {
            document.getElementById('query-input').value = text;
            handleSend(new Event('submit'));
        }

        function appendMessage(role, text, meta) {
            const chat = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = `message msg-${role}`;
            
            let html = `<strong>${role === 'user' ? 'You' : 'DocFlow Agent'}:</strong><br/>` + text.replace(/\\n/g, "<br/>");
            if (meta) {
                html += `<div class="msg-meta">${meta}</div>`;
            }
            div.innerHTML = html;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function updateMemory() {
            try {
                const res = await fetch(`/api/memory/${sessionId}`);
                const data = await res.json();
                document.getElementById('memory-state').innerText = JSON.stringify(data, null, 2);
            } catch (e) {}
        }

        async function runBenchmark() {
            const pre = document.getElementById('benchmark-output');
            pre.style.display = 'block';
            pre.innerText = "Running multi-step tool benchmark...";
            try {
                const res = await fetch('/api/benchmark');
                const data = await res.json();
                pre.innerText = 
                    `Sequential Time: ${data.sequential_ms} ms\\n` +
                    `Parallel Time:   ${data.parallel_ms} ms\\n` +
                    `Latency Cut:     ${data.speedup_percent}%\\n` +
                    `Tools Executed:  ${data.tools_tested.join(', ')}`;
            } catch (e) {
                pre.innerText = "Benchmark failed: " + e;
            }
        }
        updateMemory();
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_DASHBOARD)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "DocFlow-Agent Microservice",
        "indexed_chunks": len(indexer.chunks),
        "gemini_active": agent.gemini_client is not None,
        "default_model": config.DEFAULT_MODEL
    })

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default_session")

    if not query:
        return jsonify({"error": "Query string is required"}), 400

    # Retrieve & update custom memory
    session_mem = memory_registry.get_or_create(session_id, max_turns=config.MAX_WORKING_MEMORY_TURNS)
    memory_context = session_mem.format_for_llm()

    # Run LangGraph workflow
    result = agent.run(query=query, session_id=session_id, memory_str=memory_context)

    # Record turn in custom memory
    session_mem.add_turn(role="user", content=query)
    session_mem.add_turn(role="assistant", content=result["final_response"])

    # If active entities found, store them
    if "tools_to_run" in result and "calculate_metrics" in result["tools_to_run"]:
        session_mem.update_entity("last_tool", "calculate_metrics")
    if result.get("retrieved_docs"):
        session_mem.set_active_topic(result["retrieved_docs"][0].get("doc_id", "General"))

    return jsonify({
        "response": result["final_response"],
        "routing_decision": result.get("routing_decision"),
        "retrieved_docs_count": len(result.get("retrieved_docs", [])),
        "latency_breakdown": result.get("latency_breakdown", {}),
        "tool_outputs": result.get("tool_outputs", {})
    })

@app.route("/api/ingest", methods=["POST"])
def ingest_endpoint():
    data = request.get_json() or {}
    doc_id = data.get("doc_id", f"doc_{int(time.time())}")
    content = data.get("content", "").strip()
    metadata = data.get("metadata", {})

    if not content:
        return jsonify({"error": "Content is required"}), 400

    count = indexer.ingest_document(doc_id=doc_id, content=content, metadata=metadata)
    return jsonify({
        "status": "success",
        "doc_id": doc_id,
        "chunks_indexed": count,
        "total_index_size": len(indexer.chunks)
    })

@app.route("/api/memory/<session_id>", methods=["GET"])
def memory_endpoint(session_id):
    session_mem = memory_registry.get_or_create(session_id)
    return jsonify(session_mem.get_optimized_context())

@app.route("/api/benchmark", methods=["GET"])
def benchmark_endpoint():
    test_tools = ["retrieve_docs", "calculate_metrics", "fact_checker"]
    test_query = "Calculate latency reduction percentages from 320 to 180 ms and verify architecture docs."
    
    seq_res = tool_registry.execute_tools_sequential(test_tools, test_query)
    par_res = tool_registry.execute_tools_parallel(test_tools, test_query)
    
    seq_time = seq_res["total_execution_ms"]
    par_time = par_res["total_execution_ms"]
    
    speedup = round(((seq_time - par_time) / seq_time) * 100, 1) if seq_time > 0 else 0.0

    return jsonify({
        "tools_tested": test_tools,
        "sequential_ms": seq_time,
        "parallel_ms": par_time,
        "speedup_percent": max(0.0, speedup)
    })

if __name__ == "__main__":
    print(f"[Microservice] Starting DocFlow-Agent server on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=False)
