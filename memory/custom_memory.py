"""
Custom Memory Structures for RAG & Agentic Workflows
Implements a dual-layer memory system:
1. Sliding Working Memory: Fast FIFO buffer of recent dialogue turns.
2. Entity & Topic Memory: Key-value semantic state tracking user preferences, active documents, and extracted variables.
3. Memory Condenser: Prevents context window bloat and reduces LLM inference latency.
"""
import time
from typing import List, Dict, Any, Optional

class DialogueTurn:
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

class CustomAgentMemory:
    """
    Session-specific custom memory manager.
    Optimizes context retrievals by separating recent dialogue from persistent entities and topic focus.
    """
    def __init__(self, session_id: str, max_turns: int = 5):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[DialogueTurn] = []
        self.entities: Dict[str, str] = {}
        self.active_topic: Optional[str] = None
        self.session_summary: Optional[str] = None

    def add_turn(self, role: str, content: str):
        self.turns.append(DialogueTurn(role=role, content=content))
        # Auto-compact if turns exceed max_turns
        if len(self.turns) > self.max_turns * 2:
            self._compact_stale_turns()

    def update_entity(self, key: str, value: str):
        self.entities[key] = value

    def set_active_topic(self, topic: str):
        self.active_topic = topic

    def _compact_stale_turns(self):
        """
        Lightweight compaction: retains the most recent turns while summarizing older turns.
        """
        stale = self.turns[:-self.max_turns]
        self.turns = self.turns[-self.max_turns:]
        
        # Simple extraction of key conversation points
        stale_snippets = [f"{t.role}: {t.content[:60]}..." for t in stale]
        compacted_note = f"Prior discussion touched on: {' | '.join(stale_snippets)}"
        if self.session_summary:
            self.session_summary += f" | {compacted_note}"
        else:
            self.session_summary = compacted_note

    def get_optimized_context(self) -> Dict[str, Any]:
        """
        Produces clean, pruned context for agent prompt injection without redundant token overhead.
        """
        recent_history = [
            {"role": t.role, "content": t.content}
            for t in self.turns[-self.max_turns:]
        ]
        
        return {
            "session_id": self.session_id,
            "active_topic": self.active_topic,
            "tracked_entities": self.entities,
            "condensed_summary": self.session_summary,
            "recent_turns": recent_history
        }

    def format_for_llm(self) -> str:
        """
        Formats memory state into a compact block for prompt injection.
        """
        blocks = []
        if self.active_topic:
            blocks.append(f"[Active Topic]: {self.active_topic}")
        if self.entities:
            ent_str = ", ".join([f"{k}={v}" for k, v in self.entities.items()])
            blocks.append(f"[Tracked Parameters]: {ent_str}")
        if self.session_summary:
            blocks.append(f"[Session History Summary]: {self.session_summary}")
            
        turn_strs = []
        for t in self.turns:
            turn_strs.append(f"{t.role.capitalize()}: {t.content}")
        
        if turn_strs:
            blocks.append("[Recent Turns]:\n" + "\n".join(turn_strs))
            
        return "\n".join(blocks) if blocks else "No prior conversation history."


class MemoryStore:
    """
    In-memory registry managing multi-user session state.
    """
    def __init__(self):
        self._sessions: Dict[str, CustomAgentMemory] = {}

    def get_or_create(self, session_id: str, max_turns: int = 5) -> CustomAgentMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = CustomAgentMemory(session_id=session_id, max_turns=max_turns)
        return self._sessions[session_id]

    def clear(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

# Global singleton for microservice
memory_registry = MemoryStore()
