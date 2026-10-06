"""
DocFlow-Agent Configuration
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Vector store & RAG settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.55
    
    # Memory Settings
    MAX_WORKING_MEMORY_TURNS: int = 5
    ENABLE_ENTITY_TRACKING: bool = True
    
    # Performance & Microservice
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    SAMPLE_DOCS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")

config = Config()
