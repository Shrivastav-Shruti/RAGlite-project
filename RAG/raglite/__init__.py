"""
RAGLite: A CPU-optimized Retrieval-Augmented Generation pipeline.

This package provides a complete RAG system using:
- sentence-transformers for embeddings
- ChromaDB for vector storage
- Groq API for LLM inference
- Streamlit for the user interface
"""

__version__ = "1.0.0"
__author__ = "RAGLite Team"

from .loaders import DocumentLoader
from .embeddings import EmbeddingModel
from .vectorstore import ChromaManager
from .retrieval import Retriever
from .llm import GroqClient
from .utils import PromptBuilder

__all__ = [
    "DocumentLoader",
    "EmbeddingModel", 
    "ChromaManager",
    "Retriever",
    "GroqClient",
    "PromptBuilder"
] 