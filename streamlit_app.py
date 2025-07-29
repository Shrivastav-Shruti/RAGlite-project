"""
RAGLite Streamlit Application Entry Point
"""

import os
import sys
from pathlib import Path

# Add compatibility fixes for Python 3.13
import typing
if not hasattr(typing, "ParamSpec"):
    from typing_extensions import ParamSpec
    setattr(typing, "ParamSpec", ParamSpec)

# Add the RAG directory to Python path
repo_root = Path(__file__).parent
rag_path = repo_root / "RAG"
sys.path.append(str(rag_path))

# Import and run the main app
try:
    from raglite.api.app import RAGLiteApp
    
    # Initialize and run the app
    if __name__ == "__main__":
        app = RAGLiteApp()
        app.run()
except Exception as e:
    import streamlit as st
    st.error(f"Failed to initialize RAGLite: {str(e)}")
    st.info("Please check the logs for more details.") 