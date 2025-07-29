"""
RAGLite Streamlit Application Entry Point
"""

import os
import sys
from pathlib import Path
import streamlit as st
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add compatibility fixes for Python 3.13
import typing
if not hasattr(typing, "ParamSpec"):
    from typing_extensions import ParamSpec
    setattr(typing, "ParamSpec", ParamSpec)

# Change to RAG directory and add to Python path
os.chdir(os.path.join(os.path.dirname(__file__), 'RAG'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'RAG'))

# Try importing required packages
def install_chromadb():
    """Install ChromaDB and its dependencies."""
    try:
        packages = [
            "chromadb",
            "onnxruntime",
            "chroma-hnswlib",
            "pysqlite3-binary"
        ]
        for package in packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
        return True
    except Exception as e:
        logger.error(f"Failed to install packages: {e}")
        return False

# Try importing ChromaDB
try:
    # First try importing pysqlite3 for SQLite compatibility
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        logger.info("Using pysqlite3 for better SQLite compatibility")
    except ImportError:
        logger.warning("pysqlite3 not available, using default sqlite3")
    
    # Now try importing ChromaDB
    import chromadb
    from chromadb.config import Settings
    st.success("✅ ChromaDB imported successfully")
except ImportError as e:
    st.error(f"❌ Failed to import ChromaDB: {e}")
    st.info("Installing ChromaDB and dependencies...")
    if install_chromadb():
        try:
            import chromadb
            from chromadb.config import Settings
            st.success("✅ ChromaDB installed and imported successfully")
        except ImportError as e:
            st.error(f"❌ Failed to import ChromaDB after installation: {e}")
            st.stop()
    else:
        st.error("❌ Failed to install ChromaDB and dependencies")
        st.stop()

# Import and run the main app
try:
    from raglite.api.app import RAGLiteApp
    
    # Initialize and run the app
    if __name__ == "__main__":
        app = RAGLiteApp()
        app.run()
except Exception as e:
    st.error(f"Failed to initialize RAGLite: {str(e)}")
    logger.error(f"Application error: {e}")
    st.info("Please check the logs for more details.") 