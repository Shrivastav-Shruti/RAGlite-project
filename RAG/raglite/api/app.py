"""
RAGLite Streamlit Application

A beautiful and intuitive web interface for the RAGLite RAG system.
"""

import streamlit as st
import logging
import os
import time
import json
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, continue without it
    pass

# Add the parent directory to the Python path to import raglite
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from raglite import (
        DocumentLoader, 
        EmbeddingModel, 
        ChromaManager, 
        Retriever, 
        GroqClient, 
        PromptBuilder
    )
except ImportError as e:
    st.error(f"Failed to import RAGLite components: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAGLite - AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* ChatGPT-like dark theme */
    .main {
        background-color: #343541 !important;
    }
    
    .stApp {
        background-color: #343541 !important;
    }
    
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #40414f;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: white;
    }
    
    .document-card {
        background: #40414f;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
        color: white;
    }
    
    .answer-box {
        background: #40414f;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #ececf1;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .source-box {
        background: #40414f;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #9c27b0;
        margin: 0.5rem 0;
        font-size: 0.9em;
        color: #8e8ea0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat interface styles */
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 15px;
        max-width: 80%;
    }
    
    .user-message {
        background: #10a37f;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background: #343541;
        color: white;
        margin-right: auto;
    }
    
    .chat-input {
        border: 2px solid #565869;
        border-radius: 10px;
        padding: 1rem;
        font-size: 16px;
        transition: border-color 0.3s ease;
        background: #40414f;
        color: white;
    }
    
    .chat-input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 0.2rem rgba(16, 163, 127, 0.25);
    }
    
    /* Chat layout fixes */
    .main .block-container {
        padding-bottom: 2rem;
    }
    
    /* Ensure chat messages are properly spaced */
    .chat-message-container {
        margin-bottom: 2rem;
    }
    
    /* Fix input area at bottom */
    .chat-input-area {
        position: sticky;
        bottom: 0;
        background: #343541;
        padding: 1rem 0;
        border-top: 1px solid #565869;
        z-index: 100;
    }
    
    /* Chat message spacing */
    .chat-message {
        margin: 1.5rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Bot message styling */
    .bot-message-container {
        display: flex;
        justify-content: flex-start;
        margin: 1.5rem 0;
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* User message styling */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin: 1.5rem 0;
        animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Message bubble improvements */
    .message-bubble {
        border-radius: 18px;
        padding: 1rem 1.5rem;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        word-wrap: break-word;
        line-height: 1.6;
    }
    
    .bot-bubble {
        background: #343541;
        color: white;
        margin-right: auto;
    }
    
    .user-bubble {
        background: #10a37f;
        color: white;
        margin-left: auto;
    }
    
    .bot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
    }
    
    .conversation-history {
        background: #40414f;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s ease;
        color: white;
    }
    
    .conversation-history:hover {
        background: #565869;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #202123 !important;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #40414f !important;
        color: white !important;
        border: 1px solid #565869 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 0.2rem rgba(16, 163, 127, 0.25) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    .stButton > button:hover {
        background-color: #0d8a6f !important;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 80px;
    }
    
    .loading-dots div {
        position: absolute;
        top: 33px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #10a37f;
        animation-timing-function: cubic-bezier(0, 1, 1, 0);
    }
    
    .loading-dots div:nth-child(1) {
        left: 8px;
        animation: loading-dots1 0.6s infinite;
    }
    
    .loading-dots div:nth-child(2) {
        left: 8px;
        animation: loading-dots2 0.6s infinite;
    }
    
    .loading-dots div:nth-child(3) {
        left: 32px;
        animation: loading-dots2 0.6s infinite;
    }
    
    .loading-dots div:nth-child(4) {
        left: 56px;
        animation: loading-dots3 0.6s infinite;
    }
    
    @keyframes loading-dots1 {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    
    @keyframes loading-dots3 {
        0% { transform: scale(1); }
        100% { transform: scale(0); }
    }
    
    @keyframes loading-dots2 {
        0% { transform: translate(0, 0); }
        100% { transform: translate(24px, 0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model to avoid re-initialization."""
    try:
        from raglite import EmbeddingModel
        return EmbeddingModel()
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def initialize_chroma_manager():
    """Initialize and cache ChromaDB manager."""
    try:
        # Import directly to ensure ChromaDB is available
        import chromadb
        from raglite import ChromaManager
        
        # Create the ChromaManager with a specific path
        chroma_path = os.path.abspath("./data/chroma_db")
        logger.info(f"Initializing ChromaDB at path: {chroma_path}")
        
        return ChromaManager(persist_directory=chroma_path)
    except ImportError as e:
        st.error(f"Failed to import ChromaDB: {e}")
        st.info("Installing ChromaDB...")
        os.system("pip install chromadb")
        try:
            import chromadb
            from raglite import ChromaManager
            return ChromaManager(persist_directory="./data/chroma_db")
        except Exception as e:
            st.error(f"Failed to initialize ChromaDB after installation: {e}")
            return None
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None

class RAGLiteApp:
    """Main RAGLite Streamlit application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.initialize_session_state()
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        if 'groq_client' not in st.session_state:
            st.session_state.groq_client = None
        if 'prompt_builder' not in st.session_state:
            st.session_state.prompt_builder = PromptBuilder()
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
        if 'deleted_files' not in st.session_state:
            st.session_state.deleted_files = set()
        if 'upload_key' not in st.session_state:
            st.session_state.upload_key = 0
        if 'chat_input_key' not in st.session_state:
            st.session_state.chat_input_key = 0
        if 'question_sent' not in st.session_state:
            st.session_state.question_sent = False
    
    def check_existing_documents(self):
        """Check if documents already exist in the system and update session state."""
        try:
            if st.session_state.retriever is not None:
                existing_docs = st.session_state.retriever.list_documents()
                if existing_docs and len(existing_docs) > 0:
                    st.session_state.documents_loaded = True
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking existing documents: {e}")
            return False
    
    def initialize_components(self):
        """Initialize RAGLite components using cached models."""
        try:
            if st.session_state.retriever is None:
                with st.spinner("Loading AI models (cached after first load)..."):
                    # Use cached models to avoid re-initialization
                    embedding_model = load_embedding_model()
                    chroma_manager = initialize_chroma_manager()
                    
                    if embedding_model and chroma_manager:
                        st.session_state.retriever = Retriever(
                            embedding_model=embedding_model,
                            chroma_manager=chroma_manager
                        )
                        st.success("✅ AI models loaded successfully (using cache)!")
                        
                        # Check for existing documents after retriever is initialized
                        if self.check_existing_documents():
                            st.success("✅ Found existing documents in the system!")
                    else:
                        st.error("❌ Failed to load required models")
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
    
    def setup_groq_client(self, api_key: str = None) -> bool:
        """Setup Groq client with API key from parameter or environment."""
        try:
            # Try environment variable first, then parameter
            groq_api_key = api_key or os.getenv("GROQ_API_KEY")
            
            if not groq_api_key or groq_api_key == "your_groq_api_key_here":
                st.error("❌ Groq API key not found. Please set GROQ_API_KEY in .env file or enter it in the sidebar.")
                return False
            
            st.session_state.groq_client = GroqClient(api_key=groq_api_key)
            if st.session_state.groq_client.test_connection():
                st.success("✅ Groq API connection successful!")
                return True
            else:
                st.error("❌ Failed to connect to Groq API")
                return False
        except Exception as e:
            st.error(f"❌ Groq API setup failed: {e}")
            return False
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 RAGLite - AI Document Assistant</h1>
            <p>Upload documents, ask questions, get intelligent answers powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration from sidebar settings."""
        return {
            'temperature': st.session_state.get('temperature', 0.2),
            'max_tokens': st.session_state.get('max_tokens', 1024),
            'top_k': st.session_state.get('top_k', 5),
            'similarity_threshold': st.session_state.get('similarity_threshold', 0.3)
        }
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Auto-setup API key from environment if not already done
            if st.session_state.groq_client is None:
                env_api_key = os.getenv("GROQ_API_KEY")
                if env_api_key and env_api_key != "your_groq_api_key_here":
                    self.setup_groq_client()
                else:
                    st.error("❌ Groq API key not found in environment. Please set GROQ_API_KEY in .env file.")
            
            # API Status (no manual input needed)
            st.subheader("🤖 AI Model Status")
            
            if st.session_state.groq_client is not None:
                st.success("✅ Groq API Connected")
                model_info = st.session_state.groq_client.get_model_info()
                
                # Show model with environment variable info
                env_model = os.getenv("LLM_MODEL", "Not set")
                st.caption(f"Model: {model_info['model']}")
                if env_model != "Not set":
                    st.caption(f"Config: {env_model}")
                else:
                    st.caption("⚠️ LLM_MODEL not set in .env")
            else:
                st.error("❌ Groq API Not Connected")
                st.caption("Check GROQ_API_KEY in .env file")
            
            st.divider()
            
            # Model Configuration
            st.subheader("🧠 Model Settings")
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Higher values make responses more creative but less focused"
            )
            
            max_tokens = st.slider(
                "Max Response Length",
                min_value=256,
                max_value=2048,
                value=1024,
                step=128,
                help="Maximum number of tokens in the response"
            )
            
            top_k = st.slider(
                "Documents to Retrieve",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of relevant documents to use for answering"
            )
            
            similarity_threshold = st.slider(
                "Relevance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum similarity score to include documents (higher = more relevant only)"
            )
            
            # Threshold guidance
            if similarity_threshold <= 0.3:
                st.info("🔵 **Low Threshold**: Including loosely related content")
            elif similarity_threshold <= 0.6:
                st.success("🟢 **Balanced**: Recommended for most queries")
            else:
                st.warning("🟡 **High Threshold**: Only highly relevant matches")
            
            st.divider()
            
            # System Stats
            if st.session_state.retriever:
                st.subheader("📊 System Stats")
                try:
                    stats = st.session_state.retriever.get_retriever_stats()
                    if 'error' not in stats:
                        st.metric(
                            "Documents Loaded", 
                            stats['vector_store']['document_count']
                        )
                        st.metric(
                            "Embedding Dimension", 
                            stats['embedding_model']['embedding_dimension']
                        )
                        
                        # Add Total Files metric
                        try:
                            analytics = st.session_state.retriever.get_document_analytics()
                            if analytics.get('total_files', 0) > 0:
                                st.metric(
                                    "Total Files",
                                    analytics['total_files']
                                )
                            else:
                                st.metric(
                                    "Total Files",
                                    0
                                )
                        except Exception as e:
                            st.metric(
                                "Total Files",
                                "N/A"
                            )
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
            
            # Reset Button
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Reset System", help="Clear all documents and reset the system", key="reset_system"):
                    if st.session_state.retriever:
                        st.session_state.retriever.reset_vector_store()
                    st.session_state.documents_loaded = False
                    st.session_state.chat_history = []
                    st.success("System reset successfully!")
                    st.rerun()
            
            with col2:
                if st.button("🔧 Reset Database", help="Reset ChromaDB database if deletion fails", key="reset_database"):
                    try:
                        if st.session_state.retriever:
                            st.session_state.retriever.chroma_manager.reset_collection()
                        st.success("Database reset successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Database reset failed: {e}")
            
            return {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_k': top_k,
                'similarity_threshold': similarity_threshold
            }
    
    def render_document_upload(self):
        """Render document upload interface."""
        st.header("📁 Document Upload")
        
        # Clear uploader if there are deleted files
        if st.session_state.deleted_files:
            # Clear any existing file uploader states
            for key in list(st.session_state.keys()):
                if key.startswith("file_uploader_"):
                    del st.session_state[key]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload your documents",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT",
                key=f"file_uploader_{st.session_state.upload_key}"
            )
            
            if uploaded_files:
                # Filter out deleted files
                active_files = [f for f in uploaded_files if f.name not in st.session_state.deleted_files]
                
                if active_files:
                    st.info(f"📄 {len(active_files)} file(s) selected")
                    
                    if st.button("🚀 Process Documents", type="primary", key="process_documents"):
                        self.process_documents(active_files)
                else:
                    st.warning("📄 All selected files have been deleted. Please select new files.")
                    # Clear the uploader if all files are deleted
                    st.session_state.upload_key += 1
                    st.rerun()
        
        with col2:
            st.markdown("""
            **💡 Tips:**
            - Upload multiple documents at once
            - Supported formats: PDF, DOCX, TXT
            - Documents are automatically chunked for optimal retrieval
            - Your data stays secure and private
            """)
            
            # Show deleted files info
            if st.session_state.deleted_files:
                st.info(f"🗑️ **{len(st.session_state.deleted_files)} file(s) deleted**\nCheck Advanced tab for document management")
                
                if st.button("🔄 Clear Deleted Files List", key="clear_deleted_files"):
                    st.session_state.deleted_files.clear()
                    st.rerun()
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents with detailed progress tracking."""
        try:
            # Create processing status containers
            st.info("🚀 Starting document processing pipeline...")
            
            # Main progress container
            progress_container = st.container()
            metrics_container = st.container()
            
            with progress_container:
                # Overall progress
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                # Step-by-step progress
                step_container = st.container()
                
                loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
                all_documents = []
                processing_results = []
                
                start_time = time.time()
                total_files = len(uploaded_files)
                
                with step_container:
                    st.markdown("### 📋 Processing Steps:")
                    step1 = st.empty()
                    step2 = st.empty()
                    step3 = st.empty()
                    
                    step1.info("🔄 **Step 1:** Loading and chunking documents...")
                    step2.warning("⏳ **Step 2:** Generating embeddings (pending)")
                    step3.warning("⏳ **Step 3:** Storing in vector database (pending)")
                
                # Process each file with detailed tracking
                for i, uploaded_file in enumerate(uploaded_files):
                    file_start_time = time.time()
                    
                    # Update progress
                    file_progress = (i / total_files) * 0.6  # 60% for file processing
                    overall_progress.progress(file_progress)
                    status_text.text(f"📄 Processing file {i+1}/{total_files}: {uploaded_file.name}")
                    
                    # Read file bytes
                    file_bytes = uploaded_file.read()
                    file_size_mb = len(file_bytes) / (1024*1024)
                    
                    # Real-time file metrics
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📁 Current File", f"{i+1}/{total_files}")
                        with col2:
                            st.metric("📏 File Size", f"{file_size_mb:.1f} MB")
                        with col3:
                            st.metric("📄 Total Chunks", len(all_documents))
                        with col4:
                            elapsed = time.time() - start_time
                            st.metric("⏱️ Elapsed", f"{elapsed:.1f}s")
                    
                    # Load and chunk document
                    status_text.text(f"🔪 Chunking {uploaded_file.name}...")
                    documents = loader.load_from_bytes(file_bytes, uploaded_file.name)
                    all_documents.extend(documents)
                    
                    file_time = time.time() - file_start_time
                    
                    processing_results.append({
                        'filename': uploaded_file.name,
                        'chunks': len(documents),
                        'size': f"{file_size_mb:.1f} MB",
                        'time': f"{file_time:.2f}s",
                        'chunks_per_sec': f"{len(documents)/file_time:.1f}"
                    })
                    
                    # Show immediate feedback
                    st.success(f"✅ **{uploaded_file.name}**: {len(documents)} chunks created in {file_time:.2f}s")
                
                # Update step 1 completion
                step1.success("✅ **Step 1:** Document loading and chunking completed!")
                step2.info("🔄 **Step 2:** Generating embeddings...")
                
                # Add documents to vector store with progress tracking
                overall_progress.progress(0.7)
                status_text.text("🧠 Generating embeddings for all document chunks...")
                
                # Create a progress placeholder for embedding generation
                embedding_progress = st.progress(0)
                embedding_status = st.empty()
                
                vector_start_time = time.time()
                
                # Custom progress tracking for embeddings
                def progress_callback(current, total):
                    progress = current / total
                    embedding_progress.progress(progress)
                    embedding_status.text(f"🧠 Processing batch {current}/{total} ({progress*100:.1f}%)")
                
                # Add documents to retriever
                st.session_state.retriever.add_documents(all_documents)
                
                vector_time = time.time() - vector_start_time
                
                # Update final steps
                step2.success(f"✅ **Step 2:** Embeddings generated in {vector_time:.2f}s!")
                step3.info("🔄 **Step 3:** Storing in vector database...")
                
                overall_progress.progress(0.95)
                status_text.text("💾 Finalizing vector database storage...")
                
                time.sleep(0.5)  # Brief pause for visual effect
                
                step3.success("✅ **Step 3:** Vector database storage completed!")
                overall_progress.progress(1.0)
                total_time = time.time() - start_time
                
                # Clear processing status
                status_text.empty()
                overall_progress.empty()
                embedding_progress.empty()
                embedding_status.empty()
                
                st.session_state.documents_loaded = True
                
                # Show completion summary (without balloons)
                st.success("🎉 **Document Processing Pipeline Completed Successfully!**")
                
                # Enhanced summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="📁 Files Processed",
                        value=len(uploaded_files),
                        delta=f"{len(uploaded_files)} new files"
                    )
                
                with col2:
                    st.metric(
                        label="📄 Total Chunks",
                        value=len(all_documents),
                        delta=f"Avg: {len(all_documents)//len(uploaded_files)} per file"
                    )
                
                with col3:
                    st.metric(
                        label="⏱️ Total Time",
                        value=f"{total_time:.1f}s",
                        delta=f"{len(all_documents)/total_time:.1f} chunks/sec"
                    )
                
                with col4:
                    st.metric(
                        label="🧠 Embedding Time",
                        value=f"{vector_time:.1f}s",
                        delta=f"{len(all_documents)/vector_time:.0f} chunks/sec"
                    )
                
                # Performance insights
                st.markdown("### 📊 Processing Performance Analysis")
                st.markdown("""
                <style>
                .performance-analysis {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    color: white;
                    margin: 10px 0;
                }
                .performance-analysis h4 {
                    color: white;
                    margin-bottom: 15px;
                }
                .performance-analysis ul {
                    color: white;
                }
                .performance-analysis li {
                    color: white;
                    margin: 5px 0;
                }
                </style>
                """, unsafe_allow_html=True)
                
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    # File processing breakdown
                    st.markdown("**📁 File Processing Breakdown:**")
                    for result in processing_results:
                        efficiency = float(result['chunks_per_sec'])
                        efficiency_color = "#28a745" if efficiency > 50 else "#ffc107" if efficiency > 20 else "#dc3545"
                        
                        st.markdown(f"""
                        <div class="performance-analysis">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="color: white;">📄 {result['filename']}</strong>
                                <span style="background: {efficiency_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                                    {result['chunks_per_sec']} chunks/s
                                </span>
                            </div>
                            <div style="margin-top: 0.5rem; color: white;">
                                📦 {result['chunks']} chunks | 📏 {result['size']} | ⏱️ {result['time']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with perf_col2:
                    # Overall system metrics
                    avg_chunk_size = sum(len(doc['text']) for doc in all_documents) // len(all_documents)
                    total_chars = sum(len(doc['text']) for doc in all_documents)
                    chars_per_sec = total_chars / total_time
                    
                    st.markdown(f"""
                    <div class="performance-analysis">
                        <h4 style="color: white;">🔧 System Performance Metrics</h4>
                        <ul style="color: white;">
                            <li><strong>Total characters processed:</strong> {total_chars:,}</li>
                            <li><strong>Average chunk size:</strong> {avg_chunk_size} characters</li>
                            <li><strong>Processing speed:</strong> {chars_per_sec:,.0f} chars/sec</li>
                            <li><strong>Embedding efficiency:</strong> {len(all_documents)/vector_time:.1f} chunks/sec</li>
                            <li><strong>Memory usage:</strong> ~{len(all_documents) * 384 / (1024*1024):.1f} MB (embeddings)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Next steps with clear call-to-action
                st.markdown("---")
                st.markdown("### 🚀 What's Next?")
                
                next_col1, next_col2, next_col3 = st.columns(3)
                
                with next_col1:
                    st.info("💬 **Ask Questions**\nGo to the Chat tab to start asking questions about your documents")
                
                with next_col2:
                    st.info("📊 **View Analytics**\nCheck the Advanced tab for usage statistics and system info")
                
                with next_col3:
                    st.info("📄 **Add More Docs**\nUpload additional documents to expand your knowledge base")
                
                # Navigation options
                nav_col1, nav_col2, nav_col3 = st.columns(3)
                
                with nav_col1:
                    if st.button("💬 **Go to Chat**", type="primary", key="go_to_chat"):
                        st.session_state.active_tab = "Chat"
                        st.rerun()
                
                with nav_col2:
                    if st.button("📊 **View Analytics**", type="secondary", key="view_analytics"):
                        st.info("💡 Click on the 'Advanced' tab > 'Document Analytics' to see details!")
                
                with nav_col3:
                    if st.button("🔄 **Process More**", type="secondary", key="process_more"):
                        st.info("💡 Upload more documents above to expand your knowledge base!")
                
                # Keep analytics visible without auto-refresh
                st.info("✨ **Analytics will remain visible!** Check the 'Advanced' tab > 'Document Analytics' for detailed management.")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")
            logger.error(f"Document processing error: {e}")
    
    def render_chat_interface(self, config):
        """Render the main chat interface with ChatGPT-like design."""
        st.header("💬 Ask Questions About Your Documents")
        
        # Check if documents are loaded, and if not, check for existing documents
        if not st.session_state.documents_loaded:
            # Try to check for existing documents
            if self.check_existing_documents():
                st.success("✅ Found existing documents in the system!")
            else:
                st.warning("📤 Please upload and process documents first!")
                return
        
        # Ensure the Groq client is set up
        if st.session_state.groq_client is None:
            env_api_key = os.getenv("GROQ_API_KEY")
            if env_api_key and env_api_key != "your_groq_api_key_here":
                self.setup_groq_client()
            else:
                st.warning("🔑 Please configure your Groq API key in the sidebar!")
                return
        
        # Create a two-column layout: sidebar for history, main for chat
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self.render_chat_sidebar()
        
        with col2:
            self.render_main_chat_area(config)
    
    def render_chat_sidebar(self):
        """Render the chat sidebar with ChatGPT-like design."""
        
        # New Chat button at the top (like ChatGPT)
        if st.button("🆕 New Chat", type="primary", key="new_chat_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Chats section (like ChatGPT)
        if st.session_state.chat_history:
            st.markdown("**💬 Chats**")
            
            # Display conversation history
            for i, chat in enumerate(st.session_state.chat_history):
                # Show conversation preview
                question_preview = chat['question'][:50] + "..." if len(chat['question']) > 50 else chat['question']
                
                # Make the entire text clickable
                chat_key = f"chat_{i}"
                if st.button(question_preview, key=chat_key):
                    st.session_state.selected_chat = i
                    st.rerun()
        else:
            st.info("No conversations yet. Start asking questions!")
        
        # Clear All button at bottom (like ChatGPT)
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("🗑️ Clear All", type="secondary", key="clear_all_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    def render_main_chat_area(self, config):
        """Render the main chat area with ChatGPT-like interface."""
        
        # Force Chat tab to be active when processing questions
        if st.session_state.get('is_processing', False) or st.session_state.get('question_sent', False):
            st.session_state.active_tab = "Chat"
        
        # Create a container for the entire chat area
        main_chat_container = st.container()
        
        with main_chat_container:
            # Chat messages area (scrollable)
            messages_container = st.container()
            
            with messages_container:
                # Display chat messages
                self.display_chat_messages()
            
            # Input area at bottom - always visible
            st.markdown("---")
            
            # Create a container for the input area
            with st.container():
                # Set active tab to Chat before form is submitted
                st.session_state.active_tab = "Chat"
                
                # Create a custom input and button layout
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    # Initialize the input value based on processing state
                    if 'is_processing' in st.session_state:
                        input_value = ''
                    else:
                        input_value = ''  # Always start with empty input
                    
                    # Text area with Telegram-like styling
                    question = st.text_area(
                        "Chat Input",
                        value=input_value,
                        placeholder="Ask me anything about your documents...",
                        key=f"question_input_{st.session_state.get('chat_input_key', 0)}",  # Dynamic key to force refresh
                        height=50,
                        label_visibility="collapsed",
                        help="Type your question here and press Enter or click the send button"
                    )
                
                with col2:
                    # Send button positioned next to the text area
                    if st.button("➤", key="send_button", help="Send your question"):
                        if question.strip():
                            # Set current tab to Chat to prevent redirects
                            st.session_state.active_tab = "Chat"
                            st.session_state.is_processing = True
                            st.session_state.question_sent = True
                            
                            # Show processing message
                            with st.spinner("🤖 RAG is analyzing your query..."):
                                try:
                                    self.process_question_with_loading(question, config)
                                except Exception as e:
                                    st.error(f"Error processing question: {e}")
                                    st.session_state.is_processing = False
                                    st.session_state.question_sent = False
                            
                            # Ensure we stay in Chat tab
                            st.session_state.active_tab = "Chat"
                            st.rerun()
            
            # Also handle Enter key submission
            if question and st.session_state.get('question_input') != input_value:
                # Set current tab to Chat to prevent redirects
                st.session_state.active_tab = "Chat"
                st.session_state.is_processing = True
                st.session_state.question_sent = True
                
                # Process the question and stay in chat tab
                try:
                    self.process_question_with_loading(question, config)
                except Exception as e:
                    st.error(f"Error processing question: {e}")
                    st.session_state.is_processing = False
                    st.session_state.question_sent = False
                
                # Ensure we stay in Chat tab
                st.session_state.active_tab = "Chat"
    
    def handle_send_click(self):
        """Handle send button click to prevent tab switching."""
        # This function ensures we stay in the chat tab
        pass
    
    def display_chat_messages(self):
        """Display chat messages in a ChatGPT-like format."""
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #8e8ea0;">
                <div style="margin-bottom: 2rem;">
                    <span style="font-size: 3rem;">🤖</span>
                </div>
                <h2 style="color: white; margin-bottom: 1rem;">Welcome to RAGLite Assistant!</h2>
                <p style="font-size: 1.1em; line-height: 1.6;">I'm here to help you understand your documents. Ask me anything about the uploaded documents!</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Create a container for chat messages
        chat_messages_container = st.container()
        
        with chat_messages_container:
            # Display messages in chronological order (oldest first)
            for i, chat in enumerate(st.session_state.chat_history):
                # User message (right side) - show first
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                    <div style="
                        background: #10a37f; 
                        color: white; 
                        padding: 1rem 1.5rem; 
                        border-radius: 18px; 
                        max-width: 80%; 
                        text-align: left;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="font-weight: 500; margin-bottom: 0.5rem;">You</div>
                        <div style="line-height: 1.6; font-size: 1rem;">
                            {chat['question']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bot message (left side) - show after user question
                # First replace any HTML-like tags with safe equivalents
                answer_text = chat['answer']
                # Replace <br> tags with proper line breaks
                answer_text = answer_text.replace("<br>", "\n")
                # Convert any other HTML tags to safe versions
                answer_text = answer_text.replace("<", "&lt;").replace(">", "&gt;")
                # Now convert newlines to proper <br> tags for HTML display
                answer_text = answer_text.replace("\n", "<br>")
                
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
                    <div style="
                        background: #343541; 
                        color: white; 
                        padding: 1rem 1.5rem; 
                        border-radius: 18px; 
                        max-width: 80%; 
                        text-align: left;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="
                                background: #10a37f; 
                                color: white; 
                                border-radius: 50%; 
                                width: 32px; 
                                height: 32px; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center; 
                                margin-right: 0.75rem; 
                                font-size: 16px;
                            ">🤖</span>
                            <span style="font-weight: 500; color: #10a37f;">RAGLite Assistant</span>
                        </div>
                        <div style="color: #ececf1; line-height: 1.6; font-size: 1rem;">
                            {answer_text}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Sources (if available) - collapsible
                if chat.get('sources'):
                    with st.expander("📚 View Sources", expanded=False):
                        for j, source in enumerate(chat['sources'], 1):
                            similarity = source.get('similarity', 0)
                            relevance_color = "#10a37f" if similarity > 0.7 else "#f59e0b" if similarity > 0.5 else "#ef4444"
                            
                            st.markdown(f"""
                            <div style="
                                margin: 0.5rem 0; 
                                padding: 1rem; 
                                background: #40414f; 
                                border-radius: 8px; 
                                border-left: 4px solid {relevance_color};
                            ">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <strong style="color: #ececf1;">📄 {source.get('filename', 'Unknown')}</strong>
                                    <span style="
                                        background: {relevance_color}; 
                                        color: white; 
                                        padding: 2px 8px; 
                                        border-radius: 12px; 
                                        font-size: 0.75em;
                                    ">
                                        {similarity:.3f}
                                    </span>
                                </div>
                                <div style="color: #8e8ea0; font-size: 0.9em; line-height: 1.5;">
                                    {source.get('text', '')[:200]}{'...' if len(source.get('text', '')) > 200 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Metadata row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"⏱️ {chat.get('response_time', 0):.2f}s")
                with col2:
                    st.caption(f"📄 {chat.get('num_documents', 0)} docs")
                with col3:
                    st.caption(f"🎯 {chat.get('model', 'Unknown')}")
                
                st.markdown("---")
    
    def process_question_with_loading(self, question: str, config: Dict[str, Any]):
        """Process a user question with loading animation."""
        try:
            # Add timestamp to the question
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Show loading message with custom styling
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
                <div style="background: #40414f; padding: 15px 30px; border-radius: 10px; display: flex; align-items: center;">
                    <div style="margin-right: 10px;">🤖</div>
                    <div>RAG is analyzing your query...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Process the actual question
            start_time = time.time()
            
            # Retrieve relevant documents
            retrieved_docs = st.session_state.retriever.retrieve_similar(
                query=question,
                top_k=config['top_k'],
                similarity_threshold=config['similarity_threshold']
            )
            
            # Generate response using Groq
            response = st.session_state.groq_client.generate_rag_response(
                query=question,
                retrieved_documents=retrieved_docs,
                include_sources=True
            )
            
            response_time = time.time() - start_time
            
            # Prepare sources for display
            sources = []
            for doc in retrieved_docs:
                source = {
                    'filename': doc.get('metadata', {}).get('filename', 'Unknown'),
                    'text': doc.get('text', ''),
                    'similarity': doc.get('similarity', 0)
                }
                sources.append(source)
            
            # Add to chat history
            chat_entry = {
                'question': question,
                'answer': response['response'],
                'sources': sources,
                'response_time': response_time,
                'num_documents': len(retrieved_docs),
                'model': response.get('model', 'Unknown'),
                'timestamp': timestamp
            }
            
            st.session_state.chat_history.append(chat_entry)
            
            # Clear the input field and increment chat input key
            if "question_input" in st.session_state:
                del st.session_state.question_input
            st.session_state.chat_input_key += 1
            
            # Rerun to show the new message
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing question: {e}")
            if "question_input" in st.session_state:
                del st.session_state.question_input
            st.session_state.chat_input_key += 1
    
    def render_advanced_features(self):
        """Render advanced features tab."""
        st.header("🔧 Advanced Features")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Document Analytics", "📋 Prompt Templates", "📊 Usage Analytics", "⚙️ System Info"])
        
        with tab1:
            self.render_document_analytics()
        
        with tab2:
            self.render_prompt_templates("advanced")
        
        with tab3:
            self.render_analytics()
        
        with tab4:
            self.render_system_info()
    
    def render_document_analytics(self):
        """Render comprehensive document analytics and management."""
        st.subheader("📁 Document Library & Analytics")
        
        if not st.session_state.retriever:
            st.warning("⚠️ Retriever not initialized. Please restart the app.")
            return
        
        # Get document analytics
        try:
            analytics = st.session_state.retriever.get_document_analytics()
            
            if analytics.get('error'):
                st.error(f"❌ Error loading analytics: {analytics['error']}")
                return
            
            files = analytics.get('files', {})
            total_files = analytics.get('total_files', 0)
            total_chunks = analytics.get('total_chunks', 0)
            
            if total_files == 0:
                st.info("📂 No documents uploaded yet. Go to the 'Upload Documents' tab to add some!")
                return
            
            # Overview metrics - always visible
            st.markdown("### 📊 Collection Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📁 Total Files",
                    value=total_files,
                    delta=f"{total_files} documents"
                )
            
            with col2:
                st.metric(
                    label="📄 Total Chunks",
                    value=total_chunks,
                    delta=f"Avg: {total_chunks//total_files if total_files > 0 else 0} per file"
                )
            
            with col3:
                avg_chars = sum(file_data['total_chars'] for file_data in files.values()) // total_files if total_files > 0 else 0
                st.metric(
                    label="📝 Avg File Size",
                    value=f"{avg_chars:,} chars",
                    delta="characters"
                )
            
            with col4:
                # Memory estimation (384 dimensions per chunk)
                memory_mb = total_chunks * 384 * 4 / (1024 * 1024)  # 4 bytes per float
                st.metric(
                    label="🧠 Memory Usage",
                    value=f"{memory_mb:.1f} MB",
                    delta="embeddings"
                )
            
            st.markdown("---")
            
            # Document management section
            st.markdown("### 📋 Document Management")
            
            # Control buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Refresh button
                if st.button("🔄 Refresh Analytics", type="secondary", key="refresh_analytics"):
                    st.rerun()
            
            with col2:
                # Export button (placeholder)
                if st.button("📤 Export List", key="export_list"):
                    file_list = "\n".join([f"- {filename} ({data['chunk_count']} chunks)" 
                                         for filename, data in files.items()])
                    st.download_button(
                        label="💾 Download File List",
                        data=file_list,
                        file_name="document_list.txt",
                        mime="text/plain"
                    )
            
            with col3:
                # Danger zone button
                if st.button("🗑️ Clear All", type="secondary", key="clear_all_docs"):
                    st.warning("⚠️ This will delete all documents!")
            
            # Individual file management
            st.markdown("### 📄 Individual Documents")
            
            # Sort options
            sort_col1, sort_col2 = st.columns([1, 3])
            
            with sort_col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Upload Time", "Filename", "Chunk Count", "File Size"],
                    key="doc_sort_selectbox"
                )
            
            with sort_col2:
                search_query = st.text_input(
                    "🔍 Search documents:",
                    placeholder="Filter by filename...",
                    key="doc_search"
                )
            
            # Filter and sort files
            filtered_files = files.copy()
            
            if search_query:
                filtered_files = {
                    filename: data for filename, data in files.items()
                    if search_query.lower() in filename.lower()
                }
            
            # Sort files
            if sort_by == "Upload Time":
                sorted_files = sorted(filtered_files.items(), 
                                    key=lambda x: x[1].get('upload_time', ''), reverse=True)
            elif sort_by == "Filename":
                sorted_files = sorted(filtered_files.items(), key=lambda x: x[0])
            elif sort_by == "Chunk Count":
                sorted_files = sorted(filtered_files.items(), 
                                    key=lambda x: x[1].get('chunk_count', 0), reverse=True)
            else:  # File Size
                sorted_files = sorted(filtered_files.items(), 
                                    key=lambda x: x[1].get('total_chars', 0), reverse=True)
            
            if not sorted_files:
                st.info("🔍 No documents match your search criteria.")
                return
            
            # Display each document
            for filename, data in sorted_files:
                with st.expander(f"📄 **{filename}**", expanded=False):
                    # File details
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.markdown(f"""
                        **📊 Statistics:**
                        - **Chunks:** {data.get('chunk_count', 0)}
                        - **Total Characters:** {data.get('total_chars', 0):,}
                        - **Avg Chunk Size:** {data.get('avg_chunk_size', 0)} chars
                        """)
                    
                    with detail_col2:
                        st.markdown(f"""
                        **📝 File Info:**
                        - **Type:** {data.get('file_type', 'Unknown')}
                        - **Uploaded:** {data.get('upload_time', 'Unknown')[:19] if data.get('upload_time') != 'Unknown' else 'Unknown'}
                        """)
                    
                    with detail_col3:
                        # Action buttons
                        st.markdown("**⚡ Actions:**")
                        
                        # View details button
                        if st.button(f"👁️ View Details", key=f"view_{filename}"):
                            details = st.session_state.retriever.get_document_details(filename)
                            if details.get('exists'):
                                st.json(details)
                            else:
                                st.error(f"❌ {details.get('error', 'File not found')}")
                        
                        # Delete button with confirmation
                        if st.button(f"🗑️ Delete", key=f"delete_{filename}", type="secondary"):
                            # Use session state for confirmation
                            st.session_state[f"confirm_delete_{filename}"] = True
                        
                        # Confirmation dialog
                        if st.session_state.get(f"confirm_delete_{filename}", False):
                            st.warning(f"⚠️ **Confirm deletion of '{filename}'?**")
                            st.markdown(f"This will permanently delete **{data.get('chunk_count', 0)} chunks** from the vector database.")
                            
                            confirm_col1, confirm_col2 = st.columns(2)
                            
                            with confirm_col1:
                                if st.button(f"✅ Yes, Delete", key=f"confirm_yes_{filename}", type="primary"):
                                    # Perform deletion
                                    with st.spinner(f"Deleting {filename}..."):
                                        result = st.session_state.retriever.delete_document(filename)
                                    
                                    if result.get('success'):
                                        # Add to deleted files set and clear uploader
                                        st.session_state.deleted_files.add(filename)
                                        st.session_state.upload_key += 1  # Force uploader refresh
                                        
                                        # Clear the file uploader widget state
                                        if f"file_uploader_{st.session_state.upload_key-1}" in st.session_state:
                                            del st.session_state[f"file_uploader_{st.session_state.upload_key-1}"]
                                        
                                        # Clear any other file uploader states
                                        for key in list(st.session_state.keys()):
                                            if key.startswith("file_uploader_"):
                                                del st.session_state[key]
                                        
                                        st.success(f"✅ Deleted '{filename}' ({result.get('chunks_deleted', 0)} chunks)")
                                        st.session_state[f"confirm_delete_{filename}"] = False
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete '{filename}': {result.get('error', 'Unknown error')}")
                            
                            with confirm_col2:
                                if st.button(f"❌ Cancel", key=f"confirm_no_{filename}"):
                                    st.session_state[f"confirm_delete_{filename}"] = False
                                    st.rerun()
                    
                    # Performance insights
                    chunk_count = data.get('chunk_count', 0)
                    total_chars = data.get('total_chars', 0)
                    
                    if chunk_count > 0:
                        st.markdown("**📈 Performance Insights:**")
                        
                        # Efficiency indicators
                        efficiency_items = []
                        
                        if chunk_count > 100:
                            efficiency_items.append("🟡 Large file - may affect search speed")
                        elif chunk_count > 50:
                            efficiency_items.append("🟢 Medium file - good balance")
                        else:
                            efficiency_items.append("🟢 Small file - fast search")
                        
                        avg_chunk = data.get('avg_chunk_size', 0)
                        if avg_chunk > 800:
                            efficiency_items.append("🟡 Large chunks - good context, slower processing")
                        elif avg_chunk > 400:
                            efficiency_items.append("🟢 Optimal chunk size")
                        else:
                            efficiency_items.append("🟡 Small chunks - less context per chunk")
                        
                        for item in efficiency_items:
                            st.markdown(f"- {item}")
            
            # End of document analytics section
        
        except Exception as e:
            st.error(f"❌ Error loading document analytics: {e}")
            logger.error(f"Document analytics error: {e}")

    def render_prompt_templates(self, context="main"):
        """Render prompt template configuration."""
        st.subheader("🎯 Prompt Templates")
        
        available_templates = st.session_state.prompt_builder.get_available_templates()
        
        selected_template = st.selectbox(
            "Choose a prompt template:",
            options=available_templates,
            help="Different templates optimize for different types of responses",
            key=f"prompt_template_selectbox_{context}"
        )
        
        template_content = st.session_state.prompt_builder.get_template(selected_template)
        
        st.text_area(
            "Template Preview:",
            value=template_content,
            height=200,
            disabled=True,
            key=f"template_preview_{context}"
        )
        
        # Custom template
        st.subheader("➕ Create Custom Template")
        
        custom_name = st.text_input("Template Name:", key=f"custom_name_{context}")
        custom_template = st.text_area(
            "Template Content:",
            placeholder="Use {question} and {context} placeholders",
            height=150,
            key=f"custom_template_{context}"
        )
        
        if st.button("💾 Save Custom Template", key=f"save_template_{context}") and custom_name and custom_template:
            st.session_state.prompt_builder.add_custom_template(custom_name, custom_template)
            st.success(f"✅ Template '{custom_name}' saved successfully!")
    
    def render_analytics(self):
        """Render analytics dashboard."""
        st.subheader("📈 Usage Analytics")
        
        if st.session_state.chat_history:
            # Response time analytics
            response_times = [chat.get('response_time', 0) for chat in st.session_state.chat_history]
            avg_response_time = sum(response_times) / len(response_times)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(st.session_state.chat_history))
            with col2:
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            with col3:
                avg_docs = sum(chat.get('num_documents', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
                st.metric("Avg Documents/Query", f"{avg_docs:.1f}")
            
            # Most recent questions
            st.subheader("🕒 Recent Questions")
            for chat in st.session_state.chat_history[-5:]:
                st.write(f"• {chat['question']}")
        else:
            st.info("📈 Analytics will appear after you start asking questions!")
    
    def render_system_info(self):
        """Render system information."""
        st.subheader("⚙️ System Information")
        
        if st.session_state.retriever:
            try:
                stats = st.session_state.retriever.get_retriever_stats()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🧠 Embedding Model:**")
                    st.json(stats.get('embedding_model', {}))
                
                with col2:
                    st.markdown("**💾 Vector Store:**")
                    st.json(stats.get('vector_store', {}))
                
                if st.session_state.groq_client:
                    st.markdown("**🤖 LLM Configuration:**")
                    st.json(st.session_state.groq_client.get_model_info())
                
            except Exception as e:
                st.error(f"Error loading system info: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        # Initialize session state and components
        self.initialize_session_state()
        self.initialize_components()
        
        # Set page config
        st.set_page_config(
            page_title="RAGLite Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown("""<style>...""", unsafe_allow_html=True)
        
        # Render header
        self.render_header()
        
        # Create tabs
        tab_labels = [
            "📤 Upload Documents", 
            "💬 Chat", 
            "🔧 Advanced"
        ]
        
        # Create tabs
        tabs = st.tabs(tab_labels)
        tab1, tab2, tab3 = tabs
        
        # Upload Documents tab
        with tab1:
            self.render_document_upload()
        
        # Chat tab
        with tab2:
            # Render sidebar first to get configuration
            config = self.render_sidebar()
            
            # Check if documents are loaded
            if not st.session_state.documents_loaded:
                if self.check_existing_documents():
                    st.success("✅ Found existing documents in the system!")
                    self.render_chat_interface(config)
                else:
                    st.warning("📤 Please upload and process documents first!")
                    st.info("Go to the 'Upload Documents' tab to add your documents.")
            else:
                self.render_chat_interface(config)
        
        # Advanced Features tab
        with tab3:
            self.render_advanced_features()


# Main execution
if __name__ == "__main__":
    try:
        app = RAGLiteApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}") 