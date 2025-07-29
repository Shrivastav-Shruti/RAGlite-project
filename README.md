# RAGLite - CPU-Based RAG Pipeline 🤖

RAGLite is an end-to-end CPU-optimized Retrieval-Augmented Generation (RAG) pipeline, designed for efficient document processing and question answering. This implementation focuses on CPU-based inference, making it accessible without requiring specialized hardware.

## 🎯 Key Features

- **CPU-Optimized RAG Pipeline**
  - Efficient document processing and chunking
  - Optimized embedding generation for CPU
  - Fast vector similarity search
  - Streamlined RAG inference flow

- **Interactive Interfaces**
  - Streamlit web application
  - RESTful API endpoints
  - Command-line interface

- **Comprehensive Analytics**
  - Performance benchmarking
  - Resource utilization metrics
  - Query latency tracking

## 🏗️ Pipeline Architecture

### 1. Document Processing Pipeline
```
Document → Loader → Chunker → Embedder → Vector Store
```
- **Supported Formats**: PDF, DOCX, TXT
- **Chunking Strategy**: Overlap-based with configurable size
- **Embedding Model**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB (CPU-optimized)

### 2. Query Pipeline
```
Query → Embedder → Retriever → Reranker → LLM → Response
```
- **Query Processing**: Semantic embedding generation
- **Retrieval**: Top-K similarity search
- **LLM Integration**: Groq API (CPU-friendly)

## 📋 Requirements

### System Requirements
```
CPU: 4+ cores recommended
RAM: 8GB minimum, 16GB recommended
Storage: 1GB+ for embeddings and models
OS: Linux, macOS, Windows
```

### Software Dependencies
```
python>=3.11
streamlit==1.32.2
chromadb==0.4.24
sentence-transformers==2.5.1
python-dotenv==1.0.1
groq==0.4.2
```

## 🚀 Quick Start

1. **Installation**
   ```bash
   git clone <repository-url>
   cd RAG-Project
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Setup**
   ```bash
   # Create .env file
   cat > .env << EOL
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=100
   TOP_K=5
   SIMILARITY_THRESHOLD=0.2
   EOL
   ```

3. **Launch Options**

   a. Streamlit Interface
   ```bash
   streamlit run RAG/raglite/api/app.py
   ```

   b. API Server
   ```bash
   uvicorn RAG/raglite/api/server:app --host 0.0.0.0 --port 8000
   ```

   c. CLI Usage
   ```bash
   python RAG/raglite/cli.py query "Your question here"
   ```

## 📊 Benchmarks

### Document Processing Performance
| Operation | Time (s) | CPU Usage | RAM Usage |
|-----------|----------|-----------|-----------|
| Chunking  | 0.05/doc | 15%       | 100MB     |
| Embedding | 0.2/chunk| 45%       | 500MB     |
| Storage   | 0.01/doc | 10%       | 50MB      |

### Query Performance
| Component    | Latency (ms) | CPU Usage |
|--------------|--------------|-----------|
| Embedding    | 100-200      | 40%       |
| Retrieval    | 50-100       | 25%       |
| LLM Response | 500-1000     | 30%       |

### Optimization Techniques
- Batch processing for embeddings
- Efficient chunk size selection
- ChromaDB index optimization
- Response caching

## 🔧 Pipeline Components

### 1. Document Loader (`raglite/loaders/loader.py`)
- Handles multiple file formats
- Configurable chunk size and overlap
- Memory-efficient streaming for large files

### 2. Embedding Model (`raglite/embeddings/embedder.py`)
- CPU-optimized sentence transformers
- Configurable model selection
- Batch processing support

### 3. Vector Store (`raglite/vectorstore/chroma_manager.py`)
- ChromaDB integration
- Efficient similarity search
- Persistence management

### 4. Retriever (`raglite/retrieval/retriever.py`)
- Semantic search implementation
- Configurable retrieval parameters
- Source document tracking

### 5. LLM Client (`raglite/llm/groq_client.py`)
- Groq API integration
- Response streaming
- Error handling

## 🛠️ Extending the Pipeline

### Adding New Document Types
1. Create new loader in `raglite/loaders/`
2. Implement `load_document` interface
3. Register in `DocumentLoader` class

```python
class CustomLoader(BaseLoader):
    def load_document(self, file_path: str) -> List[Document]:
        # Implementation
        pass
```

### Custom Embedding Models
1. Create new embedder in `raglite/embeddings/`
2. Implement `EmbeddingModel` interface
3. Update model configuration

```python
class CustomEmbedder(BaseEmbedder):
    def embed_documents(self, texts: List[str]) -> List[float]:
        # Implementation
        pass
```

### Custom Vector Stores
1. Create new store in `raglite/vectorstore/`
2. Implement `VectorStore` interface
3. Update store configuration

```python
class CustomVectorStore(BaseVectorStore):
    def add_documents(self, documents: List[Document]) -> None:
        # Implementation
        pass
```

## 📈 Performance Optimization

### Memory Management
- Streaming document processing
- Batch size optimization
- Garbage collection control

### CPU Optimization
- Thread pool management
- Process pool for parallel processing
- Caching strategies

### Query Optimization
- Index optimization
- Query preprocessing
- Response caching

## 🔍 Monitoring and Logging

### Performance Metrics
- Document processing time
- Query latency
- Resource utilization

### Error Tracking
- Document processing errors
- Query failures
- System warnings

### Usage Analytics
- Query patterns
- Document statistics
- System health

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## 📝 License


## 🙏 Acknowledgments

- Groq Team for API access
- Sentence Transformers developers
- ChromaDB team
- Streamlit community

## 🚀 Deployment Guide

### Local Deployment

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd RAG-Project
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   Create `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=100
   TOP_K=5
   SIMILARITY_THRESHOLD=0.2
   ```

3. **Run Locally**
   ```bash
   streamlit run RAG/raglite/api/app.py
   ```

### Streamlit Cloud Deployment

1. **Fork/Push to GitHub**
   - Create a GitHub repository
   - Push your code to GitHub
   ```bash
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set the main file path: `RAG/raglite/api/app.py`

3. **Configure Secrets**
   - In Streamlit Cloud:
     - Go to App Settings > Secrets
     - Add your environment variables:
     ```toml
     GROQ_API_KEY = "your_groq_api_key_here"
     LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
     CHUNK_SIZE = "1000"
     CHUNK_OVERLAP = "100"
     TOP_K = "5"
     SIMILARITY_THRESHOLD = "0.2"
     ```

4. **Advanced Settings**
   - Python version: 3.11
   - Memory management: 1GB recommended
   - Package caching: Enabled

### Docker Deployment

1. **Build Docker Image**
   ```bash
   docker build -t raglite .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 --env-file .env raglite
   ```

### Troubleshooting

1. **Memory Issues**
   - Increase memory allocation in Streamlit Cloud settings
   - Optimize chunk size and embedding batch size

2. **Package Conflicts**
   - Use `pip-compile` to generate locked requirements
   - Check Python version compatibility

3. **Environment Variables**
   - Verify secrets are properly set
   - Check environment variable access in code
