#!/bin/bash

# RAGLite Setup Script
# This script sets up the complete RAGLite environment

echo "🚀 RAGLite Setup Script"
echo "======================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is installed
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        required_version="3.11"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
            print_success "Python $python_version found"
            return 0
        else
            print_error "Python 3.11+ required, found $python_version"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.11+"
        return 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_status "Using existing virtual environment"
            return 0
        fi
    fi
    
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        return 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    if [ $? -eq 0 ]; then
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        return 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "raglite/requirements.txt" ]; then
        pip install -r raglite/requirements.txt
        if [ $? -eq 0 ]; then
            print_success "Dependencies installed successfully"
        else
            print_error "Failed to install dependencies"
            return 1
        fi
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/chroma_db
    mkdir -p data/uploads
    mkdir -p benchmarks
    mkdir -p logs
    
    print_success "Directories created"
}

# Set up environment variables
setup_env() {
    print_status "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# RAGLite Environment Configuration

# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# ChromaDB Configuration
CHROMA_DB_PERSIST_DIRECTORY=./data/chroma_db

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=./logs

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=meta-llama/llama-4-maverick-17b-128k-instruct
TEMPERATURE=0.2
MAX_TOKENS=1024

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EOF
        print_success "Environment file created (.env)"
        print_warning "Please edit .env file and add your Groq API key"
    else
        print_warning ".env file already exists"
    fi
}

# Download models (optional)
download_models() {
    print_status "Pre-downloading embedding model (this may take a while)..."
    
    python3 -c "
import warnings
warnings.filterwarnings('ignore')
try:
    from sentence_transformers import SentenceTransformer
    print('Downloading sentence-transformers/all-MiniLM-L6-v2...')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✅ Model downloaded successfully')
except Exception as e:
    print(f'❌ Error downloading model: {e}')
"
}

# Create launch script
create_launch_script() {
    print_status "Creating launch script..."
    
    cat > launch_raglite.sh << 'EOF'
#!/bin/bash

# RAGLite Launch Script

echo "🤖 Starting RAGLite..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Set environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
fi

# Change to the correct directory
cd "$(dirname "$0")"

# Check if Groq API key is set
if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
    echo "⚠️  Warning: GROQ_API_KEY not set in .env file"
    echo "Please add your Groq API key to use the LLM features"
fi

# Launch Streamlit app
echo "🚀 Launching RAGLite web interface..."
echo "Opening browser at: http://localhost:8501"

streamlit run raglite/api/app.py --server.port 8501 --server.address localhost

EOF

    chmod +x launch_raglite.sh
    print_success "Launch script created (launch_raglite.sh)"
}

# Create test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > test_raglite.py << 'EOF'
#!/usr/bin/env python3
"""
RAGLite Test Script

Quick test to verify the installation is working correctly.
"""

import sys
import os
sys.path.append('raglite')

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from raglite import DocumentLoader, EmbeddingModel, ChromaManager, Retriever
        print("✅ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedding_model():
    """Test embedding model loading."""
    print("🧪 Testing embedding model...")
    
    try:
        from raglite import EmbeddingModel
        model = EmbeddingModel()
        
        # Test with sample text
        embedding = model.embed_text("This is a test sentence.")
        print(f"✅ Embedding model working (dimension: {len(embedding)})")
        return True
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False

def test_document_loader():
    """Test document loader."""
    print("🧪 Testing document loader...")
    
    try:
        from raglite import DocumentLoader
        loader = DocumentLoader()
        
        # Create a test file
        test_content = "This is a test document for RAGLite testing."
        with open("test_doc.txt", "w") as f:
            f.write(test_content)
        
        # Load the test file
        docs = loader.load_document("test_doc.txt")
        
        # Clean up
        os.remove("test_doc.txt")
        
        print(f"✅ Document loader working ({len(docs)} chunks created)")
        return True
    except Exception as e:
        print(f"❌ Document loader error: {e}")
        return False

def test_chroma_manager():
    """Test ChromaDB manager."""
    print("🧪 Testing ChromaDB...")
    
    try:
        from raglite import ChromaManager
        chroma = ChromaManager(persist_directory="./test_chroma")
        
        info = chroma.get_collection_info()
        print(f"✅ ChromaDB working (collection: {info['collection_name']})")
        
        # Clean up test directory
        import shutil
        if os.path.exists("test_chroma"):
            shutil.rmtree("test_chroma")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAGLite Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_embedding_model,
        test_document_loader,
        test_chroma_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! RAGLite is ready to use.")
        print("Run './launch_raglite.sh' to start the web interface.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

    chmod +x test_raglite.py
    print_success "Test script created (test_raglite.py)"
}

# Main setup function
main() {
    echo
    print_status "Starting RAGLite setup..."
    echo
    
    # Check Python version
    if ! check_python; then
        print_error "Setup failed: Python 3.11+ required"
        exit 1
    fi
    
    # Create virtual environment
    if ! create_venv; then
        print_error "Setup failed: Could not create virtual environment"
        exit 1
    fi
    
    # Activate virtual environment
    if ! activate_venv; then
        print_error "Setup failed: Could not activate virtual environment"
        exit 1
    fi
    
    # Install dependencies
    if ! install_dependencies; then
        print_error "Setup failed: Could not install dependencies"
        exit 1
    fi
    
    # Create directories
    create_directories
    
    # Set up environment
    setup_env
    
    # Download models
    read -p "Do you want to pre-download the embedding model? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        download_models
    fi
    
    # Create scripts
    create_launch_script
    create_test_script
    
    echo
    print_success "RAGLite setup completed successfully! 🎉"
    echo
    echo "Next steps:"
    echo "1. Edit .env file and add your Groq API key"
    echo "2. Run './test_raglite.py' to test the installation"
    echo "3. Run './launch_raglite.sh' to start the web interface"
    echo
    print_warning "Don't forget to activate the virtual environment: source venv/bin/activate"
}

# Run main function
main "$@" 