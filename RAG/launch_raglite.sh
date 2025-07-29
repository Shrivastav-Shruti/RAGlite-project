#!/bin/bash

# RAGLite Launch Script

echo "🤖 Starting RAGLite..."

# Activate virtual environment
if [ -d "env" ]; then
    source env/bin/activate
    echo "✅ Virtual environment activated"
elif [ -d "venv" ]; then
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

