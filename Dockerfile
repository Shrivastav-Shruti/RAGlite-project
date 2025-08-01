FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["streamlit", "run", "RAG/raglite/api/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 