"""ChromaDB Vector Store Manager"""

import sys
import sqlite3
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    logger.info(f"ChromaDB version: {chromadb.__version__}")
except ImportError as e:
    logger.error(f"Failed to import ChromaDB: {e}")
    chromadb = None

class ChromaManager:
    """Manages interactions with ChromaDB vector store."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """Initialize ChromaDB with the specified persistence directory."""
        if not chromadb:
            raise ImportError("ChromaDB is not installed. Please install it with 'pip install chromadb'")
            
        try:
            # Ensure persist directory exists
            persist_directory = os.path.abspath(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            logger.info(f"Using ChromaDB persist directory: {persist_directory}")
            
            # Initialize ChromaDB client with settings
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Get or create the default collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
        
        # Prepare document batches
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(doc.get('text', ''))
            
            # Ensure metadata is a dict and add timestamp
            metadata = doc.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata['timestamp'] = datetime.now().isoformat()
            metadatas.append(metadata)
            
            # Get embeddings if provided
            if 'embedding' in doc:
                embeddings.append(doc['embedding'])
        
        # Add documents to collection
        if embeddings:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        documents = []
        for idx, doc in enumerate(results['documents'][0]):
            documents.append({
                'text': doc,
                'metadata': results['metadatas'][0][idx] if results['metadatas'] else {},
                'distance': results['distances'][0][idx] if results['distances'] else None,
                'id': results['ids'][0][idx]
            })
        
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': self.collection.name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            return {'error': str(e)}

    def reset_collection(self) -> None:
        """Reset the collection."""
        try:
            self.client.reset()
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Failed to reset collection: {e}")

    def delete_document(self, document_id: str) -> None:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[document_id])
        except Exception as e:
            raise Exception(f"Failed to delete document: {e}")

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        try:
            results = self.collection.get(ids=[document_id])
            if results['documents']:
                return {
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': document_id
                }
            return None
        except Exception as e:
            raise Exception(f"Failed to get document: {e}") 