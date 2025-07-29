"""
ChromaDB manager for RAGLite.

Handles vector storage and similarity search using ChromaDB.
"""

import sys
import sqlite3
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api import API as ChromaAPI
    from chromadb.api.models.Collection import Collection
except ImportError:
    chromadb = None
    ChromaAPI = None
    Collection = None

# Set up logging
logger = logging.getLogger(__name__)


class ChromaManager:
    """ChromaDB vector store manager."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """Initialize ChromaDB manager."""
        if not chromadb:
            raise ImportError("ChromaDB is not installed. Please install it with 'pip install chromadb'")
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with settings
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

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