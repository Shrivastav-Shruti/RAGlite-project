"""ChromaDB Vector Store Manager"""

import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pysqlite3 and replace sqlite3 module for ChromaDB compatibility
try:
    import pysqlite3
    # Replace sqlite3 with pysqlite3 in sys.modules
    sys.modules['sqlite3'] = pysqlite3
    logger.info("Using pysqlite3 for better SQLite compatibility")
except ImportError:
    logger.warning("pysqlite3 not available, using default sqlite3 - this may cause issues")
    import sqlite3

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.types import Collection
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
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            return {
                'name': self.collection.name,
                'document_count': count,
                'metadata': self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}
    
    def list_unique_filenames(self) -> List[str]:
        """Get a list of unique filenames in the collection."""
        try:
            # Get all documents' metadata
            results = self.collection.get()
            if not results or 'metadatas' not in results:
                return []
            
            # Extract unique filenames from metadata
            filenames = set()
            for metadata in results['metadatas']:
                if metadata and 'filename' in metadata:
                    filenames.add(metadata['filename'])
            
            return sorted(list(filenames))
        except Exception as e:
            logger.error(f"Error listing filenames: {e}")
            return []
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about documents in the collection."""
        try:
            results = self.collection.get()
            if not results:
                return {
                    'files': {},
                    'total_files': 0,
                    'total_chunks': 0
                }
            
            # Initialize statistics
            stats = {
                'files': {},
                'total_files': 0,
                'total_chunks': 0
            }
            
            # Process each document
            for i, metadata in enumerate(results['metadatas']):
                if not metadata or 'filename' not in metadata:
                    continue
                
                filename = metadata['filename']
                if filename not in stats['files']:
                    stats['files'][filename] = {
                        'chunk_count': 0,
                        'total_chars': 0,
                        'file_type': metadata.get('file_type', 'Unknown'),
                        'upload_time': metadata.get('upload_time', 'Unknown')
                    }
                
                # Update statistics
                stats['files'][filename]['chunk_count'] += 1
                if 'text' in results and len(results['documents']) > i:
                    stats['files'][filename]['total_chars'] += len(results['documents'][i])
            
            # Calculate totals
            stats['total_files'] = len(stats['files'])
            stats['total_chunks'] = sum(f['chunk_count'] for f in stats['files'].values())
            
            # Calculate averages
            for file_stats in stats['files'].values():
                if file_stats['chunk_count'] > 0:
                    file_stats['avg_chunk_size'] = file_stats['total_chars'] // file_stats['chunk_count']
            
            return stats
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {
                'files': {},
                'total_files': 0,
                'total_chunks': 0,
                'error': str(e)
            }
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the collection."""
        try:
            # Prepare documents for ChromaDB format
            ids = []
            texts = []
            embeddings = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', str(i))
                ids.append(doc_id)
                texts.append(doc['text'])
                if 'embedding' in doc:
                    embeddings.append(doc['embedding'])
                metadata = doc.get('metadata', {})
                metadata['upload_time'] = datetime.now().isoformat()
                metadatas.append(metadata)
            
            # Add to collection
            if embeddings:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise RuntimeError(f"Failed to add documents: {e}")
    
    def search_similar(self, 
                      query_embedding: List[float],
                      top_k: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using embedding."""
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            # Format results
            documents = []
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': float(results['distances'][0][i]) if 'distances' in results else 0.0
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting all documents."""
        try:
            self.collection.delete()
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise RuntimeError(f"Reset failed: {e}")
    
    def delete_documents_by_filename(self, filename: str) -> Dict[str, Any]:
        """Delete all documents with the specified filename."""
        try:
            # Get documents with matching filename
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if not results or not results['ids']:
                return {
                    'filename': filename,
                    'chunks_deleted': 0,
                    'success': True,
                    'message': 'No documents found'
                }
            
            # Delete documents
            self.collection.delete(
                ids=results['ids']
            )
            
            return {
                'filename': filename,
                'chunks_deleted': len(results['ids']),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error deleting documents for {filename}: {e}")
            return {
                'filename': filename,
                'chunks_deleted': 0,
                'success': False,
                'error': str(e)
            }
    
    def get_file_details(self, filename: str) -> Dict[str, Any]:
        """Get detailed information about a specific file."""
        try:
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if not results or not results['ids']:
                return {
                    'filename': filename,
                    'exists': False,
                    'message': 'File not found'
                }
            
            # Calculate statistics
            total_chars = sum(len(doc) for doc in results['documents'])
            avg_chunk_size = total_chars // len(results['documents']) if results['documents'] else 0
            
            return {
                'filename': filename,
                'exists': True,
                'chunk_count': len(results['ids']),
                'total_chars': total_chars,
                'avg_chunk_size': avg_chunk_size,
                'metadata': results['metadatas'][0] if results['metadatas'] else {},
                'upload_time': results['metadatas'][0].get('upload_time', 'Unknown') if results['metadatas'] else 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting file details for {filename}: {e}")
            return {
                'filename': filename,
                'exists': False,
                'error': str(e)
            } 