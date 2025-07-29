"""
ChromaDB manager for RAGLite.

Handles vector storage and similarity search using ChromaDB.
"""

import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

# Set up logging
logger = logging.getLogger(__name__)


class ChromaManager:
    """
    ChromaDB manager for storing and retrieving document embeddings.
    Uses cosine similarity for document retrieval.
    """
    
    def __init__(self, 
                 persist_directory: str = "./data/chroma_db",
                 collection_name: str = "documents"):
        """
        Initialize ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to store documents
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Ensure persist directory exists with proper permissions
        os.makedirs(persist_directory, exist_ok=True)
        
        # Ensure the directory is writable
        if not os.access(persist_directory, os.W_OK):
            raise PermissionError(f"Directory {persist_directory} is not writable")
        
        # Set directory permissions to ensure write access
        try:
            os.chmod(persist_directory, 0o755)
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing ChromaDB client with persist directory: {self.persist_directory} (attempt {attempt + 1})")
                
                # Create client with persistence and proper permissions
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                
                # Get or create collection with cosine similarity
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                
                logger.info(f"ChromaDB collection '{self.collection_name}' initialized successfully")
                return
                
            except Exception as e:
                logger.error(f"Error initializing ChromaDB (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # On final attempt, try to reset the database
                    try:
                        logger.info("Attempting to reset ChromaDB database...")
                        import shutil
                        if os.path.exists(self.persist_directory):
                            shutil.rmtree(self.persist_directory)
                        os.makedirs(self.persist_directory, exist_ok=True)
                        
                        # Try one more time after reset
                        self.client = chromadb.PersistentClient(
                            path=self.persist_directory,
                            settings=Settings(
                                anonymized_telemetry=False,
                                allow_reset=True,
                                is_persistent=True
                            )
                        )
                        
                        self.collection = self.client.get_or_create_collection(
                            name=self.collection_name,
                            metadata={"hnsw:space": "cosine"}
                        )
                        
                        logger.info("ChromaDB database reset and initialized successfully")
                        return
                        
                    except Exception as reset_error:
                        logger.error(f"Failed to reset ChromaDB: {reset_error}")
                        raise
                else:
                    # Wait before retry
                    import time
                    time.sleep(1)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents with embeddings to the collection.
        
        Args:
            documents: List of document dictionaries with 'text', 'embedding', and 'metadata'
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            if 'embedding' not in doc:
                raise ValueError("Document missing 'embedding' field")
            if 'text' not in doc:
                raise ValueError("Document missing 'text' field")
            
            # Generate unique ID if not provided
            doc_id = doc.get('id', str(uuid.uuid4()))
            ids.append(doc_id)
            
            # Convert embedding to list if it's numpy array
            embedding = doc['embedding']
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            embeddings.append(embedding)
            
            # Add text content
            documents_text.append(doc['text'])
            
            # Prepare metadata
            metadata = doc.get('metadata', {}).copy()
            # Ensure all metadata values are JSON serializable
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            metadatas.append(metadata)
        
        # Add to collection with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_text,
                    metadatas=metadatas
                )
                
                logger.info(f"Successfully added {len(documents)} documents to ChromaDB collection")
                return
                
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # On final attempt, try to reinitialize the collection
                    try:
                        logger.info("Attempting to reinitialize ChromaDB collection...")
                        self._initialize_client()
                        
                        # Try adding again
                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            documents=documents_text,
                            metadatas=metadatas
                        )
                        
                        logger.info(f"Successfully added {len(documents)} documents to ChromaDB after reinitialization")
                        return
                        
                    except Exception as reinit_error:
                        logger.error(f"Failed to reinitialize and add documents: {reinit_error}")
                        raise
                else:
                    # Wait before retry
                    import time
                    time.sleep(1)
    
    def search_similar(self, 
                      query_embedding: np.ndarray, 
                      top_k: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top similar documents to return
            where: Optional metadata filter
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    similar_docs.append(doc)
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise
    
    def search_by_text(self, 
                      query_text: str, 
                      top_k: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text query.
        Note: This requires an embedding model to be passed or integrated.
        
        Args:
            query_text: Query text
            top_k: Number of top similar documents to return
            where: Optional metadata filter
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        # This method would require integration with the embedding model
        # For now, we'll provide a placeholder that raises an error
        raise NotImplementedError(
            "Text search requires an embedding model. Use search_similar() with pre-computed embeddings."
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'persist_directory': self.persist_directory,
                'error': str(e)
            }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection (delete and recreate)."""
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass  # Collection might not exist
            
            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Reset collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def get_documents_by_metadata(self, where: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve documents by metadata filter.
        
        Args:
            where: Metadata filter conditions
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=where,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            documents = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    doc = {
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents by metadata filter")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents by metadata: {e}")
            raise
    
    def update_document(self, doc_id: str, 
                       text: Optional[str] = None,
                       embedding: Optional[np.ndarray] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            text: New text content (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)
        """
        try:
            update_data = {}
            
            if text is not None:
                update_data['documents'] = [text]
            
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                update_data['embeddings'] = [embedding]
            
            if metadata is not None:
                # Ensure metadata values are JSON serializable
                clean_metadata = {}
                for key, value in metadata.items():
                    if not isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = str(value)
                    else:
                        clean_metadata[key] = value
                update_data['metadatas'] = [clean_metadata]
            
            if update_data:
                self.collection.update(
                    ids=[doc_id],
                    **update_data
                )
                logger.info(f"Updated document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise 
    
    def delete_documents_by_filename(self, filename: str) -> Dict[str, Any]:
        """
        Delete all chunks from a specific document/file.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Check if collection exists and is writable
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # First, get all documents with this filename
            results = self.collection.get(
                where={"filename": filename},
                include=['metadatas']
            )
            
            chunk_count = len(results['ids']) if results['ids'] else 0
            
            if chunk_count > 0:
                try:
                    # Delete all chunks from this file
                    self.collection.delete(
                        where={"filename": filename}
                    )
                    logger.info(f"Deleted {chunk_count} chunks from file: {filename}")
                except Exception as delete_error:
                    logger.error(f"Error during deletion: {delete_error}")
                    # Try alternative deletion method
                    try:
                        # Delete by IDs if where clause fails
                        self.collection.delete(
                            ids=results['ids']
                        )
                        logger.info(f"Deleted {chunk_count} chunks using ID-based deletion")
                    except Exception as alt_error:
                        logger.error(f"Alternative deletion also failed: {alt_error}")
                        raise Exception(f"Failed to delete document: {delete_error}")
            else:
                logger.info(f"No chunks found for file: {filename}")
            
            return {
                'filename': filename,
                'chunks_deleted': chunk_count,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return {
                'filename': filename,
                'chunks_deleted': 0,
                'success': False,
                'error': str(e)
            }
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all documents in the collection.
        
        Returns:
            Dictionary with document statistics by filename
        """
        try:
            # Get all documents with metadata
            results = self.collection.get(
                include=['metadatas']
            )
            
            if not results['metadatas']:
                return {}
            
            # Group by filename and calculate statistics
            file_stats = {}
            total_chunks = 0
            
            for metadata in results['metadatas']:
                filename = metadata.get('filename', 'Unknown')
                file_type = metadata.get('file_type', 'Unknown')
                upload_time = metadata.get('upload_time', 'Unknown')
                chunk_size = metadata.get('chunk_size', 0)
                
                if filename not in file_stats:
                    file_stats[filename] = {
                        'filename': filename,
                        'file_type': file_type,
                        'upload_time': upload_time,
                        'chunk_count': 0,
                        'total_chars': 0,
                        'avg_chunk_size': 0,
                        'first_upload': upload_time
                    }
                
                file_stats[filename]['chunk_count'] += 1
                file_stats[filename]['total_chars'] += int(chunk_size) if chunk_size else 0
                total_chunks += 1
            
            # Calculate averages
            for filename, stats in file_stats.items():
                if stats['chunk_count'] > 0:
                    stats['avg_chunk_size'] = stats['total_chars'] // stats['chunk_count']
            
            return {
                'files': file_stats,
                'total_files': len(file_stats),
                'total_chunks': total_chunks,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {
                'files': {},
                'total_files': 0,
                'total_chunks': 0,
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def list_unique_filenames(self) -> List[str]:
        """
        Get a list of all unique filenames in the collection.
        
        Returns:
            List of unique filenames
        """
        try:
            results = self.collection.get(
                include=['metadatas']
            )
            
            if not results['metadatas']:
                return []
            
            filenames = set()
            for metadata in results['metadatas']:
                filename = metadata.get('filename', 'Unknown')
                if filename != 'Unknown':
                    filenames.add(filename)
            
            sorted_filenames = sorted(list(filenames))
            logger.info(f"Found {len(sorted_filenames)} unique files")
            return sorted_filenames
            
        except Exception as e:
            logger.error(f"Error listing filenames: {e}")
            return []
    
    def get_file_details(self, filename: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary with file details
        """
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=['metadatas', 'documents']
            )
            
            if not results['metadatas']:
                return {
                    'filename': filename,
                    'exists': False,
                    'error': 'File not found'
                }
            
            # Calculate file statistics
            chunk_count = len(results['metadatas'])
            total_chars = sum(len(doc) for doc in results['documents'])
            
            # Get file metadata from first chunk
            first_metadata = results['metadatas'][0]
            
            return {
                'filename': filename,
                'exists': True,
                'chunk_count': chunk_count,
                'total_characters': total_chars,
                'avg_chunk_size': total_chars // chunk_count if chunk_count > 0 else 0,
                'file_type': first_metadata.get('file_type', 'Unknown'),
                'upload_time': first_metadata.get('upload_time', 'Unknown'),
                'first_chunk_preview': results['documents'][0][:200] + '...' if results['documents'] else ''
            }
            
        except Exception as e:
            logger.error(f"Error getting file details for {filename}: {e}")
            return {
                'filename': filename,
                'exists': False,
                'error': str(e)
            } 