"""
Document retriever for RAGLite.

Combines embedding generation and vector search for document retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from ..embeddings import EmbeddingModel
from ..vectorstore import ChromaManager

# Set up logging
logger = logging.getLogger(__name__)


class Retriever:
    """
    Document retriever that combines embedding generation with vector search.
    Provides a unified interface for document storage and retrieval.
    """
    
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 chroma_manager: Optional[ChromaManager] = None,
                 default_top_k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: EmbeddingModel instance for generating embeddings
            chroma_manager: ChromaManager instance for vector storage
            default_top_k: Default number of documents to retrieve
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chroma_manager = chroma_manager or ChromaManager()
        self.default_top_k = default_top_k
        
        logger.info("Retriever initialized successfully")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store with automatic embedding generation.
        
        Args:
            documents: List of document dictionaries with 'text' and optional 'metadata'
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            start_time = time.time()
            
            # Generate embeddings for documents that don't have them
            docs_to_embed = []
            enriched_docs = []
            
            for doc in documents:
                if 'embedding' not in doc:
                    docs_to_embed.append(doc)
                else:
                    enriched_docs.append(doc)
            
            # Generate embeddings for documents that need them
            if docs_to_embed:
                embedded_docs = self.embedding_model.embed_documents(docs_to_embed)
                enriched_docs.extend(embedded_docs)
            
            # Add all documents to ChromaDB
            self.chroma_manager.add_documents(enriched_docs)
            
            add_time = time.time() - start_time
            logger.info(f"Successfully added {len(documents)} documents in {add_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def retrieve_similar(self, 
                         query: str, 
                         top_k: Optional[int] = None,
                         similarity_threshold: float = 0.3,
                         metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents based on a text query.
        
        Args:
            query: Text query to search for
            top_k: Number of documents to retrieve (uses default if None)
            similarity_threshold: Minimum similarity score to include
            metadata_filter: Optional metadata filter for ChromaDB
            
        Returns:
            List of similar documents with similarity scores
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        top_k = top_k or self.default_top_k
        
        try:
            logger.info(f"Retrieving similar documents for query: '{query[:50]}...'")
            start_time = time.time()
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed_text(query)
            
            # Search for similar documents
            similar_docs = self.chroma_manager.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                where=metadata_filter
            )
            
            # Filter by similarity threshold
            filtered_docs = [
                doc for doc in similar_docs 
                if doc.get('similarity', 0) >= similarity_threshold
            ]
            
            retrieval_time = time.time() - start_time
            
            # Log filtering results
            num_filtered_out = len(similar_docs) - len(filtered_docs)
            if num_filtered_out > 0:
                logger.info(f"Filtered out {num_filtered_out} low-relevance documents (< {similarity_threshold:.2f} similarity)")
            
            logger.info(f"Retrieved {len(filtered_docs)} high-quality documents in {retrieval_time:.2f} seconds")
            
            # Add retrieval metadata to each document
            for doc in filtered_docs:
                doc['retrieval_metadata'] = {
                    'similarity_threshold': similarity_threshold,
                    'retrieval_time': retrieval_time,
                    'total_candidates': len(similar_docs),
                    'filtered_count': num_filtered_out
                }
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {e}")
            raise
    
    def retrieve_with_context(self, 
                             query: str, 
                             top_k: Optional[int] = None,
                             include_metadata: bool = True,
                             context_window: int = 0) -> Dict[str, Any]:
        """
        Retrieve similar documents with additional context information.
        
        Args:
            query: Text query to search for
            top_k: Number of documents to retrieve
            include_metadata: Whether to include document metadata
            context_window: Number of additional chunks to include around each result
            
        Returns:
            Dictionary with retrieved documents and context information
        """
        try:
            # Retrieve similar documents
            similar_docs = self.retrieve_similar(query, top_k)
            
            result = {
                'query': query,
                'num_results': len(similar_docs),
                'documents': similar_docs,
                'retrieval_metadata': {
                    'embedding_model': self.embedding_model.model_name,
                    'vector_store': 'ChromaDB',
                    'similarity_metric': 'cosine',
                    'top_k': top_k or self.default_top_k
                }
            }
            
            # Add context information if requested
            if context_window > 0:
                result['context_window'] = context_window
                # Note: Context window implementation would require additional metadata
                # about document chunk relationships
            
            # Remove metadata if not requested
            if not include_metadata:
                for doc in result['documents']:
                    doc.pop('metadata', None)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving documents with context: {e}")
            raise
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever and vector store.
        
        Returns:
            Dictionary with retriever statistics
        """
        try:
            collection_info = self.chroma_manager.get_collection_info()
            embedding_info = self.embedding_model.get_model_info()
            
            stats = {
                'vector_store': collection_info,
                'embedding_model': embedding_info,
                'default_top_k': self.default_top_k
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting retriever stats: {e}")
            return {'error': str(e)}
    
    def reset_vector_store(self) -> None:
        """Reset the vector store (delete all documents)."""
        try:
            self.chroma_manager.reset_collection()
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            raise
    
    def delete_document(self, filename: str) -> Dict[str, Any]:
        """
        Delete a specific document and all its chunks from the vector store.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            result = self.chroma_manager.delete_documents_by_filename(filename)
            logger.info(f"Document deletion result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return {
                'filename': filename,
                'chunks_deleted': 0,
                'success': False,
                'error': str(e)
            }
    
    def get_document_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics about all documents in the collection.
        
        Returns:
            Dictionary with document analytics
        """
        try:
            stats = self.chroma_manager.get_document_statistics()
            return stats
        except Exception as e:
            logger.error(f"Error getting document analytics: {e}")
            return {
                'files': {},
                'total_files': 0,
                'total_chunks': 0,
                'error': str(e)
            }
    
    def list_documents(self) -> List[str]:
        """
        Get a list of all document filenames in the collection.
        
        Returns:
            List of filenames
        """
        try:
            return self.chroma_manager.list_unique_filenames()
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_document_details(self, filename: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific document.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary with file details
        """
        try:
            return self.chroma_manager.get_file_details(filename)
        except Exception as e:
            logger.error(f"Error getting document details for {filename}: {e}")
            return {
                'filename': filename,
                'exists': False,
                'error': str(e)
            }
    
    def search_by_metadata(self, 
                          metadata_filter: Dict[str, Any], 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filter only.
        
        Args:
            metadata_filter: Metadata filter conditions
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        try:
            return self.chroma_manager.get_documents_by_metadata(metadata_filter, limit)
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            raise
    
    def get_document_by_source(self, source: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get documents from a specific source file.
        
        Args:
            source: Source filename or path
            top_k: Maximum number of documents to return
            
        Returns:
            List of documents from the specified source
        """
        try:
            metadata_filter = {"source": source}
            return self.search_by_metadata(metadata_filter, limit=top_k)
        except Exception as e:
            logger.error(f"Error getting documents by source: {e}")
            raise
    
    def hybrid_search(self, 
                     query: str, 
                     metadata_filter: Optional[Dict[str, Any]] = None,
                     top_k: Optional[int] = None,
                     rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic similarity and metadata filtering.
        
        Args:
            query: Text query for semantic search
            metadata_filter: Metadata filter to apply
            top_k: Number of documents to retrieve
            rerank: Whether to rerank results by combining scores
            
        Returns:
            List of documents with combined relevance scores
        """
        try:
            # Perform semantic search
            semantic_results = self.retrieve_similar(
                query=query, 
                top_k=top_k, 
                metadata_filter=metadata_filter
            )
            
            if not rerank:
                return semantic_results
            
            # For now, return semantic results as-is
            # In a more advanced implementation, we could combine with BM25 or other scoring methods
            logger.info(f"Hybrid search returned {len(semantic_results)} documents")
            return semantic_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise 