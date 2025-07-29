"""
Embedding model module for RAGLite.

Uses sentence-transformers/all-MiniLM-L6-v2 for CPU-optimized embeddings.
"""

import logging
import numpy as np
from typing import List, Union, Optional
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Set up logging
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    CPU-optimized embedding model using sentence-transformers.
    Uses all-MiniLM-L6-v2 for efficient and accurate embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name)
            
            # Set to CPU mode for consistency
            self.model = self.model.to('cpu')
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array containing embeddings for all texts
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        if not valid_texts:
            raise ValueError("No valid texts found after filtering")
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            start_time = time.time()
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f} seconds")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        Generate embeddings for a list of document dictionaries.
        
        Args:
            documents: List of document dictionaries with 'text' field
            
        Returns:
            List of document dictionaries with added 'embedding' field
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Extract texts from documents
        texts = []
        for i, doc in enumerate(documents):
            if 'text' not in doc:
                raise ValueError(f"Document {i} is missing 'text' field")
            texts.append(doc['text'])
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to documents
        enriched_documents = []
        for doc, embedding in zip(documents, embeddings):
            enriched_doc = doc.copy()
            enriched_doc['embedding'] = embedding
            enriched_documents.append(enriched_doc)
        
        return enriched_documents
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer representing the embedding dimension
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Generate a test embedding to get dimensions
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        return test_embedding.shape[0]
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top similar embeddings to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity
        """
        if len(candidate_embeddings) == 0:
            return []
        
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'device': str(self.model.device) if self.model else 'Not loaded'
        } 