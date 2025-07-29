"""Sentence Transformer Embedding Model"""

import os
import logging
from typing import List, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Manages sentence transformer model for text embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model_name = model_name
            
            # Get model path - check if it exists locally first
            model_path = os.path.join("models", model_name.replace("/", "--"))
            if os.path.exists(model_path):
                logger.info(f"Loading model from local path: {model_path}")
                self.model = SentenceTransformer(model_path)
            else:
                logger.info(f"Downloading model: {model_name}")
                self.model = SentenceTransformer(model_name)
                # Save model locally for future use
                os.makedirs("models", exist_ok=True)
                self.model.save(model_path)
            
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        try:
            # Get basic model info
            info = {
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dimension,
                "max_seq_length": getattr(self.model, 'max_seq_length', 512),  # Default to 512 if not available
                "device": str(self.model.device)
            }
            
            # Try to get additional model configuration if available
            try:
                if hasattr(self.model, 'config'):
                    config = self.model.config
                    info.update({
                        "model_type": getattr(config, 'model_type', 'unknown'),
                        "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                        "hidden_size": getattr(config, 'hidden_size', 'unknown')
                    })
            except Exception:
                pass  # Ignore if additional config info is not available
                
            return info
            
        except Exception as e:
            logger.warning(f"Error getting complete model info: {e}")
            # Return basic info if detailed info retrieval fails
            return {
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dimension
            } 