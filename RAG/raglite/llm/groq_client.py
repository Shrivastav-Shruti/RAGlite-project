"""
Groq LLM client for RAGLite.

Interfaces with Groq's API using llama-3.3-70b-versatile model (stable and reliable).
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

# Set up logging
logger = logging.getLogger(__name__)


class GroqClient:
    """
    Client for Groq's LLM API using llama-3.3-70b-versatile.
    Provides text generation capabilities for RAG applications.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = None,
                 temperature: float = 0.2,
                 max_tokens: int = 1024,
                 timeout: int = 30):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (will use GROQ_API_KEY env var if not provided)
            model: Model name to use
            temperature: Temperature for text generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter."
            )
        
        # Get model from environment variable or use default
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Groq client initialized with model: {self.model}")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response using the Groq API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            start_time = time.time()
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": False
            }
            
            # Make API request
            logger.info(f"Sending request to Groq API with {len(messages)} messages")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            generation_time = time.time() - start_time
            
            # Extract response content
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0]["message"]["content"]
                
                result = {
                    "response": content,
                    "model": self.model,
                    "generation_time": generation_time,
                    "usage": response_data.get("usage", {}),
                    "finish_reason": response_data["choices"][0].get("finish_reason"),
                    "raw_response": response_data
                }
                
                logger.info(f"Generated response in {generation_time:.2f} seconds")
                return result
            else:
                raise ValueError("Invalid response format from Groq API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def answer_question(self, 
                       question: str, 
                       context: str,
                       system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer a question based on provided context.
        
        Args:
            question: User's question
            context: Relevant context information
            system_prompt: Optional system prompt (uses default if not provided)
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        if not context.strip():
            logger.warning("Empty context provided")
        
        # Default conversational system prompt for RAG
        default_system_prompt = """You are a helpful and conversational AI assistant that answers questions based on the provided context. 

Guidelines:
1. Use only the information given in the context to answer the question
2. If the context doesn't contain enough information to answer the question, say so clearly and politely
3. Be conversational and natural in your tone - write as if you're having a friendly conversation
4. Use phrases like "Based on the information I found..." or "From what I can see..."
5. Reference specific page numbers when available (e.g., "According to Page 5..." or "As mentioned on Page 12...")
6. Be accurate, helpful, and engaging in your responses
7. If asked about something not in the context, acknowledge the limitation politely"""
        
        system_prompt = system_prompt or default_system_prompt
        
        # Construct user message with context and question
        user_message = f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context in a conversational manner."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            result = self.generate_response(messages)
            result["question"] = question
            result["context_length"] = len(context)
            return result
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    def generate_rag_response(self, 
                             query: str, 
                             retrieved_documents: List[Dict[str, Any]],
                             system_prompt: Optional[str] = None,
                             include_sources: bool = True) -> Dict[str, Any]:
        """
        Generate a RAG response using retrieved documents.
        
        Args:
            query: User's query
            retrieved_documents: List of retrieved document chunks
            system_prompt: Optional system prompt
            include_sources: Whether to include source information in the response
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for greetings or casual conversation
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        how_are_you_patterns = ['how are you', 'how do you do', 'how\'s it going', 'what\'s up']
        
        query_lower = query.lower().strip()
        
        # Handle greetings
        if any(query_lower.startswith(pattern) for pattern in greeting_patterns):
            friendly_greeting = {
                "response": "Hello! 👋 I'm your friendly RAGLite Assistant. I'm here to help you explore and understand your documents. What would you like to know about?",
                "query": query,
                "num_documents": 0,
                "sources": [],
                "document_similarities": [],
                "model": self.model,
                "generation_time": 0
            }
            return friendly_greeting
        
        # Handle "how are you" type questions
        if any(pattern in query_lower for pattern in how_are_you_patterns):
            friendly_response = {
                "response": "I'm doing great, thanks for asking! 😊 I'm excited to help you explore your documents. What would you like to learn about?",
                "query": query,
                "num_documents": 0,
                "sources": [],
                "document_similarities": [],
                "model": self.model,
                "generation_time": 0
            }
            return friendly_response
        
        # Handle case when no relevant documents are found
        if not retrieved_documents:
            logger.warning("No relevant documents found for query")
            friendly_response = {
                "response": "I wish I could help with that! 🤔 While I don't have any specific information about that in the documents, I'd love to help you find what you're looking for. Maybe we could:\n\n" +
                          "• Try rephrasing the question in a different way\n" +
                          "• Look for related topics in the available documents\n" +
                          "• Upload some new documents about this topic\n\n" +
                          "What would you like to try? I'm here to help! 😊",
                "query": query,
                "num_documents": 0,
                "sources": [],
                "document_similarities": [],
                "model": self.model,
                "generation_time": 0
            }
            return friendly_response
        
        # Process documents and build context
        context = ""
        sources = set()
        page_references = []
        
        for doc in retrieved_documents:
            # Extract text and metadata
            text = doc.get('text', '').strip()
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            page_numbers = metadata.get('page_numbers', [])
            
            if text:
                # Add document text to context
                context += f"\nDocument: {filename}\n"
                if page_numbers:
                    context += f"Pages: {', '.join(map(str, page_numbers))}\n"
                context += f"{text}\n"
                
                # Track sources and page references
                sources.add(filename)
                if page_numbers:
                    page_references.extend([
                        {'file': filename, 'page': page} 
                        for page in page_numbers
                    ])
        
        # Enhanced conversational system prompt for RAG
        rag_system_prompt = """You are a friendly and helpful AI assistant that provides accurate answers based on retrieved documents. Your responses should be:

1. Conversational and engaging - like chatting with a knowledgeable friend
2. Clear and easy to understand
3. Backed by information from the documents
4. Honest about limitations - if you're not sure, say so in a friendly way

When responding:
• Start with a brief acknowledgment or transition phrase
• Reference specific pages when available (e.g., "I found in page 5...")
• Use natural language and a warm tone
• Include relevant emojis occasionally to make the conversation friendly
• If information is limited, suggest helpful alternatives

Remember to maintain a consistent, helpful personality throughout the conversation."""
        
        system_prompt = system_prompt or rag_system_prompt
        
        try:
            result = self.answer_question(query, context, system_prompt)
            
            # Add RAG-specific metadata
            result.update({
                "query": query,
                "num_documents": len(retrieved_documents),
                "sources": list(sources) if include_sources else None,
                "document_similarities": [doc.get('similarity', 0) for doc in retrieved_documents],
                "page_references": page_references
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            friendly_error_response = {
                "response": "Oops! 😅 Looks like I hit a small bump while processing that. Let's try again! You could:\n\n" +
                          "• Ask the question again\n" +
                          "• Rephrase it slightly\n" +
                          "• Try a different question\n\n" +
                          "I'm all ears and ready to help! 🤗",
                "query": query,
                "num_documents": len(retrieved_documents),
                "sources": [],
                "document_similarities": [],
                "model": self.model,
                "generation_time": 0,
                "error": str(e)
            }
            return friendly_error_response
    
    def _extract_page_number(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Extract page number from document text or metadata.
        
        Args:
            text: Document text that may contain page markers
            metadata: Document metadata that may contain page numbers
            
        Returns:
            Page number if found, None otherwise
        """
        import re
        
        # First check if page numbers are available in metadata
        if metadata and 'page_numbers' in metadata:
            page_numbers = metadata['page_numbers']
            if page_numbers and len(page_numbers) > 0:
                return page_numbers[0]  # Return the first page number
        
        # Fall back to extracting from text
        page_patterns = [
            r'--- Page (\d+) ---',
            r'Page (\d+)',
            r'page (\d+)',
            r'PAGE (\d+)'
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Have a conversation using multiple messages.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        conversation = []
        
        # Add system prompt if provided
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        
        # Add conversation messages
        conversation.extend(messages)
        
        try:
            return self.generate_response(conversation)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "base_url": self.base_url
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hello, can you respond with just 'OK'?"}
            ]
            
            result = self.generate_response(test_messages, max_tokens=10)
            logger.info("Groq API connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Groq API connection test failed: {e}")
            return False 