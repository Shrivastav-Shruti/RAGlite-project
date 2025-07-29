"""
Prompt building utilities for RAGLite.

Provides templates and builders for creating effective RAG prompts.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builder class for creating optimized prompts for RAG applications.
    Includes various templates and customization options.
    """
    
    def __init__(self):
        """Initialize the prompt builder with default templates."""
        self.default_templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default prompt templates."""
        return {
            "basic_rag": """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:""",
            
            "detailed_rag": """You are an AI assistant that provides accurate answers based on the given context. Use only the information provided in the context to answer the question.

Context Information:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context
- If the context doesn't contain enough information, state this clearly
- Be concise but comprehensive
- Include relevant details from the context

Answer:""",
            
            "conversational_rag": """You are a helpful AI assistant. Use the provided context to answer the user's question in a natural, conversational way.

Relevant Information:
{context}

User Question: {question}

Please provide a helpful and informative response based on the context above:""",
            
            "analytical_rag": """Analyze the provided context and answer the question with supporting evidence.

Context Documents:
{context}

Question: {question}

Provide a well-reasoned answer that:
1. Directly addresses the question
2. Cites specific information from the context
3. Explains your reasoning
4. Acknowledges any limitations in the available information

Analysis and Answer:""",
            
            "comparative_rag": """Compare and synthesize information from multiple sources to answer the question.

Sources:
{context}

Question: {question}

Instructions:
- Compare information across sources
- Identify agreements and contradictions
- Synthesize a comprehensive answer
- Note the source of key claims

Comparative Answer:""",
            
            "system_prompt": """You are an expert AI assistant specializing in answering questions based on provided documents. 

Your capabilities:
- Accurate information extraction from context
- Clear and concise communication
- Appropriate handling of uncertain information
- Citation of sources when relevant

Guidelines:
- Only use information from the provided context
- If information is incomplete, acknowledge this limitation
- Maintain accuracy over completeness
- Provide helpful and actionable responses"""
        }
    
    def build_basic_prompt(self, question: str, context: str) -> str:
        """
        Build a basic RAG prompt.
        
        Args:
            question: User's question
            context: Retrieved context information
            
        Returns:
            Formatted prompt string
        """
        return self.default_templates["basic_rag"].format(
            question=question,
            context=context
        )
    
    def build_detailed_prompt(self, question: str, context: str) -> str:
        """
        Build a detailed RAG prompt with specific instructions.
        
        Args:
            question: User's question
            context: Retrieved context information
            
        Returns:
            Formatted prompt string
        """
        return self.default_templates["detailed_rag"].format(
            question=question,
            context=context
        )
    
    def build_conversational_prompt(self, question: str, context: str) -> str:
        """
        Build a conversational RAG prompt.
        
        Args:
            question: User's question
            context: Retrieved context information
            
        Returns:
            Formatted prompt string
        """
        return self.default_templates["conversational_rag"].format(
            question=question,
            context=context
        )
    
    def build_analytical_prompt(self, question: str, context: str) -> str:
        """
        Build an analytical RAG prompt that encourages reasoning.
        
        Args:
            question: User's question
            context: Retrieved context information
            
        Returns:
            Formatted prompt string
        """
        return self.default_templates["analytical_rag"].format(
            question=question,
            context=context
        )
    
    def build_document_context(self, 
                              documents: List[Dict[str, Any]], 
                              include_metadata: bool = True,
                              include_similarity: bool = True,
                              max_docs: Optional[int] = None) -> str:
        """
        Build formatted context from retrieved documents.
        
        Args:
            documents: List of retrieved document dictionaries
            include_metadata: Whether to include document metadata
            include_similarity: Whether to include similarity scores
            max_docs: Maximum number of documents to include
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        # Limit number of documents if specified
        if max_docs:
            documents = documents[:max_docs]
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            doc_text = doc.get('text', '').strip()
            if not doc_text:
                continue
            
            # Build document header
            header_parts = [f"Document {i}"]
            
            if include_metadata and 'metadata' in doc:
                metadata = doc['metadata']
                source = metadata.get('filename', metadata.get('source', 'Unknown'))
                header_parts.append(f"Source: {source}")
                
                if 'chunk_id' in metadata:
                    header_parts.append(f"Section: {metadata['chunk_id'] + 1}")
            
            if include_similarity and 'similarity' in doc:
                similarity = doc['similarity']
                header_parts.append(f"Relevance: {similarity:.3f}")
            
            header = f"[{' | '.join(header_parts)}]"
            
            # Add document with header
            context_parts.append(f"{header}\n{doc_text}")
        
        return "\n\n".join(context_parts)
    
    def build_messages_for_chat(self, 
                               question: str, 
                               context: str,
                               template: str = "detailed_rag",
                               system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build message list for chat-based APIs.
        
        Args:
            question: User's question
            context: Retrieved context information
            template: Template to use for the user message
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message
        system_content = system_prompt or self.default_templates["system_prompt"]
        messages.append({"role": "system", "content": system_content})
        
        # Add user message with context and question
        if template in self.default_templates:
            user_content = self.default_templates[template].format(
                question=question,
                context=context
            )
        else:
            # Fallback to basic template
            user_content = self.build_basic_prompt(question, context)
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def build_context_with_sources(self, 
                                  documents: List[Dict[str, Any]],
                                  citation_style: str = "numbered") -> str:
        """
        Build context with citation-friendly formatting.
        
        Args:
            documents: List of retrieved document dictionaries
            citation_style: Style for citations ("numbered", "bracketed", "inline")
            
        Returns:
            Formatted context string with citations
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            doc_text = doc.get('text', '').strip()
            if not doc_text:
                continue
            
            metadata = doc.get('metadata', {})
            source = metadata.get('filename', metadata.get('source', f'Document {i}'))
            
            if citation_style == "numbered":
                citation = f"[{i}]"
                header = f"{citation} {source}"
            elif citation_style == "bracketed":
                citation = f"[{source}]"
                header = citation
            else:  # inline
                citation = f"({source})"
                header = f"Source: {source}"
            
            context_parts.append(f"{header}\n{doc_text}")
        
        return "\n\n".join(context_parts)
    
    def build_few_shot_prompt(self, 
                             question: str, 
                             context: str,
                             examples: List[Dict[str, str]]) -> str:
        """
        Build a few-shot prompt with examples.
        
        Args:
            question: User's question
            context: Retrieved context information
            examples: List of example dictionaries with 'question', 'context', 'answer'
            
        Returns:
            Few-shot prompt string
        """
        prompt_parts = [
            "Here are some examples of how to answer questions based on context:",
            ""
        ]
        
        # Add examples
        for i, example in enumerate(examples, 1):
            example_text = f"""Example {i}:
Context: {example['context']}
Question: {example['question']}
Answer: {example['answer']}"""
            prompt_parts.append(example_text)
        
        prompt_parts.extend([
            "",
            "Now answer the following question using the same approach:",
            "",
            f"Context: {context}",
            f"Question: {question}",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)
    
    def build_chain_of_thought_prompt(self, question: str, context: str) -> str:
        """
        Build a chain-of-thought prompt that encourages step-by-step reasoning.
        
        Args:
            question: User's question
            context: Retrieved context information
            
        Returns:
            Chain-of-thought prompt string
        """
        template = """Please answer the question step by step using the provided context.

Context:
{context}

Question: {question}

Think through this step by step:
1. First, identify the key information in the context that relates to the question
2. Then, analyze how this information answers the question
3. Finally, provide a clear and complete answer

Step-by-step reasoning:"""
        
        return template.format(question=question, context=context)
    
    def optimize_context_length(self, 
                               context: str, 
                               max_tokens: int = 2000,
                               preserve_sentences: bool = True) -> str:
        """
        Optimize context length to fit within token limits.
        
        Args:
            context: Original context string
            max_tokens: Maximum number of tokens (approximate)
            preserve_sentences: Whether to preserve complete sentences
            
        Returns:
            Truncated context string
        """
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        if preserve_sentences:
            # Try to cut at sentence boundaries
            sentences = context.split('. ')
            truncated = ""
            
            for sentence in sentences:
                test_length = len(truncated + sentence + '. ')
                if test_length > max_chars:
                    break
                truncated += sentence + '. '
            
            return truncated.rstrip('. ')
        else:
            # Simple character truncation
            return context[:max_chars].rsplit(' ', 1)[0] + "..."
    
    def get_available_templates(self) -> List[str]:
        """Get list of available prompt templates."""
        return list(self.default_templates.keys())
    
    def add_custom_template(self, name: str, template: str) -> None:
        """
        Add a custom prompt template.
        
        Args:
            name: Template name
            template: Template string with {question} and {context} placeholders
        """
        self.default_templates[name] = template
        logger.info(f"Added custom template: {name}")
    
    def get_template(self, name: str) -> Optional[str]:
        """
        Get a specific template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template string or None if not found
        """
        return self.default_templates.get(name) 