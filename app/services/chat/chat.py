import json
import os
from typing import Any, Dict, List, Optional
import logging

from dotenv import load_dotenv
from groq import Groq
import openai

from .chat_schema import chatbot_request, chatbot_response, conversation_history_request, conversation_history_response, Message
from app.vectordb.manager import vector_db
from app.utils.cache_manager import cache_manager
from app.core.config import settings


load_dotenv()
logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Dynamic LLM client that switches between Groq and OpenAI.
    Controlled by settings.LLM_PROVIDER
    """
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        
        if self.provider == "groq":
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            self.model = settings.GROQ_MODEL
            logger.info(f"[LLM] Using Groq with model: {self.model}")
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
            logger.info(f"[LLM] Using OpenAI with model: {self.model}")
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {self.provider}. Use 'groq' or 'openai'")
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Unified chat completion interface for both providers
        """
        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"[LLM] Error calling {self.provider}: {str(e)}")
            raise


class ChatbotAgent:
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.reasoning_model = self.llm_client.model  # For backward compatibility

    def chat(self, request: chatbot_request) -> chatbot_response:
        """Generate RAG-enhanced chat response with conversation history."""
        
        # Retrieve conversation history
        conversation_history = cache_manager.get_conversation(request.chat_id) or []
        
        # Retrieve relevant knowledge from vector DB
        relevant_chunks = vector_db.search(request.message, n_results=5)
        
        # Build context from retrieved chunks
        context = self.build_context(relevant_chunks)
        
        # Create messages for OpenAI
        messages = self.create_messages(request.message, context, conversation_history)
        
        # Get AI response
        response_text = self.get_ai_response(messages)
        
        if not response_text:
            response_text = "I apologize, I'm having trouble generating a response. Please try again."
        
        # Save to conversation history
        cache_manager.add_message(request.chat_id, request.user_id, "user", request.message)
        cache_manager.add_message(request.chat_id, request.user_id, "assistant", response_text)
        
        return chatbot_response(response=response_text)
    
    def build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        if not chunks:
            return ""
        
        context_parts = ["Here is relevant information from the agricultural knowledge base:\n"]
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            doc_name = metadata.get("document_name", "Unknown")
            section = metadata.get("section_title", "")
            
            context_parts.append(
                f"\n[Source {i}: {doc_name} - {section}]\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def create_messages(self, user_query: str, context: str, history: List[Dict]) -> List[Dict]:
        """Create message list for OpenAI API"""
        
        system_prompt = """You are Krishi AI, an expert agricultural assistant specialized in crop management, farming techniques, and sustainable agriculture practices.

Your role is to:
- Provide accurate, practical advice based on the provided knowledge base
- Recommend solutions based on local farming conditions and best practices
- Help farmers optimize crop yields and reduce losses
- Explain complex agricultural concepts in simple, actionable terms
- Consider environmental sustainability and cost-effectiveness
- Always cite sources when using information from the knowledge base

When answering:
1. Use the provided context/sources to give accurate information
2. If the context doesn't contain relevant information, use your general agricultural knowledge
3. Be specific and actionable in your recommendations
4. Maintain a helpful, professional tone"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add last 5 conversation turns for context (if available)
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current query with context
        user_message = user_query
        if context:
            user_message = f"{context}\n\nUser Question: {user_query}"
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def get_conversation(self, request: conversation_history_request) -> conversation_history_response:
        """Retrieve full conversation history for a user and chat_id"""
        
        # Get conversation with user verification
        messages = cache_manager.get_conversation_by_user(request.user_id, request.chat_id)
        
        if messages is None:
            # Return empty conversation if not found or unauthorized
            return conversation_history_response(messages=[])
        
        # Convert to Message objects
        message_objects = [
            Message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"]
            )
            for msg in messages
        ]
        
        return conversation_history_response(messages=message_objects)

    def get_ai_response(self, messages: List[Dict]) -> str | None:
        """Call LLM API to generate response (Groq or OpenAI based on config)."""
        try:
            content = self.llm_client.chat(messages, temperature=0.7)
            return content
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return None


# Global instance
chatbot_agent = ChatbotAgent()
