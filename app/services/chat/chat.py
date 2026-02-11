import json
import os
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import logging

from dotenv import load_dotenv
from groq import Groq
import openai
from langgraph.graph import StateGraph, END
from operator import add

from .chat_schema import chatbot_request, chatbot_response, conversation_history_request, conversation_history_response, Message
from app.vectordb.manager import vector_db
from app.utils.cache_manager import cache_manager
from app.core.config import settings


load_dotenv()
logger = logging.getLogger(__name__)




class UnifiedLLMClient:
    """
    Dynamic LLM client with round-robin API key rotation.
    Supports multiple API keys for load balancing.
    Thread-safe for concurrent requests.
    """
    
    def __init__(self):
        import threading
        
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.GROQ_MODEL if self.provider == "groq" else settings.OPENAI_MODEL
        
        # Load multiple API keys
        self.api_keys = settings.load_api_keys(self.provider)
        
        if not self.api_keys:
            raise ValueError(f"No API keys found for provider: {self.provider}")
        
        # Thread-safe counter for round-robin
        self._counter = 0
        self._lock = threading.Lock()
        
        logger.info(f"[LLM] Using {self.provider.upper()} with {len(self.api_keys)} API keys")
        logger.info(f"[LLM] Model: {self.model}")
        logger.info(f"[LLM] Round-robin rotation enabled")
    
    def _get_next_key(self) -> str:
        """Get next API key in round-robin fashion (thread-safe)"""
        with self._lock:
            key = self.api_keys[self._counter % len(self.api_keys)]
            self._counter += 1
            return key
    
    def _create_client(self, api_key: str):
        """Create client instance with specific API key"""
        if self.provider == "groq":
            return Groq(api_key=api_key)
        elif self.provider == "openai":
            return openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Invalid provider: {self.provider}")
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Unified chat completion with automatic key rotation
        """
        # Get next API key
        api_key = self._get_next_key()
        key_index = (self._counter - 1) % len(self.api_keys) + 1
        
        logger.debug(f"[LLM] Using API key #{key_index}/{len(self.api_keys)}")
        
        try:
            # Create client with rotated key
            client = self._create_client(api_key)
            
            # Make API call
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"[LLM] Error with {self.provider} key #{key_index}: {str(e)}")
            raise


class AgricultureChatState(TypedDict):
    """LangGraph state for agriculture chatbot"""
    user_query: str
    user_id: str
    chat_id: str
    detected_language: Optional[str]  # Language of user query
    translated_query: Optional[str]  # Query translated for vector search
    is_agriculture_query: bool
    is_followup: bool
    skip_retrieval: bool
    retrieved_contexts: List[Dict]
    final_response: str
    metadata: Dict[str, Any]


class ChatbotAgent:
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.reasoning_model = self.llm_client.model  # For backward compatibility
        self.graph = self._build_graph()  # Build LangGraph workflow
        logger.info("[AGENT] Agriculture chatbot initialized with LangGraph")
    
    # ==================== LangGraph Nodes ====================
    
    def detect_language_and_guardrail_node(self, state: AgricultureChatState) -> AgricultureChatState:
        """
        Combined node: Detect language + Agriculture guardrail + Follow-up detection
        Single LLM call for efficiency
        """
        logger.info(f"[GUARDRAIL] Starting check for: {state['user_query'][:50]}...")
        
        # Get recent history for context
        recent_history = cache_manager.get_conversation_by_user(state['user_id'], state['chat_id']) or []
        recent_history = recent_history[-3:] if len(recent_history) > 3 else recent_history
        
        history_context = ""
        if recent_history:
            history_lines = [f"User: {h['content']}" if h['role'] == 'user' else f"Assistant: {h['content']}" 
                           for h in recent_history]
            history_context = "\n".join(history_lines)
        
        prompt = f"""You are a classifier for an agriculture chatbot. Analyze the user's query.

Conversation History:
{history_context if history_context else "[No previous conversation]"}

Current User Query: {state['user_query']}

Answer THREE questions in JSON format:
1. What is the language of the current query? Options: "bangla", "romanized_bangla", "english", "other"
2. Is this query related to AGRICULTURE topics? (crops, farming, pests, soil, irrigation, livestock, etc.)
3. Is this a FOLLOW-UP question referencing previous conversation?

Respond with ONLY valid JSON:
{{"detected_language": "bangla|romanized_bangla|english|other", "is_agriculture": true/false, "is_followup": true/false, "reason": "brief explanation"}}"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a precise classifier. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat(messages, temperature=0.3)
            result = json.loads(response)
            
            state["detected_language"] = result.get("detected_language", "english")
            state["is_agriculture_query"] = result.get("is_agriculture", True)
            state["is_followup"] = result.get("is_followup", False)
            state["skip_retrieval"] = result.get("is_followup", False)
            state["metadata"]["classification"] = result.get("reason", "")
            
            logger.info(f"[GUARDRAIL] Language: {state['detected_language']}, "
                       f"Agriculture: {state['is_agriculture_query']}, "
                       f"Followup: {state['is_followup']}")
            
        except Exception as e:
            logger.error(f"[GUARDRAIL] Classification error: {str(e)}")
            # Safe defaults
            state["detected_language"] = "english"
            state["is_agriculture_query"] = True
            state["is_followup"] = False
            state["skip_retrieval"] = False
        
        return state
    
    def translate_query_node(self, state: AgricultureChatState) -> AgricultureChatState:
        """
        Translate query to vector DB language if needed
        """
        target_lang = settings.VECTOR_DB_LANGUAGE.lower()
        detected_lang = state["detected_language"]
        
        logger.info(f"[TRANSLATE] Detected: {detected_lang}, Target: {target_lang}")
        
        # If already in target language, no translation needed
        if (target_lang == "english" and detected_lang in ["english"]) or \
           (target_lang == "bangla" and detected_lang in ["bangla", "romanized_bangla"]):
            state["translated_query"] = state["user_query"]
            logger.info(f"[TRANSLATE] No translation needed")
            return state
        
        # Need translation
        lang_map = {
            "bangla": "Bengali/Bangla",
            "romanized_bangla": "Bengali/Bangla (using romanized text)",
            "english": "English",
            "other": detected_lang
        }
        
        target_lang_name = "English" if target_lang == "english" else "Bengali/Bangla"
        source_lang_name = lang_map.get(detected_lang, "the detected language")
        
        prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}.
Keep agricultural terminology accurate. Only output the translation, nothing else.

Text: {state['user_query']}"""
        
        try:
            messages = [
                {"role": "system", "content": f"You are a translator. Translate to {target_lang_name}."},
                {"role": "user", "content": prompt}
            ]
            
            translated = self.llm_client.chat(messages, temperature=0.3)
            state["translated_query"] = translated.strip()
            logger.info(f"[TRANSLATE] Translated: {state['translated_query'][:50]}...")
            
        except Exception as e:
            logger.error(f"[TRANSLATE] Translation error: {str(e)}")
            state["translated_query"] = state["user_query"]  # Fallback to original
        
        return state
    
    def retrieve_knowledge_node(self, state: AgricultureChatState) -> AgricultureChatState:
        """
        Search vector DB for relevant agricultural knowledge
        """
        query = state["translated_query"] or state["user_query"]
        logger.info(f"[RETRIEVAL] Searching for: {query[:50]}...")
        
        try:
            results = vector_db.search(query, n_results=5)
            state["retrieved_contexts"] = results if results else []
            logger.info(f"[RETRIEVAL] Found {len(state['retrieved_contexts'])} results")
        
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error: {str(e)}")
            state["retrieved_contexts"] = []
            state["metadata"]["retrieval_error"] = str(e)
        
        return state
    
    def generate_response_node(self, state: AgricultureChatState) -> AgricultureChatState:
        """
        Generate response in Bangla using LLM with RAG context
        """
        logger.info(f"[GENERATION] Generating Bangla response...")
        
        # Build context from retrieved knowledge
        context = self.build_context(state["retrieved_contexts"])
        
        # Get conversation history for context
        conversation_history = cache_manager.get_conversation_by_user(
            state['user_id'], state['chat_id']
        ) or []
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        # Create messages
        system_prompt = """তুমি কৃষি বিষয়ক বিশেষজ্ঞ সহায়ক। তোমার কাজ হল:
- সবসময় বাংলায় উত্তর দিতে হবে
- প্রদত্ত তথ্য থেকে সঠিক ও ব্যবহারিক পরামর্শ দিতে হবে  
- স্থানীয় কৃষি পদ্ধতি ও পরিবেশ বিবেচনা করতে হবে
- জটিল বিষয় সহজভাবে ব্যাখ্যা করতে হবে

You are an agriculture expert assistant. Your task:
- Always respond in Bengali/Bangla language
- Provide accurate, practical advice from the knowledge base
- Consider local farming practices and environment
- Explain complex topics simply"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current query with context
        user_message = state['user_query']
        if context:
            user_message = f"প্রাসঙ্গিক তথ্য:\n{context}\n\nপ্রশ্ন: {state['user_query']}"
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response_text = self.llm_client.chat(messages, temperature=0.7)
            state["final_response"] = response_text
            logger.info(f"[GENERATION] Response generated ({len(response_text)} chars)")
        
        except Exception as e:
            logger.error(f"[GENERATION] Error: {str(e)}")
            state["final_response"] = "দুঃখিত, একটি ত্রুটি হয়েছে। আবার চেষ্টা করুন। (Sorry, an error occurred. Please try again.)"
            state["metadata"]["generation_error"] = str(e)
        
        return state
    
    def reject_query_node(self, state: AgricultureChatState) -> AgricultureChatState:
        """
        Politely reject non-agriculture queries in Bangla
        """
        state["final_response"] = """দুঃখিত, আমি শুধুমাত্র কৃষি সম্পর্কিত প্রশ্নের উত্তর দিতে পারি। 

আপনি আমাকে এইসব বিষয়ে জিজ্ঞাসা করতে পারেন:
- ফসল চাষাবাদ
- কীটপতঙ্গ ও রোগ নিয়ন্ত্রণ
- মাটির যত্ন ও সার
- সেচ ব্যবস্থা
- পশুপালন

(Sorry, I can only answer agriculture-related questions. You can ask me about crops, pests, soil, irrigation, livestock, etc.)"""
        
        logger.info(f"[REJECT] Query rejected - not agriculture related")
        return state
    
    # ==================== Routing Logic ====================
    
    def route_after_guardrail(self, state: AgricultureChatState) -> str:
        """Route based on guardrail results"""
        if not state['is_agriculture_query']:
            logger.info(f"[ROUTING] → reject (non-agriculture)")
            return "reject"
        
        if state['skip_retrieval'] or state['is_followup']:
            logger.info(f"[ROUTING] → generate (follow-up, skip retrieval)")
            return "skip_to_generate"
        
        logger.info(f"[ROUTING] → translate (new query, need retrieval)")
        return "translate"
    
    # ==================== Graph Builder ====================
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgricultureChatState)
        
        # Add nodes
        workflow.add_node("guardrail", self.detect_language_and_guardrail_node)
        workflow.add_node("translate", self.translate_query_node)
        workflow.add_node("retrieve", self.retrieve_knowledge_node)
        workflow.add_node("generate", self.generate_response_node)
        workflow.add_node("reject", self.reject_query_node)
        
        # Set entry point
        workflow.set_entry_point("guardrail")
        
        # Conditional routing after guardrail
        workflow.add_conditional_edges(
            "guardrail",
            self.route_after_guardrail,
            {
                "reject": "reject",
                "skip_to_generate": "generate",
                "translate": "translate"
            }
        )
        
        # Linear flow: translate → retrieve → generate
        workflow.add_edge("translate", "retrieve")
        workflow.add_edge("retrieve", "generate")
        
        # End nodes
        workflow.add_edge("generate", END)
        workflow.add_edge("reject", END)
        
        return workflow.compile()
    
    # ==================== Main Chat Interface ====================

    def chat(self, request: chatbot_request) -> chatbot_response:
        """
        LangGraph-based chat interface
        - Detects language
        - Checks agriculture relevance
        - Translates query for vector search
        - Generates response in Bangla
        - Saves to Redis for frontend
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[CHAT] New request from user: {request.user_id}")
        logger.info(f"[CHAT] Chat ID: {request.chat_id}")
        logger.info(f"[CHAT] Message: {request.message}")
        logger.info(f"{'='*80}")
        
        # Initialize state
        initial_state: AgricultureChatState = {
            "user_query": request.message,
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "detected_language": None,
            "translated_query": None,
            "is_agriculture_query": False,
            "is_followup": False,
            "skip_retrieval": False,
            "retrieved_contexts": [],
            "final_response": "",
            "metadata": {}
        }
        
        # Execute LangGraph workflow
        logger.info(f"[CHAT] Starting LangGraph execution...")
        final_state = self.graph.invoke(initial_state)
        logger.info(f"[CHAT] LangGraph execution completed")
        
        # Save to Redis for frontend display
        cache_manager.add_message(request.chat_id, request.user_id, "user", request.message)
        cache_manager.add_message(request.chat_id, request.user_id, "assistant", final_state['final_response'])
        
        logger.info(f"[CHAT] Response length: {len(final_state['final_response'])} chars")
        if final_state['metadata']:
            logger.warning(f"[CHAT] Metadata: {final_state['metadata']}")
        logger.info(f"{'='*80}\n")
        
        return chatbot_response(response=final_state['final_response'])
    
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


# Global instance
chatbot_agent = ChatbotAgent()
