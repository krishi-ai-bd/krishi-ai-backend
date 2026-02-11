import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # LLM Provider Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai", "groq", or "both"
    
    # OpenAI - Single key fallback (deprecated, use multiple keys)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Groq - Single key fallback (deprecated, use multiple keys)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    
    # Language Configuration
    VECTOR_DB_LANGUAGE: str = os.getenv("VECTOR_DB_LANGUAGE", "english")  # Language of PDFs in vector DB
    RESPONSE_LANGUAGE: str = os.getenv("RESPONSE_LANGUAGE", "bangla")  # Always respond in this language
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    USER_INACTIVE_DAYS: int = 60  # Remove conversations after 60 days of inactivity
    CONVERSATION_TTL_DAYS: int = 90  # Keep conversations for max 90 days
    
    # Vector DB Configuration
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    COLLECTION_NAME: str = "agricultural_knowledge"
    
    # PDF Processing
    CHUNK_SIZE: int = 800  # tokens per chunk
    CHUNK_OVERLAP: int = 150  # overlap tokens
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5  # number of chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.7
    
    def load_api_keys(self, provider: str) -> List[str]:
        """
        Load multiple API keys for a provider from environment variables.
        Looks for keys like: OPENAI_API_KEY_1, OPENAI_API_KEY_2, etc.
        Falls back to single key if multiple keys not found.
        """
        keys = []
        prefix = f"{provider.upper()}_API_KEY"
        
        # Try to load numbered keys (_1, _2, _3, ...)
        i = 1
        while True:
            key = os.getenv(f"{prefix}_{i}", "")
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        # If no numbered keys found, use single key
        if not keys:
            single_key = os.getenv(prefix, "")
            if single_key:
                keys.append(single_key)
        
        return [k for k in keys if k]  # Filter empty strings

settings = Settings()    