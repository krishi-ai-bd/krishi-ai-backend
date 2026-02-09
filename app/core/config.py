import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
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

settings = Settings()    