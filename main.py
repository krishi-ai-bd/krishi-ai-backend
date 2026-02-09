from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.chat.chat_route import router as chat_router
from app.services.documents.document_route import router as document_router

# Initialize FastAPI app
app = FastAPI(
    title="Krishi AI Backend",
    description="Agricultural AI Assistant with RAG and Conversation Management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(document_router, prefix="/api", tags=["Documents"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Krishi AI Backend",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    from app.vectordb.manager import vector_db
    from app.utils.cache_manager import cache_manager
    
    health_status = {
        "api": "healthy",
        "redis": "healthy" if cache_manager.redis_client else "unavailable",
        "vector_db": "healthy"
    }
    
    try:
        stats = vector_db.get_collection_stats()
        health_status["vector_db_chunks"] = stats.get("total_chunks", 0)
    except:
        health_status["vector_db"] = "error"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
