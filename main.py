from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from app.services.chat.chat_route import router as chat_router
from app.utils.knowledge.knowledge_route import router as knowledge_router
from app.services.daily_suggestion.daily_suggestion_route import router as daily_suggestion_router, daily_suggestion_agent
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop APScheduler for daily suggestion generation."""
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        func=daily_suggestion_agent.generate_scheduled,
        trigger="cron",
        hour=settings.DAILY_SUGGESTION_HOUR,
        minute=0,
        id="daily_suggestion",
        replace_existing=True,
    )
    scheduler.start()
    print(f"[SCHEDULER] Daily suggestion scheduled at {settings.DAILY_SUGGESTION_HOUR:02d}:00 UTC "
          f"(≈ {(settings.DAILY_SUGGESTION_HOUR + 6) % 24:02d}:00 Bangladesh time)")
    yield  # App runs here
    scheduler.shutdown()
    print("[SCHEDULER] Scheduler stopped.")


# Initialize FastAPI app
app = FastAPI(
    title="Krishi AI Backend",
    description="Agricultural AI Assistant with RAG and Conversation Management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount audio folder as static files (for TTS audio URL access)
audio_dir = Path(settings.AUDIO_DIR)
audio_dir.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")

# Include routers
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(knowledge_router, prefix="/api", tags=["Knowledge"])
app.include_router(daily_suggestion_router, prefix="/api", tags=["Daily Suggestion"])


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
