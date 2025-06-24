"""UAV Log Viewer Chatbot API
Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("🚀 Starting UAV Log Viewer Chatbot API...")
    print(f"📊 OpenAI Model: {settings.openai_model}")
    print(f"🔗 Redis: {settings.redis_host}:{settings.redis_port}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down UAV Log Viewer Chatbot API...")


# Create FastAPI app
app = FastAPI(
    title="UAV Log Viewer Chatbot API",
    description="Agentic chatbot for analyzing UAV log data using LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "UAV Log Viewer Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat/message",
            "stream": "/api/chat/stream",
            "history": "/api/chat/history/{session_id}",
            "upload_messages": "/api/upload/messages",
            "get_messages": "/api/upload/messages/{file_id}",
            "list_messages": "/api/upload/messages",
            "delete_messages": "/api/upload/messages/{file_id}",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "uav-chatbot-api"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not settings.openai_api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. Chatbot functionality will be limited.")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
