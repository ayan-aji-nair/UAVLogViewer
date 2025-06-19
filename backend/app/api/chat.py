from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
import uuid
import asyncio

from ..models.chat import ChatRequest, ChatResponse, ChatMessage
from ..services.chatbot import UAVChatbotService

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize chatbot service
chatbot_service = UAVChatbotService()


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message to the chatbot and get a response."""
    try:
        # Generate session ID if not provided
        sessionId = request.sessionId or str(uuid.uuid4())
        
        # Process message with chatbot
        response_text = await chatbot_service.process_message(
            message=request.message,
            sessionId=sessionId,
            contextData=request.contextData
        )
        
        return ChatResponse(
            message=response_text,
            sessionId=sessionId
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """Stream a chatbot response token by token."""
    try:
        sessionId = request.sessionId or str(uuid.uuid4())
        
        async def generate_stream():
            # Process message and stream response
            response_text = await chatbot_service.process_message(
                message=request.message,
                sessionId=sessionId,
                contextData=request.contextData
            )
            
            # Stream the response token by token
            tokens = response_text.split()
            for token in tokens:
                yield f"data: {token} \n\n"
                await asyncio.sleep(0.1)  # Simulate streaming delay
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")


@router.get("/history/{sessionId}", response_model=List[ChatMessage])
async def get_chat_history(sessionId: str):
    """Get chat history for a specific session."""
    try:
        history = chatbot_service.get_session_history(sessionId)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")


@router.post("/context/{sessionId}")
async def update_context(sessionId: str, contextData: dict):
    """Update context data for a chat session."""
    try:
        success = chatbot_service.update_session_context(sessionId, contextData)
        if success:
            return {"message": "Context updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating context: {str(e)}")


@router.post("/session")
async def create_session(contextData: Optional[dict] = None):
    """Create a new chat session."""
    try:
        sessionId = str(uuid.uuid4())
        chatbot_service.session_manager.create_session(sessionId, contextData)
        return {"sessionId": sessionId, "message": "Session created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}") 