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
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process message with chatbot
        response_text = await chatbot_service.process_message(
            message=request.message,
            session_id=session_id,
            context_data=request.context_data
        )
        
        return ChatResponse(
            message=response_text,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """Stream a chatbot response token by token."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        async def generate_stream():
            # Process message and stream response
            response_text = await chatbot_service.process_message(
                message=request.message,
                session_id=session_id,
                context_data=request.context_data
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


@router.get("/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str):
    """Get chat history for a specific session."""
    try:
        history = chatbot_service.get_session_history(session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")


@router.post("/context/{session_id}")
async def update_context(session_id: str, context_data: dict):
    """Update context data for a chat session."""
    try:
        success = chatbot_service.update_session_context(session_id, context_data)
        if success:
            return {"message": "Context updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating context: {str(e)}")


@router.post("/session")
async def create_session(context_data: Optional[dict] = None):
    """Create a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        chatbot_service.session_manager.create_session(session_id, context_data)
        return {"session_id": session_id, "message": "Session created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}") 