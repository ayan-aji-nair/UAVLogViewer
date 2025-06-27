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


@router.post("/test-search")
async def test_vector_search(query: str):
    """
    Test endpoint to verify vector search improvements.
    """
    try:
        from ..services.chatbot import UAVChatbotService
        
        chatbot = UAVChatbotService()
        
        # Test the smart vector search
        results = chatbot.smart_vector_search(query)
        
        # Test query preprocessing
        enhanced_query = chatbot.preprocess_query(query)
        
        # Test query expansion
        expanded_queries = chatbot.expand_query(query)
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "expanded_queries": expanded_queries,
            "search_results": results,
            "result_count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing search: {str(e)}")


@router.post("/reinitialize-vector-db")
async def reinitialize_vector_db():
    """
    Manually reinitialize the vector database with improved field-level documents.
    """
    try:
        from ..services.vector_initializer import initialize_vector_db_from_dataframe
        
        success = initialize_vector_db_from_dataframe()
        
        if success:
            return {"message": "Vector database reinitialized successfully with field-level documents"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reinitialize vector database")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reinitializing vector database: {str(e)}")


@router.post("/test-modular-search")
async def test_modular_search(query: str):
    """
    Test endpoint to verify modular semantic search improvements.
    """
    try:
        from ..services.chatbot import UAVChatbotService
        from ..services.ardupilot_scraper import search_vector_db
        
        # Test the enhanced vector search
        vector_results = search_vector_db(query, n_results=5)
        
        # Test the DataFrameQueryTool's semantic analysis
        chatbot = UAVChatbotService()
        df_tool = chatbot.tools[2]  # DataFrameQueryTool
        
        # Create a mock vector search result format
        mock_vector_result = "Here's relevant documentation for your query:\n\n"
        for i, result in enumerate(vector_results, 1):
            mock_vector_result += f"{i}. **Message Code: {result['message_code']}**\n"
            mock_vector_result += f"   Relevance: {result['relevance_score']:.2f}\n"
            mock_vector_result += f"   Documentation: {result['documentation']}\n\n"
        
        # Test the dataframe query tool
        test_input = f"{query}|||{mock_vector_result}"
        df_analysis = df_tool._run(test_input)
        
        return {
            "original_query": query,
            "vector_search_results": vector_results,
            "dataframe_analysis": df_analysis,
            "modular_approach": "The system now uses semantic search results to dynamically find relevant columns without hardcoded patterns"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing modular search: {str(e)}")


@router.post("/test-indexed-dataframes")
async def test_indexed_dataframes(query: str):
    """
    Test endpoint to verify indexed dataframe handling and aggregation.
    """
    try:
        from ..services.chatbot import UAVChatbotService
        from ..services.ardupilot_scraper import search_vector_db
        from ..services.dataframe_storage import load_dataframes_from_pickle
        
        # Test the enhanced vector search
        vector_results = search_vector_db(query, n_results=5)
        
        # Load actual dataframes to test indexed handling
        df_path = "dataframes/log_data.pkl"
        dataframes = load_dataframes_from_pickle(df_path)
        
        # Test the DataFrameQueryTool's indexed dataframe analysis
        chatbot = UAVChatbotService()
        df_tool = chatbot.tools[2]  # DataFrameQueryTool
        
        # Create a mock vector search result format
        mock_vector_result = "Here's relevant documentation for your query:\n\n"
        for i, result in enumerate(vector_results, 1):
            mock_vector_result += f"{i}. **Message Code: {result['message_code']}**\n"
            mock_vector_result += f"   Relevance: {result['relevance_score']:.2f}\n"
            mock_vector_result += f"   Documentation: {result['documentation']}\n\n"
        
        # Test the dataframe query tool
        test_input = f"{query}|||{mock_vector_result}"
        df_analysis = df_tool._run(test_input)
        
        # Test the indexed dataframe matching
        search_info = df_tool._extract_search_info_from_vector_results(mock_vector_result)
        matching_dataframes = df_tool._find_matching_dataframes(dataframes, search_info)
        
        return {
            "original_query": query,
            "vector_search_results": vector_results,
            "available_dataframes": list(dataframes.keys()),
            "matching_dataframes": list(matching_dataframes.keys()),
            "dataframe_analysis": df_analysis,
            "indexed_handling": "The system now handles both indexed (GPS[0]) and non-indexed (GPS) dataframes and aggregates results"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing indexed dataframes: {str(e)}") 