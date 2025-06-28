from fastapi import APIRouter, HTTPException, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import logging
import pickle
from ..services.parser import Parser
from ..services.dataframe_storage import save_dataframes_to_pickle
from ..services.ardupilot_scraper import scrape_ardupilot_log_messages, insert_into_vector_db
from ..services.vector_initializer import start_vector_db_initialization, get_vector_db_status
import os

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Setup logging
logger = logging.getLogger(__name__)

# Ensure uploads directory exists
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Ensure messages directory exists
MESSAGES_DIR = Path("/tmp/messages")
MESSAGES_DIR.mkdir(exist_ok=True)

# Clear messages directory on startup
def clear_messages_directory():
    """Clear all files in the messages directory on startup."""
    try:
        if MESSAGES_DIR.exists():
            for file_path in MESSAGES_DIR.glob("*.json"):
                file_path.unlink()
                logger.info(f"Cleared old message file: {file_path}")
        logger.info("Messages directory cleared on startup")
    except Exception as e:
        logger.warning(f"Error clearing messages directory: {e}")

# Call cleanup on module import
clear_messages_directory()

def register_startup_cleanup(app: FastAPI):
    @app.on_event("startup")
    async def startup_event():
        clear_messages_directory()

def initialize_vector_db_for_codes(codes):
    """Initialize vector database with only the specified message codes."""
    try:
        if not codes:
            logger.warning("No message codes provided, skipping vector DB initialization")
            return False
        
        logger.info(f"Initializing vector database for {len(codes)} message codes")
        
        # Scrape only the specified codes
        messages = scrape_ardupilot_log_messages(codes=codes)
        logger.info(f"Scraped {len(messages)} messages for the specified codes")
        
        # Insert into vector database
        insert_into_vector_db(messages)
        logger.info("Vector database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        return False

class MessageData(BaseModel):
    """Model for incoming message data from frontend."""
    fileId: Optional[str] = None
    messages: Dict[str, Any]


class MessageResponse(BaseModel):
    """Model for API response."""
    success: bool
    message: str
    fileId: str
    messageCount: int
    timestamp: str

@router.post("/messages", response_model=MessageResponse)
async def upload_messages(message_data: MessageData):
    """
    Upload and store messages from the frontend.
    
    This endpoint receives the messages from this.state.messages and stores them
    as a JSON file for later processing and analysis.
    """
    try:
        logger.info("Starting to process messages upload")
        
        # Generate fileId if not provided
        file_id = message_data.fileId or f"msg_{uuid.uuid4().hex[:8]}"
        
        # Clean the messages data to ensure JSON compatibility
        logger.info("Cleaning messages data for JSON compatibility...")
        logger.info("Messages data cleaning completed")
        
        # Create the data structure to store
        data_to_store = {
            "fileId": file_id,
            "timestamp": datetime.now().isoformat(),
            "messages": message_data.messages,
            "messageCount": len(message_data.messages) if isinstance(message_data.messages, dict) else 0
        }

        logger.info(f"Data to store with {data_to_store['messageCount']} message types")

        # Save to file
        file_path = MESSAGES_DIR / f"{file_id}.json"
        logger.info(f"Saving messages to: {file_path}")
        
        with open(file_path, 'w') as f:
            json.dump(data_to_store, f, indent=2, default=str)
        
        logger.info(f"Successfully saved messages for {file_id}")

        # Parse the data and save dataframes
        dfs = Parser(file_path)
        save_dataframes_to_pickle(dfs, "dataframes/log_data.pkl")
        
        # Extract message codes and initialize vector database
        message_codes = set([x[:x.index('[')] if '[' in x else x for x in dfs.keys()])
        logger.info("Extracted message codes: " + str(message_codes))
        
        if message_codes:
            logger.info("Initializing vector database with extracted message codes...")
            vector_init_success = initialize_vector_db_for_codes(message_codes)
            if vector_init_success:
                logger.info("Vector database initialized successfully with message-specific codes")
            else:
                logger.warning("Vector database initialization failed")
        else:
            logger.warning("No message codes found, skipping vector database initialization")
        
        return MessageResponse(
            success=True,
            message="Messages uploaded and stored successfully",
            fileId=file_id,
            messageCount=data_to_store["messageCount"],
            timestamp=data_to_store["timestamp"]
        )

    except Exception as e:
        logger.error(f"Error uploading messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload messages: {str(e)}")


@router.get("/messages/{file_id}")
async def get_messages(file_id: str):
    """Retrieve stored messages by file ID."""
    try:
        file_path = MESSAGES_DIR / f"{file_id}.json"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Messages not found")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "data": data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving messages for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")


@router.get("/messages")
async def list_message_files():
    """List all stored message files."""
    try:
        files = []
        for file_path in MESSAGES_DIR.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    files.append({
                            "fileId": data.get("fileId"),
                            "fileName": data.get("fileName"),
                            "timestamp": data.get("timestamp"),
                            "messageCount": data.get("messageCount", 0)
                        })
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error listing message files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list message files: {str(e)}")


@router.delete("/messages/{file_id}")
async def delete_messages(file_id: str):
    """Delete stored messages by file ID."""
    try:
        file_path = MESSAGES_DIR / f"{file_id}.json"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Messages not found")
        
        file_path.unlink()
        
        return {
            "success": True,
            "message": f"Messages for {file_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting messages for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete messages: {str(e)}")


@router.delete("/messages")
async def clear_all_messages():
    """Clear all stored messages."""
    try:
        cleared_count = 0
        for file_path in MESSAGES_DIR.glob("*.json"):
            file_path.unlink()
            cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} message files")
        
        return {
            "success": True,
            "message": f"Cleared {cleared_count} message files",
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing all messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear messages: {str(e)}")

@router.post("/initialize-vector-db")
async def initialize_vector_database():
    """Manually trigger vector database initialization for existing data."""
    try:
        logger.info("Manual vector database initialization requested")
        
        # Check if dataframe file exists
        df_path = "dataframes/log_data.pkl.gz"
        if not os.path.exists(df_path):
            raise HTTPException(status_code=404, detail="No log data found. Please upload data first.")
        
        # Start vector database initialization
        start_vector_db_initialization()
        
        return {
            "success": True,
            "message": "Vector database initialization started in background",
            "status": get_vector_db_status()
        }
        
    except Exception as e:
        logger.error(f"Error starting vector database initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start vector database initialization: {str(e)}")


@router.get("/vector-db-status")
async def get_vector_db_status_endpoint():
    """Get the current status of the vector database."""
    try:
        status = get_vector_db_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Error getting vector database status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get vector database status: {str(e)}")