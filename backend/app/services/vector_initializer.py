import asyncio
import threading
import time
import os
import pickle
from typing import Optional
import logging
from .ardupilot_scraper import scrape_ardupilot_log_messages, insert_into_vector_db
from .dataframe_storage import load_dataframes_from_pickle
import chromadb
from chromadb.config import Settings

# Disable ChromaDB telemetry to prevent PostHog errors
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBInitializer:
    """Background service to initialize and maintain the vector database."""
    
    def __init__(self):
        self.is_initialized = False
        self.is_initializing = False
        self.initialization_thread: Optional[threading.Thread] = None
        self.last_update = None
        self.update_interval = 24 * 60 * 60  # 24 hours in seconds
        
    def start_initialization(self):
        """Start the vector database initialization in a background thread."""
        if self.is_initializing or self.is_initialized:
            logger.info("Vector DB initialization already in progress or completed")
            return
            
        self.is_initializing = True
        self.initialization_thread = threading.Thread(
            target=self._initialize_vector_db,
            daemon=True
        )
        self.initialization_thread.start()
        logger.info("Started vector database initialization in background")
    
    def _get_available_message_codes(self) -> list:
        """Get message codes from the current dataframe dictionary."""
        try:
            df_path = "dataframes/log_data.pkl"
            dataframes = load_dataframes_from_pickle(df_path)
            
            if not dataframes:
                logger.info("No dataframes found, skipping vector DB initialization")
                return []
            
            codes = [x[:-3] if '[' in x else x for x in dataframes.keys()]
            logger.info(f"Found {len(codes)} message codes in dataframe: {codes}")
            return codes
            
        except Exception as e:
            logger.error(f"Error reading dataframe file: {e}")
            return []
    
    def _initialize_vector_db(self):
        """Initialize the vector database with ArduPilot documentation for available message codes."""
        try:
            logger.info("Starting vector database initialization...")
            
            # Create vectorstore directory if it doesn't exist
            vectorstore_dir = "./vectorstore"
            os.makedirs(vectorstore_dir, exist_ok=True)
            
            # Get available message codes from dataframe
            available_codes = self._get_available_message_codes()
            
            if not available_codes:
                logger.info("No message codes available, skipping vector DB initialization")
                self.is_initialized = False
                return
            
            # Scrape ArduPilot log messages for only the available codes
            logger.info(f"Scraping ArduPilot log messages for {len(available_codes)} codes: {available_codes}")
            messages = scrape_ardupilot_log_messages(codes=available_codes)
            logger.info(f"Scraped {len(messages)} log messages")
            
            if not messages:
                logger.warning("No messages scraped, skipping vector DB initialization")
                self.is_initialized = False
                return
            
            # Insert into vector database
            logger.info("Inserting messages into vector database...")
            insert_into_vector_db(messages)
            
            self.is_initialized = True
            self.last_update = time.time()
            logger.info("Vector database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            self.is_initialized = False
        finally:
            self.is_initializing = False
    
    def check_and_update(self):
        """Check if vector database needs updating and start update if needed."""
        if self.is_initializing:
            return
            
        current_time = time.time()
        needs_update = (
            not self.is_initialized or 
            (self.last_update and current_time - self.last_update > self.update_interval)
        )
        
        if needs_update:
            logger.info("Vector database needs initialization/update, starting...")
            self.start_initialization()
    
    def get_status(self) -> dict:
        """Get the current status of the vector database."""
        return {
            "is_initialized": self.is_initialized,
            "is_initializing": self.is_initializing,
            "last_update": self.last_update,
            "update_interval": self.update_interval
        }
    
    def wait_for_initialization(self, timeout: int = 300) -> bool:
        """Wait for vector database initialization to complete."""
        start_time = time.time()
        while self.is_initializing and (time.time() - start_time) < timeout:
            time.sleep(1)
        return self.is_initialized


# Global instance
vector_initializer = VectorDBInitializer()


def start_vector_db_initialization():
    """Start the vector database initialization process."""
    vector_initializer.start_initialization()


def get_vector_db_status() -> dict:
    """Get the current status of the vector database."""
    return vector_initializer.get_status()


def ensure_vector_db_ready() -> bool:
    """Ensure the vector database is ready, starting initialization if needed."""
    vector_initializer.check_and_update()
    return vector_initializer.wait_for_initialization(timeout=60)


def initialize_vector_db_from_dataframe():
    """
    Initialize the vector database with ArduPilot documentation for message codes found in the dataframe.
    """
    try:
        # Load the dataframes to get available message codes
        df_path = "dataframes/log_data.pkl"
        dataframes = load_dataframes_from_pickle(df_path)
        
        if not dataframes:
            logger.error("No dataframes available for vector database initialization")
            return False
        
        # Get unique message codes from the dataframes
        available_codes = list(dataframes.keys())
        logger.info(f"Found {len(available_codes)} unique message codes in dataframes")
        
        # Scrape ArduPilot documentation for these specific codes
        messages = scrape_ardupilot_log_messages(codes=available_codes)
        if not messages:
            logger.error("Failed to scrape ArduPilot documentation")
            return False
        
        # Filter messages to only include codes present in the dataframes
        filtered_messages = [msg for msg in messages if msg['name'] in available_codes]
        logger.info(f"Filtered to {len(filtered_messages)} messages that exist in dataframes")
        
        if not filtered_messages:
            logger.error("No matching messages found between documentation and dataframes")
            return False
        
        # Insert into vector database with enhanced field-level documents
        insert_into_vector_db(filtered_messages)
        
        logger.info("Vector database initialized successfully with field-level documents")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        return False 