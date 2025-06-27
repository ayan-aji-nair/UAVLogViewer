#!/usr/bin/env python3
"""
Standalone script to initialize the vector database with ArduPilot documentation.
This can be run independently or as part of the application startup.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Import directly from the module to avoid relative import issues
from app.services.ardupilot_scraper import scrape_ardupilot_log_messages, insert_into_vector_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to initialize the vector database."""
    logger.info("Starting vector database initialization...")
    
    try:
        # Create vectorstore directory if it doesn't exist
        vectorstore_dir = "./vectorstore"
        os.makedirs(vectorstore_dir, exist_ok=True)
        logger.info(f"Vectorstore directory: {vectorstore_dir}")
        
        # Scrape ArduPilot log messages
        logger.info("Scraping ArduPilot log messages...")
        messages = scrape_ardupilot_log_messages()
        logger.info(f"Scraped {len(messages)} log messages")
        
        # Insert into vector database
        logger.info("Inserting messages into vector database...")
        insert_into_vector_db(messages)
        
        logger.info("✅ Vector database initialization completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Vector database initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 