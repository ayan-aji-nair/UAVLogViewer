#!/usr/bin/env python3
"""
Test script to verify vector database initialization.
"""

import sys
import time
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.vector_initializer import VectorDBInitializer


def test_vector_initialization():
    """Test the vector database initialization."""
    print("🧪 Testing vector database initialization...")
    
    # Create initializer
    initializer = VectorDBInitializer()
    
    # Check initial status
    status = initializer.get_status()
    print(f"Initial status: {status}")
    
    # Start initialization
    print("Starting initialization...")
    initializer.start_initialization()
    
    # Monitor progress
    for i in range(30):  # Wait up to 30 seconds
        status = initializer.get_status()
        print(f"Status after {i+1}s: {status}")
        
        if status['is_initialized']:
            print("✅ Vector database initialized successfully!")
            return True
        elif not status['is_initializing']:
            print("❌ Initialization failed or stopped unexpectedly")
            return False
            
        time.sleep(1)
    
    print("⏰ Initialization timed out")
    return False


if __name__ == "__main__":
    success = test_vector_initialization()
    sys.exit(0 if success else 1) 