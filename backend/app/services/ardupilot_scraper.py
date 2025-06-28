import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
import re
import collections
import logging

# Configure logging
logger = logging.getLogger(__name__)

ARDUPILOT_LOG_URL = "https://ardupilot.org/copter/docs/logmessages.html"
VECTOR_COLLECTION_NAME = "ardupilot_log_messages"


def scrape_ardupilot_log_messages(url=ARDUPILOT_LOG_URL, codes=None):
    """
    Scrape the ArduPilot log message documentation and return a list of dicts with message info.
    If codes is provided (list of message names), only return documentation for those codes.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    messages = []
    
    # Find all section elements that contain message documentation
    sections = soup.find_all("section")
    
    for section in sections:
        # Look for h2 tags that contain message names
        h2_tag = section.find("h2")
        if not h2_tag:
            continue
            
        # Extract message name from h2 text
        msg_name = h2_tag.get_text(strip=True)[:-2]
        # Skip if we're filtering by codes and this one isn't in the list
        if codes is not None and msg_name not in codes:
            continue
            
        # Get description from the paragraph after h2
        description = ""
        desc_p = h2_tag.find_next_sibling("p")
        if desc_p:
            description = desc_p.get_text(strip=True)
            
        # Find the table with field information
        table = section.find("table")
        if not table:
            continue
            
        # Extract fields from table rows
        fields = []
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:  # Should have name, units, description
                field_name = cols[0].get_text(strip=True)
                field_units = cols[1].get_text(strip=True)
                field_desc = cols[2].get_text(strip=True)
                fields.append({
                    "name": field_name,
                    "units": field_units,
                    "description": field_desc
                })
        
        messages.append({
            "name": msg_name,
            "description": description,
            "fields": fields
        })
        
    return messages


def insert_into_vector_db(messages, collection_name=VECTOR_COLLECTION_NAME):
    """
    Insert the scraped messages into a Chroma vector database collection.
    If duplicate IDs are found, only the last occurrence is stored.
    """
    client = chromadb.Client(Settings(persist_directory="./vectorstore"))
    
    # Check if collection exists, delete and recreate to ensure clean state
    if collection_name in [c.name for c in client.list_collections()]:
        logger.info(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
    
    # Create new collection
    collection = client.create_collection(collection_name)
    logger.info(f"Created new collection: {collection_name}")
    
    # Create comprehensive documents for better search
    documents = []
    metadatas = []
    ids = []
    
    # Track used IDs to prevent duplicates
    used_ids = set()
    
    for msg in messages:
        # Create main document with message info
        main_doc = f"Message: {msg['name']}\nDescription: {msg['description']}\n"
        field_info = []
        
        for field in msg['fields']:
            field_text = f"Field: {field['name']} ({field['units']}) - {field['description']}"
            field_info.append(field_text)
            
            # Create individual field documents for better search
            field_doc = f"Message: {msg['name']}\nField: {field['name']}\nUnits: {field['units']}\nDescription: {field['description']}"
            
            # Ensure unique ID for field document
            field_id = f"{msg['name']}_{field['name']}"
            if field_id in used_ids:
                # If duplicate, add a counter
                counter = 1
                while f"{field_id}_{counter}" in used_ids:
                    counter += 1
                field_id = f"{field_id}_{counter}"
            used_ids.add(field_id)
            
            documents.append(field_doc)
            metadatas.append({
                "name": msg['name'],
                "field": field['name'],
                "type": "field",
                "description": field['description']
            })
            ids.append(field_id)
        
        # Add main message document with unique ID
        main_doc += "\nFields:\n" + "\n".join(field_info)
        
        # Ensure unique ID for main message document
        msg_id = msg['name']
        if msg_id in used_ids:
            # If duplicate, add a counter
            counter = 1
            while f"{msg_id}_{counter}" in used_ids:
                counter += 1
            msg_id = f"{msg_id}_{counter}"
        used_ids.add(msg_id)
        
        documents.append(main_doc)
        metadatas.append({
            "name": msg['name'],
            "type": "message",
            "description": msg['description']
        })
        ids.append(msg_id)
    
    # Insert into vector DB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Inserted {len(documents)} documents into vector DB collection '{collection_name}'")


def search_vector_db(query, collection_name=VECTOR_COLLECTION_NAME, n_results=5):
    """
    Search the vector database for relevant documentation based on a user query.
    Returns the most relevant message codes and their documentation.
    Falls back to mock search if collection doesn't exist.
    """
    try:
        client = chromadb.Client(Settings(persist_directory="./vectorstore"))
        
        # Check if collection exists
        if collection_name not in [c.name for c in client.list_collections()]:
            print(f"Collection '{collection_name}' not found in vector database")
            return []
        
        collection = client.get_collection(collection_name)
        
        # Search for similar documents
        results = collection.query(
            query_texts=[query],
            n_results=n_results * 2,  # Get more results to filter
            include=["documents", "metadatas", "distances"]
        )
        
        # Process and deduplicate results
        seen_messages = set()
        search_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                message_name = metadata.get('name', 'Unknown')
                
                # Only add each message once (prefer field-level matches)
                if message_name not in seen_messages:
                    seen_messages.add(message_name)
                    search_results.append({
                        'message_code': message_name,
                        'documentation': doc,
                        'relevance_score': 1 - distance,
                        'field': metadata.get('field', ''),
                        'type': metadata.get('type', 'message')
                    })
                    
                    if len(search_results) >= n_results:
                        break
        
        print(f"Found {len(search_results)} relevant results for query: '{query}'")
        return search_results
        
    except Exception as e:
        print(f"Error searching vector database: {e}")
        return []


def get_relevant_message_codes(query, collection_name=VECTOR_COLLECTION_NAME, n_results=5):
    """
    Get relevant message codes for a user query.
    Returns a list of message codes that are most relevant to the query.
    """
    search_results = search_vector_db(query, collection_name, n_results)
    return [result['message_code'] for result in search_results]


if __name__ == "__main__":
    print("Scraping ArduPilot log messages...")
    messages = scrape_ardupilot_log_messages(codes=['FILE', 'PARM', 'GPS'])
    print(f"Scraped {len(messages)} log messages.")
    print("Inserting into vector database...")
    insert_into_vector_db(messages)
    
    # Test search functionality
    print("\nTesting search functionality...")
    test_query = "GPS position and altitude"
    results = search_vector_db(test_query)
    for result in results:
        print(f"Message Code: {result['message_code']}")
        print(f"Relevance Score: {result['relevance_score']:.3f}")
        print(f"Documentation: {result['documentation'][:200]}...")
        print("-" * 50) 