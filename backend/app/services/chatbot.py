import os
import uuid
import pandas as pd
import pickle
import numpy as np
import gzip
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field
from ..models.chat import ChatMessage, MessageRole, ChatSession
from .session_manager import SessionManager
from .ardupilot_scraper import search_vector_db, get_relevant_message_codes


class UAVAnalysisTool(BaseTool):
    name = "uav_analyzer"
    description = "Analyze UAV log data by searching documentation and querying flight data to answer user questions about flight performance, altitude, GPS data, and other UAV metrics"
    
    def _run(self, user_question: str) -> str:
        """Analyze UAV log data by combining vector search and dataframe query."""
        try:
            # Step 1: Search for relevant documentation
            search_results = search_vector_db(user_question, n_results=5)
            
            if not search_results:
                return "I am unsure of how to answer that question. The documentation doesn't contain relevant information for your query."
            
            # Step 2: Extract message types and fields from search results
            message_types, fields = self._extract_message_types_and_fields(search_results)
            
            if not message_types:
                return "I found some documentation but couldn't identify specific message types to analyze. Please try rephrasing your question."
            
            # Step 3: Query dataframes with extracted information
            dataframe_tool = DataFrameQueryTool()
            result = dataframe_tool._query_with_extracted_info(user_question, message_types, fields)
            
            return result
            
        except Exception as e:
            return f"I encountered an error while analyzing your question: {str(e)}"
    
    def _extract_message_types_and_fields(self, search_results: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, str]]]:
        """Extract message types and fields from vector search results."""
        message_types = []
        fields = []
        
        for result in search_results:
            message_code = result.get('message_code', '')
            documentation = result.get('documentation', '')
            field = result.get('field', '')
            
            if message_code and message_code not in message_types:
                message_types.append(message_code)
            
            # Extract field information from documentation
            if field:
                # Clean up field name - extract only the field name before any description
                field_name = field.strip()
                # Remove description after dash or parenthesis
                if ' - ' in field_name:
                    field_name = field_name.split(' - ')[0].strip()
                elif ' (' in field_name:
                    field_name = field_name.split(' (')[0].strip()
                elif ' (' in field_name:
                    field_name = field_name.split(' (')[0].strip()
                
                if field_name:
                    fields.append({
                        'message_code': message_code,
                        'name': field_name
                    })
            else:
                # Try to extract fields from documentation text
                lines = documentation.split('\n')
                for line in lines:
                    if line.startswith('Field:') and message_code:
                        field_name = line.split('Field:')[1].strip()
                        # Remove description after dash or parenthesis
                        if ' - ' in field_name:
                            field_name = field_name.split(' - ')[0].strip()
                        elif ' (' in field_name:
                            field_name = field_name.split(' (')[0].strip()
                        
                        if field_name:
                            fields.append({
                                'message_code': message_code,
                                'name': field_name
                            })
        
        print(f"DEBUG: Extracted message types: {message_types}")
        print(f"DEBUG: Extracted fields: {fields}")
        
        return message_types, fields


class UAVChatbotService:
    def __init__(self):
        self.session_manager = SessionManager()
        self.search_patterns = {}  # Store successful search patterns
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = [
            UAVAnalysisTool()
        ]
        
        # Initialize agent with memory
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10
            )
        )
        
        # System prompt for UAV analysis
        self.system_prompt = """
        You are an expert UAV (Unmanned Aerial Vehicle) log analyst assistant. Your role is to help users understand their flight data and identify potential issues or insights.

        Key Responsibilities:
        1. Analyze UAV log data for patterns, anomalies, and performance metrics
        2. Provide focused, concise answers about GPS data, signal loss, and flight performance
        3. Look for specific events like GPS signal loss, altitude changes, and status changes
        4. Provide direct answers with minimal context to prevent rate limiting
        5. Use the uav_analyzer tool for ALL user questions about flight data

        Analysis Approach:
        - Be concise and focused in your responses
        - Provide direct answers to specific questions
        - Focus on GPS-related queries with minimal context
        - Avoid lengthy explanations that could cause rate limiting
        - Prioritize accuracy over verbosity

        TOOL USAGE:
        - Use the uav_analyzer tool for ALL user questions about flight data
        - Simply pass the user's question directly to the tool
        - The tool will automatically search documentation and analyze data
        - No special formatting required - just use the tool with the user's question

        Example usage:
        - User asks: "When did the GPS signal first get lost?"
        - Use: uav_analyzer with "When did the GPS signal first get lost?"
        - The tool handles everything internally

        Always be helpful, concise, and provide direct answers.
        """

    def store_search_pattern(self, query: str, search_results: List[str], success: bool):
        """Store successful search patterns for learning."""
        if success and search_results:
            if query not in self.search_patterns:
                self.search_patterns[query] = []
            self.search_patterns[query].extend(search_results)
            # Keep only unique patterns
            self.search_patterns[query] = list(set(self.search_patterns[query]))

    def get_learned_patterns(self) -> str:
        """Get learned search patterns as context."""
        if not self.search_patterns:
            return ""
        
        patterns = []
        patterns.append("Learned Search Patterns:")
        for query, results in self.search_patterns.items():
            patterns.append(f"- Query: '{query}' → Found: {', '.join(results[:3])}")  # Show first 3 results
        
        return "\n".join(patterns)

    async def process_message(self, message: str, sessionId: str, contextData: Optional[Dict[str, Any]] = None) -> str:
        """Process a user message and return an agentic response."""
        
        # Check if vector database is ready
        if not self._is_vector_db_ready():
            return """I'm currently setting up the documentation database for your flight data. 

Please wait a moment while I prepare the analysis tools. This happens automatically when you upload flight data.

If you've already uploaded data and are still seeing this message, you can:
1. Wait a few more moments for the setup to complete
2. Try uploading your flight data again
3. Contact support if the issue persists

The system needs to analyze your specific flight data to provide accurate answers about your UAV's performance."""
        
        # Get or create session
        session = self.session_manager.get_session(sessionId)
        if not session:
            session = self.session_manager.create_session(sessionId, contextData)
        
        # Add user message to session
        self.session_manager.add_message(sessionId, MessageRole.USER, message)
        
        # Prepare context for the LLM including learned patterns
        context = self._prepare_context(session, contextData)
        learned_patterns = self.get_learned_patterns()
        
        if learned_patterns:
            context += "\n\n" + learned_patterns
        
        # Create messages for the agent
        messages = [
            SystemMessage(content=self.system_prompt + "\n\n" + context),
            *self._convert_session_messages(session.messages[-10:])  # Last 10 messages
        ]
        
        try:
            # Get response from agent
            response = await self.agent.ainvoke({
                "input": message,
                "chat_history": messages
            })
            
            response_text = response.get("output", "I'm sorry, I couldn't process your request.")
            
            # Store successful search patterns if the response indicates success
            if "relevant message codes" in response_text or "Found data" in response_text:
                # Extract message codes from response for learning
                import re
                codes = re.findall(r'Message Code: (\w+)', response_text)
                if codes:
                    self.store_search_pattern(message, codes, True)
            
            # Add assistant response to session
            self.session_manager.add_message(sessionId, MessageRole.ASSISTANT, response_text)
            
            return response_text
            
        except Exception as e:
            error_message = f"I encountered an error while processing your request: {str(e)}"
            self.session_manager.add_message(sessionId, MessageRole.ASSISTANT, error_message)
            return error_message
    
    def _prepare_context(self, session: ChatSession, contextData: Optional[Dict[str, Any]]) -> str:
        """Prepare context information for the LLM."""
        context_parts = []
        
        # Add session context data
        if session.contextData:
            context_parts.append("Session Context:")
            for key, value in session.contextData.items():
                context_parts.append(f"- {key}: {value}")
        
        # Add new context data
        if contextData:
            context_parts.append("Current Context:")
            for key, value in contextData.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts) if context_parts else "No additional context available."
    
    def _convert_session_messages(self, messages: List[ChatMessage]) -> List:
        """Convert session messages to LangChain message format."""
        converted = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                converted.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                converted.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                converted.append(SystemMessage(content=msg.content))
        return converted
    
    def get_session_history(self, sessionId: str) -> List[ChatMessage]:
        """Get chat history for a session."""
        return self.session_manager.get_session_history(sessionId)
    
    def update_session_context(self, sessionId: str, contextData: Dict[str, Any]) -> bool:
        """Update the context data for a session."""
        return self.session_manager.update_context(sessionId, contextData)
    
    def _is_vector_db_ready(self) -> bool:
        """Check if the vector database is properly initialized and ready for queries."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.Client(Settings(persist_directory="./vectorstore"))
            
            # Check if the collection exists
            if "ardupilot_log_messages" not in [c.name for c in client.list_collections()]:
                return False
            
            # Check if the collection has any documents
            collection = client.get_collection("ardupilot_log_messages")
            count = collection.count()
            
            return count > 0
            
        except Exception as e:
            # If there's any error checking the vector database, assume it's not ready
            return False 


class DataFrameQueryTool(BaseTool):
    name = "dataframe_query"
    description = "Query UAV log dataframes using pandas operations to analyze flight data and answer user questions. Use this after vector_search to analyze actual data. Input format: 'USER_QUESTION|||VECTOR_SEARCH_RESULTS'"
    
    def _run(self, input_text: str) -> str:
        """Query dataframes based on user question and vector search results."""
        try:
            # Parse input: user_question|||vector_search_results
            if '|||' in input_text:
                user_question, vector_search_results = input_text.split('|||', 1)
                user_question = user_question.strip()
                vector_search_results = vector_search_results.strip()
            else:
                return """ERROR: Incorrect input format for dataframe_query tool.

The dataframe_query tool requires input in the format: 'USER_QUESTION|||VECTOR_SEARCH_RESULTS'

You must:
1. FIRST use the vector_search tool to find relevant message codes
2. THEN use dataframe_query with the COMPLETE results from vector_search

Example: "What is the max altitude?|||Here's relevant documentation for your query:

1. **Message Code: GPS**
   Relevance: 0.85
   Documentation: Message: GPS
Field: Alt
Units: m
Description: GPS altitude...

Please use vector_search first, then dataframe_query with the complete output."""
            
            # Load the dataframes from the pickle file
            df_path = "dataframes/log_data.pkl.gz"
            if not os.path.exists(df_path):
                return "No log data available. Please upload log data first."
            
            with gzip.open(df_path, 'rb') as f:
                dataframes = pickle.load(f)
            
            if not dataframes:
                return "No dataframes found in the log data."
            
            # Extract relevant codes and field information from vector search results
            search_info = self._extract_search_info_from_vector_results(vector_search_results)
            
            if not search_info['message_codes']:
                return "No relevant message codes found in vector search results. Please ensure vector_search was used first and returned valid results."
            
            # Analyze dataframes semantically based on search results
            analysis = self._semantic_dataframe_analysis(dataframes, search_info, user_question)
            
            return analysis
            
        except Exception as e:
            return f"Error querying dataframe: {str(e)}"
    
    def _query_with_extracted_info(self, user_question: str, message_types: List[str], fields: List[Dict[str, str]]) -> str:
        """Query dataframes using pre-extracted message types and fields."""
        try:
            # Load the dataframes from the pickle file
            df_path = "dataframes/log_data.pkl.gz"
            if not os.path.exists(df_path):
                return "No log data available. Please upload log data first."
            
            with gzip.open(df_path, 'rb') as f:
                dataframes = pickle.load(f)
            
            if not dataframes:
                return "No dataframes found in the log data."
            
            # Create search_info structure from extracted data
            search_info = {
                'message_codes': message_types,
                'fields': fields,
                'documentation': []
            }
            
            if not search_info['message_codes']:
                return "No relevant message codes provided for analysis."
            
            # Analyze dataframes semantically based on search results
            analysis = self._semantic_dataframe_analysis(dataframes, search_info, user_question)
            
            return analysis
            
        except Exception as e:
            return f"Error querying dataframe: {str(e)}"
    
    def _extract_search_info_from_vector_results(self, vector_results: str) -> Dict[str, Any]:
        """Extract message codes and field information from vector search results."""
        search_info = {
            'message_codes': [],
            'fields': [],
            'documentation': []
        }
        
        print(f"DEBUG: Parsing vector results: {vector_results[:200]}...")  # Show first 200 chars
        
        lines = vector_results.split('\n')
        print(f"DEBUG: Lines: {lines}")
        current_message = None
        current_documentation = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle numbered results: "1. **Message Code: GPS**"
            if line.startswith(('1.', '2.', '3.', '4.', '5.')) and '**Message Code:' in line:
                # Extract code from "1. **Message Code: GPS**"
                code_part = line.split('**Message Code:')[1]
                code = code_part.replace('**', '').strip()
                search_info['message_codes'].append(code)
                current_message = code
                current_documentation = []
                print(f"DEBUG: Extracted message code: {code}")
                
            # Handle message codes with asterisks: "**Message Code: GPS**"
            elif '**Message Code:' in line:
                # Extract code from "**Message Code: GPS**"
                code_part = line.split('**Message Code:')[1]
                code = code_part.replace('**', '').strip()
                search_info['message_codes'].append(code)
                current_message = code
                current_documentation = []
                print(f"DEBUG: Extracted message code: {code}")
                
            # Handle field information: "Field: Alt" or just "Alt"
            elif line.startswith('Field:') and current_message:
                field_name = line.split('Field:')[1].strip()
                search_info['fields'].append({
                    'message_code': current_message,
                    'field_info': field_name
                })
                print(f"DEBUG: Extracted field info: {field_name}")
                
            # Handle standalone field names (without "Field:" prefix)
            elif current_message and not line.startswith('Relevance:') and not line.startswith('Documentation:') and not line.startswith('Units:') and not line.startswith('Description:'):
                # Check if this looks like a field name (not a number, not empty)
                if line and not line.replace('.', '').replace('-', '').isdigit() and len(line) < 50:
                    # This might be a field name
                    search_info['fields'].append({
                        'message_code': current_message,
                        'field_info': line
                    })
                    print(f"DEBUG: Extracted potential field info: {line}")
                    
            # Handle documentation
            elif line.startswith('Documentation:') and current_message:
                doc = line.split('Documentation:')[1].strip()
                current_documentation.append(doc)
                
            # Handle description lines that are part of documentation
            elif line.startswith('Description:') and current_message:
                desc = line.split('Description:')[1].strip()
                current_documentation.append(desc)
                
            # Handle units lines that are part of documentation
            elif line.startswith('Units:') and current_message:
                units = line.split('Units:')[1].strip()
                current_documentation.append(f"Units: {units}")
        
        # Add accumulated documentation
        if current_message and current_documentation:
            search_info['documentation'].append({
                'message_code': current_message,
                'documentation': ' '.join(current_documentation)
            })
        
        # Fallback: if no message codes were extracted, try to find them in the raw text
        if not search_info['message_codes']:
            print("DEBUG: No message codes found in structured parsing, trying fallback extraction")
            import re
            # Look for patterns like "Message Code: GPS" or "GPS" in the text
            message_patterns = [
                r'Message Code:\s*(\w+)',
                r'\*\*Message Code:\s*(\w+)\*\*',
                r'(\b[A-Z]{2,}\b)'  # Look for uppercase words that might be message codes
            ]
            
            for pattern in message_patterns:
                matches = re.findall(pattern, vector_results)
                for match in matches:
                    if match not in search_info['message_codes']:
                        search_info['message_codes'].append(match)
                        print(f"DEBUG: Fallback extracted message code: {match}")
        
        print(f"DEBUG: Final extracted message codes: {search_info['message_codes']}")
        print(f"DEBUG: Final extracted fields: {[f['field_info'] for f in search_info['fields']]}")
        return search_info
    
    def _find_relevant_columns(self, df: pd.DataFrame, user_question: str, search_info: Dict[str, Any]) -> List[str]:
        """Find relevant columns with fallback strategy: semantic -> primary -> exact match."""
        all_columns = list(df.columns)
        relevant_columns = []
        
        # First attempt: Full semantic analysis (including cross-references)
        print("DEBUG: Attempting full semantic column matching")
        relevant_columns = self._find_semantic_columns(df, user_question, search_info)
        
        if relevant_columns:
            print(f"DEBUG: Found {len(relevant_columns)} semantic columns: {relevant_columns}")
            return relevant_columns
        
        # Second attempt: Primary columns only (remove cross-references)
        print("DEBUG: Semantic matching failed, trying primary columns only")
        relevant_columns = self._find_primary_columns_only(df, user_question, search_info)
        
        if relevant_columns:
            print(f"DEBUG: Found {len(relevant_columns)} primary columns: {relevant_columns}")
            return relevant_columns
        
        # Third attempt: Exact matches only
        print("DEBUG: Primary matching failed, trying exact matches only")
        relevant_columns = self._find_exact_match_columns_only(df, user_question, search_info)
        
        if relevant_columns:
            print(f"DEBUG: Found {len(relevant_columns)} exact match columns: {relevant_columns}")
            return relevant_columns
        
        print("DEBUG: No relevant columns found with any fallback strategy")
        return []

    def _find_semantic_columns(self, df: pd.DataFrame, user_question: str, search_info: Dict[str, Any]) -> List[str]:
        """Find columns using focused semantic analysis for GPS and UAV-specific queries."""
        all_columns = list(df.columns)
        relevant_columns = []
        
        # Step 1: Direct field matches from search results (highest priority)
        for field_info in search_info.get('fields', []):
            field_name = field_info.get('name', '').lower()
            for col in all_columns:
                if field_name == col.lower():  # Exact match only
                    relevant_columns.append(col)
                    print(f"DEBUG: Found exact match: {field_name} -> {col}")
        
        # Step 2: If no exact matches, look for GPS-specific semantic matches
        if not relevant_columns:
            gps_keywords = ['gps', 'satellite', 'signal', 'status', 'lock', 'fix', 'accuracy', 'dop', 'sats']
            user_question_lower = user_question.lower()
            
            # Check if this is a GPS-related query
            if any(keyword in user_question_lower for keyword in gps_keywords):
                # Look for GPS-specific columns
                gps_columns = [col for col in all_columns if any(gps_term in col.lower() for gps_term in gps_keywords)]
                relevant_columns.extend(gps_columns)
                print(f"DEBUG: Found GPS columns: {gps_columns}")
        
        # Step 3: Add essential time column if not present
        time_columns = [col for col in all_columns if 'time' in col.lower() or 'boot' in col.lower()]
        if time_columns and time_columns[0] not in relevant_columns:
            relevant_columns.append(time_columns[0])
            print(f"DEBUG: Added time column: {time_columns[0]}")
        
        # Step 4: Filter and limit results to prevent rate limiting
        filtered_columns = self._filter_inappropriate_columns(relevant_columns, df)
        
        # Step 5: Limit to maximum 5 columns to prevent rate limiting
        if len(filtered_columns) > 5:
            # Prioritize GPS-related columns
            gps_priority = [col for col in filtered_columns if any(gps_term in col.lower() for gps_term in ['gps', 'sat', 'signal', 'status'])]
            time_priority = [col for col in filtered_columns if 'time' in col.lower() or 'boot' in col.lower()]
            other_columns = [col for col in filtered_columns if col not in gps_priority and col not in time_priority]
            
            final_columns = []
            final_columns.extend(gps_priority[:3])  # Max 3 GPS columns
            final_columns.extend(time_priority[:1])  # Max 1 time column
            final_columns.extend(other_columns[:1])  # Max 1 other column
            
            filtered_columns = final_columns[:5]  # Ensure max 5 total
        
        print(f"DEBUG: Found {len(filtered_columns)} focused semantic columns: {filtered_columns}")
        return filtered_columns

    def _find_primary_columns_only(self, df: pd.DataFrame, user_question: str, search_info: Dict[str, Any]) -> List[str]:
        """Find only primary columns with focused GPS matching."""
        all_columns = list(df.columns)
        relevant_columns = []
        
        # Direct field matches from search results only
        for field_info in search_info.get('fields', []):
            field_name = field_info.get('name', '').lower()
            for col in all_columns:
                if field_name == col.lower():  # Exact match only
                    relevant_columns.append(col)
                    print(f"DEBUG: Found primary match: {field_name} -> {col}")
        
        # Add essential time column if GPS-related query
        user_question_lower = user_question.lower()
        gps_keywords = ['gps', 'satellite', 'signal', 'status', 'lock', 'fix', 'accuracy', 'dop', 'sats']
        if any(keyword in user_question_lower for keyword in gps_keywords):
            time_columns = [col for col in all_columns if 'time' in col.lower() or 'boot' in col.lower()]
            if time_columns and time_columns[0] not in relevant_columns:
                relevant_columns.append(time_columns[0])
                print(f"DEBUG: Added primary time column: {time_columns[0]}")
        
        # Filter and limit results
        filtered_columns = self._filter_inappropriate_columns(relevant_columns, df)
        
        # Limit to maximum 3 columns
        if len(filtered_columns) > 3:
            filtered_columns = filtered_columns[:3]
        
        print(f"DEBUG: Found {len(filtered_columns)} primary columns: {filtered_columns}")
        return filtered_columns

    def _find_exact_match_columns_only(self, df: pd.DataFrame, user_question: str, search_info: Dict[str, Any]) -> List[str]:
        """Find only exact match columns."""
        all_columns = list(df.columns)
        relevant_columns = []
        
        # Only exact field matches from search results
        for field_info in search_info.get('fields', []):
            field_name = field_info.get('name', '').lower()
            for col in all_columns:
                if field_name == col.lower():
                    relevant_columns.append(col)
                    print(f"DEBUG: Found exact match: {field_name} -> {col}")
        
        # Filter and limit results
        filtered_columns = self._filter_inappropriate_columns(relevant_columns, df)
        
        # Limit to maximum 2 columns
        if len(filtered_columns) > 2:
            filtered_columns = filtered_columns[:2]
        
        print(f"DEBUG: Found {len(filtered_columns)} exact match columns: {filtered_columns}")
        return filtered_columns

    def _filter_inappropriate_columns(self, columns: List[str], df: pd.DataFrame) -> List[str]:
        """Filter out columns that are unlikely to be useful for analysis."""
        inappropriate_patterns = [
            'name', 'id', 'index', 'row', 'column', 'label', 'title', 'description', 'comment',
            'note', 'remark', 'info', 'metadata', 'header', 'footer', 'summary'
        ]
        
        filtered = []
        for col in columns:
            col_lower = col.lower()
            
            # Skip columns that match inappropriate patterns
            if any(pattern in col_lower for pattern in inappropriate_patterns):
                print(f"DEBUG: Filtering out inappropriate column: {col}")
                continue
            
            # Skip columns that are mostly empty or have very low variance
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                if non_null_count < 2:  # Need at least 2 non-null values
                    print(f"DEBUG: Filtering out low-data column: {col} (only {non_null_count} non-null values)")
                    continue
                
                # For categorical columns, check if they have meaningful variance
                if not pd.api.types.is_numeric_dtype(df[col]):
                    unique_count = df[col].nunique()
                    if unique_count <= 1:  # Only one unique value
                        print(f"DEBUG: Filtering out low-variance categorical column: {col} (only {unique_count} unique values)")
                        continue
            
            filtered.append(col)
        
        return filtered
    
    def _find_cross_reference_columns(self, df: pd.DataFrame, user_question: str, primary_columns: List[str]) -> List[str]:
        """Find minimal cross-reference columns for GPS analysis."""
        cross_reference_columns = []
        all_columns = list(df.columns)
        user_question_lower = user_question.lower()
        
        # Only add essential cross-reference columns for GPS queries
        gps_keywords = ['gps', 'satellite', 'signal', 'status', 'lock', 'fix', 'accuracy', 'dop', 'sats']
        if any(keyword in user_question_lower for keyword in gps_keywords):
            # Add time column if not already present
            time_columns = [col for col in all_columns if 'time' in col.lower() or 'boot' in col.lower()]
            if time_columns and time_columns[0] not in primary_columns:
                cross_reference_columns.append(time_columns[0])
            
            # Add status column if not already present
            status_columns = [col for col in all_columns if 'status' in col.lower()]
            if status_columns and status_columns[0] not in primary_columns:
                cross_reference_columns.append(status_columns[0])
        
        # Limit cross-reference columns to prevent rate limiting
        if len(cross_reference_columns) > 2:
            cross_reference_columns = cross_reference_columns[:2]
        
        return cross_reference_columns

    def _find_matching_dataframes(self, dataframes: Dict[str, pd.DataFrame], search_info: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Find all dataframes that match the search criteria, handling indexed and non-indexed patterns."""
        matching_dataframes = {}
        
        for message_code in search_info['message_codes']:
            # Direct match (e.g., "GPS" matches "GPS")
            if message_code in dataframes:
                matching_dataframes[message_code] = dataframes[message_code]
            
            # Indexed matches (e.g., "GPS" matches "GPS[0]", "GPS[1]", etc.)
            for df_name, df in dataframes.items():
                # Check if df_name follows the pattern CODE[number]
                if '[' in df_name:
                    base_code = df_name.split('[')[0]
                    if base_code == message_code:
                        matching_dataframes[df_name] = df
                # Also check if df_name starts with the message_code (for partial matches)
                elif df_name.startswith(message_code):
                    matching_dataframes[df_name] = df
        
        return matching_dataframes

    def _aggregate_dataframe_analysis(self, dataframes: Dict[str, pd.DataFrame], user_question: str, search_info: Dict[str, Any]) -> str:
        """Aggregate analysis from multiple dataframes into a comprehensive result."""
        results = []
        
        # Find all matching dataframes (including indexed ones)
        matching_dataframes = self._find_matching_dataframes(dataframes, search_info)
        
        if not matching_dataframes:
            return f"No relevant data found. Available message codes: {list(dataframes.keys())}"
        
        results.append(f"**Analysis for: {user_question}**")
        results.append(f"Found {len(matching_dataframes)} relevant dataframes: {list(matching_dataframes.keys())}")
        results.append("")
        
        # Group dataframes by base code for better organization
        base_code_groups = {}
        for df_name, df in matching_dataframes.items():
            if '[' in df_name:
                base_code = df_name.split('[')[0]
            else:
                base_code = df_name
            
            if base_code not in base_code_groups:
                base_code_groups[base_code] = []
            base_code_groups[base_code].append((df_name, df))
        
        # Analyze each group
        for base_code, df_list in base_code_groups.items():
            results.append(f"**{base_code} Message Analysis:**")
            
            if len(df_list) == 1:
                # Single dataframe
                df_name, df = df_list[0]
                results.append(f"Dataframe: {df_name}")
                analysis = self._analyze_single_dataframe(df, user_question, search_info)
                results.append(analysis)
            else:
                # Multiple indexed dataframes
                results.append(f"Multiple dataframes found: {[name for name, _ in df_list]}")
                
                # Aggregate analysis across all dataframes
                aggregated_analysis = self._analyze_multiple_dataframes(df_list, user_question, search_info)
                results.append(aggregated_analysis)
            
            results.append("")
        
        return "\n".join(results)

    def _analyze_single_dataframe(self, df: pd.DataFrame, user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze a single dataframe with cross-referencing capabilities."""
        if df.empty:
            return "No data available in this dataframe."
        
        # Find relevant columns
        relevant_columns = self._find_relevant_columns(df, user_question, search_info)
        
        if relevant_columns:
            results = [f"Relevant columns: {', '.join(relevant_columns)}"]
            
            # Check if cross-referencing analysis is needed
            if len(relevant_columns) > 1:
                cross_analysis = self._analyze_cross_references(df, user_question, relevant_columns)
                if cross_analysis:
                    results.append("")
                    results.append(cross_analysis)
                    results.append("")
            
            # Also provide individual column analysis for context
            results.append("**Individual Column Analysis:**")
            for col in relevant_columns:
                if col in df.columns:
                    col_analysis = self._analyze_column(df, col, user_question)
                    results.append(col_analysis)
        else:
            # General analysis
            results = [
                f"Total records: {len(df)}",
                f"Available columns: {list(df.columns)}"
            ]
            
            # Show basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                results.append("Key metrics:")
                for col in numeric_cols[:5]:  # Show first 5 numeric columns
                    if col in df.columns:
                        results.append(f"  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
        
        return "\n".join(results)

    def _analyze_multiple_dataframes(self, df_list: List[tuple], user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze multiple dataframes and provide aggregated results with cross-referencing."""
        results = []
        
        # Collect all relevant columns across all dataframes
        all_relevant_columns = set()
        dataframe_summaries = []
        
        for df_name, df in df_list:
            if df.empty:
                continue
            
            # Find relevant columns for this dataframe
            relevant_columns = self._find_relevant_columns(df, user_question, search_info)
            all_relevant_columns.update(relevant_columns)
            
            # Create summary for this dataframe
            summary = f"  {df_name}: {len(df)} records"
            if relevant_columns:
                summary += f", relevant columns: {', '.join(relevant_columns[:3])}"  # Show first 3
            dataframe_summaries.append(summary)
        
        results.append("Dataframe summaries:")
        results.extend(dataframe_summaries)
        results.append("")
        
        # Cross-reference analysis across dataframes
        if len(df_list) > 1:
            cross_df_analysis = self._analyze_cross_dataframe_references(df_list, user_question, search_info)
            if cross_df_analysis:
                results.append("**Cross-Dataframe Analysis:**")
                results.append(cross_df_analysis)
                results.append("")
        
        # Aggregate analysis for common columns
        if all_relevant_columns:
            results.append("Aggregated analysis for common columns:")
            
            for col in sorted(all_relevant_columns):
                # Find all dataframes that have this column
                col_dataframes = [(name, df) for name, df in df_list if col in df.columns and not df.empty]
                
                if col_dataframes:
                    # Aggregate statistics across all dataframes
                    all_values = []
                    for _, df in col_dataframes:
                        all_values.extend(df[col].dropna().tolist())
                    
                    if all_values:
                        import numpy as np
                        all_values = np.array(all_values)
                        results.append(f"  {col}:")
                        results.append(f"    - Combined records: {len(all_values)}")
                        results.append(f"    - Min: {all_values.min():.2f}")
                        results.append(f"    - Max: {all_values.max():.2f}")
                        results.append(f"    - Mean: {all_values.mean():.2f}")
                        results.append(f"    - Found in {len(col_dataframes)} dataframes")
        
        return "\n".join(results)
    
    def _analyze_cross_dataframe_references(self, df_list: List[tuple], user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze cross-references between multiple dataframes."""
        results = []
        user_question_lower = user_question.lower()
        
        # Check if this is a temporal event query that spans multiple dataframes
        if any(term in user_question_lower for term in ['when', 'first', 'last', 'instance', 'occurrence']):
            temporal_analysis = self._analyze_cross_dataframe_temporal_events(df_list, user_question, search_info)
            if temporal_analysis:
                results.append(temporal_analysis)
        
        # Check if this is a state change query across dataframes
        if any(term in user_question_lower for term in ['lost', 'gained', 'changed', 'status', 'state']):
            state_analysis = self._analyze_cross_dataframe_state_changes(df_list, user_question, search_info)
            if state_analysis:
                results.append(state_analysis)
        
        return "\n\n".join(results)
    
    def _analyze_cross_dataframe_temporal_events(self, df_list: List[tuple], user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze temporal events across multiple dataframes."""
        results = []
        
        # Find time columns across all dataframes
        all_time_columns = []
        for df_name, df in df_list:
            time_columns = [col for col in df.columns if any(time_term in col.lower() for time_term in ['time', 'boot', 'ms', 'timestamp'])]
            for time_col in time_columns:
                all_time_columns.append((df_name, time_col))
        
        if not all_time_columns:
            return ""
        
        results.append("**Cross-Dataframe Temporal Analysis:**")
        
        # Find the earliest event across all dataframes
        earliest_events = []
        for df_name, df in df_list:
            time_columns = [col for col in df.columns if any(time_term in col.lower() for time_term in ['time', 'boot', 'ms', 'timestamp'])]
            
            if time_columns:
                time_col = time_columns[0]
                min_time = df[time_col].min()
                earliest_events.append((df_name, time_col, min_time))
        
        if earliest_events:
            # Sort by time to find the earliest
            earliest_events.sort(key=lambda x: x[2])
            earliest_df, earliest_time_col, earliest_time = earliest_events[0]
            
            results.append(f"Earliest timestamp across all dataframes:")
            results.append(f"- Dataframe: {earliest_df}")
            results.append(f"- Time column: {earliest_time_col}")
            results.append(f"- Earliest time: {earliest_time}")
        
        return "\n".join(results)
    
    def _analyze_cross_dataframe_state_changes(self, df_list: List[tuple], user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze state changes across multiple dataframes."""
        results = []
        
        # Find state/status columns across all dataframes
        all_state_changes = []
        
        for df_name, df in df_list:
            state_columns = [col for col in df.columns if any(state_term in col.lower() for state_term in ['status', 'error', 'flag', 'state', 'signal'])]
            
            for state_col in state_columns:
                if len(df) > 1:
                    state_changes = df[state_col] != df[state_col].shift(1)
                    change_indices = state_changes[state_changes].index
                    
                    if len(change_indices) > 0:
                        # Get timestamp if available
                        time_col = None
                        for time_term in ['time_boot_ms', 'time', 'timestamp']:
                            if time_term in df.columns:
                                time_col = time_term
                                break
                        
                        for idx in change_indices[:3]:  # First 3 changes per column
                            new_state = df.loc[idx, state_col]
                            previous_state = df.loc[idx - 1, state_col] if idx > 0 else "N/A"
                            timestamp = df.loc[idx, time_col] if time_col else "N/A"
                            
                            all_state_changes.append({
                                'df_name': df_name,
                                'column': state_col,
                                'timestamp': timestamp,
                                'previous_state': previous_state,
                                'new_state': new_state,
                                'index': idx
                            })
        
        if all_state_changes:
            results.append("**Cross-Dataframe State Changes:**")
            results.append(f"Found {len(all_state_changes)} state changes across all dataframes")
            
            # Sort by timestamp if available
            timestamped_changes = [c for c in all_state_changes if c['timestamp'] != "N/A"]
            if timestamped_changes:
                # Sort by timestamp (assuming numeric timestamps)
                try:
                    timestamped_changes.sort(key=lambda x: float(x['timestamp']) if isinstance(x['timestamp'], (int, float)) else 0)
                except:
                    pass  # If sorting fails, keep original order
                
                results.append("First few state changes (chronological):")
                for i, change in enumerate(timestamped_changes[:5]):
                    results.append(f"- {change['df_name']}.{change['column']}: '{change['previous_state']}' → '{change['new_state']}' at {change['timestamp']}")
        
        return "\n".join(results)

    def _semantic_dataframe_analysis(self, dataframes: Dict[str, pd.DataFrame], search_info: Dict[str, Any], user_question: str) -> str:
        """Analyze dataframes semantically with focused GPS analysis to prevent rate limiting."""
        try:
            # Find matching dataframes
            matching_dataframes = self._find_matching_dataframes(dataframes, search_info)
            
            if not matching_dataframes:
                return "No relevant GPS data found in the log files."
            
            # Focus on GPS-specific analysis
            user_question_lower = user_question.lower()
            gps_keywords = ['gps', 'satellite', 'signal', 'status', 'lock', 'fix', 'accuracy', 'dop', 'sats']
            
            if any(keyword in user_question_lower for keyword in gps_keywords):
                return self._analyze_gps_specific_data(matching_dataframes, user_question, search_info)
            else:
                # For non-GPS queries, use focused analysis
                return self._analyze_focused_data(matching_dataframes, user_question, search_info)
            
        except Exception as e:
            print(f"DEBUG: Error in semantic dataframe analysis: {str(e)}")
            return "I encountered an error while analyzing the data. Please try rephrasing your question."

    def _analyze_gps_specific_data(self, dataframes: Dict[str, pd.DataFrame], user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze GPS-specific data with minimal context to prevent rate limiting."""
        results = []
        
        for df_name, df in dataframes.items():
            # Find GPS-specific columns only
            gps_columns = [col for col in df.columns if any(term in col.lower() for term in ['gps', 'sat', 'signal', 'status', 'lock', 'fix', 'accuracy', 'dop', 'sats'])]
            time_columns = [col for col in df.columns if 'time' in col.lower() or 'boot' in col.lower()]
            
            if not gps_columns:
                continue
            
            # Limit to most relevant columns
            relevant_columns = gps_columns[:3]  # Max 3 GPS columns
            if time_columns and time_columns[0] not in relevant_columns:
                relevant_columns.append(time_columns[0])
            
            # Analyze each GPS column
            for col in relevant_columns[:3]:  # Limit analysis to 3 columns
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        analysis = self._analyze_gps_numeric_column(df, col, user_question)
                        if analysis:
                            results.append(f"**{df_name}.{col}:** {analysis}")
                    else:
                        analysis = self._analyze_gps_categorical_column(df, col, user_question)
                        if analysis:
                            results.append(f"**{df_name}.{col}:** {analysis}")
                except Exception as e:
                    print(f"DEBUG: Error analyzing column {col}: {str(e)}")
                    continue
        
        if not results:
            return "No GPS data could be analyzed. The GPS columns may be empty or contain incompatible data types."
        
        return "\n\n".join(results)

    def _analyze_gps_numeric_column(self, df: pd.DataFrame, column: str, user_question: str) -> str:
        """Analyze GPS numeric column with minimal output."""
        if df[column].isnull().all():
            return "No data available"
        
        user_question_lower = user_question.lower()
        
        # Focus on specific GPS queries
        if any(term in user_question_lower for term in ['lost', 'first', 'when']):
            # Look for signal loss events
            if 'time' in df.columns or 'time_boot_ms' in df.columns:
                time_col = 'time' if 'time' in df.columns else 'time_boot_ms'
                return self._find_gps_signal_loss(df, column, time_col, user_question)
        
        # Basic statistics
        try:
            current = df[column].iloc[-1]
            max_val = df[column].max()
            min_val = df[column].min()
            
            if any(term in user_question_lower for term in ['maximum', 'max', 'highest']):
                return f"Maximum: {max_val:.2f}, Current: {current:.2f}"
            elif any(term in user_question_lower for term in ['minimum', 'min', 'lowest']):
                return f"Minimum: {min_val:.2f}, Current: {current:.2f}"
            else:
                return f"Current: {current:.2f}, Range: {min_val:.2f} to {max_val:.2f}"
        except:
            return "Data analysis failed"

    def _analyze_gps_categorical_column(self, df: pd.DataFrame, column: str, user_question: str) -> str:
        """Analyze GPS categorical column with minimal output."""
        if df[column].isnull().all():
            return "No data available"
        
        try:
            current = df[column].iloc[-1]
            unique_values = df[column].value_counts().head(3).to_dict()
            
            if any(term in user_question_lower for term in ['lost', 'first', 'when']):
                # Look for status changes
                if 'time' in df.columns or 'time_boot_ms' in df.columns:
                    time_col = 'time' if 'time' in df.columns else 'time_boot_ms'
                    return self._find_gps_status_change(df, column, time_col, user_question)
            
            return f"Current: {current}, Common values: {list(unique_values.keys())}"
        except:
            return "Data analysis failed"

    def _find_gps_signal_loss(self, df: pd.DataFrame, data_col: str, time_col: str, user_question: str) -> str:
        """Find GPS signal loss events with minimal analysis."""
        try:
            # Look for significant drops in GPS values
            if df[data_col].dtype in ['int64', 'float64']:
                # Find first significant drop (e.g., below 50% of max)
                max_val = df[data_col].max()
                threshold = max_val * 0.5
                
                # Find first occurrence below threshold
                loss_indices = df[df[data_col] < threshold].index
                if len(loss_indices) > 0:
                    first_loss_idx = loss_indices[0]
                    first_loss_time = df.loc[first_loss_idx, time_col]
                    first_loss_value = df.loc[first_loss_idx, data_col]
                    return f"First signal loss at time {first_loss_time} (value: {first_loss_value:.2f})"
            
            return "No clear signal loss detected"
        except:
            return "Signal loss analysis failed"

    def _find_gps_status_change(self, df: pd.DataFrame, data_col: str, time_col: str, user_question: str) -> str:
        """Find GPS status changes with minimal analysis."""
        try:
            # Look for status changes
            status_values = df[data_col].unique()
            if len(status_values) > 1:
                # Find first occurrence of each status
                first_occurrences = {}
                for status in status_values:
                    first_idx = df[df[data_col] == status].index[0]
                    first_occurrences[status] = df.loc[first_idx, time_col]
                
                return f"Status changes: {first_occurrences}"
            
            return f"Status stable: {df[data_col].iloc[0]}"
        except:
            return "Status change analysis failed"

    def _analyze_focused_data(self, dataframes: Dict[str, pd.DataFrame], user_question: str, search_info: Dict[str, Any]) -> str:
        """Analyze non-GPS data with focused approach."""
        results = []
        
        for df_name, df in dataframes.items():
            # Find relevant columns based on search info
            relevant_columns = self._find_primary_columns_only(df, user_question, search_info)
            
            if not relevant_columns:
                continue
            
            # Analyze each relevant column
            for col in relevant_columns[:2]:  # Limit to 2 columns
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        analysis = self._analyze_gps_numeric_column(df, col, user_question)
                        if analysis:
                            results.append(f"**{df_name}.{col}:** {analysis}")
                except Exception as e:
                    print(f"DEBUG: Error analyzing column {col}: {str(e)}")
                    continue
        
        if not results:
            return "No relevant data could be analyzed."
        
        return "\n\n".join(results)

    def _semantic_dataframe_analysis_primary(self, dataframes: Dict[str, pd.DataFrame], search_info: Dict[str, Any], user_question: str) -> str:
        """Primary semantic dataframe analysis method - now uses focused approach."""
        return self._semantic_dataframe_analysis(dataframes, search_info, user_question)

    def _semantic_dataframe_analysis_fallback(self, dataframes: Dict[str, pd.DataFrame], search_info: Dict[str, Any], user_question: str) -> str:
        """Fallback analysis with minimal data to prevent rate limiting."""
        try:
            results = []
            
            for df_name, df in dataframes.items():
                # Only analyze essential columns
                essential_cols = [col for col in df.columns if any(term in col.lower() for term in ['time', 'status', 'gps', 'signal'])]
                
                if not essential_cols:
                    continue
                
                # Limit to 2 essential columns
                essential_cols = essential_cols[:2]
                
                for col in essential_cols:
                    try:
                        if df[col].dtype in ['int64', 'float64']:
                            current = df[col].iloc[-1]
                            results.append(f"**{df_name}.{col}:** Current value: {current:.2f}")
                        else:
                            current = df[col].iloc[-1]
                            results.append(f"**{df_name}.{col}:** Current value: {current}")
                    except:
                        continue
            
            return "\n\n".join(results) if results else ""
            
        except Exception as e:
            print(f"DEBUG: Error in fallback analysis: {str(e)}")
            return ""

    def _semantic_dataframe_analysis_broader(self, dataframes: Dict[str, pd.DataFrame], user_question: str) -> str:
        """Broader analysis with minimal output to prevent rate limiting."""
        try:
            results = []
            
            for df_name, df in dataframes.items():
                # Only provide basic info
                total_rows = len(df)
                
                # Find one relevant column
                relevant_cols = [col for col in df.columns if any(term in col.lower() for term in ['time', 'status', 'gps', 'signal'])]
                
                if relevant_cols:
                    col = relevant_cols[0]
                    try:
                        current = df[col].iloc[-1]
                        results.append(f"**{df_name}:** {total_rows} rows, {col}: {current}")
                    except:
                        results.append(f"**{df_name}:** {total_rows} rows")
                else:
                    results.append(f"**{df_name}:** {total_rows} rows")
            
            return "\n\n".join(results) if results else ""
            
        except Exception as e:
            print(f"DEBUG: Error in broader analysis: {str(e)}")
            return ""

    def _analyze_column(self, df: pd.DataFrame, column: str, user_question: str) -> str:
        """Analyze a specific column based on user question and provide direct answers."""
        if column not in df.columns:
            return f"Column '{column}' not found in dataframe."
        
        analysis = []
        user_question_lower = user_question.lower()
        
        # Check if dataframe has data
        if df.empty or df[column].isnull().all():
            return f"No data available in column '{column}'."
        
        # Provide direct answers based on user question
        if df[column].dtype in ['int64', 'float64']:
            # Handle specific queries
            if any(term in user_question_lower for term in ['maximum', 'max', 'highest']):
                max_val = df[column].max()
                analysis.append(f"**Maximum {column}: {max_val:.2f}**")
                
                # Additional context
                min_val = df[column].min()
                mean_val = df[column].mean()
                current_val = df[column].iloc[-1]
                analysis.append(f"- Range: {min_val:.2f} to {max_val:.2f}")
                analysis.append(f"- Average: {mean_val:.2f}")
                analysis.append(f"- Current: {current_val:.2f}")
                
            elif any(term in user_question_lower for term in ['minimum', 'min', 'lowest']):
                min_val = df[column].min()
                analysis.append(f"**Minimum {column}: {min_val:.2f}**")
                
                # Additional context
                max_val = df[column].max()
                mean_val = df[column].mean()
                current_val = df[column].iloc[-1]
                analysis.append(f"- Range: {min_val:.2f} to {max_val:.2f}")
                analysis.append(f"- Average: {mean_val:.2f}")
                analysis.append(f"- Current: {current_val:.2f}")
                
            elif any(term in user_question_lower for term in ['average', 'mean', 'typical']):
                mean_val = df[column].mean()
                analysis.append(f"**Average {column}: {mean_val:.2f}**")
                
                # Additional context
                min_val = df[column].min()
                max_val = df[column].max()
                std_val = df[column].std()
                analysis.append(f"- Range: {min_val:.2f} to {max_val:.2f}")
                analysis.append(f"- Standard deviation: {std_val:.2f}")
                
            elif any(term in user_question_lower for term in ['current', 'latest', 'now']):
                current_val = df[column].iloc[-1]
                analysis.append(f"**Current {column}: {current_val:.2f}**")
                
                # Additional context
                min_val = df[column].min()
                max_val = df[column].max()
                mean_val = df[column].mean()
                analysis.append(f"- Range: {min_val:.2f} to {max_val:.2f}")
                analysis.append(f"- Average: {mean_val:.2f}")
                
            else:
                # General analysis for other queries
                analysis.append(f"**{column} Analysis:**")
                min_val = df[column].min()
                max_val = df[column].max()
                mean_val = df[column].mean()
                std_val = df[column].std()
                current_val = df[column].iloc[-1]
                
                analysis.append(f"- Min: {min_val:.2f}")
                analysis.append(f"- Max: {max_val:.2f}")
                analysis.append(f"- Mean: {mean_val:.2f}")
                analysis.append(f"- Std: {std_val:.2f}")
                analysis.append(f"- Current: {current_val:.2f}")
            
            # Check for anomalies
            anomalies = self._detect_anomalies(df, column, user_question)
            if anomalies:
                analysis.append(f"- Anomalies: {anomalies}")
                
        else:
            # For non-numeric columns
            analysis.append(f"**{column} Analysis:**")
            analysis.append(f"- Data type: {df[column].dtype}")
            analysis.append(f"- Unique values: {df[column].nunique()}")
            analysis.append(f"- Most common: {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}")
        
        return "\n".join(analysis)
    
    def _detect_anomalies(self, df: pd.DataFrame, column: str, user_question: str) -> str:
        """Detect anomalies based on data characteristics rather than hardcoded rules."""
        if df[column].dtype not in ['int64', 'float64']:
            return ""
        
        anomalies = []
        
        # Statistical outliers using IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
        if len(outliers) > 0:
            anomalies.append(f"{len(outliers)} statistical outliers detected")
        
        # Sudden changes (adaptive threshold based on data variance)
        if len(df) > 1:
            changes = df[column].diff().abs()
            # Use 2 standard deviations as threshold for sudden changes
            threshold = changes.mean() + 2 * changes.std()
            sudden_changes = changes[changes > threshold]
            if len(sudden_changes) > 0:
                anomalies.append(f"{len(sudden_changes)} sudden changes detected")
        
        # Missing or invalid data
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            anomalies.append(f"{missing_count} missing values detected")
        
        # Zero or constant values (potential sensor issues)
        unique_values = df[column].nunique()
        if unique_values == 1:
            anomalies.append("Constant value detected (potential sensor issue)")
        elif unique_values < 5 and len(df) > 10:
            anomalies.append(f"Very low variance detected ({unique_values} unique values)")
        
        return "; ".join(anomalies) if anomalies else ""

    def _analyze_cross_references(self, df: pd.DataFrame, user_question: str, relevant_columns: List[str]) -> str:
        """Analyze cross-references between columns to provide comprehensive insights."""
        if len(relevant_columns) < 2:
            return ""  # No cross-referencing needed
        
        analysis_results = []
        user_question_lower = user_question.lower()
        
        # Check if this is a temporal event query
        if any(term in user_question_lower for term in ['when', 'first', 'last', 'instance', 'occurrence']):
            temporal_analysis = self._analyze_temporal_events(df, user_question, relevant_columns)
            if temporal_analysis:
                analysis_results.append(temporal_analysis)
        
        # Check if this is a state change query
        if any(term in user_question_lower for term in ['lost', 'gained', 'changed', 'status', 'state']):
            state_analysis = self._analyze_state_changes(df, user_question, relevant_columns)
            if state_analysis:
                analysis_results.append(state_analysis)
        
        # Check if this is a correlation query
        if any(term in user_question_lower for term in ['correlation', 'relationship', 'between', 'affect', 'impact']):
            correlation_analysis = self._analyze_correlations(df, user_question, relevant_columns)
            if correlation_analysis:
                analysis_results.append(correlation_analysis)
        
        # If no specific analysis was triggered, do general cross-reference analysis
        if not analysis_results:
            general_analysis = self._analyze_general_cross_references(df, user_question, relevant_columns)
            if general_analysis:
                analysis_results.append(general_analysis)
        
        return "\n\n".join(analysis_results) if analysis_results else ""
    
    def _analyze_temporal_events(self, df: pd.DataFrame, user_question: str, relevant_columns: List[str]) -> str:
        """Analyze temporal events by cross-referencing time columns with other data."""
        results = []
        
        # Find time column
        time_columns = [col for col in relevant_columns if any(time_term in col.lower() for time_term in ['time', 'boot', 'ms', 'timestamp'])]
        if not time_columns:
            return ""
        
        time_col = time_columns[0]  # Use first time column found
        other_columns = [col for col in relevant_columns if col != time_col]
        
        if not other_columns:
            return ""
        
        results.append(f"**Temporal Event Analysis:**")
        results.append(f"Using time column: {time_col}")
        
        # Analyze each other column for events
        for col in other_columns:
            if col not in df.columns:
                continue
            
            col_analysis = self._find_temporal_events_in_column(df, time_col, col, user_question)
            if col_analysis:
                results.append(col_analysis)
        
        return "\n".join(results)
    
    def _find_temporal_events_in_column(self, df: pd.DataFrame, time_col: str, data_col: str, user_question: str) -> str:
        """Find temporal events in a specific column."""
        try:
            # Check if columns exist
            if time_col not in df.columns or data_col not in df.columns:
                return ""
            
            # Check if columns have data
            if df[time_col].isnull().all() or df[data_col].isnull().all():
                return ""
            
            # Ensure time column is numeric for comparisons
            if not pd.api.types.is_numeric_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                except:
                    return ""  # Skip if time column can't be converted to numeric
            
            # Check data column type and analyze accordingly
            if pd.api.types.is_numeric_dtype(df[data_col]):
                return self._find_numeric_temporal_events(df, time_col, data_col, user_question)
            else:
                return self._find_categorical_temporal_events(df, time_col, data_col, user_question)
        except Exception as e:
            print(f"DEBUG: Error in temporal event analysis for {data_col}: {str(e)}")
            return ""
    
    def _find_numeric_temporal_events(self, df: pd.DataFrame, time_col: str, data_col: str, user_question: str) -> str:
        """Find temporal events in numeric columns with improved type safety."""
        try:
            user_question_lower = user_question.lower()
            results = []
            
            # Ensure both columns are numeric with proper error handling
            if not pd.api.types.is_numeric_dtype(df[data_col]):
                try:
                    df[data_col] = pd.to_numeric(df[data_col], errors='coerce')
                except:
                    return ""  # Skip if can't convert to numeric
            
            if not pd.api.types.is_numeric_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                except:
                    return ""  # Skip if can't convert to numeric
            
            # Remove rows with NaN values for analysis
            clean_df = df[[time_col, data_col]].dropna()
            if len(clean_df) < 2:
                return ""
            
            # Ensure we have numeric data after cleaning
            if not pd.api.types.is_numeric_dtype(clean_df[data_col]) or not pd.api.types.is_numeric_dtype(clean_df[time_col]):
                return ""
            
            # Look for significant changes
            try:
                changes = clean_df[data_col].diff().abs()
                if len(changes.dropna()) > 0:
                    threshold = changes.mean() + 2 * changes.std()  # 2 standard deviations
                    significant_changes = changes[changes > threshold]
                    
                    if len(significant_changes) > 0:
                        # Find the first significant change
                        first_change_idx = significant_changes.index[0]
                        first_change_time = clean_df.loc[first_change_idx, time_col]
                        first_change_value = clean_df.loc[first_change_idx, data_col]
                        previous_value = clean_df.loc[first_change_idx - 1, data_col] if first_change_idx > clean_df.index[0] else "N/A"
                        
                        results.append(f"**{data_col} - First Significant Change:**")
                        results.append(f"- Time: {first_change_time}")
                        results.append(f"- Value changed from {previous_value} to {first_change_value}")
                        results.append(f"- Change magnitude: {significant_changes.iloc[0]:.2f}")
                        results.append(f"- Total significant changes: {len(significant_changes)}")
            except Exception as e:
                print(f"DEBUG: Error in change detection: {str(e)}")
            
            # Look for extreme values
            try:
                if any(term in user_question_lower for term in ['maximum', 'max', 'highest']):
                    max_idx = clean_df[data_col].idxmax()
                    max_time = clean_df.loc[max_idx, time_col]
                    max_value = clean_df.loc[max_idx, data_col]
                    
                    results.append(f"**{data_col} - Maximum Value:**")
                    results.append(f"- Time: {max_time}")
                    results.append(f"- Value: {max_value}")
                
                elif any(term in user_question_lower for term in ['minimum', 'min', 'lowest']):
                    min_idx = clean_df[data_col].idxmin()
                    min_time = clean_df.loc[min_idx, time_col]
                    min_value = clean_df.loc[min_idx, data_col]
                    
                    results.append(f"**{data_col} - Minimum Value:**")
                    results.append(f"- Time: {min_time}")
                    results.append(f"- Value: {min_value}")
            except Exception as e:
                print(f"DEBUG: Error in extreme value detection: {str(e)}")
            
            return "\n".join(results) if results else ""
        except Exception as e:
            print(f"DEBUG: Error in numeric temporal event analysis: {str(e)}")
            return ""
    
    def _find_categorical_temporal_events(self, df: pd.DataFrame, time_col: str, data_col: str, user_question: str) -> str:
        """Find temporal events in categorical columns with improved type safety."""
        try:
            user_question_lower = user_question.lower()
            results = []
            
            # Ensure time column is numeric
            if not pd.api.types.is_numeric_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                except:
                    return ""
            
            # Remove rows with NaN values
            clean_df = df[[time_col, data_col]].dropna()
            if len(clean_df) < 2:
                return ""
            
            # Ensure time column is still numeric after cleaning
            if not pd.api.types.is_numeric_dtype(clean_df[time_col]):
                return ""
            
            # Look for state changes
            try:
                state_changes = clean_df[data_col] != clean_df[data_col].shift(1)
                change_indices = state_changes[state_changes].index
                
                if len(change_indices) > 0:
                    # Find first state change
                    first_change_idx = change_indices[0]
                    first_change_time = clean_df.loc[first_change_idx, time_col]
                    new_state = clean_df.loc[first_change_idx, data_col]
                    previous_state = clean_df.loc[first_change_idx - 1, data_col] if first_change_idx > clean_df.index[0] else "N/A"
                    
                    results.append(f"**{data_col} - First State Change:**")
                    results.append(f"- Time: {first_change_time}")
                    results.append(f"- State changed from '{previous_state}' to '{new_state}'")
                    results.append(f"- Total state changes: {len(change_indices)}")
                    
                    # Look for specific state changes based on query (using semantic matching)
                    if any(term in user_question_lower for term in ['lost', 'error', 'fail', 'bad', 'off', 'disconnect']):
                        # Use semantic matching to find error-like states
                        error_patterns = ['error', 'fail', 'lost', 'bad', 'off', 'disconnect', '0', 'false', 'none']
                        error_states = clean_df[clean_df[data_col].astype(str).str.lower().str.contains('|'.join(error_patterns), na=False)]
                        
                        if len(error_states) > 0:
                            first_error_idx = error_states.index[0]
                            first_error_time = clean_df.loc[first_error_idx, time_col]
                            error_state = clean_df.loc[first_error_idx, data_col]
                            
                            results.append(f"**{data_col} - First Error State:**")
                            results.append(f"- Time: {first_error_time}")
                            results.append(f"- Error state: '{error_state}'")
            except Exception as e:
                print(f"DEBUG: Error in state change detection: {str(e)}")
            
            return "\n".join(results) if results else ""
        except Exception as e:
            print(f"DEBUG: Error in categorical temporal event analysis: {str(e)}")
            return ""
    
    def _analyze_state_changes(self, df: pd.DataFrame, user_question: str, relevant_columns: List[str]) -> str:
        """Analyze state changes across multiple columns."""
        results = []
        
        # Find status/state columns
        state_columns = [col for col in relevant_columns if any(state_term in col.lower() for state_term in ['status', 'error', 'flag', 'state', 'signal'])]
        
        if not state_columns:
            return ""
        
        results.append(f"**State Change Analysis:**")
        
        for col in state_columns:
            if col not in df.columns:
                continue
            
            col_analysis = self._analyze_single_state_column(df, col, user_question)
            if col_analysis:
                results.append(col_analysis)
        
        return "\n\n".join(results)
    
    def _analyze_single_state_column(self, df: pd.DataFrame, col: str, user_question: str) -> str:
        """Analyze state changes in a single column."""
        if len(df) <= 1:
            return ""
        
        results = []
        state_changes = df[col] != df[col].shift(1)
        change_indices = state_changes[state_changes].index
        
        if len(change_indices) == 0:
            return ""
        
        results.append(f"**{col} State Changes:**")
        results.append(f"- Total state changes: {len(change_indices)}")
        
        # Show first few state changes
        for i, idx in enumerate(change_indices[:5]):  # Show first 5 changes
            new_state = df.loc[idx, col]
            previous_state = df.loc[idx - 1, col] if idx > 0 else "N/A"
            
            # Try to get timestamp if available
            time_col = None
            for time_term in ['time_boot_ms', 'time', 'timestamp']:
                if time_term in df.columns:
                    time_col = time_term
                    break
            
            if time_col:
                timestamp = df.loc[idx, time_col]
                results.append(f"- Change {i+1}: '{previous_state}' → '{new_state}' at {timestamp}")
            else:
                results.append(f"- Change {i+1}: '{previous_state}' → '{new_state}'")
        
        if len(change_indices) > 5:
            results.append(f"- ... and {len(change_indices) - 5} more changes")
        
        return "\n".join(results)
    
    def _analyze_correlations(self, df: pd.DataFrame, user_question: str, relevant_columns: List[str]) -> str:
        """Analyze correlations between numeric columns."""
        results = []
        
        # Find numeric columns
        numeric_columns = [col for col in relevant_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_columns) < 2:
            return ""
        
        results.append(f"**Correlation Analysis:**")
        
        # Calculate correlations between all pairs
        correlations = []
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                if col1 != col2:
                    corr = df[col1].corr(df[col2])
                    if not pd.isna(corr):
                        correlations.append((col1, col2, corr))
        
        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Show top correlations
        for col1, col2, corr in correlations[:5]:  # Show top 5 correlations
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            direction = "positive" if corr > 0 else "negative"
            results.append(f"- {col1} ↔ {col2}: {corr:.3f} ({strength} {direction} correlation)")
        
        return "\n".join(results)
    
    def _analyze_general_cross_references(self, df: pd.DataFrame, user_question: str, relevant_columns: List[str]) -> str:
        """General cross-reference analysis when no specific pattern is detected."""
        results = []
        
        if len(relevant_columns) < 2:
            return ""
        
        results.append(f"**Cross-Reference Analysis:**")
        results.append(f"Analyzing relationships between: {', '.join(relevant_columns)}")
        
        # Find time column for temporal context
        time_columns = [col for col in relevant_columns if any(time_term in col.lower() for time_term in ['time', 'boot', 'ms', 'timestamp'])]
        
        if time_columns:
            time_col = time_columns[0]
            results.append(f"Using time reference: {time_col}")
            
            # Analyze each column with temporal context
            for col in relevant_columns:
                if col != time_col and col in df.columns:
                    col_summary = self._summarize_column_with_time(df, time_col, col)
                    if col_summary:
                        results.append(col_summary)
        
        return "\n".join(results)
    
    def _summarize_column_with_time(self, df: pd.DataFrame, time_col: str, data_col: str) -> str:
        """Summarize a column with temporal context."""
        if df[data_col].dtype in ['int64', 'float64']:
            # Numeric column
            min_idx = df[data_col].idxmin()
            max_idx = df[data_col].idxmin()
            
            min_time = df.loc[min_idx, time_col]
            max_time = df.loc[max_idx, time_col]
            min_val = df.loc[min_idx, data_col]
            max_val = df.loc[max_idx, data_col]
            
            return f"- {data_col}: min={min_val} at {min_time}, max={max_val} at {max_time}"
        else:
            # Categorical column
            unique_states = df[data_col].nunique()
            most_common = df[data_col].mode().iloc[0] if not df[data_col].mode().empty else "N/A"
            
            return f"- {data_col}: {unique_states} unique states, most common: '{most_common}'"

    def _analyze_columns_general(self, df: pd.DataFrame, user_question: str, relevant_columns: List[str]) -> str:
        """General analysis of relevant columns with type safety."""
        try:
            if not relevant_columns:
                return ""
            
            results = []
            user_question_lower = user_question.lower()
            
            # Filter columns for type safety
            type_safe_columns = self._filter_columns_for_type_safety(df, relevant_columns, user_question)
            
            # If no type-safe columns found, try with original columns but be more careful
            if not type_safe_columns:
                print(f"DEBUG: No type-safe columns found, trying with original columns: {relevant_columns}")
                type_safe_columns = relevant_columns
            
            for col in type_safe_columns:
                if col not in df.columns:
                    continue
                
                try:
                    col_analysis = self._analyze_single_column_safe(df, col, user_question_lower)
                    if col_analysis:
                        results.append(col_analysis)
                except Exception as e:
                    print(f"DEBUG: Error analyzing column {col}: {str(e)}")
                    continue
            
            return "\n\n".join(results) if results else "Unable to analyze any columns due to data type incompatibilities."
            
        except Exception as e:
            print(f"DEBUG: Error in general column analysis: {str(e)}")
            return ""

    def _filter_columns_for_type_safety(self, df: pd.DataFrame, columns: List[str], user_question: str) -> List[str]:
        """Filter columns to ensure type safety for the specific query."""
        if not columns:
            return []
        
        user_question_lower = user_question.lower()
        type_safe_columns = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Check if column has data
            if df[col].isnull().all():
                continue
            
            # Determine if this column is safe for the query
            if self._is_column_safe_for_query(df, col, user_question_lower):
                type_safe_columns.append(col)
        
        return type_safe_columns

    def _is_column_safe_for_query(self, df: pd.DataFrame, col: str, user_question_lower: str) -> bool:
        """Check if a column is safe to use for the specific query."""
        try:
            # Check if query requires numeric comparisons
            numeric_comparison_terms = ['maximum', 'max', 'minimum', 'min', 'average', 'mean', 'greater', 'less', 'higher', 'lower', '>', '<', '=', 'compare']
            requires_numeric = any(term in user_question_lower for term in numeric_comparison_terms)
            
            if requires_numeric:
                # For numeric comparisons, ensure column is numeric or convertible
                if pd.api.types.is_numeric_dtype(df[col]):
                    return True
                else:
                    # Try to convert to numeric - be more permissive
                    try:
                        # Only try conversion if the column has some numeric-like values
                        sample_values = df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            # Check if at least some values look numeric
                            numeric_like = 0
                            for val in sample_values:
                                try:
                                    float(str(val))
                                    numeric_like += 1
                                except:
                                    pass
                            
                            # If more than 50% look numeric, allow the column
                            if numeric_like / len(sample_values) > 0.5:
                                return True
                    except:
                        pass
                    
                    # For altitude, GPS, and other common UAV fields, be more permissive
                    altitude_terms = ['alt', 'height', 'gps', 'position', 'speed', 'velocity', 'accel', 'gyro', 'mag', 'temp', 'pressure', 'voltage', 'current']
                    if any(term in col.lower() for term in altitude_terms):
                        return True
                    
                    return False
            else:
                # For non-numeric queries, any column type is safe
                return True
                
        except Exception as e:
            print(f"DEBUG: Error checking column safety for {col}: {str(e)}")
            # Be permissive on errors - allow the column
            return True

    def _analyze_single_column_safe(self, df: pd.DataFrame, col: str, user_question_lower: str) -> str:
        """Analyze a single column with comprehensive type safety."""
        try:
            # Check if column has data
            if df[col].isnull().all():
                return ""
            
            # Determine column type and analyze accordingly
            if pd.api.types.is_numeric_dtype(df[col]):
                return self._analyze_numeric_column_safe(df, col, user_question_lower)
            else:
                # Try to convert to numeric if the query requires it
                if any(term in user_question_lower for term in ['maximum', 'max', 'minimum', 'min', 'average', 'mean']):
                    try:
                        # Try to convert to numeric
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_data.isnull().all():
                            # If conversion worked, analyze as numeric
                            return self._analyze_numeric_column_safe(df, col, user_question_lower)
                    except:
                        pass
                
                # Fall back to categorical analysis
                return self._analyze_categorical_column_safe(df, col, user_question_lower)
                
        except Exception as e:
            print(f"DEBUG: Error analyzing single column {col}: {str(e)}")
            return ""

    def _analyze_numeric_column_safe(self, df: pd.DataFrame, col: str, user_question_lower: str) -> str:
        """Analyze numeric columns with type safety."""
        try:
            # Handle both numeric and potentially convertible columns
            if pd.api.types.is_numeric_dtype(df[col]):
                clean_data = df[col].dropna()
            else:
                # Try to convert to numeric
                try:
                    clean_data = pd.to_numeric(df[col], errors='coerce').dropna()
                except:
                    return ""
            
            if len(clean_data) == 0:
                return ""
            
            # Basic statistics
            min_val = clean_data.min()
            max_val = clean_data.max()
            mean_val = clean_data.mean()
            current_val = clean_data.iloc[-1] if len(clean_data) > 0 else "N/A"
            
            result = f"**{col} Analysis:**\n"
            result += f"- Type: Numeric\n"
            result += f"- Range: {min_val} to {max_val}\n"
            result += f"- Average: {mean_val:.2f}\n"
            result += f"- Current value: {current_val}\n"
            
            # Check for specific patterns based on question
            if any(term in user_question_lower for term in ['maximum', 'max', 'highest']):
                max_idx = clean_data.idxmax()
                result += f"- Maximum value: {max_val} at index {max_idx}\n"
            
            elif any(term in user_question_lower for term in ['minimum', 'min', 'lowest']):
                min_idx = clean_data.idxmin()
                result += f"- Minimum value: {min_val} at index {min_idx}\n"
            
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in numeric column analysis for {col}: {str(e)}")
            return ""

    def _analyze_categorical_column_safe(self, df: pd.DataFrame, col: str, user_question_lower: str) -> str:
        """Analyze categorical columns with type safety."""
        try:
            # Remove NaN values
            clean_data = df[col].dropna()
            if len(clean_data) == 0:
                return ""
            
            # Basic statistics
            unique_count = clean_data.nunique()
            most_common = clean_data.mode().iloc[0] if len(clean_data.mode()) > 0 else "N/A"
            current_val = clean_data.iloc[-1] if len(clean_data) > 0 else "N/A"
            
            result = f"**{col} Analysis:**\n"
            result += f"- Type: Categorical\n"
            result += f"- Unique values: {unique_count}\n"
            result += f"- Most common: '{most_common}'\n"
            result += f"- Current value: '{current_val}'\n"
            
            # Show top values if not too many
            if unique_count <= 10:
                value_counts = clean_data.value_counts().head(5)
                result += f"- Top values: {dict(value_counts)}\n"
            
            # Check for specific patterns based on question
            if any(term in user_question_lower for term in ['error', 'fail', 'lost', 'bad', 'off']):
                # Look for error-like values
                error_patterns = ['error', 'fail', 'lost', 'bad', 'off', 'disconnect', '0', 'false', 'none']
                error_values = clean_data[clean_data.astype(str).str.lower().str.contains('|'.join(error_patterns), na=False)]
                if len(error_values) > 0:
                    result += f"- Error-like values found: {len(error_values)} occurrences\n"
                    result += f"- First error: '{error_values.iloc[0]}' at index {error_values.index[0]}\n"
            
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in categorical column analysis for {col}: {str(e)}")
            return "" 