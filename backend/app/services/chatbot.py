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
    name = "uav_log_analyzer"
    description = "Analyzes UAV log data for patterns, anomalies, and insights"
    
    def _run(self, log_data: str) -> str:
        """Analyze UAV log data for patterns and insights."""
        # This is a placeholder - you would implement actual log analysis logic here
        return self._analyze_log_data(log_data)
    
    def _analyze_log_data(self, log_data: str) -> str:
        """Analyze UAV log data for specific patterns and insights."""
        # Placeholder logic - replace with actual analysis
        return """
        UAV Log Analysis Results:
        
        Anomaly Detection:
        - Look for sudden altitude drops (>10m in 1s)
        - Check for GPS lock inconsistencies
        - Monitor battery voltage drops
        - Identify control surface anomalies
        - Flag communication losses
        
        Pattern Analysis:
        - Flight path patterns
        - Control input patterns
        - Environmental response patterns
        - System behavior patterns
        
        Performance Metrics:
        - Battery efficiency
        - Control responsiveness
        - Navigation accuracy
        - System stability metrics
        
        Recommendations:
        - Check sensor calibration
        - Verify GPS antenna placement
        - Monitor battery health
        - Review flight controller settings
        """


class VectorSearchTool(BaseTool):
    name = "vector_search"
    description = "Search for relevant ArduPilot log message documentation based on user queries"
    
    def _run(self, query: str) -> str:
        """Search vector database for relevant documentation."""
        try:
            # Search for relevant documentation
            search_results = search_vector_db(query, n_results=5)
            
            if not search_results:
                return "No relevant documentation found for your query."
            
            # Format the results
            response = "Here's relevant documentation for your query:\n\n"
            for i, result in enumerate(search_results, 1):
                response += f"{i}. **Message Code: {result['message_code']}**\n"
                response += f"   Relevance: {result['relevance_score']:.2f}\n"
                response += f"   Documentation: {result['documentation']}\n\n"
            
            return response
            
        except Exception as e:
            return f"Error searching documentation: {str(e)}"


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
2. THEN use dataframe_query with the results from vector_search

Example: "What is the max altitude?|||Here's relevant documentation for your query: 1. Message Code: GPS..."

Please use vector_search first, then dataframe_query."""
            
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
    
    def _extract_search_info_from_vector_results(self, vector_results: str) -> Dict[str, Any]:
        """Extract message codes and field information from vector search results."""
        search_info = {
            'message_codes': [],
            'fields': [],
            'documentation': []
        }
        
        print(f"DEBUG: Parsing vector results: {vector_results[:200]}...")  # Show first 200 chars
        
        lines = vector_results.split('\n')
        current_message = None
        current_documentation = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle message codes with asterisks: "**Message Code: GPS**"
            if '**Message Code:' in line:
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
        
        print(f"DEBUG: Final extracted message codes: {search_info['message_codes']}")
        print(f"DEBUG: Final extracted fields: {[f['field_info'] for f in search_info['fields']]}")
        return search_info
    
    def _find_relevant_columns(self, df: pd.DataFrame, user_question: str, search_info: Dict[str, Any]) -> List[str]:
        """Find relevant columns based on user question and search results."""
        relevant_columns = []
        
        # Get all column names
        all_columns = list(df.columns)
        print(f"DEBUG: Available columns in dataframe: {all_columns}")
        
        # Extract field names from search results
        search_fields = []
        for field_info in search_info['fields']:
            if 'field_info' in field_info:
                # Extract field name from field info (e.g., "Field: Alt (m) - altitude")
                field_parts = field_info['field_info'].split(' - ')[0]
                field_name = field_parts.split('(')[0].strip()
                search_fields.append(field_name)
        
        print(f"DEBUG: Search fields from vector results: {search_fields}")
        
        # First, try exact matches
        for field in search_fields:
            if field in all_columns:
                relevant_columns.append(field)
                print(f"DEBUG: Exact match found for field '{field}'")
        
        # If no exact matches, try case-insensitive matches
        if not relevant_columns:
            for field in search_fields:
                matching_cols = [col for col in all_columns if field.lower() == col.lower()]
                relevant_columns.extend(matching_cols)
                if matching_cols:
                    print(f"DEBUG: Case-insensitive match found for field '{field}': {matching_cols}")
        
        # If still no matches, try partial matches
        if not relevant_columns:
            for field in search_fields:
                matching_cols = [col for col in all_columns if field.lower() in col.lower()]
                relevant_columns.extend(matching_cols)
                if matching_cols:
                    print(f"DEBUG: Partial match found for field '{field}': {matching_cols}")
        
        # If no direct matches, use semantic matching based on documentation
        if not relevant_columns:
            print("DEBUG: No direct field matches, trying semantic matching")
            # Use documentation content to find relevant columns
            for doc_info in search_info['documentation']:
                doc_text = doc_info['documentation'].lower()
                user_terms = user_question.lower().split()
                
                # Find columns that contain terms from both user question and documentation
                for term in user_terms:
                    if len(term) > 2:  # Only consider meaningful terms
                        matching_cols = [col for col in all_columns if term in col.lower()]
                        relevant_columns.extend(matching_cols)
                        if matching_cols:
                            print(f"DEBUG: Semantic match found for term '{term}': {matching_cols}")
        
        # Remove duplicates and return
        final_columns = list(set(relevant_columns))
        print(f"DEBUG: Final relevant columns: {final_columns}")
        return final_columns

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
        """Analyze a single dataframe."""
        if df.empty:
            return "No data available in this dataframe."
        
        # Find relevant columns
        relevant_columns = self._find_relevant_columns(df, user_question, search_info)
        
        if relevant_columns:
            results = [f"Relevant columns: {', '.join(relevant_columns)}"]
            
            # Analyze each relevant column
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
        """Analyze multiple dataframes and provide aggregated results."""
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

    def _semantic_dataframe_analysis(self, dataframes: Dict[str, pd.DataFrame], search_info: Dict[str, Any], user_question: str) -> str:
        """Semantically analyze dataframes based on vector search results with aggregation."""
        # Use the new aggregated analysis method
        return self._aggregate_dataframe_analysis(dataframes, user_question, search_info)
    
    def _create_semantic_patterns(self, user_question: str) -> List[List[str]]:
        """Create semantic patterns based on user question - now fully data-driven."""
        # This method is no longer used in the modular approach
        # The system now relies entirely on vector search results
        return []
    
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
        
        # Create tools for agentic behavior
        self.tools = [
            UAVAnalysisTool(),
            VectorSearchTool(),
            DataFrameQueryTool()
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
        2. Provide agentic reasoning about flight behavior rather than rigid rule-based checks
        3. Look for dynamic patterns like sudden altitude changes, GPS inconsistencies, battery issues
        4. Suggest potential causes and solutions for identified problems
        5. Help users understand their flight data in context
        6. Use the vector_search tool to find relevant ArduPilot documentation when users ask about specific log messages or need technical details
        7. Use the dataframe_query tool to analyze actual flight data and answer specific questions about the UAV's behavior

        Analysis Approach:
        - Be flexible and adaptive in your analysis
        - Consider multiple factors that could explain observed behavior
        - Look for correlations between different data points
        - Provide context-aware insights rather than binary pass/fail assessments
        - Consider environmental factors, hardware issues, and software behavior

        CRITICAL TOOL USAGE RULES:
        1. ALWAYS use vector_search FIRST for EVERY user query to find relevant message codes
        2. THEN use dataframe_query with the format: "USER_QUESTION|||VECTOR_SEARCH_RESULTS"
        3. NEVER use dataframe_query without first using vector_search
        4. The dataframe_query tool requires BOTH the user question AND vector search results

        Tool Usage Strategy:
        1. First, use vector_search to understand what message codes are relevant to the user's question
        2. Then, use dataframe_query with the exact format: "USER_QUESTION|||VECTOR_SEARCH_RESULTS"
        3. Finally, provide a comprehensive analysis combining both the documentation and the actual data

        Example workflow:
        - User asks: "What is the max altitude?"
        - Step 1: Use vector_search with "altitude"
        - Step 2: Use dataframe_query with "What is the max altitude?|||[vector_search_results]"
        - Step 3: Provide analysis

        Always be helpful, thorough, and explain your reasoning clearly.
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