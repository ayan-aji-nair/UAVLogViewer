import os
import uuid
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


class UAVChatbotService:
    def __init__(self):
        self.session_manager = SessionManager()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create tools for agentic behavior
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
        2. Provide agentic reasoning about flight behavior rather than rigid rule-based checks
        3. Look for dynamic patterns like sudden altitude changes, GPS inconsistencies, battery issues
        4. Suggest potential causes and solutions for identified problems
        5. Help users understand their flight data in context

        Analysis Approach:
        - Be flexible and adaptive in your analysis
        - Consider multiple factors that could explain observed behavior
        - Look for correlations between different data points
        - Provide context-aware insights rather than binary pass/fail assessments
        - Consider environmental factors, hardware issues, and software behavior

        Always be helpful, thorough, and explain your reasoning clearly.
        """

    async def process_message(self, message: str, sessionId: str, contextData: Optional[Dict[str, Any]] = None) -> str:
        """Process a user message and return an agentic response."""
        
        # Get or create session
        session = self.session_manager.get_session(sessionId)
        if not session:
            session = self.session_manager.create_session(sessionId, contextData)
        
        # Add user message to session
        self.session_manager.add_message(sessionId, MessageRole.USER, message)
        
        # Prepare context for the LLM
        context = self._prepare_context(session, contextData)
        
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