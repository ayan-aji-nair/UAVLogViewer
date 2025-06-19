import json
import redis
from typing import Optional, Dict, Any
from datetime import datetime
from ..models.chat import ChatSession, ChatMessage, MessageRole
import os


class SessionManager:
    def __init__(self):
        # Initialize Redis connection (fallback to in-memory if Redis not available)
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True
            )
            self.redis_client.ping()  # Test connection
            self.use_redis = True
        except Exception:
            print("Redis not available, using in-memory storage")
            self.use_redis = False
            self.sessions: Dict[str, ChatSession] = {}

    def get_session(self, sessionId: str) -> Optional[ChatSession]:
        """Retrieve a chat session by ID."""
        if self.use_redis:
            try:
                session_data = self.redis_client.get(f"session:{sessionId}")
                if session_data:
                    return ChatSession(**json.loads(session_data))
            except Exception as e:
                print(f"Error retrieving session: {e}")
        else:
            return self.sessions.get(sessionId)
        return None

    def create_session(self, sessionId: str, contextData: Optional[Dict[str, Any]] = None) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            id=sessionId,
            contextData=contextData or {}
        )
        
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{sessionId}",
                    3600,  # 1 hour TTL
                    session.model_dump_json()
                )
            except Exception as e:
                print(f"Error saving session: {e}")
        else:
            self.sessions[sessionId] = session
        
        return session

    def add_message(self, sessionId: str, role: MessageRole, content: str) -> ChatMessage:
        """Add a message to an existing session."""
        session = self.get_session(sessionId)
        if not session:
            session = self.create_session(sessionId)
        
        message = ChatMessage(
            role=role,
            content=content,
            sessionId=sessionId
        )
        
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        # Save updated session
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{sessionId}",
                    3600,
                    session.model_dump_json()
                )
            except Exception as e:
                print(f"Error updating session: {e}")
        else:
            self.sessions[sessionId] = session
        
        return message

    def get_session_history(self, sessionId: str) -> list[ChatMessage]:
        """Get all messages for a session."""
        session = self.get_session(sessionId)
        return session.messages if session else []

    def update_context(self, sessionId: str, contextData: Dict[str, Any]) -> bool:
        """Update the context data for a session."""
        session = self.get_session(sessionId)
        if not session:
            return False
        
        session.contextData.update(contextData)
        session.updated_at = datetime.utcnow()
        
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{sessionId}",
                    3600,
                    session.model_dump_json()
                )
                return True
            except Exception as e:
                print(f"Error updating context: {e}")
                return False
        else:
            self.sessions[sessionId] = session
            return True 