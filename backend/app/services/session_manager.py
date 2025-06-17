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

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve a chat session by ID."""
        if self.use_redis:
            try:
                session_data = self.redis_client.get(f"session:{session_id}")
                if session_data:
                    return ChatSession(**json.loads(session_data))
            except Exception as e:
                print(f"Error retrieving session: {e}")
        else:
            return self.sessions.get(session_id)
        return None

    def create_session(self, session_id: str, context_data: Optional[Dict[str, Any]] = None) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            id=session_id,
            context_data=context_data or {}
        )
        
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    3600,  # 1 hour TTL
                    session.model_dump_json()
                )
            except Exception as e:
                print(f"Error saving session: {e}")
        else:
            self.sessions[session_id] = session
        
        return session

    def add_message(self, session_id: str, role: MessageRole, content: str) -> ChatMessage:
        """Add a message to an existing session."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        message = ChatMessage(
            role=role,
            content=content,
            session_id=session_id
        )
        
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        # Save updated session
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    3600,
                    session.model_dump_json()
                )
            except Exception as e:
                print(f"Error updating session: {e}")
        else:
            self.sessions[session_id] = session
        
        return message

    def get_session_history(self, session_id: str) -> list[ChatMessage]:
        """Get all messages for a session."""
        session = self.get_session(session_id)
        return session.messages if session else []

    def update_context(self, session_id: str, context_data: Dict[str, Any]) -> bool:
        """Update the context data for a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.context_data.update(context_data)
        session.updated_at = datetime.utcnow()
        
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    3600,
                    session.model_dump_json()
                )
                return True
            except Exception as e:
                print(f"Error updating context: {e}")
                return False
        else:
            self.sessions[session_id] = session
            return True 