from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sessionId: str


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = Field(default_factory=list)
    contextData: Optional[dict] = Field(default_factory=dict)  # For storing UAV log context


class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    contextData: Optional[dict] = None  # Additional UAV log data to analyze


class ChatResponse(BaseModel):
    message: str
    sessionId: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_insights: Optional[List[str]] = Field(default_factory=list)  # Agent's insights about UAV data 