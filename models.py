"""Database models for RAG Chatbot"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from database import Base
from datetime import datetime, timezone


class ChatHistory(Base):
    """Store all chat conversations"""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class WebsiteContent(Base):
    """Track scraped websites"""
    __tablename__ = "website_content"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(500), unique=True, nullable=False, index=True)
    content = Column(Text, nullable=True)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_indexed = Column(Boolean, default=False)


class UserContact(Base):
    """Store user contact information"""
    __tablename__ = "user_contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=False)
    submitted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
