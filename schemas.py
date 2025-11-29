"""Pydantic schemas for request/response validation"""
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime


# ========================================
# Request Schemas
# ========================================
class QuestionRequest(BaseModel):
    """Request schema for asking questions"""
    question: str
    session_id: str


class ContactInfoRequest(BaseModel):
    """Request schema for submitting contact info"""
    session_id: str
    name: str
    email: EmailStr
    phone: str


class WebsiteInitRequest(BaseModel):
    """Request schema for initializing RAG with a website"""
    url: str = "https://syngrid.com/"
    max_pages: int = 25


# ========================================
# Response Schemas
# ========================================
class AnswerResponse(BaseModel):
    """Response schema for question answers"""
    question: str
    answer: str
    timestamp: datetime
    requires_contact: bool = False
    contact_message: Optional[str] = None


class ContactResponse(BaseModel):
    """Response schema for contact submission"""
    message: str
    success: bool


class StatusResponse(BaseModel):
    """Response schema for RAG status"""
    ready: bool
    message: str
    pages_scraped: int = 0


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    session_id: str
    question: str
    answer: str
    timestamp: datetime


class ContactInfoResponse(BaseModel):
    """Response schema for contact info"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    session_id: str
    name: str
    email: str
    phone: str
    submitted_at: datetime
