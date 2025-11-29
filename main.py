"""Main FastAPI application - RAG Chatbot with Contact Collection"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone
import time
import os

from database import SessionLocal, Base, engine
from models import ChatHistory, WebsiteContent, UserContact
from schemas import (
    QuestionRequest, ContactInfoRequest, WebsiteInitRequest,
    AnswerResponse, ContactResponse, StatusResponse,
    ChatHistoryResponse, ContactInfoResponse
)
from rag_service import rag_service

# Create all tables
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")
except Exception as e:
    print(f"❌ Error creating tables: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="AI-powered chatbot with contact collection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ========================================
# DATABASE DEPENDENCY
# ========================================
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========================================
# HELPER FUNCTIONS
# ========================================
def count_user_questions(session_id: str, db: Session) -> int:
    """Count questions asked by this session"""
    try:
        return db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        ).count()
    except Exception as e:
        print(f"Error counting questions: {e}")
        return 0


def has_submitted_contact(session_id: str, db: Session) -> bool:
    """Check if user already submitted contact info"""
    try:
        return db.query(UserContact).filter(
            UserContact.session_id == session_id
        ).first() is not None
    except Exception as e:
        print(f"Error checking contact: {e}")
        return False


# ========================================
# API ENDPOINTS
# ========================================

@app.get("/")
def root():
    """Serve frontend chatbot UI"""
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    """Health check for Render"""
    try:
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "connected"
    except Exception as e:
        print(f"Database health check failed: {e}")
        db_status = "error"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "rag_status": rag_service.status["message"]
    }


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get RAG initialization status"""
    status = rag_service.get_status()
    return StatusResponse(
        ready=status["ready"],
        message=status["message"],
        pages_scraped=status.get("pages_scraped", 0)
    )


@app.post("/initialize", response_model=StatusResponse)
def initialize_rag(
    request: WebsiteInitRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Initialize RAG with website content"""
    
    # Check if already initialized
    if rag_service.status["ready"]:
        return StatusResponse(
            ready=True,
            message="Already initialized",
            pages_scraped=rag_service.status.get("pages_scraped", 0)
        )
    
    # Check if website already indexed
    try:
        existing = db.query(WebsiteContent).filter(
            WebsiteContent.url == request.url
        ).first()

        if existing and existing.is_indexed:
            return StatusResponse(
                ready=True,
                message="Website already indexed",
                pages_scraped=rag_service.status.get("pages_scraped", 0)
            )
    except Exception as e:
        print(f"Error checking existing website: {e}")

    def init_task():
        """Background task for initialization"""
        db_task = SessionLocal()
        try:
            success = rag_service.initialize_rag(request.url, request.max_pages)
            
            if success:
                # Update database
                existing = db_task.query(WebsiteContent).filter(
                    WebsiteContent.url == request.url
                ).first()

                if existing:
                    existing.is_indexed = True
                    existing.scraped_at = datetime.now(timezone.utc)
                else:
                    website = WebsiteContent(
                        url=request.url,
                        content="",
                        is_indexed=True
                    )
                    db_task.add(website)
                db_task.commit()
                print(f"✅ Website {request.url} marked as indexed")
                
        except Exception as e:
            print(f"❌ Error in init_task: {e}")
            db_task.rollback()
        finally:
            db_task.close()

    background_tasks.add_task(init_task)

    return StatusResponse(
        ready=False,
        message="Initialization started in background",
        pages_scraped=0
    )


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question to the chatbot"""
    
    if not rag_service.status["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not ready. Please wait for initialization to complete."
        )

    # Get answer from RAG
    answer = rag_service.ask_question(request.question)

    # Save to database with retry logic
    for attempt in range(3):
        try:
            chat_entry = ChatHistory(
                session_id=request.session_id,
                question=request.question,
                answer=answer,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(chat_entry)
            db.commit()
            db.refresh(chat_entry)
            break
        except Exception as e:
            if attempt < 2:
                db.rollback()
                time.sleep(0.1)
            else:
                print(f"❌ Failed to save chat after 3 attempts: {e}")

    # Check if contact info required
    question_count = count_user_questions(request.session_id, db)
    has_contact = has_submitted_contact(request.session_id, db)

    requires_contact = (question_count >= 3 and not has_contact)
    contact_message = None

    if requires_contact:
        contact_message = "To continue chatting, please provide your contact information (name, email, phone number)."

    return AnswerResponse(
        question=request.question,
        answer=answer,
        timestamp=datetime.now(timezone.utc),
        requires_contact=requires_contact,
        contact_message=contact_message
    )


@app.post("/submit-contact", response_model=ContactResponse)
def submit_contact(request: ContactInfoRequest, db: Session = Depends(get_db)):
    """Submit user contact information"""
    
    try:
        # Check if already exists
        existing = db.query(UserContact).filter(
            UserContact.session_id == request.session_id
        ).first()

        if existing:
            return ContactResponse(
                message="Contact information already submitted",
                success=True
            )

        # Save new contact
        contact = UserContact(
            session_id=request.session_id,
            name=request.name.strip(),
            email=request.email.strip().lower(),
            phone=request.phone.strip(),
            submitted_at=datetime.now(timezone.utc)
        )
        
        db.add(contact)
        db.commit()
        db.refresh(contact)

        print(f"✅ Contact saved: {contact.name} ({contact.email})")

        return ContactResponse(
            message="Thank you! Your contact information has been saved.",
            success=True
        )

    except Exception as e:
        db.rollback()
        print(f"❌ Error saving contact: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save contact information. Please try again."
        )


@app.get("/history", response_model=List[ChatHistoryResponse])
def get_chat_history(
    session_id: str = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get chat history"""
    try:
        query = db.query(ChatHistory)
        
        if session_id:
            query = query.filter(ChatHistory.session_id == session_id)
        
        history = query.order_by(ChatHistory.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
        
        return history
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return []


@app.get("/contacts", response_model=List[ContactInfoResponse])
def get_contacts(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all contacts (admin endpoint)"""
    try:
        contacts = db.query(UserContact)\
            .order_by(UserContact.submitted_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
        
        return contacts
    except Exception as e:
        print(f"❌ Error fetching contacts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve contacts"
        )


@app.get("/contacts/count")
def get_contact_count(db: Session = Depends(get_db)):
    """Get total number of contacts"""
    try:
        count = db.query(UserContact).count()
        return {"total_contacts": count}
    except Exception as e:
        print(f"❌ Error counting contacts: {e}")
        return {"total_contacts": 0}


@app.delete("/contacts/{contact_id}")
def delete_contact(contact_id: int, db: Session = Depends(get_db)):
    """Delete a specific contact"""
    try:
        contact = db.query(UserContact).filter(
            UserContact.id == contact_id
        ).first()
        
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")
        
        db.delete(contact)
        db.commit()
        
        return {"message": "Contact deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting contact: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting contact: {str(e)}"
        )


@app.delete("/history")
def clear_history(db: Session = Depends(get_db)):
    """Clear all chat history"""
    try:
        deleted = db.query(ChatHistory).delete()
        db.commit()
        return {"message": f"Deleted {deleted} chat entries"}
    except Exception as e:
        db.rollback()
        print(f"❌ Error clearing history: {e}")
        return {"message": f"Error: {str(e)}"}


@app.delete("/cache")
def clear_cache():
    """Clear RAG answer cache"""
    try:
        cache_size = rag_service.clear_cache()
        return {"message": f"Cleared {cache_size} cached entries"}
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return {"message": f"Error: {str(e)}"}


# ========================================
# STARTUP EVENT - AUTO-INITIALIZE
# ========================================
@app.on_event("startup")
async def startup_event():
    """Auto-initialize chatbot on startup"""
    print("\n" + "="*60)
    print("🚀 RAG CHATBOT STARTING")
    print("="*60)
    print(f"🔑 API Key: {'✓ Configured' if rag_service.api_key else '✗ Missing'}")
    print("="*60 + "\n")
    
    if not rag_service.status["ready"]:
        print("📄 Auto-initializing with default website...")
        import threading
        thread = threading.Thread(
            target=lambda: rag_service.initialize_rag("https://syngrid.com/", 25)
        )
        thread.daemon = True
        thread.start()


# ========================================
# RUN SERVER
# ========================================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 10000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\n🌐 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
