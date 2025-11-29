"""Database configuration and connection"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Provide a default SQLite database if DATABASE_URL is not set
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./chatbot.db"
    print("⚠️  DATABASE_URL not found in environment. Using default SQLite: chatbot.db")

# Fix PostgreSQL URL for Render (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate settings
connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args = {
        "check_same_thread": False,
        "timeout": 30,
    }

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    
    # SQLite optimizations (if using SQLite)
    if "sqlite" in DATABASE_URL:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
    
    # Test the connection
    with engine.connect() as conn:
        pass
    
    print(f"✅ Database configured: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'SQLite'}")
    
except Exception as e:
    print(f"❌ Database connection error: {e}")
    raise

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
