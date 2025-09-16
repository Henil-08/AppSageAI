"""Database management with encryption and privacy features."""

import sqlite3
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from logger import logger

class PrivacyDatabase:
    """Manages database operations with encryption and privacy features."""
    
    def __init__(self, db_path: str = "appsageai.db", encryption_key: Optional[str] = None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize encryption
        if encryption_key:
            self.cipher = self._get_cipher(encryption_key)
        else:
            # Generate a new key if not provided
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            # Store key securely (in production, use key management service)
            self._store_key(key)
        
        self._init_tables()
        self._cleanup_old_data()
    
    def _get_cipher(self, password: str) -> Fernet:
        """Generate cipher from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'appsageai_salt_v2',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)
    
    def _store_key(self, key: bytes):
        """Store encryption key securely."""
        # In production, use AWS KMS, Azure Key Vault, etc.
        key_file = Path(".encryption_key")
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Read/write for owner only
    
    def _init_tables(self):
        """Initialize database tables."""
        cursor = self.conn.cursor()
        
        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Conversations table (stores encrypted data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                encrypted_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conversation_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions(session_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        # Analytics table (no PII)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_hash TEXT NOT NULL,
                action_type TEXT NOT NULL,
                response_time_ms INTEGER,
                tokens_used INTEGER,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Memory store for long-term memories
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_hash TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                encrypted_memory TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        self.conn.commit()
    
    def create_session(self, user_name: str) -> str:
        """Create a new user session with hashed identifier."""
        session_id = secrets.token_urlsafe(32)
        user_hash = hashlib.sha256(user_name.lower().encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO user_sessions (session_id, user_hash, metadata)
            VALUES (?, ?, ?)
        """, (session_id, user_hash, json.dumps({"start_time": datetime.now().isoformat()})))
        self.conn.commit()
        
        return session_id
    
    def store_conversation(self, session_id: str, message_type: str, content: Dict[str, Any]) -> int:
        """Store encrypted conversation data."""
        # Encrypt sensitive content
        encrypted_content = self.cipher.encrypt(json.dumps(content).encode()).decode()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, message_type, encrypted_content)
            VALUES (?, ?, ?)
        """, (session_id, message_type, encrypted_content))
        self.conn.commit()
        
        return cursor.lastrowid
    
    def get_user_memories(self, user_name: str, memory_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve user memories (decrypted)."""
        user_hash = hashlib.sha256(user_name.lower().encode()).hexdigest()
        
        cursor = self.conn.cursor()
        if memory_type:
            cursor.execute("""
                SELECT * FROM user_memories
                WHERE user_hash = ? AND memory_type = ?
                ORDER BY relevance_score DESC, last_accessed DESC
                LIMIT ?
            """, (user_hash, memory_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM user_memories
                WHERE user_hash = ?
                ORDER BY relevance_score DESC, last_accessed DESC
                LIMIT ?
            """, (user_hash, limit))
        
        memories = []
        for row in cursor.fetchall():
            try:
                decrypted_memory = json.loads(
                    self.cipher.decrypt(row['encrypted_memory'].encode()).decode()
                )
                memories.append({
                    'id': row['id'],
                    'type': row['memory_type'],
                    'content': decrypted_memory,
                    'relevance_score': row['relevance_score'],
                    'created_at': row['created_at'],
                    'access_count': row['access_count']
                })
                
                # Update access count and timestamp
                self._update_memory_access(row['id'])
            except Exception as e:
                logger.error(f"Failed to decrypt memory {row['id']}: {e}")
                continue
        
        return memories
    
    def store_memory(self, user_name: str, memory_type: str, content: Dict[str, Any], relevance_score: float = 1.0):
        """Store a new memory for the user."""
        user_hash = hashlib.sha256(user_name.lower().encode()).hexdigest()
        encrypted_memory = self.cipher.encrypt(json.dumps(content).encode()).decode()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO user_memories (user_hash, memory_type, encrypted_memory, relevance_score)
            VALUES (?, ?, ?, ?)
        """, (user_hash, memory_type, encrypted_memory, relevance_score))
        self.conn.commit()
    
    def _update_memory_access(self, memory_id: int):
        """Update memory access timestamp and count."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE user_memories
            SET last_accessed = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE id = ?
        """, (memory_id,))
        self.conn.commit()
    
    def store_feedback(self, session_id: str, conversation_id: int, feedback_type: str):
        """Store user feedback."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (session_id, conversation_id, feedback_type)
            VALUES (?, ?, ?)
        """, (session_id, conversation_id, feedback_type))
        self.conn.commit()
    
    def log_analytics(self, session_id: str, action_type: str, response_time_ms: int, 
                     tokens_used: int, model_used: str):
        """Log analytics without PII."""
        # Hash the session_id for analytics
        session_hash = hashlib.sha256(session_id.encode()).hexdigest()[:16]
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO analytics (session_hash, action_type, response_time_ms, tokens_used, model_used)
            VALUES (?, ?, ?, ?, ?)
        """, (session_hash, action_type, response_time_ms, tokens_used, model_used))
        self.conn.commit()
    
    def _cleanup_old_data(self, retention_days: int = 30):
        """Remove old data based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cursor = self.conn.cursor()
        
        # Clean old conversations
        cursor.execute("""
            DELETE FROM conversations
            WHERE created_at < ?
        """, (cutoff_date,))
        
        # Clean old sessions
        cursor.execute("""
            DELETE FROM user_sessions
            WHERE last_active < ?
        """, (cutoff_date,))
        
        # Clean old memories with low relevance
        cursor.execute("""
            DELETE FROM user_memories
            WHERE last_accessed < ? AND relevance_score < 0.3
        """, (cutoff_date,))
        
        self.conn.commit()
        
        logger.info(f"Cleaned up data older than {retention_days} days")
    
    def close(self):
        """Close database connection."""
        self.conn.close()