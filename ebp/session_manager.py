"""
Session management for KV cache and incremental decoding.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from .logging_config import get_logger

logger = get_logger("ebp.session_manager")


class Session:
    """Represents an inference session with KV cache."""
    
    def __init__(self, session_id: str, stage_id: str, max_length: int = 2048):
        self.session_id = session_id
        self.stage_id = stage_id
        self.max_length = max_length
        self.current_length = 0
        self.past_key_values: Optional[List] = None
        self.created_at = time.time()
        self.last_used = time.time()
        self.lock = threading.Lock()
    
    def update_last_used(self):
        """Update last used timestamp."""
        self.last_used = time.time()
    
    def is_expired(self, timeout_seconds: float = 3600.0) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_used) > timeout_seconds


class SessionManager:
    """Manages inference sessions with concurrency limits."""
    
    def __init__(self, max_sessions: int = 10, session_timeout: float = 3600.0):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()
        logger.info(f"SessionManager initialized: max_sessions={max_sessions}")
    
    def create_session(self, stage_id: str, max_length: int = 2048) -> Session:
        """Create a new session."""
        import uuid
        session_id = uuid.uuid4().hex[:16]
        
        with self.lock:
            # Clean up expired sessions
            self._cleanup_expired()
            
            # Check if we're at capacity
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest unused session
                oldest = min(self.sessions.values(), key=lambda s: s.last_used)
                logger.warning(f"Session limit reached, removing oldest session {oldest.session_id}")
                del self.sessions[oldest.session_id]
            
            session = Session(session_id, stage_id, max_length)
            self.sessions[session_id] = session
            logger.debug(f"Created session {session_id} for stage {stage_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session and session.is_expired(self.session_timeout):
                logger.debug(f"Session {session_id} expired, removing")
                del self.sessions[session_id]
                return None
            return session
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session (clear KV cache)."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.past_key_values = None
                session.current_length = 0
                session.update_last_used()
                logger.debug(f"Reset session {session_id}")
                return True
            return False
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Removed session {session_id}")
                return True
            return False
    
    def _cleanup_expired(self):
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.session_timeout)
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.debug(f"Cleaned up expired session {sid}")
    
    def get_stats(self) -> Dict:
        """Get session statistics."""
        with self.lock:
            self._cleanup_expired()
            return {
                "active_sessions": len(self.sessions),
                "max_sessions": self.max_sessions,
            }


# Global session manager instance
_session_manager: Optional[SessionManager] = None
_session_manager_lock = threading.Lock()


def get_session_manager(max_sessions: int = 10) -> SessionManager:
    """Get or create global session manager."""
    global _session_manager
    with _session_manager_lock:
        if _session_manager is None:
            _session_manager = SessionManager(max_sessions=max_sessions)
        return _session_manager
