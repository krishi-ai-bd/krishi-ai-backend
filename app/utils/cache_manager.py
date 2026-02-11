import redis
import json
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from app.core.config import settings


class ConversationCacheManager:
    """
    Redis-based conversation manager with user mapping and TTL cleanup.
    
    Structure:
    - chat:{conversation_id} -> List of messages [{role, content, timestamp}, ...]
    - user:{user_id} -> {conversation_ids: [...], last_active: timestamp}
    """
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL, db=settings.REDIS_DB, decode_responses=True)
            self.redis_client.ping()
            print("✓ Redis connection established")
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            self.redis_client = None
    
    def _chat_key(self, conversation_id: str) -> str:
        """Generate key for conversation messages"""
        return f"chat:{conversation_id}"
    
    def _user_key(self, user_id: str) -> str:
        """Generate key for user metadata"""
        return f"user:{user_id}"
    
    def get_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        """Retrieve all messages in a conversation"""
        if not self.redis_client:
            return None
        
        try:
            key = self._chat_key(conversation_id)
            data = self.redis_client.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            print(f"Error retrieving conversation {conversation_id}: {e}")
            return None
    
    def get_conversation_by_user(self, user_id: str, conversation_id: str) -> Optional[List[Dict]]:
        """Retrieve conversation with user ownership verification"""
        if not self.redis_client:
            return None
        
        try:
            # Verify conversation belongs to user
            user_conversations = self.get_user_conversations(user_id)
            if conversation_id not in user_conversations:
                print(f"Conversation {conversation_id} does not belong to user {user_id}")
                return None
            
            # Retrieve conversation
            return self.get_conversation(conversation_id)
        except Exception as e:
            print(f"Error retrieving conversation for user {user_id}: {e}")
            return None
    
    def add_message(self, conversation_id: str, user_id: str, role: str, content: str):
        """Add a message to a conversation and update user activity"""
        if not self.redis_client:
            return
        
        try:
            # Get or create conversation
            messages = self.get_conversation(conversation_id) or []
            
            # Add new message
            messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Save conversation with TTL
            chat_key = self._chat_key(conversation_id)
            ttl_seconds = settings.CONVERSATION_TTL_DAYS * 24 * 3600
            self.redis_client.setex(chat_key, ttl_seconds, json.dumps(messages))
            
            # Update user metadata
            self._update_user_activity(user_id, conversation_id)
            
        except Exception as e:
            print(f"Error adding message to {conversation_id}: {e}")
    
    def _update_user_activity(self, user_id: str, conversation_id: str):
        """Update user's last activity and conversation list"""
        if not self.redis_client:
            return
        
        try:
            user_key = self._user_key(user_id)
            user_data = self.redis_client.get(user_key)
            
            if user_data:
                user_info = json.loads(user_data)
            else:
                user_info = {"conversation_ids": []}
            
            # Add conversation ID if not already present
            if conversation_id not in user_info["conversation_ids"]:
                user_info["conversation_ids"].append(conversation_id)
            
            # Update last active timestamp
            user_info["last_active"] = datetime.utcnow().isoformat()
            
            # Save with TTL based on inactivity period
            ttl_seconds = settings.USER_INACTIVE_DAYS * 24 * 3600
            self.redis_client.setex(user_key, ttl_seconds, json.dumps(user_info))
            
        except Exception as e:
            print(f"Error updating user activity for {user_id}: {e}")
    
    def get_user_conversations(self, user_id: str) -> List[str]:
        """Get all conversation IDs for a user"""
        if not self.redis_client:
            return []
        
        try:
            user_key = self._user_key(user_id)
            user_data = self.redis_client.get(user_key)
            
            if user_data:
                user_info = json.loads(user_data)
                return user_info.get("conversation_ids", [])
            return []
        except Exception as e:
            print(f"Error retrieving user conversations for {user_id}: {e}")
            return []
    
    def delete_user_data(self, user_id: str):
        """Delete all conversations and data for a user"""
        if not self.redis_client:
            return
        
        try:
            # Get all conversation IDs
            conversation_ids = self.get_user_conversations(user_id)
            
            # Delete all conversations
            for conv_id in conversation_ids:
                chat_key = self._chat_key(conv_id)
                self.redis_client.delete(chat_key)
            
            # Delete user metadata
            user_key = self._user_key(user_id)
            self.redis_client.delete(user_key)
            
            print(f"✓ Deleted {len(conversation_ids)} conversations for user {user_id}")
            
        except Exception as e:
            print(f"Error deleting user data for {user_id}: {e}")
    
    def cleanup_inactive_users(self):
        """Remove data for users inactive for more than USER_INACTIVE_DAYS"""
        if not self.redis_client:
            return
        
        try:
            # Scan for all user keys
            cursor = 0
            inactive_threshold = datetime.utcnow() - timedelta(days=settings.USER_INACTIVE_DAYS)
            deleted_users = 0
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, match="user:*", count=100)
                
                for key in keys:
                    user_data = self.redis_client.get(key)
                    if user_data:
                        user_info = json.loads(user_data)
                        last_active = datetime.fromisoformat(user_info.get("last_active", ""))
                        
                        if last_active < inactive_threshold:
                            user_id = key.replace("user:", "")
                            self.delete_user_data(user_id)
                            deleted_users += 1
                
                if cursor == 0:
                    break
            
            print(f"✓ Cleanup complete: {deleted_users} inactive users removed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")


# Global instance
cache_manager = ConversationCacheManager()