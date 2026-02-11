from pydantic import BaseModel
from typing import Optional, Any, Dict

class HistoryItem(BaseModel):
    message: str
    response: str
    
class chatbot_request(BaseModel):
    message: str
    user_id: str
    chat_id:str  
    
class chatbot_response(BaseModel):
    response: str


class conversation_history_request(BaseModel):
    user_id: str
    chat_id: str

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

class conversation_history_response(BaseModel):
    messages: list[Message]