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
