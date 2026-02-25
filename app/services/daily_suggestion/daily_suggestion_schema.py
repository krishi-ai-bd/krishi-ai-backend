from pydantic import BaseModel
from typing import List, Optional

class daily_suggestion_request(BaseModel):
    previous_suggestions: Optional[List[str]] = None
    
class daily_suggestion_response(BaseModel):
    audio_url: str


