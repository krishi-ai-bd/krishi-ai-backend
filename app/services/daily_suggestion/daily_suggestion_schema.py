from pydantic import BaseModel
from typing import List

class daily_suggestion_request(BaseModel):
    previous_suggestions: List[str]
    
class daily_suggestion_response(BaseModel):
    response: str


