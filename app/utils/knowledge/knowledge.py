from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import json
from app.vectordb.manager import vector_db
from .knowledge_schema import ProductKnowledge

class KnowledgeManager:
    def __init__(self):
        self.collection = vector_db.get_collection()
        
    
    def flatten_metadata(self,metadata: dict) -> dict:
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                flattened[key] = ", ".join(str(v) for v in value)
            else:
                flattened[key] = value
        return flattened
    
   

knowledge_manager = KnowledgeManager()
