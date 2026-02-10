from pydantic import BaseModel
from typing import Optional


class PDFUploadRequest(BaseModel):
    """Metadata for PDF upload"""
    category: Optional[str] = None  # e.g., "pest_control", "irrigation", "soil_health"
    crop_type: Optional[str] = None  # e.g., "tomato", "wheat", "rice"
    language: Optional[str] = "english"
    description: Optional[str] = None


class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    document_name: Optional[str] = None
    chunks_created: Optional[int] = None


class DocumentDeleteRequest(BaseModel):
    document_name: str


class DocumentDeleteResponse(BaseModel):
    success: bool
    message: str
