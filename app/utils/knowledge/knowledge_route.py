from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import uuid
from pathlib import Path
from .knowledge_schema import PDFUploadResponse, DocumentDeleteRequest, DocumentDeleteResponse
from .knowledge import document_service
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    crop_type: Optional[str] = Form(None),
    language: Optional[str] = Form("english"),
    description: Optional[str] = Form(None)
):
    """
    Upload and process agricultural PDF document.
    
    The PDF will be:
    1. Chunked semantically by sections
    2. Embedded using OpenAI
    3. Stored in vector database for retrieval
    """
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Check file size (approximate)
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    try:
        # Create unique document name
        document_name = f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded file temporarily
        upload_path = Path("./uploads") / f"{document_name}.pdf"
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(contents)
        
        # Prepare metadata
        metadata = {}
        if category:
            metadata["category"] = category
        if crop_type:
            metadata["crop_type"] = crop_type
        if language:
            metadata["language"] = language
        if description:
            metadata["description"] = description
        
        # Process and index
        result = document_service.process_and_index_pdf(
            str(upload_path),
            document_name,
            metadata
        )
        
        return PDFUploadResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/delete", response_model=DocumentDeleteResponse)
async def delete_document(request: DocumentDeleteRequest):
    """Delete a document and all its chunks from the vector database"""
    
    result = document_service.delete_document(request.document_name)
    
    if result["success"]:
        return DocumentDeleteResponse(**result)
    else:
        raise HTTPException(status_code=500, detail=result["message"])


@router.get("/stats")
async def get_stats():
    """Get statistics about indexed documents"""
    return document_service.get_stats()
