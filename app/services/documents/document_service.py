import os
import shutil
from pathlib import Path
from typing import Dict, Optional
from app.utils.pdf_processor import pdf_processor
from app.vectordb.manager import vector_db


class DocumentService:
    """Handles PDF upload, processing, and indexing"""
    
    def __init__(self):
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def process_and_index_pdf(
        self, 
        pdf_path: str, 
        document_name: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process PDF and add to vector database.
        
        Args:
            pdf_path: Path to PDF file
            document_name: Unique name for document
            metadata: Additional metadata (category, crop_type, etc.)
        
        Returns:
            Dict with success status and details
        """
        try:
            # Process PDF into chunks
            chunks = pdf_processor.process_pdf(pdf_path, document_name, metadata)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "Failed to extract chunks from PDF",
                    "chunks_created": 0
                }
            
            # Prepare data for vector DB
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_metadatas = [chunk["metadata"] for chunk in chunks]
            chunk_ids = [chunk["id"] for chunk in chunks]
            
            # Add to vector database
            success = vector_db.add_chunks(chunk_texts, chunk_metadatas, chunk_ids)
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully indexed {len(chunks)} chunks",
                    "document_name": document_name,
                    "chunks_created": len(chunks)
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to add chunks to vector database",
                    "chunks_created": 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing PDF: {str(e)}",
                "chunks_created": 0
            }
        finally:
            # Clean up uploaded file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    def delete_document(self, document_name: str) -> Dict:
        """Delete all chunks of a document from vector database"""
        try:
            vector_db.delete_by_document(document_name)
            return {
                "success": True,
                "message": f"Document '{document_name}' deleted successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting document: {str(e)}"
            }
    
    def get_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        return vector_db.get_collection_stats()


# Global instance
document_service = DocumentService()
