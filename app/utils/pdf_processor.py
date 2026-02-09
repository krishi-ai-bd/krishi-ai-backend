import re
import uuid
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
import tiktoken
from app.core.config import settings


class PDFProcessor:
    """
    Processes agricultural PDF documents with semantic chunking.
    Splits by sections/headers while maintaining context.
    """
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF file.
        
        Returns:
            Tuple of (extracted_text, total_pages)
        """
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            text_content = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    # Add page marker for metadata tracking
                    text_content.append(f"[PAGE_{page_num}]\n{text}")
            
            full_text = "\n".join(text_content)
            return full_text, total_pages
            
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return "", 0
    
    def detect_sections(self, text: str) -> List[Dict]:
        """
        Detect sections in text based on headers.
        Looks for patterns like:
        - ALL CAPS HEADERS
        - Numbered sections (1. 2. 3.)
        - Chapter/Section keywords
        """
        sections = []
        
        # Patterns for agricultural document headers
        header_patterns = [
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS LINE
            r'^\d+\.\s+[A-Z][\w\s]+$',  # 1. Title Format
            r'^Chapter\s+\d+:?[\s\w]+',  # Chapter X: Title
            r'^Section\s+\d+:?[\s\w]+',  # Section X: Title
            r'^\d+\.\d+\s+[\w\s]+$',  # 1.1 Subsection
        ]
        
        lines = text.split('\n')
        current_section = {
            "title": "Introduction",
            "content": [],
            "start_line": 0,
            "page": 1
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Track page numbers
            page_match = re.match(r'\[PAGE_(\d+)\]', line)
            if page_match:
                current_section["page"] = int(page_match.group(1))
                continue
            
            # Check if line is a header
            is_header = any(re.match(pattern, line) for pattern in header_patterns)
            
            if is_header and len(line) > 5 and len(line) < 100:
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": line,
                    "content": [],
                    "start_line": i,
                    "page": current_section["page"]
                }
            else:
                if line:  # Skip empty lines
                    current_section["content"].append(line)
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
        
        # If no sections detected, treat entire document as one section
        if not sections:
            sections = [{
                "title": "Full Document",
                "content": [line for line in lines if line.strip() and not line.startswith('[PAGE_')],
                "start_line": 0,
                "page": 1
            }]
        
        return sections
    
    def chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            max_tokens: Max tokens per chunk (default: CHUNK_SIZE from settings)
        
        Returns:
            List of text chunks
        """
        if max_tokens is None:
            max_tokens = self.chunk_size
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max, split it
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by commas or semicolons
                parts = re.split(r'[,;]\s+', sentence)
                for part in parts:
                    part_tokens = self.count_tokens(part)
                    if current_tokens + part_tokens <= max_tokens:
                        current_chunk.append(part)
                        current_tokens += part_tokens
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [part]
                        current_tokens = part_tokens
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Keep overlap from previous chunk
                overlap_chunk = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_chunk.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_chunk + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_pdf(self, pdf_path: str, document_name: str, metadata: Dict = None) -> List[Dict]:
        """
        Complete PDF processing pipeline: extract → detect sections → chunk
        
        Args:
            pdf_path: Path to PDF file
            document_name: Name for the document
            metadata: Additional metadata (e.g., {"crop_type": "tomato", "category": "pest_control"})
        
        Returns:
            List of chunk dictionaries ready for vector DB
        """
        try:
            # Extract text
            full_text, total_pages = self.extract_text_from_pdf(pdf_path)
            
            if not full_text:
                raise Exception("No text extracted from PDF")
            
            # Detect sections
            sections = self.detect_sections(full_text)
            
            print(f"✓ Detected {len(sections)} sections in {total_pages} pages")
            
            # Process each section
            all_chunks = []
            
            for section in sections:
                section_text = " ".join(section["content"])
                
                # Skip very short sections
                if self.count_tokens(section_text) < 50:
                    continue
                
                # Chunk the section
                chunks = self.chunk_text(section_text)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    
                    chunk_metadata = {
                        "document_name": document_name,
                        "section_title": section["title"],
                        "page_start": section["page"],
                        "chunk_index": i,
                        "total_chunks_in_section": len(chunks),
                        "token_count": self.count_tokens(chunk)
                    }
                    
                    # Add custom metadata if provided
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    all_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": chunk_metadata
                    })
            
            print(f"✓ Created {len(all_chunks)} chunks from {document_name}")
            
            return all_chunks
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []


# Global instance
pdf_processor = PDFProcessor()
