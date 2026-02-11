import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import openai
import threading
from app.core.config import settings


class VectorDBManager:
    """
    Manages ChromaDB for agricultural knowledge storage and retrieval.
    Handles embedding generation and semantic search with API key rotation.
    """
    
    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"description": "Agricultural knowledge base with PDF chunks"}
        )
        
        # Load OpenAI API keys for embeddings
        self.openai_keys = settings.load_api_keys("openai")
        if not self.openai_keys:
            raise ValueError("No OpenAI API keys found for embeddings")
        
        # Round-robin counter for embeddings
        self._embedding_counter = 0
        self._embedding_lock = threading.Lock()
        
        print(f"✓ VectorDB initialized with {self.collection.count()} chunks")
        print(f"✓ Using {len(self.openai_keys)} OpenAI keys for embeddings")
    
    def _get_next_openai_key(self) -> str:
        """Get next OpenAI API key for embeddings (thread-safe)"""
        with self._embedding_lock:
            key = self.openai_keys[self._embedding_counter % len(self.openai_keys)]
            self._embedding_counter += 1
            return key
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI with key rotation"""
        try:
            # Get next API key
            api_key = self._get_next_openai_key()
            client = openai.OpenAI(api_key=api_key)
            
            response = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
    
    def add_chunks(self, chunks: List[str], metadatas: List[Dict], ids: List[str]):
        """Add PDF chunks to vector database"""
        try:
            # Generate embeddings
            embeddings = self.get_embeddings(chunks)
            
            if not embeddings:
                raise Exception("Failed to generate embeddings")
            
            # Add to collection
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✓ Added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding chunks to VectorDB: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        n_results: int = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: User's query
            n_results: Number of results to return (default: TOP_K_RESULTS from settings)
            filters: Metadata filters (e.g., {"crop_type": "tomato"})
        
        Returns:
            List of dicts with keys: text, metadata, score
        """
        try:
            if n_results is None:
                n_results = settings.TOP_K_RESULTS
            
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters if filters else None
            )
            
            # Format results
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    # Only include results above threshold
                    if similarity_score >= settings.SIMILARITY_THRESHOLD:
                        formatted_results.append({
                            "text": doc,
                            "metadata": results['metadatas'][0][i],
                            "score": similarity_score
                        })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching VectorDB: {e}")
            return []
    
    def delete_by_document(self, document_name: str):
        """Delete all chunks from a specific document"""
        try:
            self.collection.delete(
                where={"document_name": document_name}
            )
            print(f"✓ Deleted chunks from document: {document_name}")
        except Exception as e:
            print(f"Error deleting document chunks: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": settings.COLLECTION_NAME
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}


# Global instance
vector_db = VectorDBManager()
