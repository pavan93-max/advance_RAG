"""
Vector database for storing text and image embeddings.
Uses Chroma for local vector storage.
"""
import os
import json
import hashlib
from typing import List, Dict, Optional, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Vector database for text and image storage."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "omnirag", 
                 image_processor=None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.image_processor = image_processor
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize sentence transformer for text embeddings
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collections
        self.text_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_text",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_images",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Store metadata separately (Chroma has size limits)
        self.metadata_dir = os.path.join(persist_directory, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def _save_metadata(self, doc_id: str, metadata: Dict):
        """Save metadata to file (for large data like images)."""
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self, doc_id: str) -> Optional[Dict]:
        """Load metadata from file."""
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def add_text_chunks(self, chunks: List[Dict]):
        """
        Add text chunks to vector store.
        chunks: List of dicts with keys: text, page, section_heading, chunk_id, linked_image_ids
        """
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.text_encoder.encode(texts, show_progress_bar=True)
        
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [
            {
                'page': chunk['page'],
                'section_heading': chunk.get('section_heading', ''),
                'linked_image_ids': ','.join(chunk.get('linked_image_ids', []))
            }
            for chunk in chunks
        ]
        
        # Store full text in metadata file
        for chunk in chunks:
            self._save_metadata(chunk['chunk_id'], {
                'text': chunk['text'],
                'page': chunk['page'],
                'section_heading': chunk.get('section_heading', ''),
                'linked_image_ids': chunk.get('linked_image_ids', [])
            })
        
        self.text_collection.add(
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )
    
    def add_images(self, images: List[Dict]):
        """
        Add images to vector store.
        images: List of dicts with keys: image_id, page, vlm_caption, ocr_text, 
                surrounding_context, clip_embedding, image_bytes
        """
        if not images:
            return
        
        # Filter images with valid CLIP embeddings
        valid_images = [img for img in images if img.get('clip_embedding') is not None]
        
        if not valid_images:
            print("Warning: No images with valid CLIP embeddings to add")
            return
        
        embeddings = np.array([img['clip_embedding'] for img in valid_images])
        ids = [img['image_id'] for img in valid_images]
        
        metadatas = [
            {
                'page': img['page'],
                'vlm_caption': img.get('vlm_caption', '')[:500],  # Truncate for Chroma
                'ocr_text': img.get('ocr_text', '')[:500],
            }
            for img in valid_images
        ]
        
        # Store full metadata in files
        for img in valid_images:
            # Save image bytes to file
            image_dir = os.path.join(self.metadata_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{img['image_id']}.png")
            
            if 'image_data' in img:
                img['image_data'].save(image_path)
            
            self._save_metadata(img['image_id'], {
                'page': img['page'],
                'vlm_caption': img.get('vlm_caption', ''),
                'ocr_text': img.get('ocr_text', ''),
                'surrounding_context': img.get('surrounding_context', ''),
                'original_caption': img.get('original_caption', ''),
                'image_path': image_path
            })
        
        self.image_collection.add(
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )
    
    def search_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search in text collection."""
        query_embedding = self.text_encoder.encode([query])[0]
        
        results = self.text_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                metadata = self._load_metadata(doc_id)
                
                formatted_results.append({
                    'id': doc_id,
                    'text': metadata['text'] if metadata else '',
                    'page': results['metadatas'][0][i].get('page', 0),
                    'section_heading': metadata.get('section_heading', '') if metadata else '',
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'linked_image_ids': metadata.get('linked_image_ids', []) if metadata else []
                })
        
        return formatted_results
    
    def search_images_by_clip(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search images using CLIP (text-to-image similarity)."""
        if not self.image_collection.count():
            return []
        
        # Encode query text with CLIP if available, otherwise use text encoder
        if self.image_processor:
            query_embedding = self.image_processor.get_clip_text_embedding(query)
            if query_embedding is None:
                # Fallback to text encoder
                query_embedding = self.text_encoder.encode([query])[0]
        else:
            query_embedding = self.text_encoder.encode([query])[0]
        
        results = self.image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                img_id = results['ids'][0][i]
                metadata = self._load_metadata(img_id)
                
                formatted_results.append({
                    'id': img_id,
                    'page': results['metadatas'][0][i].get('page', 0),
                    'vlm_caption': metadata.get('vlm_caption', '') if metadata else '',
                    'ocr_text': metadata.get('ocr_text', '') if metadata else '',
                    'surrounding_context': metadata.get('surrounding_context', '') if metadata else '',
                    'image_path': metadata.get('image_path', '') if metadata else '',
                    'score': 1 - results['distances'][0][i]
                })
        
        return formatted_results
    
    def get_images_by_ids(self, image_ids: List[str]) -> List[Dict]:
        """Retrieve images by their IDs."""
        results = self.image_collection.get(ids=image_ids)
        
        formatted_results = []
        if results['ids']:
            for i, img_id in enumerate(results['ids']):
                metadata = self._load_metadata(img_id)
                formatted_results.append({
                    'id': img_id,
                    'page': results['metadatas'][i].get('page', 0),
                    'vlm_caption': metadata.get('vlm_caption', '') if metadata else '',
                    'ocr_text': metadata.get('ocr_text', '') if metadata else '',
                    'surrounding_context': metadata.get('surrounding_context', '') if metadata else '',
                    'image_path': metadata.get('image_path', '') if metadata else ''
                })
        
        return formatted_results

