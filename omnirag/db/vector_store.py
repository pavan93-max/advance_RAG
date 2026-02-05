"""
Vector database for storing text and image embeddings.
Uses Chroma for local vector storage.
"""
import os
import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
from functools import lru_cache
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
        
        # Validate embedding dimensions
        self.text_embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
        self.image_embedding_dim = 512  # CLIP dimension
        
        # Create or get collections with optimized settings
        # Use try-except to handle existing collections with different metadata format
        try:
            # Try to get existing collection first
            self.text_collection = self.client.get_collection(
                name=f"{collection_name}_text"
            )
        except Exception:
            # Collection doesn't exist, create with optimized settings
            self.text_collection = self.client.create_collection(
                name=f"{collection_name}_text",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:ef_construction": 200,  # Better index quality
                    "hnsw:M": 16  # More connections for better recall
                }
            )
        
        try:
            # Try to get existing collection first
            self.image_collection = self.client.get_collection(
                name=f"{collection_name}_images"
            )
        except Exception:
            # Collection doesn't exist, create with optimized settings
            self.image_collection = self.client.create_collection(
                name=f"{collection_name}_images",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:ef_construction": 200,
                    "hnsw:M": 16
                }
            )
        
        # Store metadata separately (Chroma has size limits)
        self.metadata_dir = os.path.join(persist_directory, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Metadata version for schema tracking
        self.metadata_version = "1.1"  # Updated for related_text_chunks
    
    def _save_metadata(self, doc_id: str, metadata: Dict):
        """Save metadata to file (for large data like images)."""
        # Add versioning and timestamp
        metadata_with_version = {
            **metadata,
            'metadata_version': self.metadata_version,
            'last_updated': datetime.now().isoformat()
        }
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_version, f, ensure_ascii=False, indent=2)
    
    @lru_cache(maxsize=1000)
    def _load_metadata_cached(self, doc_id: str) -> Optional[Dict]:
        """Load metadata from file with caching."""
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Validate metadata version and migrate if needed
                    return self._validate_and_migrate_metadata(metadata)
            except Exception as e:
                print(f"Error loading metadata for {doc_id}: {e}")
                return None
        return None
    
    def _load_metadata(self, doc_id: str) -> Optional[Dict]:
        """Load metadata from file (wrapper for cached version)."""
        return self._load_metadata_cached(doc_id)
    
    def clear_metadata_cache(self, doc_id: Optional[str] = None):
        """Clear metadata cache for a specific doc_id or all."""
        # LRU cache doesn't support per-key clearing, so we clear all
        # This is acceptable since cache is small (maxsize=1000)
        self._load_metadata_cached.cache_clear()
    
    def _validate_and_migrate_metadata(self, metadata: Dict) -> Dict:
        """Validate metadata and migrate if needed."""
        version = metadata.get('metadata_version', '1.0')
        
        # Migrate from version 1.0 to 1.1 (add related_text_chunks if missing)
        if version == '1.0':
            if 'related_text_chunks' not in metadata:
                metadata['related_text_chunks'] = []
            if 'related_chunk_ids' not in metadata:
                metadata['related_chunk_ids'] = []
            if 'related_text_content' not in metadata:
                metadata['related_text_content'] = ''
            metadata['metadata_version'] = self.metadata_version
            metadata['last_updated'] = datetime.now().isoformat()
        
        return metadata
    
    def add_text_chunks(self, chunks: List[Dict], upsert: bool = True):
        """
        Add text chunks to vector store with duplicate handling.
        
        Args:
            chunks: List of dicts with keys: text, page, section_heading, chunk_id, linked_image_ids
            upsert: If True, update existing chunks; if False, skip duplicates
        """
        if not chunks:
            return
        
        # Check for duplicates
        ids = [chunk['chunk_id'] for chunk in chunks]
        existing = self.text_collection.get(ids=ids)
        existing_ids = set(existing['ids']) if existing['ids'] else set()
        
        # Filter out duplicates if upsert=False
        if not upsert and existing_ids:
            chunks = [c for c in chunks if c['chunk_id'] not in existing_ids]
            if not chunks:
                return
            ids = [chunk['chunk_id'] for chunk in chunks]
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.text_encoder.encode(texts, show_progress_bar=True)
        
        # Validate embedding dimensions
        if embeddings.shape[1] != self.text_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.text_embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        metadatas = [
            {
                'page': chunk['page'],
                'section_heading': chunk.get('section_heading', ''),
                'linked_image_ids': ','.join(chunk.get('linked_image_ids', [])),
                'is_table': chunk.get('is_table', False),
                'related_chunk_ids': ','.join(chunk.get('related_chunk_ids', []))[:500] if chunk.get('is_table') else ''  # Store chunk IDs for tables
            }
            for chunk in chunks
        ]
        
        # Store full text in metadata file
        for chunk in chunks:
            metadata_dict = {
                'text': chunk['text'],
                'page': chunk['page'],
                'section_heading': chunk.get('section_heading', ''),
                'linked_image_ids': chunk.get('linked_image_ids', []),
                'linked_table_ids': chunk.get('linked_table_ids', [])  # Add linked table IDs
            }
            # Store table metadata if it's a table
            if chunk.get('is_table'):
                metadata_dict['is_table'] = True
                metadata_dict['table_metadata'] = chunk.get('table_metadata', {})
                # Add related text chunks metadata (similar to images)
                metadata_dict['related_text_chunks'] = chunk.get('related_text_chunks', [])
                metadata_dict['related_chunk_ids'] = chunk.get('related_chunk_ids', [])
                metadata_dict['related_text_content'] = chunk.get('related_text_content', '')
            
            self._save_metadata(chunk['chunk_id'], metadata_dict)
            # Clear cache for this chunk
            self.clear_metadata_cache(chunk['chunk_id'])
        
        # Use upsert to handle duplicates
        if upsert:
            self.text_collection.upsert(
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )
        else:
            self.text_collection.add(
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )
    
    def add_images(self, images: List[Dict], upsert: bool = True):
        """
        Add images to vector store with duplicate handling.
        
        Args:
            images: List of dicts with keys: image_id, page, vlm_caption, ocr_text, 
                    surrounding_context, clip_embedding, image_bytes
            upsert: If True, update existing images; if False, skip duplicates
        """
        if not images:
            return
        
        # Filter images with valid CLIP embeddings
        valid_images = [img for img in images if img.get('clip_embedding') is not None]
        
        if not valid_images:
            print("Warning: No images with valid CLIP embeddings to add")
            return
        
        # Check for duplicates
        ids = [img['image_id'] for img in valid_images]
        existing = self.image_collection.get(ids=ids)
        existing_ids = set(existing['ids']) if existing['ids'] else set()
        
        # Filter out duplicates if upsert=False
        if not upsert and existing_ids:
            valid_images = [img for img in valid_images 
                          if img['image_id'] not in existing_ids]
            if not valid_images:
                return
            ids = [img['image_id'] for img in valid_images]
        
        embeddings = np.array([img['clip_embedding'] for img in valid_images])
        
        # Validate embedding dimensions
        if embeddings.shape[1] != self.image_embedding_dim:
            raise ValueError(
                f"Image embedding dimension mismatch: expected {self.image_embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        metadatas = [
            {
                'page': img['page'],
                'vlm_caption': img.get('vlm_caption', '')[:500],  # Truncate for Chroma
                'ocr_text': img.get('ocr_text', '')[:500],
                'related_chunk_ids': ','.join(img.get('related_chunk_ids', []))[:500],  # Store chunk IDs
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
                'image_path': image_path,
                # Add related text chunks metadata
                'related_text_chunks': img.get('related_text_chunks', []),
                'related_chunk_ids': img.get('related_chunk_ids', []),
                'related_text_content': img.get('related_text_content', '')  # For searchability
            })
            # Clear cache for this image
            self.clear_metadata_cache(img['image_id'])
        
        # Use upsert to handle duplicates
        if upsert:
            self.image_collection.upsert(
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )
        else:
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
        
        # Encode query text with CLIP - REQUIRED for image collection (512 dim)
        # Do NOT fallback to text encoder (384 dim) as it will cause dimension mismatch
        if not self.image_processor:
            return []  # CLIP not available, cannot search image collection
        
        query_embedding = self.image_processor.get_clip_text_embedding(query)
        if query_embedding is None:
            # CLIP embedding failed, cannot search
            return []
        
        # Validate embedding dimension matches image collection (512 for CLIP)
        if len(query_embedding) != self.image_embedding_dim:
            raise ValueError(
                f"CLIP embedding dimension mismatch: expected {self.image_embedding_dim}, "
                f"got {len(query_embedding)}"
            )
        
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
                    'related_text_chunks': metadata.get('related_text_chunks', []) if metadata else [],  # Add related text
                    'related_text_content': metadata.get('related_text_content', '') if metadata else '',  # Combined text
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
                    'image_path': metadata.get('image_path', '') if metadata else '',
                    'related_text_chunks': metadata.get('related_text_chunks', []) if metadata else [],  # Add related text
                    'related_text_content': metadata.get('related_text_content', '') if metadata else ''  # Combined text
                })
        
        return formatted_results

