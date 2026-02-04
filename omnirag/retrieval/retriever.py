"""
Hybrid retrieval: text (semantic + keyword) and image search.
"""
import re
from typing import List, Dict, Tuple
from collections import Counter

from omnirag.db.vector_store import VectorStore


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def bm25_keyword_search(self, query: str, text_results: List[Dict], k: int = 5) -> List[Dict]:
        """
        Simple BM25-like keyword matching on retrieved text results.
        """
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        scored_results = []
        for result in text_results:
            text = result['text'].lower()
            text_terms = set(re.findall(r'\b\w+\b', text))
            
            # Simple term overlap score
            overlap = len(query_terms & text_terms)
            if len(query_terms) > 0:
                score = overlap / len(query_terms)
            else:
                score = 0
            
            scored_results.append({
                **result,
                'keyword_score': score
            })
        
        # Sort by keyword score
        scored_results.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored_results[:k]
    
    def search_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid text search: semantic + keyword."""
        # Semantic search
        semantic_results = self.vector_store.search_text(query, top_k=top_k * 2)
        
        # Re-rank with keyword matching
        hybrid_results = self.bm25_keyword_search(query, semantic_results, k=top_k)
        
        return hybrid_results
    
    def search_images(self, query: str, text_results: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Multi-strategy image search:
        1. CLIP similarity
        2. Caption/OCR text match
        3. Images linked to retrieved text
        """
        image_results = []
        
        # 1. CLIP-based search
        clip_results = self.vector_store.search_images_by_clip(query, top_k=top_k)
        for img in clip_results:
            img['search_method'] = 'clip'
            image_results.append(img)
        
        # 2. Search by caption/OCR text
        query_lower = query.lower()
        all_images = self.vector_store.image_collection.get()
        
        if all_images['ids']:
            for i, img_id in enumerate(all_images['ids']):
                metadata = self.vector_store._load_metadata(img_id)
                if not metadata:
                    continue
                
                caption = (metadata.get('vlm_caption', '') + ' ' + 
                          metadata.get('ocr_text', '')).lower()
                
                if any(term in caption for term in query_lower.split()):
                    img_data = {
                        'id': img_id,
                        'page': all_images['metadatas'][i].get('page', 0),
                        'vlm_caption': metadata.get('vlm_caption', ''),
                        'ocr_text': metadata.get('ocr_text', ''),
                        'surrounding_context': metadata.get('surrounding_context', ''),
                        'image_path': metadata.get('image_path', ''),
                        'score': 0.7,  # Fixed score for text match
                        'search_method': 'text_match'
                    }
                    # Avoid duplicates
                    if not any(r['id'] == img_id for r in image_results):
                        image_results.append(img_data)
        
        # 3. Images linked to retrieved text
        linked_image_ids = set()
        for text_result in text_results:
            linked_ids = text_result.get('linked_image_ids', [])
            if isinstance(linked_ids, str):
                linked_ids = linked_ids.split(',') if linked_ids else []
            linked_image_ids.update(linked_ids)
        
        if linked_image_ids:
            linked_images = self.vector_store.get_images_by_ids(list(linked_image_ids))
            for img in linked_images:
                img['search_method'] = 'linked'
                img['score'] = 0.8  # Higher score for linked images
                # Avoid duplicates
                if not any(r['id'] == img['id'] for r in image_results):
                    image_results.append(img)
        
        # Sort by score and return top_k
        image_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return image_results[:top_k]
    
    def retrieve(self, query: str, text_top_k: int = 5, image_top_k: int = 3) -> Dict:
        """
        Main retrieval function.
        Returns: {
            'text_results': [...],
            'image_results': [...]
        }
        """
        # Text retrieval
        text_results = self.search_text(query, top_k=text_top_k)
        
        # Image retrieval
        image_results = self.search_images(query, text_results, top_k=image_top_k)
        
        return {
            'text_results': text_results,
            'image_results': image_results
        }

