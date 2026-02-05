"""
Production-grade hybrid retrieval with BM25, dense search, reranking, and MMR.
"""
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("Warning: CrossEncoder not available. Reranking will be disabled.")

from omnirag.db.vector_store import VectorStore
from omnirag.utils.logger import logger


class HybridRetriever:
    """
    Production-grade hybrid retrieval combining:
    - Dense semantic search (sentence transformers)
    - Sparse keyword search (BM25)
    - Cross-encoder reranking
    - MMR for diversity
    """
    
    def __init__(self, vector_store: VectorStore, 
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_rerank: bool = True,
                 use_mmr: bool = True,
                 mmr_lambda: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store instance
            rerank_model: Cross-encoder model for reranking
            use_rerank: Whether to use reranking
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR lambda parameter (0.5 = balance relevance/diversity)
        """
        self.vector_store = vector_store
        self.use_rerank = use_rerank and RERANKER_AVAILABLE
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        
        # Initialize reranker
        if self.use_rerank:
            try:
                self.reranker = CrossEncoder(rerank_model)
                logger.info(f"Reranker initialized: {rerank_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.use_rerank = False
                self.reranker = None
        else:
            self.reranker = None
        
        # Build BM25 index
        self.bm25_index = None
        self.doc_ids = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all text chunks."""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available. Sparse search will be disabled.")
            self.bm25_index = None
            self.doc_ids = []
            return
        
        try:
            all_docs = self.vector_store.text_collection.get()
            if not all_docs['ids']:
                self.bm25_index = None
                self.doc_ids = []
                logger.info("No documents in collection. BM25 index not built.")
                return
            
            # Get full texts from metadata (include related text for tables)
            texts = []
            self.doc_ids = []
            
            for doc_id in all_docs['ids']:
                metadata = self.vector_store._load_metadata(doc_id)
                if metadata and 'text' in metadata:
                    # Tokenize text for BM25
                    text = metadata['text'].lower()
                    
                    # For tables, include related text content for better searchability
                    if metadata.get('is_table') and metadata.get('related_text_content'):
                        text += ' ' + metadata['related_text_content'].lower()
                    
                    tokens = re.findall(r'\b\w+\b', text)
                    if tokens:  # Only add non-empty documents
                        texts.append(tokens)
                        self.doc_ids.append(doc_id)
            
            if texts:
                self.bm25_index = BM25Okapi(texts)
                logger.info(f"BM25 index built with {len(texts)} documents")
            else:
                self.bm25_index = None
                self.doc_ids = []
                logger.warning("No valid texts found. BM25 index not built.")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25_index = None
            self.doc_ids = []
    
    def _expand_query(self, query: str) -> str:
        """
        Simple query expansion (can be enhanced with LLM).
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # For now, return as-is
        # Can be enhanced with:
        # - Synonym expansion
        # - LLM-based query expansion
        # - Related term extraction
        return query
    
    def search_text(self, query: str, top_k: int = 5, 
                   use_rerank: Optional[bool] = None,
                   page_filter: Optional[List[int]] = None,
                   section_filter: Optional[List[str]] = None,
                   boost_tables: bool = True) -> List[Dict]:
        """
        True hybrid search with reranking and optional filtering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_rerank: Override default reranking setting
            page_filter: Optional page number filter
            section_filter: Optional section heading filter
            
        Returns:
            List of retrieved documents with scores
        """
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank
        
        # 1. Query expansion
        expanded_query = self._expand_query(query)
        
        # 2. Dense search (semantic)
        dense_results = self.vector_store.search_text(expanded_query, top_k=top_k * 3)
        dense_scores = {r['id']: r['score'] for r in dense_results}
        
        # 3. Sparse search (BM25)
        sparse_results = []
        if self.bm25_index:
            try:
                query_terms = re.findall(r'\b\w+\b', expanded_query.lower())
                if query_terms:
                    bm25_scores = self.bm25_index.get_scores(query_terms)
                    
                    # Get top BM25 results
                    top_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]
                    
                    for idx in top_indices:
                        if idx < len(self.doc_ids) and bm25_scores[idx] > 0:
                            doc_id = self.doc_ids[idx]
                            metadata = self.vector_store._load_metadata(doc_id)
                            if metadata:
                                sparse_results.append({
                                    'id': doc_id,
                                    'text': metadata['text'],
                                    'page': metadata.get('page', 0),
                                    'section_heading': metadata.get('section_heading', ''),
                                    'bm25_score': float(bm25_scores[idx]),
                                    'dense_score': dense_scores.get(doc_id, 0.0),
                                    'linked_image_ids': metadata.get('linked_image_ids', [])
                                })
            except Exception as e:
                logger.error(f"Error in BM25 search: {e}")
        
        # 4. Combine and deduplicate
        all_results = {}
        
        # Add dense results
        for result in dense_results:
            all_results[result['id']] = {
                **result,
                'dense_score': result['score'],
                'bm25_score': 0.0
            }
        
        # Add/update with sparse results
        for result in sparse_results:
            if result['id'] in all_results:
                all_results[result['id']]['bm25_score'] = result['bm25_score']
            else:
                all_results[result['id']] = result
        
        # 5. Hybrid scoring (weighted combination)
        # Normalize scores to [0, 1] range
        if all_results:
            max_dense = max((r.get('dense_score', 0.0) for r in all_results.values()), default=1.0)
            max_bm25 = max((r.get('bm25_score', 0.0) for r in all_results.values()), default=1.0)
            
            if max_dense == 0:
                max_dense = 1.0
            if max_bm25 == 0:
                max_bm25 = 1.0
            
            # Check if query mentions tables (for boosting)
            query_lower = query.lower()
            table_keywords = ['table', 'tables', 'data', 'dataset', 'results', 'comparison', 'performance', 'metrics', 'statistics']
            is_table_query = any(keyword in query_lower for keyword in table_keywords)
            
            for result in all_results.values():
                dense_norm = result.get('dense_score', 0.0) / max_dense
                bm25_norm = result.get('bm25_score', 0.0) / max_bm25
                
                # Weighted combination (60% dense, 40% sparse)
                hybrid_score = 0.6 * dense_norm + 0.4 * bm25_norm
                
                # Boost table chunks if query mentions tables
                if boost_tables and is_table_query:
                    metadata = self.vector_store._load_metadata(result['id'])
                    if metadata and metadata.get('is_table'):
                        hybrid_score *= 1.5  # 50% boost for tables
                
                result['hybrid_score'] = hybrid_score
        
        # Sort by hybrid score
        combined_results = sorted(
            all_results.values(),
            key=lambda x: x.get('hybrid_score', 0.0),
            reverse=True
        )[:top_k * 2]
        
        # 6. Apply metadata filters
        if page_filter:
            combined_results = [r for r in combined_results 
                              if r.get('page') in page_filter]
        if section_filter:
            combined_results = [r for r in combined_results 
                              if r.get('section_heading') in section_filter]
        
        # 7. Rerank with cross-encoder
        if use_rerank and self.reranker and combined_results:
            try:
                query_text_pairs = [
                    [query, result['text']] for result in combined_results
                ]
                rerank_scores = self.reranker.predict(query_text_pairs)
                
                for i, result in enumerate(combined_results):
                    result['rerank_score'] = float(rerank_scores[i])
                
                # Sort by rerank score
                combined_results = sorted(
                    combined_results,
                    key=lambda x: x.get('rerank_score', 0.0),
                    reverse=True
                )
            except Exception as e:
                logger.error(f"Error in reranking: {e}")
        
        return combined_results[:top_k]
    
    def _mmr_diversity(self, results: List[Dict], query: str, 
                      top_k: int) -> List[Dict]:
        """
        Maximal Marginal Relevance for diversity.
        
        Args:
            results: List of search results
            query: Original query
            top_k: Number of diverse results to return
            
        Returns:
            Diverse subset of results
        """
        if len(results) <= top_k or not self.use_mmr:
            return results[:top_k]
        
        selected = [results[0]]  # Start with best result
        remaining = results[1:]
        
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance to query (use rerank_score if available, else hybrid_score)
                relevance = candidate.get('rerank_score', 
                                         candidate.get('hybrid_score', 0.0))
                
                # Diversity (max similarity to selected)
                max_sim = 0.0
                for sel in selected:
                    sim = self._text_similarity(candidate['text'], sel['text'])
                    max_sim = max(max_sim, sim)
                
                # MMR score: lambda * relevance - (1 - lambda) * max_similarity
                mmr_score = (self.mmr_lambda * relevance - 
                           (1 - self.mmr_lambda) * max_sim)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity (Jaccard similarity on words).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0, 1]
        """
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def search_images(self, query: str, text_results: List[Dict], top_k: int = 3,
                     min_clip_score: float = 0.5, min_text_score: float = 0.5) -> List[Dict]:
        """
        Multi-strategy image search with relevance filtering:
        1. CLIP similarity (with minimum score threshold)
        2. Caption/OCR text match (with higher threshold)
        3. Images linked to retrieved text (only if text is highly relevant)
        
        Args:
            query: Search query
            text_results: Retrieved text results
            top_k: Number of images to return
            min_clip_score: Minimum CLIP similarity score (0-1)
            min_text_score: Minimum text match score (0-1)
            
        Returns:
            List of relevant image results
        """
        image_results = []
        
        # 1. CLIP-based search with minimum score threshold
        clip_results = self.vector_store.search_images_by_clip(query, top_k=top_k * 2)
        for img in clip_results:
            score = img.get('score', 0.0)
            # Only include if score meets threshold
            if score >= min_clip_score:
                img['search_method'] = 'clip'
                image_results.append(img)
        
        # 2. Search by caption/OCR text with stricter matching
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'}
        query_terms = query_terms - stop_words
        
        if len(query_terms) > 0:
            all_images = self.vector_store.image_collection.get()
            if all_images['ids']:
                for i, img_id in enumerate(all_images['ids']):
                    metadata = self.vector_store._load_metadata(img_id)
                    if not metadata:
                        continue
                    
                    # Include related text content in search (caption + OCR + related text chunks)
                    related_text = metadata.get('related_text_content', '')
                    vlm_caption = metadata.get('vlm_caption', '').lower()
                    ocr_text = metadata.get('ocr_text', '').lower()
                    related_text_lower = related_text.lower()
                    
                    # Calculate separate scores for different text sources
                    vlm_terms = set(re.findall(r'\b\w+\b', vlm_caption))
                    ocr_terms = set(re.findall(r'\b\w+\b', ocr_text))
                    related_terms = set(re.findall(r'\b\w+\b', related_text_lower))
                    
                    # Weighted scoring: VLM caption (40%), OCR (30%), Related text (30%)
                    vlm_score = len(query_terms & vlm_terms) / len(query_terms | vlm_terms) if (query_terms | vlm_terms) else 0.0
                    ocr_score = len(query_terms & ocr_terms) / len(query_terms | ocr_terms) if (query_terms | ocr_terms) else 0.0
                    related_score = len(query_terms & related_terms) / len(query_terms | related_terms) if (query_terms | related_terms) else 0.0
                    
                    # Weighted combination
                    score = 0.4 * vlm_score + 0.3 * ocr_score + 0.3 * related_score
                    
                    # Boost score if related text matches (indicates strong contextual relevance)
                    if related_score > 0.3:
                        score *= 1.2  # 20% boost for strong related text match
                    
                    # Stricter threshold - require meaningful overlap
                    if score >= min_text_score:
                        img_data = {
                            'id': img_id,
                            'page': all_images['metadatas'][i].get('page', 0),
                            'vlm_caption': metadata.get('vlm_caption', ''),
                            'ocr_text': metadata.get('ocr_text', ''),
                            'surrounding_context': metadata.get('surrounding_context', ''),
                            'image_path': metadata.get('image_path', ''),
                            'related_text_chunks': metadata.get('related_text_chunks', []),  # Add related text
                            'related_text_content': metadata.get('related_text_content', ''),  # Combined text
                            'score': score,
                            'search_method': 'text_match'
                        }
                        # Avoid duplicates
                        if not any(r['id'] == img_id for r in image_results):
                            image_results.append(img_data)
        
        # 3. Images linked to retrieved text (only if text is highly relevant)
        # Only include linked images if the text result has high relevance
        linked_image_ids = set()
        for text_result in text_results:
            # Only link images from highly relevant text results
            text_score = text_result.get('rerank_score', 
                                        text_result.get('hybrid_score', 
                                                      text_result.get('score', 0.0)))
            if text_score >= 0.5:  # Only link from relevant text
                linked_ids = text_result.get('linked_image_ids', [])
                if isinstance(linked_ids, str):
                    linked_ids = linked_ids.split(',') if linked_ids else []
                linked_image_ids.update(linked_ids)
        
        if linked_image_ids:
            linked_images = self.vector_store.get_images_by_ids(list(linked_image_ids))
            for img in linked_images:
                # Verify linked image is actually relevant by checking if it appears in CLIP results
                is_relevant = any(r['id'] == img['id'] and r.get('score', 0) >= min_clip_score 
                                 for r in clip_results)
                if is_relevant:
                    img['search_method'] = 'linked'
                    img['score'] = img.get('score', 0.7)  # Moderate score for linked images
                    # Avoid duplicates
                    if not any(r['id'] == img['id'] for r in image_results):
                        image_results.append(img)
        
        # Sort by score and filter by minimum relevance
        image_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Final filter: only return images with meaningful relevance
        filtered_results = [img for img in image_results if img.get('score', 0) >= 0.4]
        
        return filtered_results[:top_k]
    
    def retrieve(self, query: str, text_top_k: int = 5, 
                image_top_k: int = 3) -> Dict:
        """
        Main retrieval function with MMR diversity.
        
        Args:
            query: Search query
            text_top_k: Number of text results
            image_top_k: Number of image results
            
        Returns:
            Dictionary with text_results and image_results
        """
        # Text retrieval
        text_results = self.search_text(query, top_k=text_top_k * 2)
        
        # Apply MMR for diversity
        text_results = self._mmr_diversity(text_results, query, top_k=text_top_k)
        
        # Image retrieval
        image_results = self.search_images(query, text_results, top_k=image_top_k)
        
        return {
            'text_results': text_results[:text_top_k],
            'image_results': image_results
        }
