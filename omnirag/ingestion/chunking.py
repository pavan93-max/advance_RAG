"""
Semantic chunking with overlap and metadata preservation.
"""
from typing import List, Dict, Optional
import re


class SemanticChunker:
    """
    Semantic chunking with overlap and metadata preservation.
    Provides better chunking than fixed-size splitting.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk text with metadata preservation and overlap.
        
        Args:
            text: Text to chunk
            metadata: Metadata to preserve for each chunk
            
        Returns:
            List of chunk dictionaries with preserved metadata
        """
        if not text or len(text.strip()) < 50:
            return []
        
        # Use recursive splitting strategy
        chunks = self._recursive_split(text)
        
        result = []
        base_chunk_id = metadata.get('chunk_id', 'chunk')
        page = metadata.get('page', 0)
        section_heading = metadata.get('section_heading', '')
        
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:  # Filter very short chunks
                continue
            
            result.append({
                'text': chunk_text.strip(),
                'chunk_id': f"{base_chunk_id}_{i}",
                'page': page,
                'section_heading': section_heading,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'original_chunk_id': base_chunk_id
            })
        
        return result
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using multiple separators.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Priority order of separators (from largest to smallest)
        separators = [
            "\n\n\n",  # Triple newline (section breaks)
            "\n\n",    # Double newline (paragraph breaks)
            "\n",      # Single newline
            ". ",      # Sentence endings
            " ",       # Word boundaries
            ""         # Character boundaries (last resort)
        ]
        
        # Try each separator
        for separator in separators:
            if separator == "":
                # Last resort: split by character
                return self._split_by_size(text)
            
            chunks = text.split(separator)
            
            # Check if chunks are appropriate size
            if self._chunks_appropriate_size(chunks):
                # Merge chunks that are too small
                merged = self._merge_small_chunks(chunks, separator)
                return merged
        
        # Fallback
        return self._split_by_size(text)
    
    def _chunks_appropriate_size(self, chunks: List[str]) -> bool:
        """Check if chunks are within acceptable size range."""
        if not chunks:
            return False
        
        # Most chunks should be within reasonable size
        sizes = [len(chunk) for chunk in chunks if chunk.strip()]
        if not sizes:
            return False
        
        avg_size = sum(sizes) / len(sizes)
        # Average should be between chunk_size/2 and chunk_size*2
        return (self.chunk_size / 2) <= avg_size <= (self.chunk_size * 2)
    
    def _merge_small_chunks(self, chunks: List[str], separator: str) -> List[str]:
        """Merge chunks that are too small."""
        result = []
        current_chunk = []
        current_size = 0
        
        for chunk in chunks:
            chunk_size = len(chunk)
            
            # If adding this chunk would exceed size, finalize current
            if current_size + chunk_size > self.chunk_size and current_chunk:
                result.append(separator.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(chunk)
            current_size += chunk_size
        
        # Add remaining chunk
        if current_chunk:
            result.append(separator.join(current_chunk))
        
        return result
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text by character count with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for sentence ending near the end
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 2
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end + 1
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_with_overlap(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk text with explicit overlap between chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to preserve
            
        Returns:
            List of overlapping chunks
        """
        chunks = self.chunk_text(text, metadata)
        
        # Add overlap information
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add previous chunk's ending as context
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk['text'][-self.chunk_overlap:]
                chunk['overlap_prefix'] = overlap_text
            
            if i < len(chunks) - 1:
                # Add next chunk's beginning as context
                next_chunk = chunks[i + 1]
                overlap_text = next_chunk['text'][:self.chunk_overlap]
                chunk['overlap_suffix'] = overlap_text
        
        return chunks

