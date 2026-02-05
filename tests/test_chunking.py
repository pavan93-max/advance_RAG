"""
Tests for semantic chunking.
"""
import pytest
from omnirag.ingestion.chunking import SemanticChunker


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
    assert chunker.chunk_size == 1000
    assert chunker.chunk_overlap == 200


def test_chunk_text():
    """Test text chunking."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    
    text = "This is a test. " * 20  # Long text
    metadata = {
        'chunk_id': 'test_chunk',
        'page': 1,
        'section_heading': 'Test Section'
    }
    
    chunks = chunker.chunk_text(text, metadata)
    
    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('page' in chunk for chunk in chunks)
    assert all(chunk['page'] == 1 for chunk in chunks)


def test_chunk_short_text():
    """Test chunking short text."""
    chunker = SemanticChunker()
    
    text = "Short text"
    metadata = {'chunk_id': 'short', 'page': 1}
    
    chunks = chunker.chunk_text(text, metadata)
    # Short text might return empty or single chunk
    assert isinstance(chunks, list)

