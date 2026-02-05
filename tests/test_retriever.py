"""
Tests for hybrid retriever.
"""
import pytest
from unittest.mock import Mock, MagicMock
from omnirag.retrieval.retriever import HybridRetriever


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.text_collection = Mock()
    store.text_collection.get = Mock(return_value={'ids': [], 'metadatas': []})
    store.image_collection = Mock()
    store.image_collection.get = Mock(return_value={'ids': [], 'metadatas': []})
    store.image_collection.count = Mock(return_value=0)
    store.search_text = Mock(return_value=[])
    store.search_images_by_clip = Mock(return_value=[])
    store.get_images_by_ids = Mock(return_value=[])
    store._load_metadata = Mock(return_value=None)
    return store


def test_retriever_initialization(mock_vector_store):
    """Test retriever initialization."""
    retriever = HybridRetriever(mock_vector_store)
    assert retriever.vector_store == mock_vector_store
    assert retriever.use_mmr == True


def test_text_similarity():
    """Test text similarity calculation."""
    retriever = HybridRetriever(Mock())
    
    # Identical texts
    assert retriever._text_similarity("hello world", "hello world") == 1.0
    
    # Different texts
    sim = retriever._text_similarity("hello world", "goodbye world")
    assert 0.0 <= sim <= 1.0
    
    # Empty texts
    assert retriever._text_similarity("", "hello") == 0.0


def test_mmr_diversity():
    """Test MMR diversity selection."""
    retriever = HybridRetriever(Mock())
    
    # Create mock results
    results = [
        {'text': 'result 1', 'hybrid_score': 0.9},
        {'text': 'result 2', 'hybrid_score': 0.8},
        {'text': 'result 3', 'hybrid_score': 0.7},
        {'text': 'result 4', 'hybrid_score': 0.6},
    ]
    
    diverse = retriever._mmr_diversity(results, "query", top_k=2)
    assert len(diverse) == 2
    assert diverse[0] == results[0]  # Best result always included

