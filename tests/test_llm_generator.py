"""
Tests for LLM generator.
"""
import pytest
from unittest.mock import Mock, patch
from omnirag.models.llm_generator import LLMGenerator


def test_prompt_building():
    """Test prompt building."""
    generator = LLMGenerator(model_type="template")
    
    query = "What is AI?"
    context = "AI is artificial intelligence."
    image_context = ["Figure 1: AI diagram"]
    
    prompt = generator._build_prompt(query, context, image_context)
    
    assert query in prompt
    assert context in prompt
    assert "INSTRUCTIONS" in prompt
    assert "[Page X]" in prompt  # Citation format


def test_citation_format():
    """Test that prompts enforce citation format."""
    generator = LLMGenerator(model_type="template")
    
    prompt = generator._build_prompt("test", "context", [])
    
    # Check for citation instructions
    assert "cite" in prompt.lower() or "citation" in prompt.lower()
    assert "Page" in prompt

