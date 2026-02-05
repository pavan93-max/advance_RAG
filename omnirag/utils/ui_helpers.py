"""
UI helper functions for formatting metadata and results for display.
"""
from typing import List, Dict, Optional


def format_image_with_metadata(image_result: Dict) -> Dict:
    """
    Format image result with metadata for display.
    
    Args:
        image_result: Image result dict from retrieval
        
    Returns:
        Formatted dict for UI display
    """
    return {
        'image': image_result.get('image_path', ''),
        'image_id': image_result.get('id', ''),
        'caption': image_result.get('vlm_caption', ''),
        'ocr_text': image_result.get('ocr_text', ''),
        'context': format_related_chunks(image_result.get('related_text_chunks', [])),
        'page': image_result.get('page', 0),
        'score': image_result.get('score', 0.0),
        'search_method': image_result.get('search_method', 'unknown')
    }


def format_related_chunks(chunks: List[Dict], max_length: int = 100) -> List[str]:
    """
    Format related chunks for display.
    
    Args:
        chunks: List of chunk dicts
        max_length: Maximum length of text preview
        
    Returns:
        List of formatted chunk strings
    """
    formatted = []
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id', 'N/A')
        section = chunk.get('section_heading', '')
        text = chunk.get('text', '')
        
        # Truncate text
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Format with section heading if available
        if section:
            formatted.append(f"[{section}] {text}")
        else:
            formatted.append(f"{text}")
    
    return formatted


def format_table_with_metadata(table_result: Dict) -> Dict:
    """
    Format table result with metadata for display.
    
    Args:
        table_result: Table result dict from retrieval
        
    Returns:
        Formatted dict for UI display
    """
    table_meta = table_result.get('table_metadata', {})
    return {
        'table_id': table_result.get('id', ''),
        'text': table_result.get('text', ''),
        'page': table_result.get('page', 0),
        'section': table_result.get('section_heading', ''),
        'context': format_related_chunks(table_result.get('related_text_chunks', [])),
        'table_html': table_meta.get('table_html', ''),
        'table_markdown': table_meta.get('table_markdown', ''),
        'num_rows': table_meta.get('num_rows', 0),
        'num_cols': table_meta.get('num_cols', 0),
        'score': table_result.get('rerank_score', table_result.get('hybrid_score', 0.0))
    }


def format_citation(citation: Dict) -> str:
    """
    Format a citation for display.
    
    Args:
        citation: Citation dict
        
    Returns:
        Formatted citation string
    """
    page = citation.get('page', 0)
    section = citation.get('section', '')
    text = citation.get('text', '')
    is_table = citation.get('is_table', False)
    
    prefix = "Table" if is_table else "Document"
    section_str = f" [{section}]" if section else ""
    
    # Truncate text for display
    if len(text) > 300:
        text = text[:300] + "..."
    
    return f"{prefix} on Page {page}{section_str}: {text}"


def format_image_caption(image_result: Dict) -> str:
    """
    Format image caption with context.
    
    Args:
        image_result: Image result dict
        
    Returns:
        Formatted caption string
    """
    caption = image_result.get('vlm_caption', '')
    ocr = image_result.get('ocr_text', '')
    page = image_result.get('page', 0)
    
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr:
        parts.append(f"OCR: {ocr[:100]}{'...' if len(ocr) > 100 else ''}")
    
    result = f"Page {page}"
    if parts:
        result += f" | {' | '.join(parts)}"
    
    return result


def get_metadata_summary(result: Dict) -> Dict:
    """
    Get a summary of metadata for a result.
    
    Args:
        result: Result dict (image or table)
        
    Returns:
        Summary dict with key metadata
    """
    return {
        'id': result.get('id', ''),
        'page': result.get('page', 0),
        'section': result.get('section_heading', ''),
        'has_related_chunks': len(result.get('related_text_chunks', [])) > 0,
        'num_related_chunks': len(result.get('related_text_chunks', [])),
        'has_vlm_caption': bool(result.get('vlm_caption')),
        'has_ocr': bool(result.get('ocr_text')),
        'is_table': result.get('is_table', False)
    }

