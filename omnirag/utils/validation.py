"""
Validation utilities for metadata integrity checks.
"""
from typing import Dict, List, Optional, Tuple
import os
import json


def validate_image_metadata(metadata: Dict) -> Tuple[bool, List[str]]:
    """
    Validate image metadata structure and content.
    
    Args:
        metadata: Image metadata dict
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_fields = ['page', 'image_path']
    optional_fields = ['vlm_caption', 'ocr_text', 'surrounding_context', 
                      'related_text_chunks', 'related_chunk_ids', 'related_text_content']
    
    # Check required fields
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
    
    # Validate page number
    if 'page' in metadata:
        page = metadata['page']
        if not isinstance(page, int) or page < 1:
            errors.append(f"Invalid page number: {page}")
    
    # Validate related_text_chunks structure
    if 'related_text_chunks' in metadata:
        chunks = metadata['related_text_chunks']
        if not isinstance(chunks, list):
            errors.append("related_text_chunks must be a list")
        else:
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    errors.append(f"related_text_chunks[{i}] must be a dict")
                else:
                    if 'chunk_id' not in chunk:
                        errors.append(f"related_text_chunks[{i}] missing chunk_id")
                    if 'text' not in chunk:
                        errors.append(f"related_text_chunks[{i}] missing text")
    
    # Validate related_chunk_ids
    if 'related_chunk_ids' in metadata:
        chunk_ids = metadata['related_chunk_ids']
        if not isinstance(chunk_ids, list):
            errors.append("related_chunk_ids must be a list")
    
    # Validate consistency: related_chunk_ids should match related_text_chunks
    if 'related_chunk_ids' in metadata and 'related_text_chunks' in metadata:
        chunk_ids = metadata['related_chunk_ids']
        chunks = metadata['related_text_chunks']
        chunk_ids_from_chunks = [chunk.get('chunk_id') for chunk in chunks if isinstance(chunk, dict)]
        
        if set(chunk_ids) != set(chunk_ids_from_chunks):
            errors.append("related_chunk_ids and related_text_chunks are inconsistent")
    
    return len(errors) == 0, errors


def validate_table_metadata(metadata: Dict) -> Tuple[bool, List[str]]:
    """
    Validate table metadata structure and content.
    
    Args:
        metadata: Table metadata dict
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_fields = ['page', 'is_table']
    
    # Check required fields
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
    
    # Validate is_table flag
    if 'is_table' in metadata:
        if not isinstance(metadata['is_table'], bool):
            errors.append("is_table must be a boolean")
        elif not metadata['is_table']:
            errors.append("Table metadata must have is_table=True")
    
    # Validate table_metadata structure
    if 'table_metadata' in metadata:
        table_meta = metadata['table_metadata']
        if not isinstance(table_meta, dict):
            errors.append("table_metadata must be a dict")
        else:
            if 'num_rows' in table_meta:
                if not isinstance(table_meta['num_rows'], int) or table_meta['num_rows'] < 0:
                    errors.append("table_metadata.num_rows must be a non-negative integer")
            if 'num_cols' in table_meta:
                if not isinstance(table_meta['num_cols'], int) or table_meta['num_cols'] < 0:
                    errors.append("table_metadata.num_cols must be a non-negative integer")
    
    # Validate related_text_chunks (same as images)
    if 'related_text_chunks' in metadata:
        chunks = metadata['related_text_chunks']
        if not isinstance(chunks, list):
            errors.append("related_text_chunks must be a list")
    
    return len(errors) == 0, errors


def validate_metadata_file(file_path: str, metadata_type: str = 'image') -> Tuple[bool, List[str]]:
    """
    Validate a metadata JSON file.
    
    Args:
        file_path: Path to metadata JSON file
        metadata_type: Type of metadata ('image' or 'table')
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    if metadata_type == 'image':
        return validate_image_metadata(metadata)
    elif metadata_type == 'table':
        return validate_table_metadata(metadata)
    else:
        return False, [f"Unknown metadata type: {metadata_type}"]


def validate_metadata_consistency(vector_store, collection_type: str = 'images') -> Dict:
    """
    Validate consistency of all metadata in a collection.
    
    Args:
        vector_store: VectorStore instance
        collection_type: 'images' or 'text' (for tables)
        
    Returns:
        Dict with validation results
    """
    results = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'errors': []
    }
    
    try:
        if collection_type == 'images':
            collection = vector_store.image_collection
        else:
            collection = vector_store.text_collection
        
        all_items = collection.get()
        
        for item_id in all_items.get('ids', []):
            results['total'] += 1
            metadata = vector_store._load_metadata(item_id)
            
            if not metadata:
                results['invalid'] += 1
                results['errors'].append(f"{item_id}: No metadata found")
                continue
            
            # Determine type
            is_table = metadata.get('is_table', False)
            metadata_type = 'table' if is_table else 'image'
            
            is_valid, errors = validate_image_metadata(metadata) if metadata_type == 'image' else validate_table_metadata(metadata)
            
            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].extend([f"{item_id}: {e}" for e in errors])
    
    except Exception as e:
        results['errors'].append(f"Validation error: {e}")
    
    return results

