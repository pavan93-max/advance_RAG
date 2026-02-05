"""
Spatial proximity utilities for linking images/tables to text chunks.
"""
import math
from typing import List, Dict, Optional, Tuple


def calculate_distance(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Euclidean distance between two bounding boxes.
    
    Args:
        bbox1: Dict with keys 'x0', 'y0', 'x1', 'y1' or center point
        bbox2: Dict with keys 'x0', 'y0', 'x1', 'y1' or center point
        
    Returns:
        Distance between center points of the bounding boxes
    """
    # Get center points
    if 'x0' in bbox1 and 'x1' in bbox1:
        center1_x = (bbox1['x0'] + bbox1['x1']) / 2
        center1_y = (bbox1['y0'] + bbox1['y1']) / 2
    else:
        center1_x = bbox1.get('x', 0)
        center1_y = bbox1.get('y', 0)
    
    if 'x0' in bbox2 and 'x1' in bbox2:
        center2_x = (bbox2['x0'] + bbox2['x1']) / 2
        center2_y = (bbox2['y0'] + bbox2['y1']) / 2
    else:
        center2_x = bbox2.get('x', 0)
        center2_y = bbox2.get('y', 0)
    
    # Calculate Euclidean distance
    dx = center2_x - center1_x
    dy = center2_y - center1_y
    return math.sqrt(dx * dx + dy * dy)


def calculate_overlap(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate overlap ratio between two bounding boxes.
    
    Args:
        bbox1: Dict with keys 'x0', 'y0', 'x1', 'y1'
        bbox2: Dict with keys 'x0', 'y0', 'x1', 'y1'
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if not all(k in bbox1 for k in ['x0', 'y0', 'x1', 'y1']):
        return 0.0
    if not all(k in bbox2 for k in ['x0', 'y0', 'x1', 'y1']):
        return 0.0
    
    # Calculate intersection
    x0 = max(bbox1['x0'], bbox2['x0'])
    y0 = max(bbox1['y0'], bbox2['y0'])
    x1 = min(bbox1['x1'], bbox2['x1'])
    y1 = min(bbox1['y1'], bbox2['y1'])
    
    if x1 <= x0 or y1 <= y0:
        return 0.0
    
    intersection = (x1 - x0) * (y1 - y0)
    
    # Calculate union
    area1 = (bbox1['x1'] - bbox1['x0']) * (bbox1['y1'] - bbox1['y0'])
    area2 = (bbox2['x1'] - bbox2['x0']) * (bbox2['y1'] - bbox2['y0'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def find_nearby_chunks(
    target_bbox: Optional[Dict],
    text_chunks: List[Dict],
    max_chunks: int = 3,
    page: Optional[int] = None
) -> List[Dict]:
    """
    Find text chunks closest to a target bounding box using spatial proximity.
    
    Args:
        target_bbox: Bounding box of the target (image/table) with keys 'x0', 'y0', 'x1', 'y1'
        text_chunks: List of text chunk dicts, optionally with 'bbox' key
        max_chunks: Maximum number of chunks to return
        page: Optional page number to filter chunks
        
    Returns:
        List of closest text chunks, sorted by distance
    """
    if not target_bbox:
        # Fallback to first N chunks on same page if no bbox
        filtered = [c for c in text_chunks if page is None or c.get('page') == page]
        return filtered[:max_chunks]
    
    # Filter chunks by page if specified
    filtered_chunks = text_chunks
    if page is not None:
        filtered_chunks = [c for c in text_chunks if c.get('page') == page]
    
    # Calculate distances for chunks with bbox
    chunk_distances = []
    for chunk in filtered_chunks:
        chunk_bbox = chunk.get('bbox')
        if chunk_bbox:
            distance = calculate_distance(target_bbox, chunk_bbox)
            overlap = calculate_overlap(target_bbox, chunk_bbox)
            # Combine distance and overlap (prefer overlapping chunks)
            score = distance * (1 - overlap * 0.5)  # Reduce distance score if overlapping
            chunk_distances.append((score, distance, overlap, chunk))
        else:
            # Chunks without bbox get a high distance (low priority)
            chunk_distances.append((999999, 999999, 0.0, chunk))
    
    # Sort by score (distance adjusted for overlap)
    chunk_distances.sort(key=lambda x: x[0])
    
    # Return closest chunks with metadata
    result = []
    for score, distance, overlap, chunk in chunk_distances[:max_chunks]:
        result.append({
            'chunk': chunk,
            'distance': distance,
            'overlap': overlap,
            'score': score
        })
    
    return result


def estimate_text_chunk_bbox(chunk: Dict, page_width: float = 612, page_height: float = 792) -> Dict:
    """
    Estimate bounding box for a text chunk if not available.
    Uses position in page and text length as heuristics.
    
    Args:
        chunk: Text chunk dict
        page_width: Default page width in points
        page_height: Default page height in points
        
    Returns:
        Estimated bbox dict
    """
    # Simple heuristic: distribute chunks vertically on page
    # This is a fallback when actual bbox is not available
    page = chunk.get('page', 1)
    chunk_id = chunk.get('chunk_id', '')
    
    # Try to extract position from chunk_id if it contains position info
    # Otherwise use a simple vertical distribution
    text_length = len(chunk.get('text', ''))
    
    # Estimate: longer chunks are higher on page, shorter are lower
    # This is a very rough estimate
    y_position = page_height * 0.1 + (text_length % 100) * 0.5
    
    return {
        'x0': page_width * 0.1,
        'y0': y_position,
        'x1': page_width * 0.9,
        'y1': y_position + min(text_length / 10, 100)  # Height based on text length
    }

