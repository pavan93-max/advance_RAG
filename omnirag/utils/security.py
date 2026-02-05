"""
Security utilities for file validation and input sanitization.
"""
import os
import re
from pathlib import Path
from typing import Optional

# Security constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'.pdf'}
BLOCKED_PATTERNS = [
    r'\.\./',  # Path traversal
    r'\.\.\\',  # Path traversal (Windows)
]

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+previous',
    r'forget\s+everything',
    r'system\s*:',
    r'assistant\s*:',
    r'you\s+are\s+now',
    r'disregard\s+instructions',
]

MAX_QUERY_LENGTH = 1000


def validate_filename(filename: str) -> bool:
    """
    Validate filename for security.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if filename is safe, False otherwise
    """
    if not filename:
        return False
    
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check for blocked patterns (path traversal)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return False
    
    # Sanitize filename - only allow alphanumeric, dots, underscores, hyphens
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    if safe_name != filename:
        return False
    
    # Check for absolute paths
    if os.path.isabs(filename):
        return False
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to safe format.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Get base name
    base_name = Path(filename).stem
    ext = Path(filename).suffix.lower()
    
    # Remove dangerous characters
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', base_name)
    
    # Limit length
    safe_name = safe_name[:100]
    
    return f"{safe_name}{ext}"


def validate_file_size(file_size: int, max_size: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """
    Validate file size.
    
    Args:
        file_size: Size of file in bytes
        max_size: Maximum allowed size (default: MAX_FILE_SIZE)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    max_size = max_size or MAX_FILE_SIZE
    
    if file_size > max_size:
        size_mb = file_size / 1024 / 1024
        max_mb = max_size / 1024 / 1024
        return False, f"File too large: {size_mb:.1f} MB. Maximum: {max_mb:.0f} MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, None


def sanitize_query(query: str) -> str:
    """
    Sanitize user query to prevent prompt injection.
    
    Args:
        query: User query to sanitize
        
    Returns:
        Sanitized query (empty string if blocked)
    """
    if not query:
        return ""
    
    query_lower = query.lower()
    
    # Check for prompt injection patterns
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return ""  # Reject query
    
    # Limit length
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
    
    # Remove excessive whitespace
    query = ' '.join(query.split())
    
    return query.strip()


def validate_path(path: str, base_dir: Optional[str] = None) -> bool:
    """
    Validate that path is within allowed directory (prevent path traversal).
    
    Args:
        path: Path to validate
        base_dir: Base directory (default: current working directory)
        
    Returns:
        True if path is safe
    """
    if not path:
        return False
    
    base_dir = base_dir or os.getcwd()
    
    try:
        # Resolve paths
        resolved_path = os.path.abspath(os.path.join(base_dir, path))
        resolved_base = os.path.abspath(base_dir)
        
        # Check if resolved path is within base directory
        return resolved_path.startswith(resolved_base)
    except Exception:
        return False

