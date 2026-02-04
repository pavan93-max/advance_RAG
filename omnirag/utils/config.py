"""
Configuration settings for OmniRAG.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
METADATA_DIR = CHROMA_DB_DIR / "metadata"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

# Model settings
DEFAULT_TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# LLM settings
DEFAULT_LLM_TYPE = "ollama"  # or "transformers" or "template"
DEFAULT_LLM_MODEL = "llama3.2"

# Processing settings
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200
MAX_IMAGE_SIZE = (1024, 1024)  # Resize large images

# Tesseract OCR path (set if needed)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

