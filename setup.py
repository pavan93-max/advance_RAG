"""
Setup script for OmniRAG.
Creates necessary directories and checks dependencies.
"""
import os
import sys
from pathlib import Path

def check_tesseract():
    """Check if Tesseract is available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("[OK] Tesseract OCR found")
        return True
    except Exception as e:
        print(f"[WARNING] Tesseract OCR not found: {e}")
        print("   Install Tesseract:")
        print("   - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Linux: sudo apt-get install tesseract-ocr")
        print("   - macOS: brew install tesseract")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    required = [
        'streamlit',
        'fitz',  # PyMuPDF
        'pdfplumber',
        'PIL',  # Pillow
        'pandas',
        'chromadb',
        'sentence_transformers',
        'torch',
        'transformers',
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'fitz':
                import fitz
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing.append(package)
    
    return len(missing) == 0

def create_directories():
    """Create necessary directories."""
    dirs = [
        "uploads",
        "chroma_db",
        "chroma_db/metadata",
        "chroma_db/metadata/images",
        ".ai/terminal"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {dir_path}")

def check_models_info():
    """Display information about models that will be downloaded."""
    print("\n" + "=" * 50)
    print("📦 Models Information")
    print("=" * 50)
    print("The following models will be auto-downloaded on first run:")
    print()
    print("1. CLIP (openai/clip-vit-base-patch32)")
    print("   Size: ~150 MB")
    print("   Purpose: Image embeddings for similarity search")
    print()
    print("2. BLIP (Salesforce/blip-image-captioning-base)")
    print("   Size: ~990 MB")
    print("   Purpose: Generate captions for images")
    print()
    print("3. Sentence Transformers (all-MiniLM-L6-v2)")
    print("   Size: ~80 MB")
    print("   Purpose: Text embeddings")
    print()
    print("Total: ~1.2 GB (one-time download)")
    print("Location: ~/.cache/huggingface/hub/")
    print("=" * 50)

def main():
    print("=" * 50)
    print("OmniRAG Setup")
    print("=" * 50)
    print()
    
    print("Checking Python dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("Checking Tesseract OCR...")
    tesseract_ok = check_tesseract()
    print()
    
    print("Creating directories...")
    create_directories()
    print()
    
    check_models_info()
    print()
    
    if deps_ok and tesseract_ok:
        print("=" * 50)
        print("[OK] Setup complete! You can now run:")
        print("   streamlit run streamlit_app.py")
        print()
        print("[NOTE] Models will be downloaded on first run (~1.2 GB)")
        print("   This is a one-time download and may take 5-10 minutes.")
        print("=" * 50)
    else:
        print("=" * 50)
        print("[WARNING] Setup incomplete. Please install missing dependencies.")
        print()
        if not deps_ok:
            print("Missing Python packages. Run: pip install -r requirements.txt")
        if not tesseract_ok:
            print("Missing Tesseract OCR. See INSTALLATION.md for instructions.")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()

