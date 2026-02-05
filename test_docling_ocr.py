"""
Test script using Docling for OCR instead of Tesseract.
Shows comprehensive metadata for each image including related text chunks.
"""
import sys
import os
from pathlib import Path
from PIL import Image
import io

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not available. Install with: pip install docling")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available")

# Try to import vector store to get metadata if available
try:
    from omnirag.db.vector_store import VectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    print("Warning: Vector store not available. Metadata from ingestion will not be shown.")


def extract_images_from_pdf(pdf_path: str):
    """Extract images from PDF using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        print("PyMuPDF not available")
        return []
    
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))
                images.append({
                    'image': pil_image,
                    'page': page_num + 1,
                    'image_id': f"p{page_num+1}_img{img_idx}"
                })
            except Exception as e:
                print(f"Error extracting image {img_idx} from page {page_num+1}: {e}")
                continue
    
    doc.close()
    return images


def ocr_with_docling(image: Image.Image) -> str:
    """
    Extract text from image using Docling OCR.
    
    Docling processes PDFs, so we convert the image to a PDF first,
    then use Docling's OCR capabilities to extract text.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text string
    """
    if not DOCLING_AVAILABLE:
        return ""
    
    import tempfile
    import os
    
    temp_path = None
    pdf_path = None
    
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Convert image to PDF using reportlab
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        
        pdf_path = temp_path.replace('.png', '.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        img_width, img_height = image.size
        
        # Scale to fit page while maintaining aspect ratio
        page_width, page_height = letter
        scale = min(page_width / img_width, page_height / img_height) * 0.9
        new_width = img_width * scale
        new_height = img_height * scale
        
        # Center the image on the page
        x_offset = (page_width - new_width) / 2
        y_offset = (page_height - new_height) / 2
        
        c.drawImage(ImageReader(image), x_offset, y_offset, 
                   width=new_width, 
                   height=new_height)
        c.save()
        
        # Configure Docling with OCR enabled
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR
        pipeline_options.do_table_structure = False  # Skip table detection for speed
        
        # Create converter with format_options (correct API)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Process PDF with Docling
        result = converter.convert(pdf_path)
        
        # Extract text from result
        text_parts = []
        
        # Docling returns a ConversionResult object
        # The document is accessed via result.document
        if hasattr(result, 'document') and result.document:
            doc = result.document
            
            # Try to get text content - Docling stores text in various ways
            # Method 1: Direct text attribute
            if hasattr(doc, 'text') and doc.text:
                text_parts.append(str(doc.text))
            
            # Method 2: Check for content items (paragraphs, text blocks, etc.)
            if hasattr(doc, 'items') and doc.items:
                for item in doc.items:
                    # Text items have a 'text' attribute
                    if hasattr(item, 'text') and item.text:
                        text_parts.append(str(item.text))
                    # Some items might have 'content' attribute
                    elif hasattr(item, 'content') and item.content:
                        if isinstance(item.content, str):
                            text_parts.append(item.content)
                        elif hasattr(item.content, 'text'):
                            text_parts.append(str(item.content.text))
            
            # Method 3: Try to get all text recursively
            if hasattr(doc, 'get_text') and callable(doc.get_text):
                try:
                    text_parts.append(doc.get_text())
                except:
                    pass
        
        # Method 4: Check result object directly
        if hasattr(result, 'text') and result.text:
            text_parts.append(str(result.text))
        
        # Remove duplicates and empty strings, then join
        unique_texts = []
        seen = set()
        for text in text_parts:
            text_str = str(text).strip()
            if text_str and text_str not in seen:
                unique_texts.append(text_str)
                seen.add(text_str)
        
        return '\n'.join(unique_texts).strip()
            
    except Exception as e:
        print(f"Docling OCR error: {e}")
        import traceback
        traceback.print_exc()
        return ""
    finally:
        # Clean up temporary files
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass


def test_docling_ocr_on_pdf(pdf_path: str, output_file: str = "docling_ocr_test.html"):
    """Test Docling OCR on images extracted from PDF."""
    if not DOCLING_AVAILABLE:
        print("Docling not available. Install with: pip install docling")
        return
    
    print(f"Testing Docling OCR on: {pdf_path}")
    print("=" * 80)
    
    # Extract images
    print("\nExtracting images from PDF...")
    images = extract_images_from_pdf(pdf_path)
    print(f"Extracted {len(images)} images")
    
    # Try to load vector store metadata if available
    vector_store_metadata = {}
    if VECTOR_STORE_AVAILABLE:
        try:
            vector_store = VectorStore(persist_directory="./chroma_db", collection_name="omnirag")
            # Try to get all images from vector store
            all_images = vector_store.image_collection.get()
            for img_id in all_images.get('ids', []):
                metadata = vector_store._load_metadata(img_id)
                if metadata:
                    vector_store_metadata[img_id] = metadata
            print(f"Loaded metadata for {len(vector_store_metadata)} images from vector store")
        except Exception as e:
            print(f"Could not load vector store metadata: {e}")
    
    # Process first 5 images with Docling OCR
    print("\nProcessing images with Docling OCR...")
    results = []
    
    # Create output directory for images
    output_dir = Path(output_file).parent / "docling_test_images"
    output_dir.mkdir(exist_ok=True)
    
    for i, img_data in enumerate(images[:5], 1):  # Test first 5 images
        print(f"\nProcessing image {i}/{min(5, len(images))} (Page {img_data['page']})...")
        ocr_text = ocr_with_docling(img_data['image'])
        
        # Save image for HTML report
        image_filename = f"{img_data['image_id']}.png"
        image_path = output_dir / image_filename
        img_data['image'].save(image_path)
        
        # Collect image metadata
        img = img_data['image']
        file_size = image_path.stat().st_size
        
        # Get image format and mode
        image_format = img.format or "Unknown"
        image_mode = img.mode
        image_size = img.size
        
        # Calculate aspect ratio
        aspect_ratio = f"{image_size[0] / image_size[1]:.2f}" if image_size[1] > 0 else "N/A"
        
        # Check if image has transparency
        has_transparency = image_mode in ('RGBA', 'LA', 'P') and 'transparency' in img.info
        
        # Get metadata from vector store if available
        stored_metadata = vector_store_metadata.get(img_data['image_id'], {})
        
        results.append({
            'image_id': img_data['image_id'],
            'page': img_data['page'],
            'ocr_text': ocr_text,
            'image_size': image_size,
            'image_path': str(image_path.relative_to(Path(output_file).parent)),
            'metadata': {
                'format': image_format,
                'mode': image_mode,
                'width': image_size[0],
                'height': image_size[1],
                'aspect_ratio': aspect_ratio,
                'file_size_bytes': file_size,
                'file_size_kb': f"{file_size / 1024:.2f}",
                'has_transparency': has_transparency,
                'ocr_text_length': len(ocr_text),
                'ocr_words': len(ocr_text.split()) if ocr_text else 0
            },
            # Add metadata from vector store if available
            'vlm_caption': stored_metadata.get('vlm_caption', ''),
            'stored_ocr_text': stored_metadata.get('ocr_text', ''),
            'surrounding_context': stored_metadata.get('surrounding_context', ''),
            'related_text_chunks': stored_metadata.get('related_text_chunks', []),
            'related_text_content': stored_metadata.get('related_text_content', ''),
            'related_chunk_ids': stored_metadata.get('related_chunk_ids', []),
            'has_stored_metadata': bool(stored_metadata)
        })
        
        print(f"  OCR Text Length: {len(ocr_text)} characters")
        if stored_metadata:
            print(f"  Found stored metadata: VLM caption, {len(stored_metadata.get('related_text_chunks', []))} related text chunks")
        if ocr_text:
            print(f"  Preview: {ocr_text[:100]}...")
        else:
            print("  No text extracted")
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Docling OCR Test Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .result-section {{
            background: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-container {{
            text-align: center;
            margin: 15px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .ocr-text {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: monospace;
        }}
        .metadata {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
            border-left: 4px solid #2196F3;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .metadata-item {{
            background: white;
            padding: 8px;
            border-radius: 4px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
        }}
        .metadata-value {{
            color: #333;
            font-size: 1em;
        }}
    </style>
</head>
<body>
    <h1>Docling OCR Test Results</h1>
    <p><strong>PDF:</strong> {Path(pdf_path).name}</p>
    <p><strong>Total Images Tested:</strong> {len(results)}</p>
"""
    
    for result in results:
        image_path = result.get('image_path', '')
        metadata = result.get('metadata', {})
        has_stored = result.get('has_stored_metadata', False)
        
        # Build basic metadata grid
        metadata_html = ""
        if metadata:
            metadata_html = '<div class="metadata-grid">'
            metadata_items = [
                ('Format', metadata.get('format', 'N/A')),
                ('Color Mode', metadata.get('mode', 'N/A')),
                ('Dimensions', f"{metadata.get('width', 0)} × {metadata.get('height', 0)} px"),
                ('Aspect Ratio', metadata.get('aspect_ratio', 'N/A')),
                ('File Size', f"{metadata.get('file_size_kb', '0')} KB ({metadata.get('file_size_bytes', 0):,} bytes)"),
                ('Transparency', 'Yes' if metadata.get('has_transparency', False) else 'No'),
                ('OCR Text Length', f"{metadata.get('ocr_text_length', 0):,} characters"),
                ('OCR Words', f"{metadata.get('ocr_words', 0):,} words"),
            ]
            for label, value in metadata_items:
                metadata_html += f'''
                <div class="metadata-item">
                    <div class="metadata-label">{label}</div>
                    <div class="metadata-value">{value}</div>
                </div>'''
            metadata_html += '</div>'
        
        # Build stored metadata section (from vector store)
        stored_metadata_html = ""
        if has_stored:
            vlm_caption = result.get('vlm_caption', '')
            stored_ocr = result.get('stored_ocr_text', '')
            surrounding = result.get('surrounding_context', '')
            related_chunks = result.get('related_text_chunks', [])
            related_content = result.get('related_text_content', '')
            related_ids = result.get('related_chunk_ids', [])
            
            stored_metadata_html = '<div class="metadata" style="background: #fff3cd; border-left-color: #ffc107;">'
            stored_metadata_html += '<h3 style="margin-top: 0;">📚 Stored Metadata (from Vector Store)</h3>'
            
            if vlm_caption:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>VLM Caption:</strong><br><em>{vlm_caption}</em></div>'
            
            if stored_ocr:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Stored OCR Text:</strong><br><div class="ocr-text" style="background: white; margin-top: 5px;">{stored_ocr[:300]}{"..." if len(stored_ocr) > 300 else ""}</div></div>'
            
            if surrounding:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Surrounding Context:</strong><br><div class="ocr-text" style="background: white; margin-top: 5px;">{surrounding[:300]}{"..." if len(surrounding) > 300 else ""}</div></div>'
            
            if related_chunks:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Related Text Chunks ({len(related_chunks)}):</strong>'
                for i, chunk in enumerate(related_chunks, 1):
                    chunk_id = chunk.get('chunk_id', 'N/A')
                    section = chunk.get('section_heading', '')
                    text = chunk.get('text', '')[:200]
                    stored_metadata_html += f'''
                    <div style="background: white; padding: 10px; margin: 5px 0; border-radius: 4px;">
                        <strong>Chunk {i} ({chunk_id})</strong> {f"[{section}]" if section else ""}<br>
                        <div style="font-size: 0.9em; color: #666; margin-top: 5px;">{text}{"..." if len(chunk.get('text', '')) > 200 else ""}</div>
                    </div>'''
                stored_metadata_html += '</div>'
            
            if related_content:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Combined Related Text:</strong><br><div class="ocr-text" style="background: white; margin-top: 5px; max-height: 200px; overflow-y: auto;">{related_content[:500]}{"..." if len(related_content) > 500 else ""}</div></div>'
            
            if related_ids:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Related Chunk IDs:</strong> {", ".join(related_ids)}</div>'
            
            stored_metadata_html += '</div>'
        
        html_content += f"""
    <div class="result-section">
        <h2>Image {result['image_id']} (Page {result['page']})</h2>
        
        <div class="metadata">
            <h3 style="margin-top: 0;">📊 Image Metadata</h3>
            {metadata_html}
        </div>
        
        {stored_metadata_html}
        
        <div class="image-container">
            <img src="{image_path}" alt="Image {result['image_id']}">
        </div>
        
        <div class="ocr-text">
            <strong>Extracted Text (Docling OCR):</strong><br>
            {result['ocr_text'] if result['ocr_text'] else '(No text extracted)'}
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n\nTest complete! Results saved to: {output_file}")
    print(f"Total images processed: {len(results)}")
    print(f"Images with text: {sum(1 for r in results if r['ocr_text'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check if file exists in uploads
        uploads_dir = Path("uploads")
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"Using PDF from uploads: {pdf_path}\n")
        else:
            print("Usage: python test_docling_ocr.py <path_to_pdf> [output_html]")
            print("\nOr place a PDF in the 'uploads' directory")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "docling_ocr_test.html"
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    test_docling_ocr_on_pdf(pdf_path, output_file)

