"""
OCR Comparison Test Script: Docling vs Tesseract
Shows comprehensive metadata for each image including OCR results from both engines.
"""
import sys
import os
import html
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

# Try to import Tesseract OCR
try:
    import pytesseract
    import os
    # Check if Tesseract is configured
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception:
        # Try to set path from environment
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            try:
                pytesseract.get_tesseract_version()
                TESSERACT_AVAILABLE = True
            except Exception:
                TESSERACT_AVAILABLE = False
        else:
            TESSERACT_AVAILABLE = False
        if not TESSERACT_AVAILABLE:
            print("Warning: Tesseract OCR not found. Tesseract OCR will be disabled.")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Tesseract OCR will be disabled.")

# Try to import vector store to get metadata if available
try:
    from omnirag.db.vector_store import VectorStore
    from omnirag.models.image_processor import ImageProcessor
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


def ocr_with_tesseract(image: Image.Image) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text string
    """
    if not TESSERACT_AVAILABLE:
        return ""
    
    try:
        # Convert PIL image to format tesseract expects
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        return ""


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


def test_ocr_comparison(pdf_path: str, output_file: str = "ocr_comparison_test.html"):
    """Test both Docling and Tesseract OCR on images extracted from PDF."""
    if not DOCLING_AVAILABLE and not TESSERACT_AVAILABLE:
        print("Neither Docling nor Tesseract available. Install at least one OCR engine.")
        return
    
    print(f"Testing OCR Comparison on: {pdf_path}")
    print("=" * 80)
    
    # Extract images
    print("\nExtracting images from PDF...")
    images = extract_images_from_pdf(pdf_path)
    print(f"Extracted {len(images)} images")
    
    # Try to load vector store metadata if available
    vector_store_metadata = {}
    image_processor = None
    vector_store = None
    if VECTOR_STORE_AVAILABLE:
        try:
            image_processor = ImageProcessor()
            vector_store = VectorStore(persist_directory="./chroma_db", collection_name="omnirag", 
                                     image_processor=image_processor)
            # Try to get all images from vector store
            all_images = vector_store.image_collection.get()
            for img_id in all_images.get('ids', []):
                metadata = vector_store._load_metadata(img_id)
                if metadata:
                    vector_store_metadata[img_id] = metadata
            print(f"Loaded metadata for {len(vector_store_metadata)} images from vector store")
        except Exception as e:
            print(f"Could not load vector store metadata: {e}")
            import traceback
            traceback.print_exc()
    
    # Process first 5 images with both OCR engines
    print("\nProcessing images with OCR engines...")
    results = []
    
    # Create output directory for images
    output_dir = Path(output_file).parent / "ocr_comparison_images"
    output_dir.mkdir(exist_ok=True)
    
    for i, img_data in enumerate(images[:5], 1):  # Test first 5 images
        print(f"\nProcessing image {i}/{min(5, len(images))} (Page {img_data['page']})...")
        
        # Process with both OCR engines
        docling_ocr_text = ocr_with_docling(img_data['image']) if DOCLING_AVAILABLE else ""
        tesseract_ocr_text = ocr_with_tesseract(img_data['image']) if TESSERACT_AVAILABLE else ""
        
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
        
        # Get CLIP embedding for similarity analysis
        clip_embedding = None
        if image_processor:
            try:
                clip_embedding = image_processor.get_clip_embedding(img_data['image'])
            except Exception as e:
                print(f"  Warning: Could not get CLIP embedding: {e}")
        
        results.append({
            'image_id': img_data['image_id'],
            'page': img_data['page'],
            'docling_ocr_text': docling_ocr_text,
            'tesseract_ocr_text': tesseract_ocr_text,
            'image_size': image_size,
            'image_path': str(image_path.relative_to(Path(output_file).parent)),
            'clip_embedding': clip_embedding,
            'metadata': {
                'format': image_format,
                'mode': image_mode,
                'width': image_size[0],
                'height': image_size[1],
                'aspect_ratio': aspect_ratio,
                'file_size_bytes': file_size,
                'file_size_kb': f"{file_size / 1024:.2f}",
                'has_transparency': has_transparency,
                'docling_ocr_length': len(docling_ocr_text),
                'docling_ocr_words': len(docling_ocr_text.split()) if docling_ocr_text else 0,
                'tesseract_ocr_length': len(tesseract_ocr_text),
                'tesseract_ocr_words': len(tesseract_ocr_text.split()) if tesseract_ocr_text else 0
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
        
        print(f"  Docling OCR: {len(docling_ocr_text)} chars, {len(docling_ocr_text.split()) if docling_ocr_text else 0} words")
        print(f"  Tesseract OCR: {len(tesseract_ocr_text)} chars, {len(tesseract_ocr_text.split()) if tesseract_ocr_text else 0} words")
        if stored_metadata:
            print(f"  Found stored metadata: VLM caption, {len(stored_metadata.get('related_text_chunks', []))} related text chunks")
        if clip_embedding is not None:
            print(f"  CLIP embedding: {len(clip_embedding)} dimensions")
    
    # Calculate image-to-image similarity matrix
    similarity_matrix = []
    if image_processor:
        print("\nCalculating image-to-image similarity matrix...")
        import numpy as np
        for i, result1 in enumerate(results):
            row = []
            for j, result2 in enumerate(results):
                if result1['clip_embedding'] is not None and result2['clip_embedding'] is not None:
                    # Cosine similarity
                    emb1 = np.array(result1['clip_embedding'])
                    emb2 = np.array(result2['clip_embedding'])
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    row.append(float(similarity))
                else:
                    row.append(0.0)
            similarity_matrix.append(row)
        
        # Test search queries
        test_queries = [
            "frequency modeling",
            "FEDformer",
            "PatchTST",
            "time series forecasting",
            "graph comparison"
        ]
        search_results = {}
        if vector_store:
            print("\nTesting search queries...")
            for query in test_queries:
                try:
                    img_results = vector_store.search_images_by_clip(query, top_k=3)
                    search_results[query] = img_results
                    print(f"  Query '{query}': Found {len(img_results)} images")
                    for img in img_results[:2]:
                        print(f"    - {img.get('id', 'N/A')}: score={img.get('score', 0):.3f}")
                except Exception as e:
                    print(f"  Query '{query}': Error - {e}")
    else:
        similarity_matrix = None
        search_results = None
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Comparison Test Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
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
            padding: 20px;
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
            margin: 10px 0;
        }}
        .ocr-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .ocr-box {{
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 15px;
        }}
        .ocr-box.docling {{
            border-color: #2196F3;
            background: #e3f2fd;
        }}
        .ocr-box.tesseract {{
            border-color: #4CAF50;
            background: #e8f5e9;
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
        .stored-metadata {{
            background: #fff3cd;
            border-left-color: #ffc107;
        }}
        .similarity-matrix {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        .similarity-matrix table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        .similarity-matrix th, .similarity-matrix td {{
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        .similarity-matrix th {{
            background: #f0f0f0;
            font-weight: bold;
        }}
        .similarity-matrix td {{
            font-family: monospace;
        }}
        .similarity-high {{
            background: #c8e6c9;
        }}
        .similarity-medium {{
            background: #fff9c4;
        }}
        .similarity-low {{
            background: #ffcdd2;
        }}
        .search-results {{
            margin: 15px 0;
        }}
        .search-query {{
            background: #e1f5fe;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            border-left: 4px solid #0288d1;
        }}
        .search-result-item {{
            background: white;
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 3px solid #4caf50;
        }}
    </style>
</head>
<body>
    <h1>OCR Comparison Test Results</h1>
    <p><strong>PDF:</strong> {Path(pdf_path).name}</p>
    <p><strong>Total Images Tested:</strong> {len(results)}</p>
    <p><strong>Engines:</strong> Docling ({'✓' if DOCLING_AVAILABLE else '✗'}), Tesseract ({'✓' if TESSERACT_AVAILABLE else '✗'})</p>
"""
    
    # Add similarity matrix section
    if similarity_matrix:
        html_content += """
    <div class="result-section">
        <h2>🔍 Image Similarity Matrix (CLIP Embeddings)</h2>
        <p style="color: #666; font-size: 0.9em;">Cosine similarity between images based on CLIP embeddings. Higher values indicate more similar images.</p>
        <div class="similarity-matrix">
            <table>
                <thead>
                    <tr>
                        <th></th>"""
        for result in results:
            html_content += f"<th>{result['image_id']}</th>"
        html_content += """
                    </tr>
                </thead>
                <tbody>"""
        for i, result in enumerate(results):
            html_content += f"""
                    <tr>
                        <th>{result['image_id']}</th>"""
            for j, sim_score in enumerate(similarity_matrix[i]):
                if i == j:
                    cell_class = "similarity-high"
                    cell_text = "1.000"
                elif sim_score >= 0.8:
                    cell_class = "similarity-high"
                    cell_text = f"{sim_score:.3f}"
                elif sim_score >= 0.6:
                    cell_class = "similarity-medium"
                    cell_text = f"{sim_score:.3f}"
                else:
                    cell_class = "similarity-low"
                    cell_text = f"{sim_score:.3f}"
                html_content += f'<td class="{cell_class}">{cell_text}</td>'
            html_content += """
                    </tr>"""
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
    
    # Add search query test results
    if search_results:
        html_content += """
    <div class="result-section">
        <h2>🔎 Search Query Test Results</h2>
        <p style="color: #666; font-size: 0.9em;">Testing how well different queries match the images using CLIP similarity.</p>
"""
        for query, results_list in search_results.items():
            query_escaped = html.escape(query)
            html_content += f"""
        <div class="search-query">
            <h3 style="margin-top: 0;">Query: "{query_escaped}"</h3>"""
            if results_list:
                for img_result in results_list:
                    score = img_result.get('score', 0)
                    img_id = html.escape(img_result.get('id', 'N/A'))
                    caption = html.escape(img_result.get('vlm_caption', '')[:100])
                    html_content += f"""
            <div class="search-result-item">
                <strong>{img_id}</strong> (Score: {score:.3f})<br>
                <span style="font-size: 0.9em; color: #666;">{caption}{"..." if len(img_result.get('vlm_caption', '')) > 100 else ""}</span>
            </div>"""
            else:
                html_content += """
            <div style="color: #999; font-style: italic;">No matching images found</div>"""
            html_content += """
        </div>"""
        html_content += """
    </div>
"""
    
    for result in results:
        image_path = result.get('image_path', '')
        metadata = result.get('metadata', {})
        has_stored = result.get('has_stored_metadata', False)
        docling_text = result.get('docling_ocr_text', '')
        tesseract_text = result.get('tesseract_ocr_text', '')
        
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
            ]
            for label, value in metadata_items:
                metadata_html += f'''
                <div class="metadata-item">
                    <div class="metadata-label">{label}</div>
                    <div class="metadata-value">{value}</div>
                </div>'''
            metadata_html += '</div>'
        
        # OCR comparison stats
        ocr_stats_html = '<div class="metadata-grid">'
        ocr_stats = [
            ('Docling OCR Length', f"{metadata.get('docling_ocr_length', 0):,} characters"),
            ('Docling OCR Words', f"{metadata.get('docling_ocr_words', 0):,} words"),
            ('Tesseract OCR Length', f"{metadata.get('tesseract_ocr_length', 0):,} characters"),
            ('Tesseract OCR Words', f"{metadata.get('tesseract_ocr_words', 0):,} words"),
        ]
        for label, value in ocr_stats:
            ocr_stats_html += f'''
            <div class="metadata-item">
                <div class="metadata-label">{label}</div>
                <div class="metadata-value">{value}</div>
            </div>'''
        ocr_stats_html += '</div>'
        
        # Build stored metadata section (from vector store)
        stored_metadata_html = ""
        if has_stored:
            vlm_caption = result.get('vlm_caption', '')
            stored_ocr = result.get('stored_ocr_text', '')
            surrounding = result.get('surrounding_context', '')
            related_chunks = result.get('related_text_chunks', [])
            related_content = result.get('related_text_content', '')
            related_ids = result.get('related_chunk_ids', [])
            
            stored_metadata_html = '<div class="metadata stored-metadata">'
            stored_metadata_html += '<h3 style="margin-top: 0;">📚 Stored Metadata (from Vector Store)</h3>'
            
            if vlm_caption:
                vlm_caption_escaped = html.escape(vlm_caption)
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>VLM Caption:</strong><br><em>{vlm_caption_escaped}</em></div>'
            
            if stored_ocr:
                stored_ocr_escaped = html.escape(stored_ocr[:300])
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Stored OCR Text:</strong><br><div class="ocr-text" style="background: white; margin-top: 5px;">{stored_ocr_escaped}{"..." if len(stored_ocr) > 300 else ""}</div></div>'
            
            if surrounding:
                surrounding_escaped = html.escape(surrounding[:300])
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Surrounding Context:</strong><br><div class="ocr-text" style="background: white; margin-top: 5px;">{surrounding_escaped}{"..." if len(surrounding) > 300 else ""}</div></div>'
            
            if related_chunks:
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Related Text Chunks ({len(related_chunks)}):</strong>'
                for i, chunk in enumerate(related_chunks, 1):
                    chunk_id = html.escape(chunk.get('chunk_id', 'N/A'))
                    section = html.escape(chunk.get('section_heading', ''))
                    text = html.escape(chunk.get('text', '')[:200])
                    section_part = f"[{section}]" if section else ""
                    stored_metadata_html += f'''
                    <div style="background: white; padding: 10px; margin: 5px 0; border-radius: 4px;">
                        <strong>Chunk {i} ({chunk_id})</strong> {section_part}<br>
                        <div style="font-size: 0.9em; color: #666; margin-top: 5px;">{text}{"..." if len(chunk.get('text', '')) > 200 else ""}</div>
                    </div>'''
                stored_metadata_html += '</div>'
            
            if related_content:
                related_content_escaped = html.escape(related_content[:500])
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Combined Related Text:</strong><br><div class="ocr-text" style="background: white; margin-top: 5px; max-height: 200px; overflow-y: auto;">{related_content_escaped}{"..." if len(related_content) > 500 else ""}</div></div>'
            
            if related_ids:
                related_ids_escaped = ", ".join(html.escape(rid) for rid in related_ids)
                stored_metadata_html += f'<div style="margin: 10px 0;"><strong>Related Chunk IDs:</strong> {related_ids_escaped}</div>'
            
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
            <div style="position: relative; display: inline-block;">
                <img src="{image_path}" alt="Image {result['image_id']}" 
                     title="Image {result['image_id']} - Page {result['page']} - {metadata.get('width', 0)}×{metadata.get('height', 0)}px">
                <div style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); color: white; padding: 8px; font-size: 0.9em;">
                    <strong>Image {result['image_id']}</strong> | Page {result['page']} | {metadata.get('width', 0)}×{metadata.get('height', 0)}px | {metadata.get('format', 'N/A')}
                </div>
            </div>
            <div style="margin-top: 10px; text-align: center; color: #666; font-size: 0.9em;">
                <div><strong>Image Properties:</strong> {metadata.get('format', 'N/A')} | {metadata.get('mode', 'N/A')} | {metadata.get('file_size_kb', '0')} KB</div>
                <div style="margin-top: 5px;">
                    Docling: {metadata.get('docling_ocr_length', 0)} chars | Tesseract: {metadata.get('tesseract_ocr_length', 0)} chars
                </div>
            </div>
        </div>
        
        <div class="metadata">
            <h3 style="margin-top: 0;">📈 OCR Statistics</h3>
            {ocr_stats_html}
        </div>
        
        <div class="ocr-comparison">
            <div class="ocr-box docling">
                <h3>🔵 Docling OCR</h3>
                <div class="ocr-text">
                    {html.escape(docling_text) if docling_text else '(No text extracted)'}
                </div>
            </div>
            
            <div class="ocr-box tesseract">
                <h3>🟢 Tesseract OCR</h3>
                <div class="ocr-text">
                    {html.escape(tesseract_text) if tesseract_text else '(No text extracted)'}
                </div>
            </div>
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
    print(f"Images with Docling text: {sum(1 for r in results if r['docling_ocr_text'])}")
    print(f"Images with Tesseract text: {sum(1 for r in results if r['tesseract_ocr_text'])}")
    if similarity_matrix:
        print(f"\nImage Similarity Analysis:")
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results):
                if i < j and similarity_matrix[i][j] > 0.7:
                    print(f"  {result1['image_id']} <-> {result2['image_id']}: {similarity_matrix[i][j]:.3f} (highly similar)")
                elif i < j and similarity_matrix[i][j] < 0.5:
                    print(f"  {result1['image_id']} <-> {result2['image_id']}: {similarity_matrix[i][j]:.3f} (very different)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check if file exists in uploads
        uploads_dir = Path("uploads")
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"Using PDF from uploads: {pdf_path}\n")
        else:
            print("Usage: python test_ocr_comparison.py <path_to_pdf> [output_html]")
            print("\nOr place a PDF in the 'uploads' directory")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "ocr_comparison_test.html"
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    test_ocr_comparison(pdf_path, output_file)

