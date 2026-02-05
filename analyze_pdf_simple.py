"""
Simple utility to analyze PDF and show extracted images and tables.
Uses only PDFParser without full pipeline dependencies.
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pandas as pd
import re

def analyze_pdf(pdf_path: str):
    """Analyze PDF and display extracted content."""
    print(f"Analyzing PDF: {pdf_path}")
    print("=" * 80)
    
    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)
    
    try:
        # Extract images
        print("\nIMAGES:")
        print("-" * 80)
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    image_rects = page.get_image_rects(xref)
                    bbox = None
                    if image_rects:
                        bbox = {
                            'x0': image_rects[0].x0,
                            'y0': image_rects[0].y0,
                            'x1': image_rects[0].x1,
                            'y1': image_rects[0].y1
                        }
                    
                    image_id = f"p{page_num+1}_img{img_idx}"
                    images.append({
                        'image_id': image_id,
                        'page': page_num + 1,
                        'format': image_ext,
                        'size': pil_image.size,
                        'bbox': bbox
                    })
                except Exception as e:
                    print(f"  Error extracting image {img_idx} from page {page_num+1}: {e}")
                    continue
        
        print(f"Total images: {len(images)}")
        for i, img in enumerate(images, 1):
            print(f"\nImage {i}:")
            print(f"  ID: {img['image_id']}")
            print(f"  Page: {img['page']}")
            print(f"  Format: {img['format']}")
            print(f"  Size: {img['size'][0]} x {img['size'][1]} pixels")
            if img.get('bbox'):
                bbox = img['bbox']
                print(f"  Position: ({bbox['x0']:.1f}, {bbox['y0']:.1f}) to ({bbox['x1']:.1f}, {bbox['y1']:.1f})")
        
        # Extract tables
        print("\n\nTABLES:")
        print("-" * 80)
        tables = []
        for page_num, page in enumerate(plumber_doc.pages):
            page_tables = page.extract_tables()
            
            for table_idx, table in enumerate(page_tables):
                if table:
                    try:
                        headers = table[0] if table and len(table) > 0 else []
                        rows = table[1:] if len(table) > 1 else []
                        
                        df = pd.DataFrame(rows, columns=headers if headers else None)
                        table_markdown = df.to_markdown(index=False) if not df.empty else str(table)
                        
                        table_id = f"p{page_num+1}_table{table_idx}"
                        tables.append({
                            'table_id': table_id,
                            'page': page_num + 1,
                            'num_rows': len(rows),
                            'num_cols': len(headers) if headers else 0,
                            'markdown': table_markdown
                        })
                    except Exception as e:
                        print(f"  Error processing table {table_idx} from page {page_num+1}: {e}")
                        continue
        
        print(f"Total tables: {len(tables)}")
        for i, table in enumerate(tables, 1):
            print(f"\nTable {i}:")
            print(f"  ID: {table['table_id']}")
            print(f"  Page: {table['page']}")
            print(f"  Rows: {table['num_rows']}")
            print(f"  Columns: {table['num_cols']}")
            if table.get('markdown'):
                all_lines = table['markdown'].split('\n')
                lines = all_lines[:8]  # Show first 8 lines
                print(f"  Preview:")
                for line in lines:
                    print(f"    {line}")
                if len(all_lines) > 8:
                    remaining = len(all_lines) - 8
                    print(f"    ... ({remaining} more lines)")
        
        # Summary
        print("\n\nSUMMARY:")
        print("-" * 80)
        print(f"Total pages: {len(doc)}")
        print(f"Images extracted: {len(images)}")
        print(f"Tables extracted: {len(tables)}")
        
        # Images per page
        if images:
            print("\n\nIMAGES BY PAGE:")
            print("-" * 80)
            from collections import Counter
            page_counts = Counter(img['page'] for img in images)
            for page, count in sorted(page_counts.items()):
                print(f"  Page {page}: {count} image(s)")
        
        # Tables per page
        if tables:
            print("\n\nTABLES BY PAGE:")
            print("-" * 80)
            from collections import Counter
            page_counts = Counter(table['page'] for table in tables)
            for page, count in sorted(page_counts.items()):
                print(f"  Page {page}: {count} table(s)")
        
    finally:
        doc.close()
        plumber_doc.close()

if __name__ == "__main__":
    import io
    import sys
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    if len(sys.argv) < 2:
        # Check if file exists in uploads
        uploads_dir = Path("uploads")
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"Using PDF from uploads: {pdf_path}\n")
        else:
            print("Usage: python analyze_pdf_simple.py <path_to_pdf>")
            print("\nOr place a PDF in the 'uploads' directory")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    analyze_pdf(pdf_path)

