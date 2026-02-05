"""
Utility script to analyze PDF and show extracted images and tables.
"""
import sys
from pathlib import Path
from omnirag.ingestion.pdf_parser import PDFParser

def analyze_pdf(pdf_path: str):
    """Analyze PDF and display extracted content."""
    print(f"Analyzing PDF: {pdf_path}")
    print("=" * 80)
    
    parser = PDFParser(pdf_path)
    
    try:
        # Extract text blocks
        print("\n📄 TEXT BLOCKS:")
        print("-" * 80)
        text_blocks = parser.extract_text_blocks()
        print(f"Total text blocks: {len(text_blocks)}")
        for i, block in enumerate(text_blocks[:5], 1):  # Show first 5
            print(f"\nBlock {i} (Page {block['page']}, Section: {block.get('section_heading', 'N/A')}):")
            print(f"  Text: {block['text'][:200]}...")
        if len(text_blocks) > 5:
            print(f"\n... and {len(text_blocks) - 5} more text blocks")
        
        # Extract images
        print("\n\n🖼️  IMAGES:")
        print("-" * 80)
        images = parser.extract_images()
        print(f"Total images: {len(images)}")
        for i, img in enumerate(images, 1):
            print(f"\nImage {i}:")
            print(f"  ID: {img['image_id']}")
            print(f"  Page: {img['page']}")
            print(f"  Format: {img.get('format', 'unknown')}")
            if img.get('bbox'):
                bbox = img['bbox']
                print(f"  Position: ({bbox['x0']:.1f}, {bbox['y0']:.1f}) to ({bbox['x1']:.1f}, {bbox['y1']:.1f})")
            print(f"  Size: {img['image_data'].size if 'image_data' in img else 'N/A'}")
        
        # Extract tables
        print("\n\n📊 TABLES:")
        print("-" * 80)
        tables = parser.extract_tables()
        print(f"Total tables: {len(tables)}")
        for i, table in enumerate(tables, 1):
            print(f"\nTable {i}:")
            print(f"  ID: {table['table_id']}")
            print(f"  Page: {table['page']}")
            print(f"  Rows: {table.get('num_rows', 'N/A')}")
            print(f"  Columns: {table.get('num_cols', 'N/A')}")
            if table.get('table_markdown'):
                # Show first few lines of markdown
                all_lines = table['table_markdown'].split('\n')
                lines = all_lines[:5]
                print(f"  Preview:")
                for line in lines:
                    print(f"    {line}")
                if len(all_lines) > 5:
                    remaining = len(all_lines) - 5
                    print(f"    ... ({remaining} more lines)")
        
        # Summary
        print("\n\n📊 SUMMARY:")
        print("-" * 80)
        print(f"Total pages processed: {len(parser.doc)}")
        print(f"Text blocks: {len(text_blocks)}")
        print(f"Images: {len(images)}")
        print(f"Tables: {len(tables)}")
        
        # Show image captions if found
        if images:
            print("\n\n🔍 IMAGE CAPTIONS:")
            print("-" * 80)
            for i, img in enumerate(images[:10], 1):  # Show first 10
                caption = parser.find_image_caption(img['page'], img.get('bbox'))
                if caption:
                    print(f"Image {i} (Page {img['page']}): {caption}")
                else:
                    print(f"Image {i} (Page {img['page']}): No caption found")
        
    finally:
        parser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check if file exists in uploads
        uploads_dir = Path("uploads")
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"Using PDF from uploads: {pdf_path}")
        else:
            print("Usage: python analyze_pdf.py <path_to_pdf>")
            print("\nOr place a PDF in the 'uploads' directory")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    analyze_pdf(pdf_path)

