"""
Export PDF images and tables to an HTML report file.
"""
import sys
import os
import io
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pandas as pd
import base64
from datetime import datetime

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def export_pdf_content(pdf_path: str, output_dir: str = "pdf_export"):
    """Export PDF images and tables to HTML report."""
    print(f"Exporting content from: {pdf_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)
    
    html_content = []
    pdf_name = Path(pdf_path).name
    export_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_pages = len(doc)
    
    html_content.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PDF Content Export</title>
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
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
        }}
        .image-section {{
            background: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-info {{
            background: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
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
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .table-section {{
            background: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 12px;
        }}
        table th, table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        table th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .summary {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .page-header {{
            background: #4CAF50;
            color: white;
            padding: 10px;
            margin: 20px 0 10px 0;
            border-radius: 4px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>PDF Content Export Report</h1>
    <div class="summary">
        <p><strong>PDF File:</strong> {pdf_name}</p>
        <p><strong>Export Date:</strong> {export_date}</p>
        <p><strong>Total Pages:</strong> {total_pages}</p>
    </div>
""")
    
    try:
        # Extract and save images
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
                    
                    # Skip very small images (likely decorative)
                    if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                        continue
                    
                    # Save image to file
                    image_id = f"p{page_num+1}_img{img_idx}"
                    image_filename = f"{image_id}.{image_ext}"
                    image_filepath = images_dir / image_filename
                    pil_image.save(image_filepath)
                    
                    # Get position info
                    image_rects = page.get_image_rects(xref)
                    bbox = None
                    if image_rects:
                        bbox = {
                            'x0': image_rects[0].x0,
                            'y0': image_rects[0].y0,
                            'x1': image_rects[0].x1,
                            'y1': image_rects[0].y1
                        }
                    
                    images.append({
                        'id': image_id,
                        'page': page_num + 1,
                        'format': image_ext,
                        'size': pil_image.size,
                        'bbox': bbox,
                        'filename': image_filename,
                        'filepath': image_filepath
                    })
                except Exception as e:
                    print(f"  Error extracting image {img_idx} from page {page_num+1}: {e}")
                    continue
        
        # Group images by page
        images_by_page = {}
        for img in images:
            page = img['page']
            if page not in images_by_page:
                images_by_page[page] = []
            images_by_page[page].append(img)
        
        # Add images to HTML
        html_content.append('<h2>Extracted Images</h2>')
        html_content.append(f'<p><strong>Total Images:</strong> {len(images)}</p>')
        
        for page_num in sorted(images_by_page.keys()):
            html_content.append(f'<div class="page-header">Page {page_num}</div>')
            for img in images_by_page[page_num]:
                html_content.append('<div class="image-section">')
                html_content.append(f'<div class="image-info">')
                html_content.append(f'<strong>Image ID:</strong> {img["id"]}<br>')
                html_content.append(f'<strong>Format:</strong> {img["format"]}<br>')
                html_content.append(f'<strong>Size:</strong> {img["size"][0]} x {img["size"][1]} pixels<br>')
                if img.get('bbox'):
                    bbox = img['bbox']
                    html_content.append(f'<strong>Position:</strong> ({bbox["x0"]:.1f}, {bbox["y0"]:.1f}) to ({bbox["x1"]:.1f}, {bbox["y1"]:.1f})')
                html_content.append('</div>')
                html_content.append('<div class="image-container">')
                # Use relative path for images
                html_content.append(f'<img src="images/{img["filename"]}" alt="{img["id"]}">')
                html_content.append('</div>')
                html_content.append('</div>')
        
        # Extract and save tables
        tables = []
        for page_num, page in enumerate(plumber_doc.pages):
            page_tables = page.extract_tables()
            
            for table_idx, table in enumerate(page_tables):
                if table:
                    try:
                        headers = table[0] if table and len(table) > 0 else []
                        rows = table[1:] if len(table) > 1 else []
                        
                        if headers and rows:
                            df = pd.DataFrame(rows, columns=headers)
                        elif rows:
                            # No headers, use generic column names
                            df = pd.DataFrame(rows)
                        else:
                            continue
                        
                        table_id = f"p{page_num+1}_table{table_idx}"
                        tables.append({
                            'id': table_id,
                            'page': page_num + 1,
                            'dataframe': df,
                            'num_rows': len(rows),
                            'num_cols': len(headers) if headers else len(rows[0]) if rows else 0
                        })
                    except Exception as e:
                        print(f"  Error processing table {table_idx} from page {page_num+1}: {e}")
                        continue
        
        # Group tables by page
        tables_by_page = {}
        for table in tables:
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            tables_by_page[page].append(table)
        
        # Add tables to HTML
        html_content.append('<h2>Extracted Tables</h2>')
        html_content.append(f'<p><strong>Total Tables:</strong> {len(tables)}</p>')
        
        for page_num in sorted(tables_by_page.keys()):
            html_content.append(f'<div class="page-header">Page {page_num}</div>')
            for table in tables_by_page[page_num]:
                html_content.append('<div class="table-section">')
                html_content.append(f'<div class="image-info">')
                html_content.append(f'<strong>Table ID:</strong> {table["id"]}<br>')
                html_content.append(f'<strong>Rows:</strong> {table["num_rows"]}<br>')
                html_content.append(f'<strong>Columns:</strong> {table["num_cols"]}')
                html_content.append('</div>')
                
                # Convert DataFrame to HTML table
                html_table = table['dataframe'].to_html(
                    classes='extracted-table',
                    index=False,
                    escape=False,
                    table_id=table['id']
                )
                html_content.append(html_table)
                html_content.append('</div>')
        
        # Summary
        html_content.append(f"""
    <h2>Summary</h2>
    <div class="summary">
        <p><strong>Total Images Extracted:</strong> {len(images)}</p>
        <p><strong>Total Tables Extracted:</strong> {len(tables)}</p>
        <p><strong>Images Directory:</strong> images/</p>
    </div>
</body>
</html>
""")
        
        # Write HTML file
        html_file = output_path / "pdf_content_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        print(f"\nExport complete!")
        print(f"HTML Report: {html_file}")
        print(f"Images saved to: {images_dir}")
        print(f"Total images: {len(images)}")
        print(f"Total tables: {len(tables)}")
        
    finally:
        doc.close()
        plumber_doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check if file exists in uploads
        uploads_dir = Path("uploads")
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"Using PDF from uploads: {pdf_path}\n")
        else:
            print("Usage: python export_pdf_content.py <path_to_pdf> [output_dir]")
            print("\nOr place a PDF in the 'uploads' directory")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pdf_export"
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    export_pdf_content(pdf_path, output_dir)

