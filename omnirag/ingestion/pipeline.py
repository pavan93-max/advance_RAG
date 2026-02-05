"""
Complete ingestion pipeline: PDF → Text + Images + Tables → Vector Store
"""
import os
from typing import List, Dict
from pathlib import Path

from omnirag.ingestion.pdf_parser import PDFParser
from omnirag.models.image_processor import ImageProcessor
from omnirag.db.vector_store import VectorStore
from omnirag.utils.spatial import find_nearby_chunks, estimate_text_chunk_bbox


class IngestionPipeline:
    """Complete pipeline for ingesting PDF documents."""
    
    def __init__(self, vector_store: VectorStore, image_processor: ImageProcessor):
        self.vector_store = vector_store
        self.image_processor = image_processor
        # Link image processor to vector store for CLIP text encoding
        self.vector_store.image_processor = image_processor
    
    def ingest_pdf(self, pdf_path: str, doc_name: str = None) -> Dict:
        """
        Ingest a PDF document.
        Returns summary of ingested content.
        """
        print(f"Starting ingestion of {pdf_path}...")
        
        parser = PDFParser(pdf_path)
        
        try:
            # Extract content
            print("Extracting text blocks...")
            text_blocks = parser.extract_text_blocks()
            print(f"Extracted {len(text_blocks)} text blocks")
            
            print("Extracting images...")
            images = parser.extract_images()
            print(f"Extracted {len(images)} images")
            
            print("Extracting tables...")
            tables = parser.extract_tables()
            print(f"Extracted {len(tables)} tables")
            
            # Process images
            print("Processing images (OCR, VLM, CLIP)...")
            processed_images = []
            image_id_to_chunk_mapping = {}
            
            for img in images:
                # Get caption if exists
                caption = parser.find_image_caption(img['page'], img['bbox'])
                
                # Get surrounding context
                text_before, text_after = parser.get_surrounding_text(
                    img['page'], img['bbox']
                )
                surrounding_context = f"{text_before}\n\n{text_after}".strip()
                
                # Find related text chunks using spatial proximity
                page_chunks = [chunk for chunk in text_blocks 
                             if chunk['page'] == img['page']]
                
                # Use spatial proximity to find closest chunks
                nearby_results = find_nearby_chunks(
                    target_bbox=img.get('bbox'),
                    text_chunks=page_chunks,
                    max_chunks=3,
                    page=img['page']
                )
                
                # Get text content from related chunks (for image metadata)
                related_text_chunks = []
                related_chunk_ids = []
                for result in nearby_results:
                    chunk = result['chunk']
                    related_text_chunks.append({
                        'chunk_id': chunk.get('chunk_id', ''),
                        'text': chunk.get('text', '')[:500],  # Truncate for storage
                        'section_heading': chunk.get('section_heading', ''),
                        'distance': result.get('distance', 0),
                        'overlap': result.get('overlap', 0)
                    })
                    related_chunk_ids.append(chunk.get('chunk_id', ''))
                    
                    # Link image to text chunk (bidirectional)
                    if 'linked_image_ids' not in chunk:
                        chunk['linked_image_ids'] = []
                    chunk['linked_image_ids'].append(img['image_id'])
                
                # Combine related text chunks into searchable text
                related_text_content = "\n\n".join([
                    f"[{chunk.get('section_heading', '')}]\n{chunk.get('text', '')}"
                    for chunk in related_text_chunks
                ])
                
                # Process image
                processed = self.image_processor.process_image(
                    img['image_data'],
                    caption=caption,
                    surrounding_context=surrounding_context
                )
                
                processed_img = {
                    **img,
                    **processed,
                    # Add related text chunks metadata
                    'related_text_chunks': related_text_chunks,
                    'related_chunk_ids': related_chunk_ids,
                    'related_text_content': related_text_content  # Combined text for search
                }
                processed_images.append(processed_img)
            
            # Add tables as text chunks with enhanced representation
            for table in tables:
                # Find related text chunks using spatial proximity
                page_chunks = [chunk for chunk in text_blocks 
                             if chunk['page'] == table['page'] and not chunk.get('is_table', False)]
                
                # Estimate table bbox if not available (tables might not have explicit bbox)
                table_bbox = table.get('bbox')
                if not table_bbox:
                    # Use a default position for tables (middle of page)
                    table_bbox = {'x0': 50, 'y0': 200, 'x1': 562, 'y1': 400}
                
                # Use spatial proximity to find closest chunks
                nearby_results = find_nearby_chunks(
                    target_bbox=table_bbox,
                    text_chunks=page_chunks,
                    max_chunks=3,
                    page=table['page']
                )
                
                # Get text content from related chunks (for table metadata)
                related_text_chunks = []
                related_chunk_ids = []
                for result in nearby_results:
                    chunk = result['chunk']
                    related_text_chunks.append({
                        'chunk_id': chunk.get('chunk_id', ''),
                        'text': chunk.get('text', '')[:500],  # Truncate for storage
                        'section_heading': chunk.get('section_heading', ''),
                        'distance': result.get('distance', 0),
                        'overlap': result.get('overlap', 0)
                    })
                    related_chunk_ids.append(chunk.get('chunk_id', ''))
                    
                    # Link table to text chunk (bidirectional)
                    if 'linked_table_ids' not in chunk:
                        chunk['linked_table_ids'] = []
                    chunk['linked_table_ids'].append(table['table_id'])
                
                # Combine related text chunks into searchable text
                related_text_content = "\n\n".join([
                    f"[{chunk.get('section_heading', '')}]\n{chunk.get('text', '')}"
                    for chunk in related_text_chunks
                ])
                
                # Create searchable text representation of table
                table_text_parts = []
                
                # Add headers as searchable text
                if table.get('table_data', {}).get('headers'):
                    headers = table['table_data']['headers']
                    table_text_parts.append(f"Table with columns: {', '.join(str(h) for h in headers if h)}")
                
                # Add markdown representation
                if table.get('table_markdown'):
                    table_text_parts.append(table['table_markdown'])
                
                # Add structured data as text for better searchability
                if table.get('table_data', {}).get('rows'):
                    rows = table['table_data']['rows']
                    # Include first few rows as text for searchability
                    for i, row in enumerate(rows[:5]):  # First 5 rows
                        row_text = ' | '.join(str(cell) for cell in row if cell)
                        table_text_parts.append(f"Row {i+1}: {row_text}")
                
                table_text = "\n".join(table_text_parts)
                
                table_chunk = {
                    'text': f"[TABLE on Page {table['page']}]\n{table_text}",
                    'page': table['page'],
                    'section_heading': 'Tables',
                    'chunk_id': table['table_id'],
                    'linked_image_ids': [],
                    'is_table': True,  # Mark as table for special handling
                    # Add related text chunks metadata (similar to images)
                    'related_text_chunks': related_text_chunks,
                    'related_chunk_ids': related_chunk_ids,
                    'related_text_content': related_text_content,  # Combined text for search
                    'table_metadata': {
                        'num_rows': table.get('num_rows', 0),
                        'num_cols': table.get('num_cols', 0),
                        'table_markdown': table.get('table_markdown', ''),
                        'table_html': table.get('table_data', {}).get('html', ''),
                        'table_csv': table.get('table_data', {}).get('csv', '')
                    }
                }
                text_blocks.append(table_chunk)
            
            # Store in vector database
            print("Storing text chunks in vector database...")
            self.vector_store.add_text_chunks(text_blocks)
            
            print("Storing images in vector database...")
            self.vector_store.add_images(processed_images)
            
            print("Ingestion complete!")
            
            return {
                'text_chunks': len(text_blocks),
                'images': len(processed_images),
                'tables': len(tables),
                'pages': max([chunk['page'] for chunk in text_blocks], default=0)
            }
        
        finally:
            parser.close()

