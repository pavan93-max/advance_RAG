"""
Complete ingestion pipeline: PDF → Text + Images + Tables → Vector Store
"""
import os
from typing import List, Dict
from pathlib import Path

from omnirag.ingestion.pdf_parser import PDFParser
from omnirag.models.image_processor import ImageProcessor
from omnirag.db.vector_store import VectorStore


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
                
                # Process image
                processed = self.image_processor.process_image(
                    img['image_data'],
                    caption=caption,
                    surrounding_context=surrounding_context
                )
                
                processed_img = {
                    **img,
                    **processed
                }
                processed_images.append(processed_img)
                
                # Link image to nearby text chunks
                # Find chunks on the same page
                page_chunks = [chunk for chunk in text_blocks 
                             if chunk['page'] == img['page']]
                for chunk in page_chunks[:3]:  # Link to up to 3 nearby chunks
                    if 'linked_image_ids' not in chunk:
                        chunk['linked_image_ids'] = []
                    chunk['linked_image_ids'].append(img['image_id'])
            
            # Add tables as text chunks
            for table in tables:
                table_chunk = {
                    'text': f"[TABLE]\n{table['table_markdown']}",
                    'page': table['page'],
                    'section_heading': 'Tables',
                    'chunk_id': table['table_id'],
                    'linked_image_ids': []
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

