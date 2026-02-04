"""
PDF Parser for extracting text, images, and tables from PDF documents.
"""
import io
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pandas as pd


class PDFParser:
    """Parse PDF documents to extract text, images, and tables."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.plumber_doc = pdfplumber.open(pdf_path)
        
    def extract_text_blocks(self) -> List[Dict]:
        """
        Extract text blocks with page numbers and section information.
        Returns list of dicts with: text, page, section_heading, chunk_id
        """
        text_blocks = []
        current_section = "Introduction"
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Detect section headings (lines that are short, bold, or all caps)
            lines = text.split('\n')
            section_heading = current_section
            
            for line in lines:
                line_stripped = line.strip()
                # Heuristic for section headings
                if (len(line_stripped) < 100 and 
                    len(line_stripped) > 0 and
                    (line_stripped.isupper() or 
                     re.match(r'^\d+\.?\s+[A-Z]', line_stripped) or
                     re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line_stripped))):
                    section_heading = line_stripped
                    current_section = section_heading
            
            # Split text into chunks (paragraphs)
            paragraphs = self._split_into_paragraphs(text)
            
            for idx, para in enumerate(paragraphs):
                if len(para.strip()) > 50:  # Filter very short chunks
                    text_blocks.append({
                        'text': para.strip(),
                        'page': page_num + 1,
                        'section_heading': section_heading,
                        'chunk_id': f"p{page_num+1}_chunk{idx}"
                    })
        
        return text_blocks
    
    def extract_images(self) -> List[Dict]:
        """
        Extract images from PDF pages.
        Returns list of dicts with: image_data, page, image_id, bbox
        """
        images = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Get image position on page
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
                        'image_data': pil_image,
                        'image_bytes': image_bytes,
                        'page': page_num + 1,
                        'image_id': image_id,
                        'bbox': bbox,
                        'format': image_ext
                    })
                except Exception as e:
                    print(f"Error extracting image {img_idx} from page {page_num+1}: {e}")
                    continue
        
        return images
    
    def extract_tables(self) -> List[Dict]:
        """
        Extract tables from PDF and convert to markdown.
        Returns list of dicts with: table_markdown, page, table_id
        """
        tables = []
        
        for page_num, page in enumerate(self.plumber_doc.pages):
            page_tables = page.extract_tables()
            
            for table_idx, table in enumerate(page_tables):
                if table:
                    try:
                        # Convert to DataFrame then to markdown
                        df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                        table_markdown = df.to_markdown(index=False)
                        
                        table_id = f"p{page_num+1}_table{table_idx}"
                        
                        tables.append({
                            'table_markdown': table_markdown,
                            'page': page_num + 1,
                            'table_id': table_id,
                            'raw_table': table
                        })
                    except Exception as e:
                        print(f"Error processing table {table_idx} from page {page_num+1}: {e}")
                        continue
        
        return tables
    
    def get_surrounding_text(self, page_num: int, bbox: Optional[Dict]) -> Tuple[str, str]:
        """
        Get text before and after an image on the same page.
        Returns: (text_before, text_after)
        """
        page = self.doc[page_num - 1]  # 0-indexed
        text = page.get_text()
        
        if not bbox:
            # If no bbox, return all page text split roughly in half
            lines = text.split('\n')
            mid = len(lines) // 2
            return '\n'.join(lines[:mid]), '\n'.join(lines[mid:])
        
        # Get text blocks and find which ones are before/after the image
        blocks = page.get_text("blocks")
        text_before = []
        text_after = []
        image_y = bbox['y0']
        
        for block in blocks:
            block_y = block[1]  # y0 of block
            block_text = block[4]  # text content
            
            if block_y < image_y - 50:  # Before image (with margin)
                text_before.append(block_text)
            elif block_y > image_y + 50:  # After image (with margin)
                text_after.append(block_text)
        
        return '\n'.join(text_before), '\n'.join(text_after)
    
    def find_image_caption(self, page_num: int, bbox: Optional[Dict]) -> Optional[str]:
        """
        Try to find caption for an image (text near the image, often starting with "Figure" or "Fig").
        """
        page = self.doc[page_num - 1]
        text = page.get_text()
        
        if not bbox:
            return None
        
        # Look for caption patterns near the image
        blocks = page.get_text("blocks")
        image_y = bbox['y1']  # Bottom of image
        
        for block in blocks:
            block_y = block[1]
            block_text = block[4].strip()
            
            # Caption is usually just below the image
            if (image_y <= block_y <= image_y + 100 and 
                len(block_text) < 200 and
                (block_text.lower().startswith('figure') or
                 block_text.lower().startswith('fig.') or
                 block_text.lower().startswith('fig ') or
                 re.match(r'^Figure\s+\d+', block_text, re.IGNORECASE))):
                return block_text
        
        return None
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or single newline if line is short
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        result = []
        for para in paragraphs:
            if len(para) > 1000:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = []
                current_len = 0
                
                for sent in sentences:
                    if current_len + len(sent) > 1000 and current_chunk:
                        result.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_len = len(sent)
                    else:
                        current_chunk.append(sent)
                        current_len += len(sent)
                
                if current_chunk:
                    result.append(' '.join(current_chunk))
            else:
                result.append(para)
        
        return result
    
    def close(self):
        """Close PDF documents."""
        self.doc.close()
        self.plumber_doc.close()

