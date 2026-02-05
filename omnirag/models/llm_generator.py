"""
LLM generation with citations and image references.
Uses OpenAI API with token optimization.
"""
import os
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. Install with: pip install openai")


class LLMGenerator:
    """Generate answers using OpenAI API with citations and token optimization."""
    
    def __init__(self, model_type: str = "openai", model_name: str = "gpt-4o-mini", 
                 openai_api_key: Optional[str] = None,
                 max_context_tokens: Optional[int] = None,
                 max_response_tokens: Optional[int] = None):
        self.model_type = model_type
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens  # None = no limit
        self.max_response_tokens = max_response_tokens  # None = no limit (model default)
        self.openai_client = None
        
        if model_type == "openai" and OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    print(f"Using OpenAI API with model: {model_name}")
                except Exception as e:
                    print(f"Error initializing OpenAI client: {e}. Falling back to template.")
                    self.model_type = "template"
            else:
                print("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass openai_api_key.")
                self.model_type = "template"
        else:
            self.model_type = "template"
            print("Using template-based generation (no LLM)")
    
    def _extract_figure_references(self, answer: str, text_results: List[Dict], 
                                   image_results: List[Dict]) -> List[Dict]:
        """
        Extract figure references from LLM answer and find corresponding images.
        
        Args:
            answer: LLM generated answer text
            text_results: Retrieved text results (for page context)
            image_results: Already retrieved image results
            
        Returns:
            List of additional image references found from figure mentions
        """
        import re
        from omnirag.db.vector_store import VectorStore
        
        figure_refs = []
        
        # Pattern to match figure references: "Figure 1", "Figure 2", "Fig. 1", etc.
        patterns = [
            r'Figure\s+(\d+)',
            r'Fig\.\s*(\d+)',
            r'Fig\s+(\d+)',
            r'figure\s+(\d+)',
        ]
        
        found_figures = set()
        for pattern in patterns:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            for match in matches:
                fig_num = int(match.group(1))
                found_figures.add(fig_num)
        
        if not found_figures:
            return []
        
        # Get pages mentioned in the answer - try multiple patterns
        mentioned_pages = set()
        
        # Pattern 1: [Page X]
        page_pattern1 = r'\[Page\s+(\d+)\]'
        for match in re.finditer(page_pattern1, answer):
            mentioned_pages.add(int(match.group(1)))
        
        # Pattern 2: Page X (without brackets)
        page_pattern2 = r'[Pp]age\s+(\d+)'
        for match in re.finditer(page_pattern2, answer):
            mentioned_pages.add(int(match.group(1)))
        
        # Pattern 3: Look for page numbers near figure mentions
        for fig_num in found_figures:
            # Look for "Figure X ... Page Y" pattern
            fig_page_pattern = rf'[Ff]igure\s+{fig_num}.*?[Pp]age\s+(\d+)'
            for match in re.finditer(fig_page_pattern, answer, re.DOTALL):
                mentioned_pages.add(int(match.group(1)))
        
        # If no pages mentioned, use pages from text results
        if not mentioned_pages and text_results:
            mentioned_pages = {r.get('page', 0) for r in text_results if r.get('page', 0) > 0}
        
        # Search for images on mentioned pages
        # We need access to vector_store, but we don't have it here
        # Instead, we'll return figure numbers and pages, and let the caller handle it
        # Or we can check existing image_results first
        
        # First, check if images are already in image_results
        existing_image_ids = {img.get('id', '') for img in image_results}
        
        # Return figure references that need to be looked up
        figure_refs = []
        for fig_num in found_figures:
            for page in mentioned_pages:
                figure_refs.append({
                    'figure_number': fig_num,
                    'page': page,
                    'mention': f"Figure {fig_num}"
                })
        
        return figure_refs
    
    def generate(self, query: str, text_results: List[Dict], 
                 image_results: List[Dict]) -> Dict:
        """
        Generate answer with citations.
        Returns: {
            'answer': str,
            'citations': List[Dict],
            'image_references': List[str]
        }
        """
        # Build context from retrieved text (no token limits)
        context_parts = []
        citations = []
        
        # Include all retrieved text results (no truncation)
        for i, result in enumerate(text_results, 1):
            text = result['text']
            page = result['page']
            section = result.get('section_heading', '')
            is_table = result.get('is_table', False)
            
            # Format table chunks differently
            if is_table:
                table_meta = result.get('table_metadata', {})
                chunk_text = f"[Table {i} - Page {page}]\n{text}"
                
                # Add related text chunks for context
                related_text = result.get('related_text_content', '')
                if related_text:
                    chunk_text += f"\n\nRelated Context:\n{related_text[:500]}"
                
                # Add table HTML if available for better representation
                if table_meta.get('table_html'):
                    chunk_text += f"\n\nTable HTML:\n{table_meta['table_html']}"
            else:
                chunk_text = f"[Document {i} - Page {page}]\n{text}"
            
            context_parts.append(chunk_text)
            citations.append({
                'text': text[:500] + "..." if len(text) > 500 else text,  # Truncate only for citation display
                'page': page,
                'section': section,
                'is_table': is_table,
                'table_metadata': result.get('table_metadata') if is_table else None,
                'related_text_chunks': result.get('related_text_chunks', []) if is_table else [],  # Add related text for tables
                'related_text_content': result.get('related_text_content', '') if is_table else ''  # Combined text
            })
        
        context = "\n\n".join(context_parts)
        
        # Build image context
        image_context = []
        image_references = []
        
        for img in image_results:
            caption = img.get('vlm_caption', '')
            ocr = img.get('ocr_text', '')
            context_text = img.get('surrounding_context', '')
            page = img.get('page', 0)
            
            img_desc = f"Figure on page {page}"
            if caption:
                img_desc += f": {caption}"
            if ocr:
                img_desc += f" (OCR: {ocr[:100]})"
            
            image_context.append(img_desc)
            image_references.append({
                'image_id': img.get('id', ''),
                'page': page,
                'caption': caption,
                'image_path': img.get('image_path', '')
            })
        
        # Build prompt (optimized for tokens)
        prompt = self._build_prompt(query, context, image_context)
        
        # Generate answer
        if self.model_type == "openai" and self.openai_client:
            answer = self._generate_openai(prompt)
        else:
            answer = self._generate_template(query, context, image_context)
        
        # Extract figure references from the answer and find corresponding images
        figure_refs = self._extract_figure_references(answer, text_results, image_results)
        
        # If figure references found, try to get images from those pages
        if figure_refs and hasattr(self, 'vector_store') and self.vector_store:
            all_images = self.vector_store.image_collection.get()
            if all_images['ids']:
                # Build a map of page -> images for quick lookup
                page_to_images = {}
                for i, img_id in enumerate(all_images['ids']):
                    img_page = all_images['metadatas'][i].get('page', 0)
                    if img_page not in page_to_images:
                        page_to_images[img_page] = []
                    metadata = self.vector_store._load_metadata(img_id)
                    if metadata:
                        page_to_images[img_page].append({
                            'id': img_id,
                            'metadata': metadata,
                            'index': len(page_to_images[img_page])  # Track order on page
                        })
                
                # Match figure references to images
                for fig_ref in figure_refs:
                    page = fig_ref.get('page', 0)
                    fig_num = fig_ref.get('figure_number', 0)
                    
                    if page > 0 and page in page_to_images:
                        page_images = page_to_images[page]
                        
                        # Try to find image matching figure number
                        matched = False
                        for img_data in page_images:
                            img_id = img_data['id']
                            metadata = img_data['metadata']
                            
                            # Check if already in image_references
                            if any(ref.get('image_id') == img_id for ref in image_references):
                                continue
                            
                            # Try to match by caption containing "Figure X" or by position
                            caption = metadata.get('vlm_caption', '').lower()
                            surrounding = metadata.get('surrounding_context', '').lower()
                            
                            # Check if caption or surrounding text mentions this figure number
                            fig_pattern = f'figure {fig_num}|fig {fig_num}|fig\. {fig_num}'
                            import re
                            if (re.search(fig_pattern, caption, re.IGNORECASE) or 
                                re.search(fig_pattern, surrounding, re.IGNORECASE) or
                                img_data['index'] == fig_num - 1):  # Match by position (Figure 1 = first image)
                                
                                image_references.append({
                                    'image_id': img_id,
                                    'page': page,
                                    'caption': fig_ref.get('mention', f"Figure {fig_num} on page {page}"),
                                    'image_path': metadata.get('image_path', '')
                                })
                                matched = True
                                break
                        
                        # If no exact match, add ALL images from page (fallback)
                        # This ensures we show images even if figure number matching fails
                        if not matched and page_images:
                            for img_data in page_images:
                                img_id = img_data['id']
                                metadata = img_data['metadata']
                                
                                if not any(ref.get('image_id') == img_id for ref in image_references):
                                    # Use figure number if available, otherwise just page
                                    if fig_num > 0:
                                        caption_text = f"Figure {fig_num} on page {page}"
                                    else:
                                        caption_text = f"Figure on page {page}"
                                    
                                    image_references.append({
                                        'image_id': img_id,
                                        'page': page,
                                        'caption': caption_text,
                                        'image_path': metadata.get('image_path', '')
                                    })
        
        return {
            'answer': answer,
            'citations': citations,
            'image_references': image_references
        }
    
    def _build_prompt(self, query: str, context: str, image_context: List[str]) -> str:
        """
        Build production-grade prompt with citation enforcement.
        """
        # Structure image context (figures are supplementary, not primary source)
        image_text = ""
        if image_context:
            image_text = "\nSUPPLEMENTARY FIGURES (for reference only, answer from text context above):\n"
            # Include all images (no limit)
            for i, img in enumerate(image_context, 1):
                image_text += f"Figure {i}: {img}\n"
            image_text += "\nNote: Figures are supplementary. Base your answer on the TEXT CONTEXT above.\n"
        
        # Production-grade prompt with explicit instructions
        prompt = f"""You are a precise document Q&A assistant. Answer using the provided TEXT CONTEXT below.

TEXT CONTEXT FROM DOCUMENT:
{context}

{image_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Read the TEXT CONTEXT above carefully - this is your PRIMARY source
2. Analyze the context to find relevant information for the question
3. If you find ANY relevant information in the TEXT CONTEXT (even if partial):
   - Provide a clear, comprehensive answer based on what you found
   - Cite the EXACT page number using format: [Page X] for each piece of information
   - If the answer is partial, mention what information is available
   - Optionally reference figures if they support your answer: [Figure X on Page Y]
   - Use exact wording from the document when possible
4. Only say "This information is not found in the document" if:
   - The context contains NO information related to the question AT ALL
   - The question asks for something completely unrelated to the provided context
5. If the context has related information but doesn't fully answer the question:
   - Provide what information IS available
   - Mention what aspects are covered and what might be missing
6. DO NOT make up information not in the context
7. Be thorough and comprehensive in your answer

Provide your answer directly:"""
        
        return prompt
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API with improved system prompt."""
        # Enhanced system prompt with citation enforcement
        system_prompt = """You are a document Q&A assistant. Your role is to:
1. Answer questions based on provided document context
2. Always cite page numbers using [Page X] format when referencing information
3. Reference figures when relevant using [Figure X on Page Y]
4. Be thorough - extract and present ALL relevant information from the context
5. If information is partially available, provide what you can find
6. Only state "not found" if the context has NO relevant information at all
7. Never hallucinate or make up information not in the context
8. Be precise, factual, and comprehensive
9. Use exact wording from the document when possible"""
        
        try:
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Lower temperature for more factual responses
                "top_p": 0.9
            }
            
            # Only add max_tokens if specified (None = no limit, use model default)
            if self.max_response_tokens is not None:
                request_params["max_tokens"] = self.max_response_tokens
            
            response = self.openai_client.chat.completions.create(**request_params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return self._generate_template("", "", [])
    
    def _generate_template(self, query: str, context: str, image_context: List[str]) -> str:
        """Template-based generation (fallback)."""
        # Simple template-based answer
        answer_parts = []
        
        if context:
            # Extract first relevant paragraph
            paragraphs = context.split('\n\n')
            if paragraphs:
                first_para = paragraphs[0]
                if len(first_para) > 100:
                    answer_parts.append(first_para[:500] + "...")
                else:
                    answer_parts.append(first_para)
        
        if image_context:
            answer_parts.append(f"\n\nRelevant figures are available: {len(image_context)} figure(s) found.")
        
        if not answer_parts:
            return "I couldn't find relevant information in the document to answer this question."
        
        answer = "\n".join(answer_parts)
        answer += "\n\n[Note: This is a template-based response. Configure OpenAI API key for better answers.]"
        
        return answer

