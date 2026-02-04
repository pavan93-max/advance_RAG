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
    
    def __init__(self, model_type: str = "openai", model_name: str = "gpt-3.5-turbo", 
                 openai_api_key: Optional[str] = None,
                 max_context_tokens: int = 3000,
                 max_response_tokens: int = 500):
        self.model_type = model_type
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens  # Limit context to save tokens
        self.max_response_tokens = max_response_tokens  # Limit response length
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
    
    def generate(self, query: str, text_results: List[Dict], 
                 image_results: List[Dict]) -> Dict:
        """
        Generate answer with citations.
        Optimized for token usage.
        Returns: {
            'answer': str,
            'citations': List[Dict],
            'image_references': List[str]
        }
        """
        # Build context from retrieved text (with token optimization)
        context_parts = []
        citations = []
        
        # Estimate tokens: ~4 characters per token
        chars_per_token = 4
        max_context_chars = self.max_context_tokens * chars_per_token
        current_chars = 0
        
        for i, result in enumerate(text_results, 1):
            text = result['text']
            page = result['page']
            section = result.get('section_heading', '')
            
            # Truncate text if needed to stay within token limit
            chunk_text = f"[Document {i} - Page {page}]\n{text}"
            chunk_chars = len(chunk_text)
            
            # If adding this chunk would exceed limit, truncate it
            if current_chars + chunk_chars > max_context_chars:
                remaining_chars = max_context_chars - current_chars - 100  # Leave some buffer
                if remaining_chars > 100:  # Only add if meaningful
                    truncated_text = text[:remaining_chars] + "..."
                    chunk_text = f"[Document {i} - Page {page}]\n{truncated_text}"
                    context_parts.append(chunk_text)
                    citations.append({
                        'text': text[:200] + "..." if len(text) > 200 else text,
                        'page': page,
                        'section': section
                    })
                break  # Stop adding more chunks
            
            context_parts.append(chunk_text)
            citations.append({
                'text': text[:200] + "..." if len(text) > 200 else text,
                'page': page,
                'section': section
            })
            current_chars += chunk_chars
        
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
        
        return {
            'answer': answer,
            'citations': citations,
            'image_references': image_references
        }
    
    def _build_prompt(self, query: str, context: str, image_context: List[str]) -> str:
        """Build prompt for LLM (optimized for token usage)."""
        # Limit image context to save tokens
        max_images = 3
        image_text = "\n".join([f"- {img}" for img in image_context[:max_images]]) if image_context else "None"
        if len(image_context) > max_images:
            image_text += f"\n... and {len(image_context) - max_images} more figures"
        
        # Concise prompt to save tokens
        prompt = f"""Answer based on this document:

CONTEXT:
{context}

FIGURES:
{image_text}

QUESTION: {query}

Answer concisely. Cite page numbers [Page X]. Reference figures when relevant. If not found, say so.

ANSWER:"""
        
        return prompt
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API with token optimization."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based on provided document context. Always cite page numbers [Page X] when referencing information. Be concise."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=self.max_response_tokens,  # Limit response tokens
                top_p=0.9
            )
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

