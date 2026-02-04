"""
Image processing: OCR, VLM captioning, and CLIP embeddings.
"""
import io
from typing import Optional, Dict
import numpy as np
from PIL import Image
import torch

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
            print("Warning: Tesseract OCR not found. OCR will be disabled.")
            print("  Set TESSERACT_CMD environment variable or install Tesseract.")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR will be disabled.")

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not available. CLIP will be disabled.")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: BLIP not available. VLM captioning will be disabled.")


class ImageProcessor:
    """Process images: OCR, VLM captioning, CLIP embeddings."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load CLIP and BLIP models."""
        # Load CLIP
        if CLIP_AVAILABLE:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.clip_model.eval()
                print("CLIP model loaded successfully")
            except Exception as e:
                print(f"Error loading CLIP: {e}")
                self.clip_model = None
        
        # Load BLIP for captioning
        if BLIP_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model.to(self.device)
                self.blip_model.eval()
                print("BLIP model loaded successfully")
            except Exception as e:
                print(f"Error loading BLIP: {e}")
                self.blip_model = None
    
    def extract_ocr_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            # Convert PIL image to format tesseract expects
            text = pytesseract.image_to_string(image, lang='eng')
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def generate_vlm_caption(self, image: Image.Image) -> str:
        """Generate caption using Vision Language Model (BLIP)."""
        if not self.blip_model or not BLIP_AVAILABLE:
            return ""
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"VLM captioning error: {e}")
            return ""
    
    def get_clip_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Get CLIP embedding for image."""
        if not self.clip_model or not CLIP_AVAILABLE:
            return None
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Try get_image_features first (simpler API)
                try:
                    image_features = self.clip_model.get_image_features(**inputs)
                    # If it returns an output object instead of tensor, extract the tensor
                    if hasattr(image_features, 'pooler_output') and image_features.pooler_output is not None:
                        image_features = image_features.pooler_output
                    elif hasattr(image_features, 'last_hidden_state'):
                        # Use CLS token (first token) for vision models
                        image_features = image_features.last_hidden_state[:, 0, :]
                except (AttributeError, TypeError):
                    # Fallback: use vision_model directly
                    vision_outputs = self.clip_model.vision_model(**inputs)
                    image_features = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None else vision_outputs.last_hidden_state[:, 0, :]
                    # Project to shared embedding space
                    image_features = self.clip_model.visual_projection(image_features)
                
                # Ensure it's a tensor
                if not isinstance(image_features, torch.Tensor):
                    raise ValueError(f"Unexpected type for image_features: {type(image_features)}")
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"CLIP embedding error: {e}")
            return None
    
    def get_clip_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for text (for text-to-image search)."""
        if not self.clip_model or not CLIP_AVAILABLE:
            return None
        
        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                # Try get_text_features first (simpler API)
                try:
                    text_features = self.clip_model.get_text_features(**inputs)
                    # If it returns an output object instead of tensor, extract the tensor
                    if hasattr(text_features, 'pooler_output') and text_features.pooler_output is not None:
                        text_features = text_features.pooler_output
                    elif hasattr(text_features, 'last_hidden_state'):
                        # Use EOS token (last non-padding token) for text models
                        # Find the last non-padding token index (assume pad_token_id is 0 if not found)
                        pad_token_id = getattr(self.clip_processor.tokenizer, 'pad_token_id', 0) if hasattr(self.clip_processor, 'tokenizer') else 0
                        seq_lengths = (inputs['input_ids'] != pad_token_id).sum(dim=1) - 1
                        text_features = text_features.last_hidden_state[torch.arange(text_features.last_hidden_state.size(0)), seq_lengths]
                except (AttributeError, TypeError):
                    # Fallback: use text_model directly
                    text_outputs = self.clip_model.text_model(**inputs)
                    pad_token_id = getattr(self.clip_processor.tokenizer, 'pad_token_id', 0) if hasattr(self.clip_processor, 'tokenizer') else 0
                    seq_lengths = (inputs['input_ids'] != pad_token_id).sum(dim=1) - 1
                    text_features = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None else text_outputs.last_hidden_state[torch.arange(text_outputs.last_hidden_state.size(0)), seq_lengths]
                    # Project to shared embedding space
                    text_features = self.clip_model.text_projection(text_features)
                
                # Ensure it's a tensor
                if not isinstance(text_features, torch.Tensor):
                    raise ValueError(f"Unexpected type for text_features: {type(text_features)}")
                
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embedding = text_features.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"CLIP text embedding error: {e}")
            return None
    
    def process_image(self, image: Image.Image, 
                     caption: Optional[str] = None,
                     surrounding_context: Optional[str] = None) -> Dict:
        """
        Process image: OCR, VLM caption, CLIP embedding.
        Returns dict with all extracted information.
        """
        result = {
            'ocr_text': self.extract_ocr_text(image),
            'vlm_caption': '',
            'clip_embedding': None,
            'original_caption': caption,
            'surrounding_context': surrounding_context
        }
        
        # Generate VLM caption if no original caption
        if not caption:
            result['vlm_caption'] = self.generate_vlm_caption(image)
        else:
            result['vlm_caption'] = caption
        
        # Get CLIP embedding
        result['clip_embedding'] = self.get_clip_embedding(image)
        
        return result

