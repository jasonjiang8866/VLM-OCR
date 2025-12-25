import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import json
import re
import os
from .models import ExtractedData, Passport, DriversLicense, NationalID, DocumentType

class OCREngineV4:
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-OCR"
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Loading V4 Engine (DeepSeek-OCR) on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Note: flash_attention_2 is not supported on MPS/CPU usually, so we omit it here for compatibility.
            # If running on CUDA, you might want to add _attn_implementation='flash_attention_2'
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                use_safetensors=True
            )
            self.model = self.model.to(self.device).eval()
            
            # For MPS, we might need to cast to float32 or bfloat16 depending on support
            if self.device == "cuda":
                self.model = self.model.to(torch.bfloat16)
            
            print("V4 Engine loaded successfully.")
        except Exception as e:
            print(f"Failed to load V4 Engine: {e}")
            self.model = None

    def extract_data(self, image_path: str) -> ExtractedData:
        if not self.model:
            return ExtractedData(confidence_score=0.0, error="Model not loaded (DeepSeek-OCR)")

        try:
            # DeepSeek-OCR uses a specific prompt for markdown conversion
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
            
            # The model has a custom .infer() method. 
            # We need to check if it exists (it comes from trust_remote_code=True)
            if hasattr(self.model, 'infer'):
                # The infer method expects image_file path
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False, # We want the return value
                    test_compress=False
                )
                # result is usually the generated text
                raw_text = result
            else:
                # Fallback if .infer is not available or different API
                return ExtractedData(confidence_score=0.0, error="Model does not have .infer() method")

            # Since DeepSeek-OCR is primarily an OCR model (image -> text/markdown),
            # it might not output JSON directly even if asked.
            # For this V4, we will return the raw markdown text.
            # In a real app, we would pass this text to an LLM (like GPT-4o) to parse into JSON.
            
            # Attempt to parse if it looks like JSON (unlikely with the default prompt)
            # Or we can try to prompt it for JSON? 
            # The docs say "Convert the document to markdown".
            
            return ExtractedData(
                confidence_score=0.8, # Placeholder confidence
                raw_text=raw_text,
                document=None # We don't have structured data from pure OCR
            )

        except Exception as e:
            return ExtractedData(confidence_score=0.0, error=str(e))
