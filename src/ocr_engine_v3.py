import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import json
import re
import os
from .models import ExtractedData, Passport, DriversLicense, NationalID, DocumentType

class OCREngineV3:
    def __init__(self):
        self.model_path = "deepseek-ai/Janus-Pro-1B"
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Loading V3 Engine (Janus-Pro-1B) on {self.device}...")
        try:
            # We use a try-except block because downloading the model might fail or take too long
            # in some environments.
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.model = self.model.to(self.device).eval()
            print("V3 Engine loaded successfully.")
        except Exception as e:
            print(f"Failed to load V3 Engine: {e}")
            self.model = None

    def extract_data(self, image_path: str) -> ExtractedData:
        if not self.model:
            return ExtractedData(confidence_score=0.0, error="Model not loaded (DeepSeek Janus-Pro-1B)")

        try:
            pil_image = Image.open(image_path)
            
            # Prompt designed to get JSON output
            prompt = """
            You are an expert OCR assistant. Analyze the image and extract data into a valid JSON object.
            
            The JSON must have this structure:
            {
                "document": {
                    "document_type": "passport" | "drivers_license" | "national_id",
                    "country": "string",
                    "full_name": "string",
                    "date_of_birth": "YYYY-MM-DD",
                    "document_number": "string",
                    "expiry_date": "YYYY-MM-DD",
                    "issuing_authority": "string",
                    "nationality": "string",
                    "vehicle_classes": ["class1", "class2"],
                    "issue_date": "YYYY-MM-DD",
                    "address": "string"
                },
                "raw_text": "all text found",
                "confidence_score": 0.9
            }
            
            If a field is missing, use null. Output ONLY the JSON.
            """
            
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>\n" + prompt,
                    "images": [image_path],
                },
                {"role": "Assistant", "content": ""},
            ]
            
            prepare_inputs = self.processor(
                conversations=conversation, 
                images=[pil_image], 
                force_batchify=True
            ).to(self.device)
            
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
            
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Attempt to find JSON in the output
            json_match = re.search(r'\{.*\}', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    return ExtractedData(**data)
                except json.JSONDecodeError:
                     return ExtractedData(confidence_score=0.0, error="Failed to decode JSON from model output", raw_text=answer)
                except Exception as e:
                     return ExtractedData(confidence_score=0.0, error=f"Validation error: {str(e)}", raw_text=answer)
            else:
                return ExtractedData(confidence_score=0.0, error="No JSON found in output", raw_text=answer)

        except Exception as e:
            return ExtractedData(confidence_score=0.0, error=str(e))
