import base64
import os
import json
from typing import Optional
from openai import OpenAI
from .models import ExtractedData, Passport, DriversLicense, NationalID, DocumentType

class OCREngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. OCR functionality will fail.")
            self.client = OpenAI(api_key="dummy")
        else:
            self.client = OpenAI(api_key=self.api_key)

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_data(self, image_path: str) -> ExtractedData:
        if not self.api_key:
             return ExtractedData(confidence_score=0.0, error="Missing OpenAI API Key")

        base64_image = self.encode_image(image_path)
        
        # We define the schema we want the LLM to adhere to.
        # Note: In a production app, we might use the Pydantic model's .model_json_schema() 
        # but for simplicity in this MVP we'll describe it in the prompt and use json_object mode.
        
        prompt = """
        Analyze the provided image of an identity document. 
        Identify the type of document (Passport, Driver's License, or National ID).
        Extract the relevant fields.
        
        Return a JSON object with the following structure:
        {
            "document": {
                "document_type": "passport" | "drivers_license" | "national_id",
                "country": "string",
                "full_name": "string",
                "date_of_birth": "YYYY-MM-DD",
                "document_number": "string",
                ... (other fields specific to the document type)
            },
            "raw_text": "string (all text found on document)",
            "confidence_score": float (0.0 to 1.0)
        }
        
        Specific fields per type:
        - Passport: expiry_date, issuing_authority, nationality
        - Driver's License: vehicle_classes (list of strings), expiry_date, issue_date
        - National ID: address, expiry_date
        
        If a field is not visible or applicable, use null.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert OCR and document parsing assistant. You output strictly valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Basic validation/conversion
            return ExtractedData(**data)

        except Exception as e:
            return ExtractedData(confidence_score=0.0, error=str(e))
