import base64
import os
import json
from typing import Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from .models import ExtractedData, Passport, DriversLicense, NationalID, DocumentType

# Define the state of our graph
class OCRState(TypedDict):
    image_path: str
    extracted_data: Optional[dict]
    final_result: Optional[ExtractedData]
    error: Optional[str]

class OCREngineV2:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. OCR functionality will fail.")
            self.llm = ChatOpenAI(model="gpt-4o", api_key="dummy", max_tokens=1000)
        else:
            self.llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key, max_tokens=1000)
        
        self.graph = self._build_graph()

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _analyze_document(self, state: OCRState) -> OCRState:
        try:
            image_path = state["image_path"]
            base64_image = self._encode_image(image_path)
            
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

            messages = [
                SystemMessage(content="You are an expert OCR and document parsing assistant. You output strictly valid JSON."),
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]

            # We use json_object mode by binding response_format
            response = self.llm.bind(response_format={"type": "json_object"}).invoke(messages)
            content = response.content
            data = json.loads(content)
            
            return {**state, "extracted_data": data}
        except Exception as e:
            return {**state, "error": str(e)}

    def _validate_output(self, state: OCRState) -> OCRState:
        if state.get("error"):
            return state
            
        data = state.get("extracted_data")
        if not data:
            return {**state, "error": "No data extracted"}
            
        try:
            # Validate against Pydantic model
            result = ExtractedData(**data)
            return {**state, "final_result": result}
        except Exception as e:
            return {**state, "error": f"Validation failed: {str(e)}"}

    def _build_graph(self):
        workflow = StateGraph(OCRState)
        
        workflow.add_node("analyze", self._analyze_document)
        workflow.add_node("validate", self._validate_output)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()

    def extract_data(self, image_path: str) -> ExtractedData:
        if not self.api_key:
             return ExtractedData(confidence_score=0.0, error="Missing OpenAI API Key")

        initial_state: OCRState = {
            "image_path": image_path,
            "extracted_data": None,
            "final_result": None,
            "error": None
        }
        
        result_state = self.graph.invoke(initial_state)
        
        if result_state.get("error"):
            return ExtractedData(confidence_score=0.0, error=result_state["error"])
            
        return result_state["final_result"]
