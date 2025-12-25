from fastapi import FastAPI, UploadFile, File, HTTPException
from .pipeline import DocumentPipeline
from .pipeline_v2 import DocumentPipelineV2
from .pipeline_v3 import DocumentPipelineV3
from .pipeline_v4 import DocumentPipelineV4
from .models import ExtractedData
import shutil
import os
import uuid

app = FastAPI(title="OCR Agent API")
pipeline = DocumentPipeline()
pipeline_v2 = DocumentPipelineV2()
# Initialize V3 pipeline (DeepSeek Janus)
# Note: This loads the model into memory, which might take time/resources.
pipeline_v3 = DocumentPipelineV3()
# Initialize V4 pipeline (DeepSeek OCR)
pipeline_v4 = DocumentPipelineV4()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/parse", response_model=ExtractedData)
async def parse_document(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = ".jpg" # Default to jpg if no extension
            
        file_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process
        result = pipeline.process_document(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/parse", response_model=ExtractedData)
async def parse_document_v2(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = ".jpg" # Default to jpg if no extension
            
        file_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process with V2 pipeline
        result = pipeline_v2.process_document(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v3/parse", response_model=ExtractedData)
async def parse_document_v3(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = ".jpg" # Default to jpg if no extension
            
        file_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process with V3 pipeline (DeepSeek Janus)
        result = pipeline_v3.process_document(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v4/parse", response_model=ExtractedData)
async def parse_document_v4(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            file_extension = ".jpg" # Default to jpg if no extension
            
        file_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process with V4 pipeline (DeepSeek OCR)
        result = pipeline_v4.process_document(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "OCR Agent API is running"}
