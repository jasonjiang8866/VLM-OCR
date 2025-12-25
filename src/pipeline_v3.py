from .ocr_engine_v3 import OCREngineV3
from .models import ExtractedData

class DocumentPipelineV3:
    def __init__(self):
        self.engine = OCREngineV3()

    def process_document(self, file_path: str) -> ExtractedData:
        return self.engine.extract_data(file_path)
