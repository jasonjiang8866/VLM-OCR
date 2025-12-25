from .ocr_engine_v2 import OCREngineV2
from .models import ExtractedData

class DocumentPipelineV2:
    def __init__(self):
        self.engine = OCREngineV2()

    def process_document(self, file_path: str) -> ExtractedData:
        return self.engine.extract_data(file_path)
