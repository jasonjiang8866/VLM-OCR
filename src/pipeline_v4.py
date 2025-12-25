from .ocr_engine_v4 import OCREngineV4
from .models import ExtractedData

class DocumentPipelineV4:
    def __init__(self):
        self.engine = OCREngineV4()

    def process_document(self, file_path: str) -> ExtractedData:
        return self.engine.extract_data(file_path)
