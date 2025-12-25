from .ocr_engine import OCREngine
from .models import ExtractedData

class DocumentPipeline:
    def __init__(self):
        self.engine = OCREngine()

    def process_document(self, file_path: str) -> ExtractedData:
        # Future: Add image preprocessing (deskewing, resizing) here
        return self.engine.extract_data(file_path)
