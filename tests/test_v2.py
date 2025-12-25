from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
from src.models import ExtractedData, Passport, DocumentType

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OCR Agent API is running"}

@patch("src.ocr_engine_v2.OCREngineV2.extract_data")
def test_v2_endpoint(mock_extract):
    # Mock the return value
    mock_extract.return_value = ExtractedData(
        document=Passport(
            document_type=DocumentType.PASSPORT,
            country="USA",
            full_name="John Doe",
            document_number="123456789",
            expiry_date="2030-01-01",
            issuing_authority="US Dept of State",
            nationality="USA"
        ),
        raw_text="Passport USA John Doe",
        confidence_score=0.99
    )

    # Create a dummy file
    files = {'file': ('test.jpg', b'dummy content', 'image/jpeg')}
    
    response = client.post("/v2/parse", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_type"] == "passport"
    assert data["document"]["full_name"] == "John Doe"
    assert data["confidence_score"] == 0.99
