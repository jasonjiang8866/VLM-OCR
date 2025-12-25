from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
from src.models import ExtractedData, Passport, DocumentType

client = TestClient(app)

@patch("src.ocr_engine_v3.OCREngineV3.extract_data")
def test_v3_endpoint(mock_extract):
    # Mock the return value
    mock_extract.return_value = ExtractedData(
        document=Passport(
            document_type=DocumentType.PASSPORT,
            country="France",
            full_name="Jane Doe",
            document_number="987654321",
            expiry_date="2032-01-01",
            issuing_authority="Prefecture",
            nationality="France"
        ),
        raw_text="Passport France Jane Doe",
        confidence_score=0.85
    )

    # Create a dummy file
    files = {'file': ('test.jpg', b'dummy content', 'image/jpeg')}
    
    response = client.post("/v3/parse", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_type"] == "passport"
    assert data["document"]["country"] == "France"
    assert data["confidence_score"] == 0.85
