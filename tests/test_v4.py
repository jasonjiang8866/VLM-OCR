from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
from src.models import ExtractedData

client = TestClient(app)

@patch("src.ocr_engine_v4.OCREngineV4.extract_data")
def test_v4_endpoint(mock_extract):
    # Mock the return value
    mock_extract.return_value = ExtractedData(
        document=None,
        raw_text="# Passport\n\n**Country:** France\n**Name:** Jane Doe",
        confidence_score=0.8
    )

    # Create a dummy file
    files = {'file': ('test.jpg', b'dummy content', 'image/jpeg')}
    
    response = client.post("/v4/parse", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["raw_text"] == "# Passport\n\n**Country:** France\n**Name:** Jane Doe"
    assert data["confidence_score"] == 0.8
