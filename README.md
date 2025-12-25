# OCR Agent

Scalable OCR and document-parsing pipeline for passports, driver’s licenses, and national IDs using LLM/VLM.

## Features

- Supports Passports, Driver's Licenses, and National IDs.
- Extracts structured data (JSON) using OpenAI GPT-4o.
- FastAPI backend for easy integration.

## Setup

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your `OPENAI_API_KEY`

## Running the API

```bash
uvicorn src.main:app --reload
```

## Usage

### V1 Endpoint (Standard)
Send a POST request to `/parse` with an image file.

```bash
curl -X POST -F "file=@/path/to/document.jpg" http://127.0.0.1:8000/parse
```

### V2 Endpoint (LangGraph)
Send a POST request to `/v2/parse` with an image file. This uses the LangGraph-based engine.

```bash
curl -X POST -F "file=@/path/to/document.jpg" http://127.0.0.1:8000/v2/parse
```

### V3 Endpoint (DeepSeek Janus-Pro)
Send a POST request to `/v3/parse` with an image file. This uses the local DeepSeek Janus-Pro-1B model.
**Note**: This requires significant RAM and may be slow on CPU.

```bash
curl -X POST -F "file=@/path/to/document.jpg" http://127.0.0.1:8000/v3/parse
```

### V4 Endpoint (DeepSeek OCR)
Send a POST request to `/v4/parse` with an image file. This uses the local DeepSeek-OCR model.
**Note**: This model is specialized for markdown output and may not return structured JSON.

```bash
curl -X POST -F "file=@/path/to/document.jpg" http://127.0.0.1:8000/v4/parse
```
