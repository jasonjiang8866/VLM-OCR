from pydantic import BaseModel, Field
from typing import Optional, List, Union
from enum import Enum

class DocumentType(str, Enum):
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UNKNOWN = "unknown"

class BaseDocument(BaseModel):
    document_type: DocumentType
    country: str = Field(..., description="Country of issue")
    full_name: str = Field(..., description="Full name of the document holder")
    date_of_birth: Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD format")
    document_number: str = Field(..., description="Unique document identifier")

class Passport(BaseDocument):
    document_type: DocumentType = DocumentType.PASSPORT
    expiry_date: Optional[str] = Field(None, description="Expiry date in YYYY-MM-DD format")
    issuing_authority: Optional[str] = Field(None, description="Authority that issued the passport")
    nationality: Optional[str] = Field(None, description="Nationality of the holder")

class DriversLicense(BaseDocument):
    document_type: DocumentType = DocumentType.DRIVERS_LICENSE
    vehicle_classes: List[str] = Field(default_factory=list, description="Classes of vehicles allowed")
    expiry_date: Optional[str] = Field(None, description="Expiry date in YYYY-MM-DD format")
    issue_date: Optional[str] = Field(None, description="Issue date in YYYY-MM-DD format")

class NationalID(BaseDocument):
    document_type: DocumentType = DocumentType.NATIONAL_ID
    address: Optional[str] = Field(None, description="Address listed on the ID")
    expiry_date: Optional[str] = Field(None, description="Expiry date in YYYY-MM-DD format")

DocumentUnion = Union[Passport, DriversLicense, NationalID]

class ExtractedData(BaseModel):
    document: Optional[DocumentUnion] = Field(None, description="The extracted document data")
    raw_text: Optional[str] = Field(None, description="Raw text extracted from the document")
    confidence_score: float = Field(..., description="Confidence score of the extraction (0-1)")
    error: Optional[str] = Field(None, description="Error message if extraction failed")
