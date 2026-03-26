"""Document upload and processing routes"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.models.schemas import DocumentUploadResponse
from app.utils.file_handler import process_uploaded_file
import uuid


class PasteContentRequest(BaseModel):
    """Request body for paste content endpoint"""
    content: str


router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        document_id = str(uuid.uuid4())
        content = await process_uploaded_file(file)
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            content=content,
            upload_status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/paste", response_model=DocumentUploadResponse)
async def paste_content(request: PasteContentRequest):
    """Process pasted text content"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Content field required and cannot be empty")
    
    document_id = str(uuid.uuid4())
    return DocumentUploadResponse(
        document_id=document_id,
        filename="pasted_text.txt",
        content=request.content,
        upload_status="success"
    )
