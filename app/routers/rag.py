from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import shutil
import os
from app.rag_engine import ingest_pdf, ingest_image, ingest_instagram, query_rag, optimize_prompt

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None

class URLRequest(BaseModel):
    url: str

class OptimizeRequest(BaseModel):
    question: str

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only PDFs and Images (PNG/JPG) are supported.")
    
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if filename.endswith(".pdf"):
        result = ingest_pdf(file_path)
    else:
        result = ingest_image(file_path)
        
    os.remove(file_path)
    return {"message": result}

@router.post("/ingest_url")
async def ingest_url(request: URLRequest):
    if "instagram.com" in request.url:
        result = ingest_instagram(request.url)
    else:
        return {"message": "Error: Only Instagram URLs are supported currently."}
    return {"message": result}

@router.post("/chat")
async def chat(request: QueryRequest):
    answer = query_rag(request.question, request.model, request.api_key, request.api_url)
    return {"answer": answer}

@router.post("/optimize_prompt")
async def optimize(request: OptimizeRequest):
    result = optimize_prompt(request.question)
    return result