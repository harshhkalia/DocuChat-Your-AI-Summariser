from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pipelines import query_rag, document_store
import ingestion
import logging
import uuid
import os
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

sccsca

app = FastAPI(
    title="Haystack RAG API",
    description="PDF Summarization and Question Answering System",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_files(
    session_id: str = Form(default=""),  
    files: List[UploadFile] = File(...)
):
    """Upload and process files with automatic session management"""
    if not session_id.strip():
        session_id = str(uuid.uuid4())
    
    if not files:
        raise HTTPException(400, detail="No files uploaded")
    
    pairs = []
    for f in files:
        try:
            content = await f.read()
            pairs.append((f.filename or "unnamed", content))
        except Exception as e:
            logger.error(f"Failed to process {f.filename}: {str(e)}")
            continue
    
    added = ingestion.ingest_files(session_id, pairs)
    return {
        "status": "success",
        "session_id": session_id,
        "documents_added": added,
        "message": f"Use this session_id for queries: {session_id}"
    }

@app.post("/query")
async def query(
    session_id: str = Form(...),
    question: str = Form(...)
):
    """Query the RAG system"""
    if not session_id.strip():
        raise HTTPException(400, detail="Session ID cannot be empty")
    
    result = query_rag(question, session_id)
    return result

@app.get("/clear")
async def clear_session(session_id: str):
    """Clear documents for a session"""
    docs_to_delete = [doc.id for doc in document_store.filter_documents({"session_id": session_id})]
    document_store.delete_documents(docs_to_delete)
    return {"status": "success", "deleted": len(docs_to_delete)}

@app.get("/healthz")
async def healthz():
    """Health check endpoint"""
    return {"status": "ok", "version": app.version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


