from typing import List, Tuple, Dict, Any
from pipelines import add_documents
import ocr
import logging

logger = logging.getLogger(__name__)

def ingest_files(session_id: str, files: List[Tuple[str, bytes]]) -> int:
    """
    Process files and ingest per-page content
    files: list of tuples (filename, file_bytes)
    """
    all_texts = []
    all_metas = []
    
    for filename, file_bytes in files:
        if not file_bytes:
            logger.warning(f"Empty file: {filename}")
            continue
            
        try:
            # Extract text (returns list of pages)
            pages = ocr.guess_and_extract(filename, file_bytes)
            
            for page_num, page_text in enumerate(pages, 1):
                if not page_text or not page_text.strip():
                    continue
                    
                all_texts.append(page_text)
                all_metas.append({
                    "session_id": session_id,
                    "filename": filename,
                    "page": page_num
                })
        except Exception as e:
            logger.error(f"Processing failed for {filename}: {e}")
    
    if not all_texts:
        logger.warning("No valid text extracted from files")
        return 0
        
    try:
        return add_documents(all_texts, all_metas)
    except Exception as e:
        logger.exception(f"Document addition failed: {e}")
        return 0