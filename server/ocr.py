import os
import tempfile
import io
import pdfplumber
from PIL import Image
import logging
from typing import List
import numpy as np
import easyocr

cache_dir = os.path.join(tempfile.gettempdir(), '.EasyOCR')
os.environ['EASYOCR_MODULE_PATH'] = cache_dir
os.environ['EASYOCR_CACHE_DIR'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)
os.chmod(cache_dir, 0o755)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}

# Lazy global reader
ocr_reader = None

def get_ocr_reader():
    """Initialize EasyOCR only when needed"""
    global ocr_reader
    if ocr_reader is None:
        logger.info("Initializing EasyOCR reader (this may take some time)...")
        try:
            # Try to force the directory by changing the current working directory temporarily
            original_cwd = os.getcwd()
            os.chdir(cache_dir)
            
            # Initialize EasyOCR with explicit paths
            ocr_reader = easyocr.Reader(
                ['en'], 
                gpu=False, 
                verbose=True,
                model_storage_directory=cache_dir,
                user_network_directory=cache_dir,
                download_enabled=True
            )
            
            # Restore original working directory
            os.chdir(original_cwd)
            logger.info("EasyOCR initialized successfully ✅")
        except Exception as e:
            logger.error(f"EasyOCR initialization failed ❌: {e}")
            # Try a more direct approach
            try:
                # Use a completely different directory approach
                alt_cache_dir = "/tmp/easyocr_cache"
                os.makedirs(alt_cache_dir, exist_ok=True)
                os.chmod(alt_cache_dir, 0o755)
                
                ocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,
                    model_storage_directory=alt_cache_dir,
                    user_network_directory=alt_cache_dir
                )
                logger.info("EasyOCR alternative initialization successful ✅")
            except Exception as alt_error:
                logger.error(f"All EasyOCR initialization attempts failed: {alt_error}")
                ocr_reader = None
    return ocr_reader

def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image using EasyOCR (handles both printed and handwritten)"""
    reader = get_ocr_reader()
    if reader is None:
        logger.error("OCR reader not available")
        return ""
    
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        image_np = np.array(image)
        
        logger.info("Running OCR on image...")
        results = reader.readtext(image_np, paragraph=True, detail=1)
        logger.info(f"OCR found {len(results)} text segments")
        
        # Filter out low-confidence results
        text_parts = []
        for result in results:
            # Handle different return formats from EasyOCR
            if len(result) == 3:
                bbox, text, confidence = result
            elif len(result) == 2:
                bbox, text = result
                confidence = 1.0  # Default confidence if not provided
            else:
                continue  # Skip unexpected formats
            
            if confidence > 0.3:  # Adjust this threshold as needed
                text_parts.append(text)
        
        return "\n".join(text_parts).strip()
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        return ""

def extract_text_from_pdf(file_bytes: bytes) -> List[str]:
    """Hybrid PDF text extraction: Uses pdfplumber for text and EasyOCR for image-based pages"""
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            logger.info(f"Opened PDF with {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                try:
                    logger.info(f"Extracting text from page {i+1}...")
                    page_text = page.extract_text() or ""
                    
                    # If text extraction seems insufficient, use OCR
                    if len(page_text.strip()) < 50:  # Threshold for handwritten content
                        reader = get_ocr_reader()
                        if reader is not None:
                            logger.info(f"Running OCR on page {i+1}...")
                            im = page.to_image(resolution=200).original.convert("RGB")
                            image_np = np.array(im)
                            results = reader.readtext(image_np, paragraph=True, detail=1)
                            
                            # Filter out low-confidence results
                            ocr_text_parts = []
                            for result in results:
                                # Handle different return formats from EasyOCR
                                if len(result) == 3:
                                    bbox, text, confidence = result
                                elif len(result) == 2:
                                    bbox, text = result
                                    confidence = 1.0  # Default confidence if not provided
                                else:
                                    continue  # Skip unexpected formats
                                
                                if confidence > 0.3:  # Adjust this threshold as needed
                                    ocr_text_parts.append(text)
                            
                            ocr_text = "\n".join(ocr_text_parts).strip()
                            text_parts.append(ocr_text if ocr_text else page_text)
                        else:
                            text_parts.append(page_text)
                    else:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.error(f"Page {i+1} extraction failed: {e}")
                    text_parts.append("")
        return text_parts
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return []

def guess_and_extract(filename: str, file_bytes: bytes) -> List[str]:
    """Extract text based on file extension with enhanced OCR support"""
    ext = ("." + filename.lower().rsplit(".", 1)[-1]) if "." in filename else ""
    
    try:
        if ext in PDF_EXTS:
            return extract_text_from_pdf(file_bytes)
        elif ext in IMAGE_EXTS:
            text = extract_text_from_image(file_bytes)
            return [text] if text else []
        else:
            # Text file processing
            for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                try:
                    return [file_bytes.decode(encoding).strip()]
                except UnicodeDecodeError:
                    continue
            return []
    except Exception as e:
        logger.error(f"Extraction failed for {filename}: {e}")
        return []