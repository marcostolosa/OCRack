"PDF text extraction with OCR support for scanned documents."

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import pdfplumber
import fitz  # PyMuPDF
try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .utils import console, log_error, log_warning, log_info


def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text for better readability."""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    import re
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text


def _should_apply_ocr(text: str, min_text_threshold: int = 50) -> bool:
    """
    Determine if OCR should be applied based on extracted text quality.
    
    Args:
        text: Extracted text from pdfplumber
        min_text_threshold: Minimum characters to consider text extraction successful
        
    Returns:
        True if OCR should be applied, False otherwise
    """
    if not text or len(text.strip()) < min_text_threshold:
        return True
    
    # Check for signs of poor text extraction (lots of special chars, garbled text)
    import re
    
    # Count letters vs total characters
    letters = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(text.strip())
    
    if total_chars == 0:
        return True
        
    letter_ratio = letters / total_chars
    
    # If less than 50% letters, probably needs OCR
    if letter_ratio < 0.5:
        return True
        
    # Check for excessive special characters that might indicate scanning artifacts
    special_chars = len(re.findall(r'[^\w\s\.\,\!\?\:\;\-\(\)]', text))
    special_ratio = special_chars / total_chars
    
    # If more than 20% special characters, might need OCR
    if special_ratio > 0.2:
        return True
        
    return False


# PaddleOCR instance (initialized once for performance)
_paddle_ocr = None


def _get_paddle_ocr(lang='en'):
    """Get PaddleOCR instance, initializing if needed."""
    global _paddle_ocr
    if _paddle_ocr is None and PADDLEOCR_AVAILABLE:
        try:
            # Updated for PaddleOCR v3.2.0 - use_textline_orientation instead of use_angle_cls
            # 'en' for English, 'ch' supports Chinese and English
            _paddle_ocr = paddleocr.PaddleOCR(use_textline_orientation=True, lang=lang)
            log_info(f"PaddleOCR v3.2.0 initialized with language: {lang}")
        except Exception as e:
            log_error(f"Failed to initialize PaddleOCR: {e}")
            return None
    return _paddle_ocr


def _extract_text_with_paddleocr_enhanced(image_path: str) -> str:
    """Enhanced PaddleOCR v3.2.0 with multiple language support and better settings."""
    if not PADDLEOCR_AVAILABLE:
        return ""
    
    try:
        # Try with multilingual support (Chinese+English covers more characters)
        ocr = _get_paddle_ocr('ch')  # 'ch' supports both Chinese and English
        if not ocr:
            return ""
        
        # PaddleOCR v3.2.0 prefers numpy arrays
        import cv2
        import numpy as np
        
        # Load image as numpy array for PaddleOCR v3.2.0
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                # Fallback to PIL if cv2 fails
                from PIL import Image
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = image_path
            
        result = ocr.predict(image)
        
        if not result or not isinstance(result, list) or len(result) == 0:
            return ""
        
        page_result = result[0]
        
        if 'rec_texts' not in page_result or 'rec_scores' not in page_result:
            return ""
        
        texts = page_result['rec_texts']
        scores = page_result['rec_scores']
        
        # More lenient confidence threshold for enhanced mode
        text_lines = []
        for text, confidence in zip(texts, scores):
            if confidence > 0.3:  # Lower threshold for enhanced mode
                text_lines.append(text.strip())
        
        extracted_text = '\n'.join(text_lines)
        log_info(f"PaddleOCR enhanced extracted {len(text_lines)} text lines with >30% confidence")
        return extracted_text
        
    except Exception as e:
        log_error(f"PaddleOCR enhanced extraction failed: {e}")
        return ""


def _check_ocrmypdf_installation() -> bool:
    """Check if OCRmyPDF is installed and available."""
    try:
        import subprocess
        result = subprocess.run(['ocrmypdf', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def _extract_text_with_ocrmypdf(image_path: str) -> str:
    """Extract text using OCRmyPDF as fallback."""
    try:
        import subprocess
        import tempfile
        
        # Create temp PDF from image
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            # Convert image to PDF first
            from PIL import Image
            img = Image.open(image_path)
            img.save(temp_pdf.name, "PDF")
            
            # Run OCRmyPDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as ocr_pdf:
                result = subprocess.run([
                    'ocrmypdf', '--force-ocr', '--language', 'eng',
                    temp_pdf.name, ocr_pdf.name
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Extract text from OCR'd PDF
                    import pdfplumber
                    with pdfplumber.open(ocr_pdf.name) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        return text.strip()
                else:
                    log_error(f"OCRmyPDF failed: {result.stderr}")
                    return ""
                    
    except Exception as e:
        log_error(f"OCRmyPDF extraction failed: {e}")
        return ""


def _extract_text_with_paddleocr(image_path: str) -> str:
    """Extract text using PaddleOCR - MUCH BETTER than Tesseract!"""
    if not PADDLEOCR_AVAILABLE:
        return ""
    
    try:
        ocr = _get_paddle_ocr('en')  # English mode for best performance
        if not ocr:
            return ""
        
        # PaddleOCR v3.2.0 prefers numpy arrays
        import cv2
        import numpy as np
        
        # Load image as numpy array for PaddleOCR v3.2.0
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                # Fallback to PIL if cv2 fails
                from PIL import Image
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = image_path
            
        # Use predict() method for PaddleOCR v3.2.0
        result = ocr.predict(image)
        
        if not result or not isinstance(result, list) or len(result) == 0:
            return ""
        
        # Extract text from the new API result structure
        page_result = result[0]  # Get first (and usually only) page result
        
        if 'rec_texts' not in page_result or 'rec_scores' not in page_result:
            log_warning("PaddleOCR result missing text fields")
            return ""
        
        texts = page_result['rec_texts']
        scores = page_result['rec_scores']
        
        # Extract text with confidence filtering
        text_lines = []
        high_confidence_count = 0
        
        for text, confidence in zip(texts, scores):
            # PaddleOCR v3.2.0 is very accurate - use 70% confidence threshold
            if confidence > 0.7:
                text_lines.append(text.strip())
                high_confidence_count += 1
            elif confidence > 0.5:  # Include medium confidence text 
                text_lines.append(text.strip())
        
        extracted_text = '\n'.join(text_lines)
        log_info(f"PaddleOCR v3.2.0 extracted {len(text_lines)} text lines ({high_confidence_count} high confidence)")
        return extracted_text
        
    except Exception as e:
        log_warning(f"PaddleOCR failed: {e}")
        return ""


def _check_tesseract_installation() -> bool:
    """Check if Tesseract is properly installed and accessible."""
    if not TESSERACT_AVAILABLE:
        return False
    
    try:
        # Try to get Tesseract version
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        try:
            # Check common Windows installation paths
            windows_paths = [
                r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
                r"C:\\Users\\{}\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe".format(os.getenv('USERNAME', ''))
            ]
            
            for path in windows_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        pytesseract.get_tesseract_version()
                        log_info(f"Found Tesseract at: {path}")
                        return True
                    except Exception:
                        continue
            
            return False
        except Exception:
            return False


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR accuracy."""
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive threshold to handle varying lighting
        threshold = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        return Image.fromarray(threshold)
        
    except Exception as e:
        log_warning(f"Image preprocessing failed, using original: {e}")
        return image


def _extract_text_from_image_ocr(image: Image.Image, lang: str = 'eng') -> str:
    """Extract text from image using OCR."""
    if not _check_tesseract_installation():
        return ""
    
    try:
        # Preprocess image for better OCR
        processed_image = _preprocess_image_for_ocr(image)
        
        # OCR configuration for better accuracy
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '
        
        # Extract text
        text = pytesseract.image_to_string(processed_image, lang=lang, config=config)
        
        return text.strip()
        
    except Exception as e:
        log_warning(f"OCR failed for image: {e}")
        return ""

def _extract_page_with_ocr(pdf_path: Path, page_num: int, lang: str = 'eng') -> str:
    """Extract text from a specific PDF page using OCR - PaddleOCR first, Tesseract as fallback."""
    try:
        # Open PDF with PyMuPDF for image extraction
        pdf_doc = fitz.open(str(pdf_path))
        
        if page_num - 1 >= len(pdf_doc):
            return ""
        
        page = pdf_doc[page_num - 1]  # Convert to 0-based indexing
        
        # Render page as image at high resolution for better OCR
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        
        # Save as temporary image file for PaddleOCR (Windows-friendly approach)
        temp_image_path = None
        try:
            temp_dir = Path(tempfile.gettempdir()) / "pdf_translate_ocr"
            temp_dir.mkdir(exist_ok=True)
            temp_image_path = temp_dir / f"page_{page_num}_{os.getpid()}.png"
            pix.save(str(temp_image_path))
            
            # Use PaddleOCR v3.2.0 - THE BEST OCR ENGINE!
            if PADDLEOCR_AVAILABLE:
                log_info(f"Using PaddleOCR v3.2.0 for page {page_num} (superior accuracy)")
                text = _extract_text_with_paddleocr(temp_image_path)
                if text and len(text.strip()) > 5:  # Even minimal text is acceptable
                    return text
                else:
                    # PaddleOCR should work for all cases - retry with different settings
                    log_info(f"Retrying PaddleOCR with enhanced settings for page {page_num}")
                    text = _extract_text_with_paddleocr_enhanced(temp_image_path)
                    if text and len(text.strip()) > 0:
                        return text
                    else:
                        log_warning(f"PaddleOCR could not extract text from page {page_num}, trying OCRmyPDF fallback")
            
            # Fallback to OCRmyPDF if PaddleOCR fails completely
            if _check_ocrmypdf_installation():
                log_info(f"Using OCRmyPDF fallback for page {page_num}")
                text = _extract_text_with_ocrmypdf(temp_image_path)
                return text
            else:
                log_error("PaddleOCR failed and OCRmyPDF not available")
                return ""
                
        finally:
            # Clean up temp file with more robust error handling
            if temp_image_path and temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except PermissionError:
                    # If we can't delete immediately, try again after a short delay
                    import time
                    time.sleep(0.2)
                    try:
                        temp_image_path.unlink()
                    except (PermissionError, FileNotFoundError):
                        # If still failing, log but don't crash - temp files will be cleaned by OS
                        log_warning(f"Could not delete temporary file {temp_image_path} - will be cleaned by OS")
                except FileNotFoundError:
                    # File already deleted, no problem
                    pass
                
        pdf_doc.close()
        
    except Exception as e:
        log_error(f"OCR extraction failed for page {page_num}: {e}")
        return ""

def extract_text_by_page(pdf_path: Path, layout_aware: bool = True, use_ocr: bool = True, ocr_lang: str = 'eng', force_ocr: bool = False) -> Dict[int, str]:
    """
    Extract text from PDF using pdfplumber first, then OCR for scanned/image-based PDFs.
    
    Args:
        pdf_path: Path to PDF file
        layout_aware: Whether to preserve layout (columns, tables)
        use_ocr: Whether to use OCR for scanned/image-based PDFs
        ocr_lang: Language for OCR (default: 'eng', use 'eng+por' for mixed content)
    
    Returns:
        Dictionary mapping page numbers (1-based) to extracted text
    
    Raises:
        Exception: If PDF cannot be read or is password protected
    """
    page_texts = {}
    
    try:
        # First attempt: Standard text extraction with pdfplumber
        log_info("Attempting standard PDF text extraction...")
        
        with pdfplumber.open(pdf_path) as pdf:
            if hasattr(pdf, 'is_encrypted') and pdf.is_encrypted:
                raise Exception("PDF is password protected")
            
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise Exception("PDF contains no pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    if layout_aware:
                        # Try to preserve layout structure
                        text = page.extract_text(
                            layout=True,
                            x_tolerance=3,
                            y_tolerance=3
                        )
                        
                        # If layout-aware extraction returns empty, fallback to basic
                        if not text or text.strip() == "":
                            text = page.extract_text()
                    else:
                        text = page.extract_text()
                    
                    # Store even empty pages (user may want to know structure)
                    page_texts[page_num] = text if text else ""
                    
                except Exception as e:
                    log_warning(f"Failed to extract text from page {page_num}: {e}")
                    page_texts[page_num] = ""
            
    except Exception as e:
        log_error(f"Failed to open PDF: {e}")
        raise
    
    # Check if we got any meaningful text
    total_text = sum(len(text.strip()) for text in page_texts.values())
    text_density = total_text / len(page_texts) if page_texts else 0
    
    # Check if we should use OCR - either forced, very little text, or likely poor quality extraction
    should_use_ocr = force_ocr
    if use_ocr and not force_ocr:
        # Check for very low density
        if text_density < 100:  # Less than 100 chars per page average
            should_use_ocr = True
            reason = f"Low text density ({text_density:.1f} chars/page)"
        else:
            # Check for poor quality extraction (lots of single chars, broken words)
            sample_text = ' '.join(list(page_texts.values())[:5])  # First 5 pages
            if len(sample_text) > 100:
                # Count single character "words" and very short words
                words = sample_text.split()
                single_chars = sum(1 for word in words if len(word) <= 1)
                short_words = sum(1 for word in words if len(word) <= 2)
                total_words = len(words)
                
                if total_words > 0:
                    single_char_ratio = single_chars / total_words
                    short_word_ratio = short_words / total_words
                    
                    # If more than 30% are single chars or 60% are very short words, likely OCR candidate
                    if single_char_ratio > 0.3 or short_word_ratio > 0.6:
                        should_use_ocr = True
                        reason = f"Poor quality text extraction (single chars: {single_char_ratio:.1%}, short words: {short_word_ratio:.1%})"
    
    if should_use_ocr:
        if force_ocr:
            log_info("Force OCR enabled. Attempting OCR extraction on all pages...")
        else:
            log_info(f"{reason}. Attempting OCR extraction...")
        
        if not PADDLEOCR_AVAILABLE and not _check_tesseract_installation():
            log_error("OCR required but neither PaddleOCR nor Tesseract available.")
            log_error("Install PaddleOCR (RECOMMENDED): pip install paddlepaddle paddleocr")
            log_error("Or install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            raise Exception("PDF appears to be scanned or image-based. No OCR engine available.")
        
        # Try OCR on pages with no or very little text
        ocr_pages_processed = 0
        
        for page_num in page_texts.keys():
            current_text = page_texts[page_num]
            
            # Decide if this page needs OCR
            should_ocr_page = force_ocr or len(current_text.strip()) < 50
            
            if should_ocr_page:
                log_info(f"Applying OCR to page {page_num}...")
                ocr_text = _extract_page_with_ocr(pdf_path, page_num, ocr_lang)
                
                # Use OCR text if it's significantly better or if forced
                if ocr_text and (force_ocr or len(ocr_text.strip()) > len(current_text.strip())):
                    page_texts[page_num] = ocr_text
                    ocr_pages_processed += 1
        
        if ocr_pages_processed > 0:
            log_info(f"Successfully applied OCR to {ocr_pages_processed} pages")
        else:
            log_warning("OCR processing did not improve text extraction")
    
    # Final check
    final_text = sum(len(text.strip()) for text in page_texts.values())
    if final_text == 0:
        error_msg = "No text extracted from PDF after trying both standard extraction and OCR"
        if not use_ocr:
            error_msg += ". Try enabling OCR with --use-ocr flag"
        elif not _check_tesseract_installation():
            error_msg += ". Tesseract OCR is not properly installed"
        raise Exception(error_msg)
    
    log_info(f"Successfully extracted {final_text} characters from {len(page_texts)} pages")
    return page_texts

def get_pdf_metadata(pdf_path: Path) -> Dict[str, any]:
    """
    Extract basic metadata from PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with title, author, subject, etc.
    """
    metadata = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            info = pdf.metadata
            if info:
                metadata = {
                    'title': info.get('Title', ''),
                    'author': info.get('Author', ''),
                    'subject': info.get('Subject', ''),
                    'creator': info.get('Creator', ''),
                    'producer': info.get('Producer', ''),
                    'creation_date': info.get('CreationDate', ''),
                    'modification_date': info.get('ModDate', ''),
                    'pages': len(pdf.pages)
                }
    except Exception as e:
        log_warning(f"Could not extract PDF metadata: {e}")
    
    return metadata


def extract_pages_range(pdf_path: Path, page_numbers: List[int], layout_aware: bool = True, use_ocr: bool = True, ocr_lang: str = 'eng', force_ocr: bool = False) -> Dict[int, str]:
    """
    Extract text from specific pages with OCR support.
    
    Args:
        pdf_path: Path to PDF file
        page_numbers: List of page numbers (1-based) to extract
        layout_aware: Whether to preserve layout
        use_ocr: Whether to use OCR for scanned/image-based PDFs
        ocr_lang: Language for OCR
        force_ocr: Force OCR even if text extraction seems to work
    
    Returns:
        Dictionary mapping page numbers to extracted text
    """
    if not page_numbers:
        return {}
    
    log_info(f"Extracting text from specific pages: {page_numbers}")
    page_texts = {}
    
    # Extract text from each specific page
    for page_num in sorted(page_numbers):
        try:
            log_info(f"Processing page {page_num}...")
            
            # Try standard text extraction first (unless force_ocr is True)
            text = ""
            if not force_ocr:
                with pdfplumber.open(pdf_path) as pdf:
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]  # Convert to 0-based index
                        text = page.extract_text() or ""
                        if layout_aware:
                            text = _clean_extracted_text(text)
            
            # Check if OCR is needed
            should_use_ocr = force_ocr or (use_ocr and _should_apply_ocr(text))
            
            if should_use_ocr:
                log_info(f"Applying OCR to page {page_num}...")
                ocr_text = _extract_page_with_ocr(pdf_path, page_num, ocr_lang)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    log_info(f"OCR improved text extraction for page {page_num}")
                elif ocr_text:
                    log_info(f"Standard extraction was better for page {page_num}")
                else:
                    log_warning(f"OCR failed for page {page_num}, using standard extraction")
            
            page_texts[page_num] = text
            
        except Exception as e:
            log_error(f"Failed to extract page {page_num}: {e}")
            page_texts[page_num] = ""
    
    log_info(f"Successfully extracted text from {len(page_texts)} pages")
    return page_texts


def validate_pdf_extractable(pdf_path: Path, check_ocr: bool = True) -> Tuple[bool, str]:
    """
    Check if PDF text can be extracted and determine if OCR is needed.
    
    Args:
        pdf_path: Path to PDF file
        check_ocr: Whether to check OCR availability
        
    Returns:
        Tuple of (can_extract, method) where method is 'text', 'ocr', or 'none'
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if hasattr(pdf, 'is_encrypted') and pdf.is_encrypted:
                return False, 'none'
            
            # Check first few pages for extractable text
            pages_to_check = min(3, len(pdf.pages))
            total_text = 0
            
            for i in range(pages_to_check):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    total_text += len(text.strip())
            
            # If we found meaningful text, standard extraction works
            if total_text > 50 * pages_to_check:  # 50+ chars per page
                return True, 'text'
            
            # Little or no text found - check if OCR is available
            if check_ocr and _check_tesseract_installation():
                return True, 'ocr'
            
            # No text and no OCR available
            return False, 'none'
            
    except Exception:
        return False, 'none'


def get_page_dimensions(pdf_path: Path) -> Dict[int, tuple]:
    """
    Get dimensions for each page in PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary mapping page numbers to (width, height) tuples
    """
    dimensions = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                dimensions[page_num] = (page.width, page.height)
    except Exception as e:
        log_warning(f"Could not extract page dimensions: {e}")
    
    return dimensions
