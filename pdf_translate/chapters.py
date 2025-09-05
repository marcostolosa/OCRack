"""Chapter detection using PDF outlines and heuristic regex patterns."""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import PyPDF2
from .utils import console, log_info, log_warning
from .extract import extract_text_by_page


def get_outlines(pdf_path: Path) -> List[Tuple[str, int, int]]:
    """
    Extract chapter outlines/bookmarks from PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of tuples: (title, start_page, end_page)
        Pages are 1-based indices
    """
    chapters = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            if not reader.outline:
                log_info("No PDF bookmarks/outlines found")
                return chapters
            
            # Parse the outline structure
            outline_items = []
            
            def extract_outline_items(outline, level=0):
                """Recursively extract outline items with their page numbers."""
                for item in outline:
                    if isinstance(item, list):
                        extract_outline_items(item, level + 1)
                    else:
                        try:
                            # Get the destination page number
                            if hasattr(item, 'page') and item.page:
                                page_num = reader.pages.index(item.page) + 1
                                outline_items.append((item.title, page_num, level))
                        except (AttributeError, ValueError):
                            try:
                                # Alternative way to get page number
                                dest = item.get('/A', item)
                                if dest and hasattr(dest, 'get'):
                                    page_ref = dest.get('/D', [None])[0]
                                    if page_ref:
                                        page_num = reader.pages.index(page_ref) + 1
                                        outline_items.append((item.title, page_num, level))
                            except:
                                continue
            
            extract_outline_items(reader.outline)
            
            # Convert outline items to chapters with end pages
            for i, (title, start_page, level) in enumerate(outline_items):
                # Find end page (start of next chapter at same or higher level)
                end_page = len(reader.pages)  # Default to last page
                
                for j in range(i + 1, len(outline_items)):
                    next_title, next_start, next_level = outline_items[j]
                    if next_level <= level:  # Same level or higher (parent)
                        end_page = next_start - 1
                        break
                
                chapters.append((title.strip(), start_page, end_page))
            
            log_info(f"Found {len(chapters)} chapters from PDF bookmarks")
            
    except Exception as e:
        log_warning(f"Could not read PDF outlines: {e}")
    
    return chapters


def detect_chapters_by_regex(page_texts: Dict[int, str]) -> List[Tuple[str, int, int]]:
    """
    Detect chapters using regex patterns on page text.
    
    Args:
        page_texts: Dictionary mapping page numbers to text content
        
    Returns:
        List of tuples: (title, start_page, end_page)
    """
    chapters = []
    
    # Regex patterns for chapter detection (case insensitive)
    patterns = [
        r'^(cap[ií]tulo\s+\d+.*?)(?:\n|$)',  # Capítulo 1, Capítulo I
        r'^(chapter\s+\d+.*?)(?:\n|$)',      # Chapter 1, Chapter I
        r'^(section\s+\d+.*?)(?:\n|$)',      # Section 1
        r'^(sec\.\s*\d+.*?)(?:\n|$)',        # Sec. 1
        r'^(parte\s+\d+.*?)(?:\n|$)',        # Parte 1
        r'^(\d+\.\s+[A-Z].*?)(?:\n|$)',      # 1. Introduction
        r'^([A-Z][A-Z\s]+)(?:\n|$)',         # ALL CAPS titles
    ]
    
    compiled_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in patterns]
    
    detected_chapters = []
    
    for page_num in sorted(page_texts.keys()):
        text = page_texts[page_num]
        if not text:
            continue
        
        # Look for chapter markers at the beginning of the text
        text_start = text.lstrip()[:500]  # First 500 chars after whitespace
        
        for pattern in compiled_patterns:
            matches = pattern.findall(text_start)
            for match in matches:
                title = match.strip()
                # Filter out very short or very long titles
                if 5 <= len(title) <= 100:
                    detected_chapters.append((title, page_num))
                    break  # Found chapter on this page, move to next page
        
        # Also check for numbered sections in the middle of pages
        if page_num not in [ch[1] for ch in detected_chapters]:
            # Look for section headers (less restrictive)
            section_pattern = re.compile(r'^(\d+\.\d+\s+[A-Za-z].*?)(?:\n|$)', re.MULTILINE)
            matches = section_pattern.findall(text)
            if matches:
                title = matches[0].strip()
                if 10 <= len(title) <= 80:
                    detected_chapters.append((title, page_num))
    
    # Convert to chapters with end pages
    chapters = []
    for i, (title, start_page) in enumerate(detected_chapters):
        # Find end page
        if i + 1 < len(detected_chapters):
            end_page = detected_chapters[i + 1][1] - 1
        else:
            end_page = max(page_texts.keys())  # Last page
        
        chapters.append((title, start_page, end_page))
    
    if chapters:
        log_info(f"Detected {len(chapters)} chapters using regex patterns")
    else:
        log_warning("No chapters detected using regex patterns")
    
    return chapters


def get_chapters(pdf_path: Path, chapter_numbers: Optional[List[int]] = None) -> List[Tuple[str, int, int]]:
    """
    Get chapters from PDF, trying outlines first, then regex detection.
    
    Args:
        pdf_path: Path to PDF file
        chapter_numbers: Optional list of specific chapter numbers to extract
        
    Returns:
        List of tuples: (title, start_page, end_page)
    """
    # First try to get chapters from PDF outlines
    chapters = get_outlines(pdf_path)
    
    # If no outlines found, try regex detection
    if not chapters:
        log_info("Attempting chapter detection using text patterns...")
        page_texts = extract_text_by_page(pdf_path, layout_aware=False, use_ocr=True, ocr_lang='eng+por', force_ocr=False)
        chapters = detect_chapters_by_regex(page_texts)
    
    # If specific chapters requested, filter the list
    if chapter_numbers:
        if len(chapters) == 0:
            log_warning("No chapters found to filter")
            return []
        
        filtered_chapters = []
        for chapter_num in chapter_numbers:
            if 1 <= chapter_num <= len(chapters):
                filtered_chapters.append(chapters[chapter_num - 1])
            else:
                log_warning(f"Chapter {chapter_num} not found (PDF has {len(chapters)} chapters)")
        
        chapters = filtered_chapters
        log_info(f"Filtered to {len(chapters)} requested chapters")
    
    return chapters


def print_chapter_summary(chapters: List[Tuple[str, int, int]]):
    """Print a summary of detected chapters."""
    if not chapters:
        console.print("[yellow]No chapters found[/yellow]")
        return
    
    console.print(f"\n[bold]Found {len(chapters)} chapters:[/bold]")
    for i, (title, start, end) in enumerate(chapters, 1):
        page_count = end - start + 1
        console.print(f"  {i:2d}. [cyan]{title}[/cyan] (pages {start}-{end}, {page_count} pages)")
    console.print()


def get_total_pages_for_chapters(chapters: List[Tuple[str, int, int]]) -> int:
    """Calculate total number of pages across all chapters."""
    if not chapters:
        return 0
    
    total = 0
    for title, start, end in chapters:
        total += end - start + 1
    
    return total


def validate_chapters(chapters: List[Tuple[str, int, int]], total_pages: int) -> List[Tuple[str, int, int]]:
    """
    Validate and fix chapter page ranges.
    
    Args:
        chapters: List of chapter tuples
        total_pages: Total number of pages in PDF
        
    Returns:
        Validated and corrected chapter list
    """
    if not chapters:
        return chapters
    
    validated = []
    
    for title, start, end in chapters:
        # Ensure start page is valid
        start = max(1, min(start, total_pages))
        
        # Ensure end page is valid and >= start
        end = max(start, min(end, total_pages))
        
        validated.append((title, start, end))
    
    return validated