"""Assembly of translated chapters and PDF generation."""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .utils import log_info, log_warning, log_error, log_success, slugify, ensure_output_dir

def assemble_full_document(output_dir: Path, chapters: List[Tuple[str, str]], 
                           output_filename_stem: str, original_title: str = "Documento Traduzido") -> Path:
    """Assemble all chapters into a single markdown document."""
    ensure_output_dir(output_dir)
    full_doc_path = output_dir / f"{output_filename_stem}.md"
    
    try:
        with open(full_doc_path, 'w', encoding='utf-8') as f:
            # Write document title
            f.write(f"# {original_title}\n\n")
            # Write content
            for _, content in chapters:
                f.write(content)
        log_success(f"Full document assembled: {full_doc_path}")
        return full_doc_path
    except IOError as e:
        log_error(f"Failed to assemble full document: {e}")
        raise

def _insert_images_into_markdown(markdown_content: str, manifest: dict, images_dir: Path) -> str:
    """
    Insert images from manifest into markdown content at appropriate positions.
    
    Args:
        markdown_content: The translated markdown content
        manifest: Image manifest with coordinates and filenames
        images_dir: Directory containing extracted images
        
    Returns:
        Markdown content with image references inserted
    """
    if not manifest or not images_dir.exists():
        return markdown_content
    
    # Extract all images from the manifest structure
    all_images = []
    pages = manifest.get('pages', {})
    for page_key, page_data in pages.items():
        images = page_data.get('images', [])
        for img in images:
            all_images.append(img)
    
    if not all_images:
        log_info("No images found in manifest to insert")
        return markdown_content
    
    log_info(f"Processing {len(all_images)} images for insertion...")
    
    # Split content into lines for processing
    lines = markdown_content.split('\n')
    result_lines = []
    
    # Track which images we've inserted to avoid duplicates
    inserted_images = set()
    
    for line in lines:
        result_lines.append(line)
        
        # Insert images after headers, at the beginning of sections
        if line.strip().startswith('#') and line.strip():
            # Find images that haven't been inserted yet
            section_images = []
            for img_info in all_images:
                img_filename = img_info.get('filename', '')
                if img_filename and img_filename not in inserted_images:
                    # Insert images that haven't been used yet
                    section_images.append(img_info)
                    inserted_images.add(img_filename)
                    
                    # Only insert a few images per section to avoid cluttering
                    if len(section_images) >= 2:
                        break
            
            # Add image references
            for img_info in section_images:
                img_filename = img_info.get('filename', '')
                if img_filename:
                    img_path = images_dir / img_filename
                    if img_path.exists():
                        # Use relative path for markdown
                        relative_path = f"img_manifest/{img_filename}"
                        result_lines.append("")
                        result_lines.append(f"![Imagem extraída]({relative_path})")
                        result_lines.append("")
    
    # If we still have uninserted images, add them at the end
    remaining_images = []
    for img_info in all_images:
        img_filename = img_info.get('filename', '')
        if img_filename and img_filename not in inserted_images:
            remaining_images.append(img_info)
    
    if remaining_images:
        result_lines.append("")
        result_lines.append("## Imagens Adicionais")
        result_lines.append("")
        
        for img_info in remaining_images:
            img_filename = img_info.get('filename', '')
            if img_filename:
                img_path = images_dir / img_filename
                if img_path.exists():
                    relative_path = f"img_manifest/{img_filename}"
                    result_lines.append(f"![Imagem extraída]({relative_path})")
                    result_lines.append("")
    
    final_content = '\n'.join(result_lines)
    log_success(f"Images successfully inserted into markdown")
    return final_content


def _extract_original_text_for_page(page_num: int) -> str:
    """Extract original text from the specified page for paragraph counting."""
    try:
        # Try to extract text from the source PDF for this specific page
        from .extract import extract_pages_range
        
        # Get source PDF path from manifest if available
        manifest_file = Path("out/img_manifest/manifest.json")
        if manifest_file.exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            source_pdf = manifest.get("source_pdf", "")
            if source_pdf and Path(source_pdf).exists():
                log_info(f"Extracting original text from page {page_num} for paragraph counting")
                page_texts = extract_pages_range(Path(source_pdf), [page_num], use_ocr=False)
                return page_texts.get(page_num, "")
    except Exception as e:
        log_warning(f"Could not extract original text for page {page_num}: {e}")
    
    return ""


def _count_paragraphs_before_image(original_text: str, image_y: float) -> int:
    """
    Count paragraphs in original text that would appear before the image position.
    Uses content analysis to find the exact position where image should be inserted.
    Returns a position that can be mapped to translated content.
    """
    if not original_text.strip():
        return 0
    
    # Split text into lines and look for specific patterns
    lines = original_text.split('\n')
    paragraphs = []
    
    # Create paragraphs by grouping non-empty lines
    current_paragraph = []
    for line in lines:
        line = line.strip()
        if not line:  # Empty line
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line)
    
    # Add final paragraph if exists
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    log_info(f"Original text has {len(paragraphs)} paragraphs")
    
    # Look for figure caption pattern - this is more universal
    # Figure captions typically follow the pattern: "Figure X-Y:" or "Figure X.Y:" 
    target_paragraph = 0
    
    for i, paragraph in enumerate(paragraphs):
        log_info(f"Paragraph {i}: {paragraph[:100]}...")
        
        # Look for figure caption patterns that are likely to be preserved after translation
        import re
        
        # Match patterns like "Figure 7-2:", "Figure 7.2:", "Figura 7-2:", etc.
        figure_pattern = r'(Figure|Figura)\s*7[-\.\s]*2\s*:'
        
        if re.search(figure_pattern, paragraph, re.IGNORECASE):
            target_paragraph = i  # Insert BEFORE the figure caption
            log_info(f"Found figure caption at paragraph {i}: {paragraph[:80]}...")
            break
    
    # If no figure caption found, look for reference patterns
    if target_paragraph == 0:
        for i, paragraph in enumerate(paragraphs):
            # Look for references to Figure 7-2 in text (these patterns might survive translation)
            figure_ref_patterns = [
                r'Figure\s*7[-\.\s]*2',
                r'shown in Figure',
                r'fuzzed ApplicationMessage field'  # Context-specific pattern
            ]
            
            for pattern in figure_ref_patterns:
                if re.search(pattern, paragraph, re.IGNORECASE):
                    target_paragraph = i + 1  # Insert AFTER the reference
                    log_info(f"Found figure reference at paragraph {i}, will insert after it")
                    break
            
            if target_paragraph > 0:
                break
    
    log_info(f"Determined image should be inserted before paragraph {target_paragraph}")
    return target_paragraph


def _insert_images_into_html(html_content: str, images_dir: Path) -> str:
    """
    Insert images into HTML content at appropriate positions by counting paragraphs from original PDF.
    
    Args:
        html_content: The translated HTML content
        images_dir: Directory containing extracted images and manifest
        
    Returns:
        HTML content with images inserted at correct positions
    """
    manifest_file = images_dir / "manifest.json"
    if not manifest_file.exists():
        log_warning("No manifest.json found for image insertion")
        return html_content
    
    try:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        pages = manifest.get('pages', {})
        if not pages:
            log_info("No images found in manifest")
            return html_content
        
        # Extract all images with their page info and Y coordinates
        images_to_insert = []
        for page_key, page_data in pages.items():
            images = page_data.get('images', [])
            for img in images:
                img_filename = img.get('filename', '')
                if img_filename:
                    img_path = images_dir / img_filename
                    if img_path.exists():
                        # Calculate average Y position from rects for better positioning
                        rects = img.get('rects', [])
                        avg_y = 0
                        if rects:
                            avg_y = sum(rect.get('y0', 0) for rect in rects) / len(rects)
                        
                        # Extract original text and count paragraphs for this page
                        page_num = int(page_key)
                        original_text = _extract_original_text_for_page(page_num)
                        paragraph_position = _count_paragraphs_before_image(original_text, avg_y)
                        
                        images_to_insert.append({
                            'filename': img_filename,
                            'page': page_num,
                            'path': img_path,
                            'rects': rects,
                            'avg_y': avg_y,
                            'paragraph_position': paragraph_position
                        })
        
        if not images_to_insert:
            log_info("No valid images found for insertion")
            return html_content
        
        # Sort images by Y coordinate within each page (higher Y = earlier in text)
        # Group by page first, then sort within each page
        from collections import defaultdict
        images_by_page = defaultdict(list)
        
        for img_info in images_to_insert:
            images_by_page[img_info['page']].append(img_info)
        
        # Sort images within each page by Y coordinate (descending = top to bottom in text)
        sorted_images = []
        for page_num, page_images in images_by_page.items():
            # Sort by Y coordinate descending (higher Y first = earlier in text)
            page_images.sort(key=lambda x: x['avg_y'], reverse=True)
            log_info(f"Page {page_num}: sorted {len(page_images)} images by Y coordinate")
            for img in page_images:
                log_info(f"  - {img['filename']}: Y={img['avg_y']:.1f}")
            sorted_images.extend(page_images)
        
        log_info(f"Inserting {len(sorted_images)} images into HTML content using intelligent pattern matching")
        
        # Process images in reverse order to maintain correct positions
        # (insert from bottom to top so positions don't shift)
        modified_content = html_content
        inserted_count = 0
        
        # Find all insertion positions first
        image_positions = []
        for img_info in sorted_images:
            insertion_pos = _find_image_insertion_position(html_content, img_info)
            if insertion_pos is not None:
                image_positions.append((insertion_pos, img_info))
            else:
                log_warning(f"Could not find insertion position for {img_info['filename']}")
        
        # Sort by position (descending) to insert from end to beginning
        image_positions.sort(key=lambda x: x[0], reverse=True)
        log_info(f"Found insertion positions for {len(image_positions)} images")
        
        for insertion_pos, img_info in image_positions:
            log_info(f"Found insertion position for {img_info['filename']} at character {insertion_pos}")
            
            # Convert image to base64 for reliable PDF rendering
            try:
                import base64
                with open(img_info["path"], 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    # Detect image format
                    img_ext = img_info["path"].suffix.lower()
                    if img_ext == '.png':
                        mime_type = 'image/png'
                    elif img_ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                    else:
                        mime_type = 'image/png'  # default
                    
                    img_src = f"data:{mime_type};base64,{img_base64}"
                    log_info(f"Converting image to base64: {img_info['filename']} ({len(img_data)} bytes)")
            except Exception as e:
                log_warning(f"Failed to convert image to base64: {e}")
                img_src = img_info["path"].resolve().as_uri()  # fallback to file URI
            
            img_tag = f'\n<div class="image-container"><img src="{img_src}" alt="Imagem da página {img_info["page"]}" class="inserted-image" /></div>\n'
            
            # Insert the image at the found position
            modified_content = modified_content[:insertion_pos] + img_tag + modified_content[insertion_pos:]
            inserted_count += 1
            log_info(f"Inserted image {img_info['filename']} at position {insertion_pos}")
        
        log_success(f"Successfully inserted {inserted_count} images using intelligent pattern matching")
        
        # Add any uninserted images at the end
        if inserted_count < len(sorted_images):
            remaining_images = [img_info for _, img_info in image_positions[inserted_count:]]
            if remaining_images:
                log_info(f"Adding {len(remaining_images)} remaining images at the end")
                modified_content = _add_images_at_end(modified_content, remaining_images)
        
        return modified_content
        
    except Exception as e:
        log_error(f"Error inserting images into HTML: {e}")
        return html_content


def _find_image_insertion_position(html_content: str, img_info: Dict) -> Optional[int]:
    """
    Find the exact position in HTML where the image should be inserted.
    Uses intelligent mapping by analyzing the original page content to determine the correct figure.
    """
    import re
    
    page_num = img_info.get('page', 0)
    filename = img_info.get('filename', '')
    avg_y = img_info.get('avg_y', 0)
    
    log_info(f"Looking for insertion position for {filename} at Y={avg_y:.1f} on page {page_num}")
    
    # Extract original text from the source page to identify which figure this should be
    original_text = _extract_original_text_for_page(page_num)
    
    # Find all figure references in the original page text
    figure_matches = re.findall(r'Figure\s*(\d+)[-\.\s]*(\d+)\s*:', original_text)
    
    if figure_matches:
        # Extract figure number from filename (e.g., page_201_img_01.png -> image 1)
        img_match = re.search(r'_img_(\d+)\.', filename)
        if img_match:
            img_number = int(img_match.group(1))
            log_info(f"Image number extracted: {img_number}")
            
            # Get the figure that corresponds to this image number on this page
            if img_number <= len(figure_matches):
                fig_major, fig_minor = figure_matches[img_number - 1]  # img_01 -> first figure
                target_figure = f"{fig_major}-{fig_minor}"
                log_info(f"Page {page_num}, image {img_number} should map to Figure {target_figure}")
                
                # Look for this specific figure in the translated HTML
                figure_patterns = [
                    rf'(Figure|Figura)\s*{fig_major}[-\.\s]*{fig_minor}\s*:',  
                    rf'(Figure|Figura)\s*{fig_major}[-\.\s]*{fig_minor}[^:]',
                ]
                
                for pattern in figure_patterns:
                    match = re.search(pattern, html_content, re.IGNORECASE)
                    if match:
                        log_info(f"Found correct figure caption for page {page_num}: {match.group()}")
                        return match.start()
                        
                log_warning(f"Could not find Figure {target_figure} in HTML for page {page_num}")
            else:
                log_warning(f"Image {img_number} on page {page_num} exceeds available figures ({len(figure_matches)})")
    else:
        log_warning(f"No figure captions found in original text for page {page_num}")
    
    # Fallback: Look for any figure caption that matches the page context
    # This is a more generic approach when specific mapping fails
    log_info(f"Using fallback strategy for {filename}")
    
    # Try to find figures that might belong to this page based on sequential order
    all_figures = list(re.finditer(r'(Figure|Figura)\s*\d+[-\.\s]*\d+\s*:', html_content, re.IGNORECASE))
    
    if all_figures:
        # Use a heuristic: pages with lower numbers get earlier figures
        # This assumes figures are roughly sequential in the document
        
        # Extract just page numbers from all images to determine relative order
        all_page_nums = sorted(set(int(re.search(r'page_(\d+)_', f['filename']).group(1)) 
                                   for f in [img_info] + []))  # Add context if available
        
        if page_num in all_page_nums:
            page_index = all_page_nums.index(page_num)
            
            # Map page index to figure index (with some safety bounds)
            if page_index < len(all_figures):
                target_figure = all_figures[page_index]
                log_info(f"Fallback mapping: page {page_num} -> figure at position {page_index}")
                return target_figure.start()
    
    # Final fallback: Use Y coordinate to determine relative position
    log_warning(f"Using Y-coordinate fallback for {filename}")
    
    # Find all paragraph breaks and estimate position based on Y coordinate
    paragraphs = re.finditer(r'<[ph]\d*[^>]*>', html_content)
    paragraph_list = list(paragraphs)
    
    if paragraph_list:
        # Estimate relative position (higher Y = earlier position)
        # Assuming page height ~800, normalize Y position
        relative_pos = min(avg_y / 800.0, 1.0)  # 0.0 = bottom, 1.0 = top
        target_idx = int((1.0 - relative_pos) * len(paragraph_list))  # Invert for text order
        target_idx = max(0, min(target_idx, len(paragraph_list) - 1))
        
        target_paragraph = paragraph_list[target_idx]
        log_info(f"Y-coordinate fallback: Y={avg_y:.1f} -> paragraph {target_idx}")
        return target_paragraph.start()
    
    log_warning("Could not find suitable insertion position")
    return None


def _add_images_at_end(html_content: str, images: List[Dict]) -> str:
    """Add images at the end of HTML content as fallback."""
    if not images:
        return html_content
    
    remaining_images_html = '\n<div class="additional-images">\n<h2>Imagens Adicionais</h2>\n'
    
    for img_info in images:
        try:
            import base64
            with open(img_info["path"], 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                img_ext = img_info["path"].suffix.lower()
                mime_type = 'image/png' if img_ext == '.png' else 'image/jpeg'
                img_src = f"data:{mime_type};base64,{img_base64}"
        except Exception as e:
            log_warning(f"Failed to convert image to base64: {e}")
            img_src = img_info["path"].resolve().as_uri()
        
        remaining_images_html += f'<div class="image-container"><img src="{img_src}" alt="Imagem da página {img_info["page"]}" class="inserted-image" /></div>\n'
    
    remaining_images_html += '</div>\n'
    return html_content + remaining_images_html


def _get_professional_css() -> str:
    """Return professional CSS for PDF generation."""
    return """
        body {
            font-family: 'Arial', 'Helvetica', sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 0;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-weight: bold;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            color: #2c3e50;
            page-break-after: avoid;
        }
        
        h1 { font-size: 24pt; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { font-size: 18pt; color: #2980b9; }
        h3 { font-size: 14pt; color: #8e44ad; }
        
        p {
            margin: 1em 0;
            text-align: justify;
        }
        
        /* Code blocks with syntax highlighting */
        pre {
            background-color: #f8f8f8;
            border: 1px solid #e1e1e8;
            border-radius: 6px;
            padding: 1em;
            margin: 1em 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }
        
        code {
            background-color: #f1f1f1;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 85%;
            color: #e74c3c;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            color: inherit;
        }
        
        /* Syntax highlighting */
        .language-python .keyword { color: #0000ff; font-weight: bold; }
        .language-javascript .keyword { color: #0000ff; font-weight: bold; }
        .language-bash .keyword { color: #0000ff; font-weight: bold; }
        
        /* Images */
        .image-container {
            text-align: center;
            margin: 1.5em 0;
            page-break-inside: avoid;
        }
        
        .inserted-image {
            width: 100%;
            max-width: 600px;
            height: auto;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 8px;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            object-fit: contain;
        }
        
        .additional-images {
            margin-top: 2em;
            border-top: 1px solid #ddd;
            padding-top: 1em;
        }
        
        /* Tables */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 10pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        /* Lists */
        ul, ol {
            margin: 1em 0;
            padding-left: 2em;
        }
        
        li {
            margin: 0.3em 0;
        }
        
        /* Blockquotes */
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 1em;
            margin: 1em 0;
            font-style: italic;
            color: #555;
            background-color: #ecf0f1;
            padding: 1em;
            border-radius: 4px;
        }
        
        @page {
            margin: 2cm;
            size: A4;
        }
        
        @media print {
            .container {
                margin: 0;
                padding: 0;
            }
        }
    """


def generate_pdf_from_html(html_content: str, output_path: Path, images_dir: Optional[Path] = None) -> Path:
    """Generate PDF from HTML using Playwright (preferred) with fallback to markdown-pdf."""
    try:
        from playwright.sync_api import sync_playwright
        
        # Insert images into HTML content at correct positions
        if images_dir and images_dir.exists():
            html_content = _insert_images_into_html(html_content, images_dir)
        
        # Create complete HTML document with professional styling
        full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documento Traduzido</title>
    <style>
        {_get_professional_css()}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""
        
        # Save HTML for debugging
        html_debug_path = output_path.with_suffix('.html')
        with open(html_debug_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        log_info(f"HTML debug file saved: {html_debug_path}")
        
        log_info(f"Generating PDF via Playwright: {output_path}")
        
        with sync_playwright() as p:
            # Launch browser with file access permissions
            browser = p.chromium.launch(args=['--allow-file-access-from-files', '--disable-web-security'])
            page = browser.new_page()
            page.set_content(full_html)
            
            # Wait for images to load
            try:
                page.wait_for_load_state('networkidle', timeout=10000)
                page.wait_for_timeout(2000)  # Additional wait for images
                log_info("Images loaded successfully")
            except Exception as e:
                log_warning(f"Image loading wait timed out: {e}")
            
            # Generate PDF with professional settings
            page.pdf(
                path=str(output_path),
                format='A4',
                margin={
                    'top': '2cm',
                    'bottom': '2cm',
                    'left': '2cm',
                    'right': '2cm'
                },
                print_background=True
            )
            
            browser.close()
        
        log_success(f"PDF generated successfully via Playwright: {output_path}")
        return output_path
        
    except ImportError:
        log_warning("Playwright not available, falling back to markdown-pdf")
        log_warning("Install Playwright with: pip install playwright && playwright install chromium")
        return _fallback_to_markdown_pdf_from_html(html_content, output_path, images_dir)
    except Exception as e:
        log_error(f"Playwright PDF generation failed: {e}")
        log_warning("Falling back to markdown-pdf method")
        return _fallback_to_markdown_pdf_from_html(html_content, output_path, images_dir)


def _fallback_to_markdown_pdf_from_html(html_content: str, output_path: Path, images_dir: Optional[Path] = None) -> Path:
    """Fallback method when Playwright is not available."""
    try:
        from markdown_pdf import MarkdownPdf, Section
        
        log_warning("FALLBACK: Using markdown-pdf instead of Playwright")
        log_warning("For better results, install Playwright: pip install playwright && playwright install chromium")
        
        # Convert HTML back to markdown (simple conversion)
        # This is a basic fallback - ideally we'd keep the markdown version
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        markdown_content = h.handle(html_content)
        
        # Apply image insertion to markdown
        if images_dir and images_dir.exists():
            manifest_file = images_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    markdown_content = _insert_images_into_markdown(markdown_content, manifest, images_dir)
                except Exception as e:
                    log_warning(f"Could not process image manifest: {e}")
        
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.save(str(output_path))
        
        log_success(f"PDF generated with fallback method: {output_path}")
        return output_path
        
    except Exception as e:
        log_error(f"Fallback PDF generation failed: {e}")
        raise


def _fallback_to_markdown_pdf(markdown_path: Path, images_dir: Optional[Path] = None) -> Path:
    """Fallback to original markdown-pdf method if HTML conversion fails."""
    try:
        from markdown_pdf import MarkdownPdf, Section
        
        output_path = markdown_path.with_suffix('.pdf')
        log_warning("Falling back to markdown-pdf method...")
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        if images_dir and images_dir.exists():
            manifest_file = images_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    markdown_content = _insert_images_into_markdown(markdown_content, manifest, images_dir)
                except Exception as e:
                    log_warning(f"Could not process image manifest: {e}")
        
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.save(str(output_path))
        
        log_success(f"PDF generated with fallback method: {output_path}")
        return output_path
        
    except Exception as e:
        log_error(f"Fallback PDF generation also failed: {e}")
        raise


# Keep the old function name for compatibility
def generate_pdf(markdown_path: Path, images_dir: Optional[Path] = None) -> Path:
    """Generate PDF with HTML method first, fallback to markdown-pdf."""
    return generate_pdf_from_html(markdown_path, images_dir)

def add_metadata_to_markdown(markdown_path: Path, title: str, author: str = "", subject: str = ""):
    """Adds YAML metadata header to a markdown file for context (not used by markdown-pdf)."""
    pass # No longer needed for this PDF generation method

def print_pdf_requirements():
    pass # No longer needed