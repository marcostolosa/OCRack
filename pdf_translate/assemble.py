"""Assembly of translated chapters and PDF generation."""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

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


def _insert_images_into_html(html_content: str, images_dir: Path) -> str:
    """
    Insert images into HTML content at appropriate positions based on manifest.
    
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
        
        # Extract all images with their page info
        images_to_insert = []
        for page_key, page_data in pages.items():
            images = page_data.get('images', [])
            for img in images:
                img_filename = img.get('filename', '')
                if img_filename:
                    img_path = images_dir / img_filename
                    if img_path.exists():
                        images_to_insert.append({
                            'filename': img_filename,
                            'page': int(page_key),
                            'path': img_path,
                            'rects': img.get('rects', [])
                        })
        
        if not images_to_insert:
            log_info("No valid images found for insertion")
            return html_content
        
        log_info(f"Inserting {len(images_to_insert)} images into HTML content")
        
        # Sort images by page order
        images_to_insert.sort(key=lambda x: x['page'])
        
        # Insert images before "Figura XYZ" references
        import re
        
        # First try to find "Figura" references to place images correctly
        figura_pattern = r'(<p[^>]*>.*?Figura\s+\d+.*?</p>)'
        figura_matches = list(re.finditer(figura_pattern, html_content, re.IGNORECASE | re.DOTALL))
        
        modified_content = html_content
        inserted_count = 0
        
        # Process from end to beginning to avoid position shifting issues
        figura_matches.reverse()
        
        # Insert images before "Figura X" references (working backwards)
        for i, match in enumerate(figura_matches):
            if inserted_count >= len(images_to_insert):
                break
                
            # Get the last available image (since we're working backwards)
            img_info = images_to_insert[-(inserted_count + 1)]
            
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
                    log_info(f"Inserting image as base64 before Figura: {img_info['filename']} ({len(img_data)} bytes)")
            except Exception as e:
                log_warning(f"Failed to convert image to base64: {e}")
                img_src = img_info["path"].resolve().as_uri()  # fallback to file URI
            
            img_tag = f'\n<div class="image-container"><img src="{img_src}" alt="Imagem da página {img_info["page"]}" class="inserted-image" /></div>\n'
            
            # Insert the image before the "Figura" reference
            insert_pos = match.start()
            modified_content = modified_content[:insert_pos] + img_tag + modified_content[insert_pos:]
            inserted_count += 1
        
        # If there are remaining images, add them at the end
        if inserted_count < len(images_to_insert):
            remaining_images_html = '\n<div class="additional-images">\n<h2>Imagens Adicionais</h2>\n'
            for img_info in images_to_insert[inserted_count:]:
                # Convert remaining images to base64
                try:
                    import base64
                    with open(img_info["path"], 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        img_ext = img_info["path"].suffix.lower()
                        mime_type = 'image/png' if img_ext == '.png' else 'image/jpeg'
                        img_src = f"data:{mime_type};base64,{img_base64}"
                except Exception as e:
                    log_warning(f"Failed to convert remaining image to base64: {e}")
                    img_src = img_info["path"].resolve().as_uri()
                
                remaining_images_html += f'<div class="image-container"><img src="{img_src}" alt="Imagem da página {img_info["page"]}" class="inserted-image" /></div>\n'
            remaining_images_html += '</div>\n'
            modified_content += remaining_images_html
        
        log_success(f"Successfully inserted {len(images_to_insert)} images into HTML")
        return modified_content
        
    except Exception as e:
        log_error(f"Error inserting images into HTML: {e}")
        return html_content


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