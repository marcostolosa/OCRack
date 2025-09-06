"""Assembly of translated chapters and PDF generation."""

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from .utils import console, log_info, log_warning, log_error, log_success, slugify, ensure_output_dir


def _insert_images_into_markdown(markdown_content: str, manifest: dict, images_dir: Path) -> str:
    """
    Insert images from manifest into markdown content at appropriate locations.
    
    Args:
        markdown_content: Original markdown content
        manifest: Image manifest with page and coordinate information
        images_dir: Directory containing extracted images
        
    Returns:
        Modified markdown content with image references
    """
    import re
    
    # Look for figure references in the markdown
    figure_patterns = [
        r'Figura\s+(\d+[-\w]*):',  # Portuguese "Figura X-Y:"
        r'Figure\s+(\d+[-\w]*):',  # English "Figure X-Y:"
    ]
    
    modified_content = markdown_content
    
    # Process each image in the manifest
    for page_str, images in manifest.items():
        if not page_str.startswith('page_'):
            continue
            
        page_num = int(page_str.replace('page_', ''))
        
        for image_info in images:
            image_name = image_info['name']
            image_path = images_dir / image_name
            
            if image_path.exists():
                # Try to find figure reference to insert image after
                for pattern in figure_patterns:
                    matches = list(re.finditer(pattern, modified_content))
                    for match in matches:
                        figure_ref = match.group(0)
                        figure_num = match.group(1)
                        
                        # Insert image after the figure reference
                        image_markdown = f"\n\n![{figure_ref}]({image_path.as_posix()})\n\n"
                        
                        # Find the end of the line containing the figure reference
                        line_end = modified_content.find('\n', match.end())
                        if line_end == -1:
                            line_end = len(modified_content)
                        
                        # Insert image after the figure reference line
                        modified_content = (
                            modified_content[:line_end] + 
                            image_markdown + 
                            modified_content[line_end:]
                        )
                        
                        log_info(f"Inserted image {image_name} after {figure_ref}")
                        break
                else:
                    # If no figure reference found, try to insert based on page context
                    # This is a fallback for images without explicit figure references
                    image_markdown = f"\n\n![Image from page {page_num}]({image_path.as_posix()})\n\n"
                    
                    # Insert at the end of content for now (simple fallback)
                    if f"page {page_num}" in modified_content.lower():
                        # Try to insert near content that mentions this page
                        page_mention = re.search(rf'page\s+{page_num}\b', modified_content, re.IGNORECASE)
                        if page_mention:
                            insert_pos = modified_content.find('\n\n', page_mention.end())
                            if insert_pos != -1:
                                modified_content = (
                                    modified_content[:insert_pos] + 
                                    image_markdown + 
                                    modified_content[insert_pos:]
                                )
                                log_info(f"Inserted image {image_name} near page {page_num} reference")
    
    return modified_content


def write_chapter_file(output_dir: Path, chapter_num: int, title: str, content: str) -> Path:
    """
    Write a single chapter to a markdown file.
    
    Args:
        output_dir: Output directory
        chapter_num: Chapter number
        title: Chapter title
        content: Translated content
        
    Returns:
        Path to written file
    """
    ensure_output_dir(output_dir)
    
    slug = slugify(title)
    filename = f"{chapter_num:02d}_{slug}.md"
    file_path = output_dir / filename
    
    # Prepare chapter content with title
    chapter_content = f"# {title}\n\n{content}"
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chapter_content)
        
        log_info(f"Chapter {chapter_num} saved to {filename}")
        return file_path
        
    except IOError as e:
        log_error(f"Failed to write chapter file {filename}: {e}")
        raise


def assemble_full_document(output_dir: Path, chapters: List[Tuple[str, str]], 
                          original_title: str = "Documento Traduzido") -> Path:
    """
    Assemble all chapters into a single markdown document.
    
    Args:
        output_dir: Output directory
        chapters: List of (title, content) tuples
        original_title: Title for the full document
        
    Returns:
        Path to assembled document
    """
    ensure_output_dir(output_dir)
    
    full_doc_path = output_dir / "LIVRO_TRADUZIDO.md"
    
    try:
        with open(full_doc_path, 'w', encoding='utf-8') as f:
            # Write document title
            f.write(f"# {original_title}\n\n")
            f.write("---\n\n")
            
            # Write table of contents
            f.write("## Sumário\n\n")
            for i, (title, _) in enumerate(chapters, 1):
                f.write(f"{i}. [{title}](#{slugify(title)})\n")
            f.write("\n---\n\n")
            
            # Write chapters
            for i, (title, content) in enumerate(chapters, 1):
                f.write(f"# {title} {{#{slugify(title)}}}\n\n")
                f.write(content)
                
                # Add page break hint for PDF generation
                if i < len(chapters):
                    f.write("\n\n\\pagebreak\n\n")
        
        log_success(f"Full document assembled: {full_doc_path}")
        return full_doc_path
        
    except IOError as e:
        log_error(f"Failed to assemble full document: {e}")
        raise


def _get_pandoc_path() -> Optional[str]:
    """
    Get the path to pandoc executable.
    
    Returns:
        Path to pandoc executable or None if not found
    """
    # First try standard PATH lookup
    pandoc_path = shutil.which("pandoc")
    if pandoc_path is not None:
        return pandoc_path
    
    # Try common Windows installation paths
    import os
    windows_paths = [
        r"C:\Program Files\Pandoc\pandoc.exe",
        r"C:\Program Files (x86)\Pandoc\pandoc.exe",
        r"C:\Users\{}\AppData\Local\Pandoc\pandoc.exe".format(os.getenv('USERNAME', ''))
    ]
    
    for path in windows_paths:
        if os.path.exists(path):
            return path
    
    return None


def check_pandoc_available() -> bool:
    """
    Check if pandoc is available in PATH or standard Windows locations.
    
    Returns:
        True if pandoc is available, False otherwise
    """
    return _get_pandoc_path() is not None


def _add_miktex_to_path():
    """
    Add MiKTeX to PATH on Windows if available.
    """
    import os
    if os.name == 'nt':  # Windows only
        miktex_path = r"C:\Users\Marcos\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
        if os.path.exists(miktex_path):
            current_path = os.environ.get('PATH', '')
            if miktex_path not in current_path:
                os.environ['PATH'] = f"{current_path};{miktex_path}"
                log_info(f"Added MiKTeX to PATH: {miktex_path}")


def check_xelatex_available() -> bool:
    """
    Check if xelatex is available in PATH.
    
    Returns:
        True if xelatex is available, False otherwise
    """
    # First try adding MiKTeX to PATH
    _add_miktex_to_path()
    return shutil.which("xelatex") is not None


def _check_pdflatex_available() -> bool:
    """
    Check if pdflatex is available in PATH.
    
    Returns:
        True if pdflatex is available, False otherwise
    """
    # First try adding MiKTeX to PATH
    _add_miktex_to_path()
    return shutil.which("pdflatex") is not None


def generate_pdf_python(markdown_path: Path, output_path: Optional[Path] = None, 
                        images_dir: Optional[Path] = None) -> Path:
    """
    Generate PDF using Python markdown-pdf library with image reinsertion (preferred method).
    
    Args:
        markdown_path: Path to markdown file
        output_path: Optional output PDF path (default: same name as markdown)
        images_dir: Optional directory containing extracted images for reinsertion
        
    Returns:
        Path to generated PDF
        
    Raises:
        Exception: If PDF generation fails
    """
    try:
        from markdown_pdf import MarkdownPdf, Section
        import json
        import re
        
        if output_path is None:
            output_path = markdown_path.with_suffix('.pdf')
        
        log_info(f"Generating PDF with markdown-pdf: {output_path}")
        
        # Read markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Check for images directory and manifest
        if images_dir and images_dir.exists():
            manifest_file = images_dir / "manifest.json"
            if manifest_file.exists():
                log_info(f"Found image manifest, processing images for reinsertion...")
                
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    
                    # Process images and insert into markdown
                    markdown_content = _insert_images_into_markdown(markdown_content, manifest, images_dir)
                    log_success(f"Images successfully inserted into markdown from manifest")
                    
                except Exception as e:
                    log_warning(f"Could not process image manifest: {e}")
        
        # Create PDF and disable TOC manually
        pdf = MarkdownPdf()
        
        # Add the translated markdown content with images
        pdf.add_section(Section(markdown_content))
        
        # Manually clear TOC to avoid hierarchy errors  
        pdf.toc = []
        
        # Generate PDF
        pdf.save(str(output_path))
        
        log_success(f"PDF generated successfully with markdown-pdf: {output_path}")
        return output_path
        
    except ImportError as e:
        log_error(f"Required libraries not installed: {e}. Install with: pip install markdown-pdf")
        raise Exception("markdown-pdf library not available")
    except Exception as e:
        log_error(f"PDF generation failed: {e}")
        # Fallback to HTML
        return _generate_html_fallback(markdown_path, output_path)


def generate_pdf(markdown_path: Path, output_path: Optional[Path] = None, 
                  images_dir: Optional[Path] = None) -> Path:
    """
    Generate PDF from markdown using Python library first, then pandoc fallback.
    
    Args:
        markdown_path: Path to markdown file
        output_path: Optional output PDF path (default: same name as markdown)
        images_dir: Optional directory containing extracted images for reinsertion
        
    Returns:
        Path to generated PDF
        
    Raises:
        Exception: If all methods fail
    """
    # Try Python method first (preferred)
    try:
        return generate_pdf_python(markdown_path, output_path, images_dir)
    except Exception as e:
        log_warning(f"Python PDF generation failed: {e}. Trying Pandoc fallback...")
    
    # Fallback to Pandoc method
    if not check_pandoc_available():
        log_error("Neither Python PDF generation nor Pandoc are available")
        # Generate HTML as final fallback
        return _generate_html_fallback(markdown_path, output_path or markdown_path.with_suffix('.pdf'))
    
    # Ensure MiKTeX is in PATH before checking
    _add_miktex_to_path()
    
    has_xelatex = check_xelatex_available()
    has_pdflatex = _check_pdflatex_available()
    
    if not has_xelatex and not has_pdflatex:
        log_warning(
            "Neither xelatex nor pdflatex found. PDF generation may fail.\n"
            "Please ensure MiKTeX or TeX Live is properly installed:\n"
            "  Windows: Install MiKTeX or TeX Live\n"
            "  Linux: sudo apt-get install texlive-xetex\n"
            "  macOS: brew install --cask mactex"
        )
    elif not has_xelatex:
        log_warning(
            "xelatex not found. PDF generation will use pdflatex.\n"
            "For better Unicode support, install XeLaTeX:\n"
            "  Windows: Install MiKTeX or TeX Live\n"
            "  Linux: sudo apt-get install texlive-xetex\n"
            "  macOS: brew install --cask mactex"
        )
    
    if output_path is None:
        output_path = markdown_path.with_suffix('.pdf')
    
    # Get pandoc executable path
    pandoc_path = _get_pandoc_path()
    if not pandoc_path:
        raise Exception("pandoc executable not found")
    
    # Pandoc command with Portuguese-friendly settings
    # Convert paths to absolute and use forward slashes for pandoc
    abs_markdown_path = str(markdown_path.resolve()).replace('\\', '/')
    abs_output_path = str(output_path.resolve()).replace('\\', '/')
    
    # Try wkhtmltopdf first, fall back to xelatex/pdflatex
    wkhtmltopdf_available = shutil.which("wkhtmltopdf") is not None
    
    if wkhtmltopdf_available:
        # Use wkhtmltopdf via pandoc (doesn't require LaTeX)
        cmd = [
            pandoc_path,
            abs_markdown_path,
            "-o", abs_output_path,
            "--pdf-engine=wkhtmltopdf",
            "--variable", "geometry:margin=1in",
            "--variable", "fontsize=11pt",
            "--toc",  # Table of contents
            "--toc-depth=3",
            "--number-sections",
            "--highlight-style=tango",
        ]
    else:
        # Fallback to LaTeX engines
        cmd = [
            pandoc_path,
            abs_markdown_path,
            "-o", abs_output_path,
            "--pdf-engine=xelatex" if has_xelatex else "--pdf-engine=pdflatex",
            "--variable", "geometry:margin=1in",
            "--variable", "fontsize=11pt",
            "--variable", "mainfont=DejaVu Serif",  # Good Unicode support
            "--variable", "sansfont=DejaVu Sans",
            "--variable", "monofont=DejaVu Sans Mono",
            "--toc",  # Table of contents
            "--toc-depth=3",
            "--number-sections",
            "--highlight-style=tango",
            "--variable", "lang=pt-BR",
            "--variable", "babel-lang=brazilian",
        ]
    
    log_info(f"Generating PDF: {output_path}")
    log_info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run pandoc
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=markdown_path.parent,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            # If LaTeX/wkhtmltopdf failed, try HTML fallback
            if not wkhtmltopdf_available and ("not found" in error_msg or "cannot be executed" in error_msg or "pdflatex" in error_msg or "xelatex" in error_msg):
                log_warning("PDF generation failed with LaTeX, generating HTML instead...")
                return _generate_html_fallback(markdown_path, output_path)
            else:
                raise Exception(f"Pandoc failed with return code {result.returncode}: {error_msg}")
        
        if result.stderr:
            log_warning(f"Pandoc warnings: {result.stderr}")
        
        if not output_path.exists():
            # Try HTML fallback if PDF wasn't created
            log_warning("PDF was not created, generating HTML instead...")
            return _generate_html_fallback(markdown_path, output_path)
        
        log_success(f"PDF generated: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("PDF generation timed out (5 minutes)")
    except Exception as e:
        log_error(f"PDF generation failed: {e}")
        # Try HTML fallback as last resort
        if "HTML" not in str(e):  # Avoid infinite recursion
            log_warning("Attempting HTML fallback...")
            try:
                return _generate_html_fallback(markdown_path, output_path)
            except:
                pass
        raise


def _generate_html_fallback(markdown_path: Path, output_path: Path) -> Path:
    """Generate HTML as fallback when PDF generation fails."""
    html_path = output_path.with_suffix('.html')
    
    pandoc_path = _get_pandoc_path()
    if not pandoc_path:
        raise Exception("pandoc executable not found")
    
    abs_markdown_path = str(markdown_path.resolve()).replace('\\', '/')
    abs_html_path = str(html_path.resolve()).replace('\\', '/')
    
    cmd = [
        pandoc_path,
        abs_markdown_path,
        "-o", abs_html_path,
        "--toc",
        "--toc-depth=3", 
        "--number-sections",
        "--highlight-style=tango",
        "--standalone",
        "--metadata", "title=From Day Zero to Zero Day",
        "--css", "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    ]
    
    log_info(f"Generating HTML fallback: {html_path}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        cwd=markdown_path.parent,
        timeout=120
    )
    
    if result.returncode != 0:
        error_msg = result.stderr or "Unknown error"
        raise Exception(f"HTML generation failed: {error_msg}")
    
    if not html_path.exists():
        raise Exception("HTML was not created")
    
    log_success(f"HTML generated: {html_path}")
    return html_path


def create_pandoc_metadata(title: str, author: str = "", subject: str = "") -> str:
    """
    Create YAML metadata header for pandoc.
    
    Args:
        title: Document title
        author: Document author
        subject: Document subject
        
    Returns:
        YAML metadata string
    """
    metadata = f"""---
title: "{title}"
author: "{author}"
subject: "{subject}"
lang: pt-BR
babel-lang: brazilian
documentclass: article
geometry: margin=1in
fontsize: 11pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
toc: true
toc-depth: 3
numbersections: true
highlight-style: tango
---

"""
    return metadata


def add_metadata_to_markdown(markdown_path: Path, title: str, 
                           author: str = "", subject: str = ""):
    """
    Add YAML metadata header to markdown file.
    
    Args:
        markdown_path: Path to markdown file
        title: Document title
        author: Document author
        subject: Document subject
    """
    try:
        # Read existing content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if metadata already exists
        if content.startswith('---'):
            log_info("Markdown file already has metadata")
            return
        
        # Add metadata
        metadata = create_pandoc_metadata(title, author, subject)
        new_content = metadata + content
        
        # Write back
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        log_info("Added metadata to markdown file")
        
    except IOError as e:
        log_error(f"Failed to add metadata: {e}")


def get_pandoc_version() -> Optional[str]:
    """Get pandoc version string."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # First line contains version
            first_line = result.stdout.split('\n')[0]
            return first_line
        
        return None
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def print_pdf_requirements():
    """Print information about PDF generation requirements."""
    console.print("\n[bold]PDF Generation Requirements:[/bold]")
    
    # Check pandoc
    pandoc_available = check_pandoc_available()
    pandoc_version = get_pandoc_version() if pandoc_available else None
    
    if pandoc_available:
        console.print(f"  Pandoc: [green]{pandoc_version}[/green]")
    else:
        console.print("  Pandoc: [red]Not found[/red]")
        console.print("    Install from: https://pandoc.org/installing.html")
    
    # Check XeLaTeX
    xelatex_available = check_xelatex_available()
    if xelatex_available:
        console.print("  XeLaTeX: [green]Available[/green]")
    else:
        console.print("  XeLaTeX: [yellow]Not found (will use pdflatex)[/yellow]")
        console.print("    Install TeX Live or MiKTeX for better Unicode support")
    
    console.print()


def clean_markdown_for_pdf(content: str) -> str:
    """
    Clean markdown content for better PDF generation.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Cleaned markdown content
    """
    import re
    
    # Fix common issues that cause problems in PDF generation
    
    # Escape underscores in URLs that aren't in code blocks
    content = re.sub(r'(?<!`)(https?://[^\s`]+?)(?!`)', 
                    lambda m: m.group(0).replace('_', r'\_'), 
                    content)
    
    # Fix standalone underscores that might be interpreted as emphasis
    content = re.sub(r'(?<!\w)_(?!\w)', r'\_', content)
    
    # Ensure proper spacing around headers
    content = re.sub(r'\n(#+\s)', r'\n\n\1', content)
    
    # Fix table formatting issues
    content = re.sub(r'\|(\s*-+\s*)\|', r'| \1 |', content)
    
    return content