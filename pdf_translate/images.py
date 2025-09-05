"""Image extraction and reinsertion using PyMuPDF."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from .utils import console, log_info, log_warning, log_error, ensure_output_dir


def extract_image_manifest(pdf_path: Path, output_dir: Path) -> Dict:
    """
    Extract all images from PDF and create a manifest.
    
    Args:
        pdf_path: Path to source PDF
        output_dir: Directory to save extracted images
        
    Returns:
        Dictionary with image manifest information
    """
    output_dir = ensure_output_dir(output_dir)
    img_dir = output_dir / "img_manifest"
    img_dir.mkdir(exist_ok=True)
    
    manifest = {
        "source_pdf": str(pdf_path),
        "extraction_date": None,
        "pages": {},
        "total_images": 0
    }
    
    try:
        doc = fitz.open(pdf_path)
        total_images = 0
        
        log_info(f"Extracting images from {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = []
            
            # Get image list for this page
            img_list = page.get_images(full=True)
            
            if img_list:
                log_info(f"Found {len(img_list)} images on page {page_num + 1}")
            
            for img_index, img in enumerate(img_list):
                try:
                    # Extract image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate filename
                    img_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:02d}.{image_ext}"
                    img_path = img_dir / img_filename
                    
                    # Save image
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Get image rectangle (bbox) on the page
                    img_rects = page.get_image_rects(img)
                    
                    # Store image info
                    img_info = {
                        "filename": img_filename,
                        "xref": xref,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "extension": image_ext,
                        "size_bytes": len(image_bytes),
                        "rects": []
                    }
                    
                    # Add rectangle information
                    for rect in img_rects:
                        img_info["rects"].append({
                            "x0": rect.x0,
                            "y0": rect.y0,
                            "x1": rect.x1,
                            "y1": rect.y1
                        })
                    
                    page_images.append(img_info)
                    total_images += 1
                    
                except Exception as e:
                    log_warning(f"Failed to extract image {img_index + 1} from page {page_num + 1}: {e}")
            
            # Store page info even if no images
            manifest["pages"][page_num + 1] = {
                "images": page_images,
                "image_count": len(page_images)
            }
        
        manifest["total_images"] = total_images
        
        # Save manifest
        manifest_path = img_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        log_info(f"✓ Extracted {total_images} images to {img_dir}")
        log_info(f"✓ Manifest saved to {manifest_path}")
        
        doc.close()
        return manifest
        
    except Exception as e:
        log_error(f"Failed to extract images: {e}")
        raise


def reinsert_images(pdf_path: Path, manifest_path: Path, page_shift: int = 0) -> Path:
    """
    Reinsert images from manifest into a PDF.
    
    Args:
        pdf_path: Path to target PDF (e.g., translated PDF)
        manifest_path: Path to image manifest.json
        page_shift: Page offset to compensate for different page counts
        
    Returns:
        Path to output PDF with images
    """
    output_path = pdf_path.parent / f"{pdf_path.stem}_IMGS.pdf"
    
    try:
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        img_dir = manifest_path.parent
        
        # Open target PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        log_info(f"Reinserting images into {total_pages} pages...")
        
        images_inserted = 0
        
        for page_num_str, page_info in manifest["pages"].items():
            source_page = int(page_num_str)
            target_page = source_page + page_shift
            
            # Check if target page exists
            if target_page < 1 or target_page > total_pages:
                log_warning(f"Target page {target_page} doesn't exist (PDF has {total_pages} pages)")
                continue
            
            page = doc[target_page - 1]  # Convert to 0-based
            
            for img_info in page_info["images"]:
                try:
                    img_path = img_dir / img_info["filename"]
                    
                    if not img_path.exists():
                        log_warning(f"Image file not found: {img_path}")
                        continue
                    
                    # Insert image at each rectangle position
                    for rect_info in img_info["rects"]:
                        rect = fitz.Rect(
                            rect_info["x0"],
                            rect_info["y0"], 
                            rect_info["x1"],
                            rect_info["y1"]
                        )
                        
                        # Insert image
                        page.insert_image(rect, filename=str(img_path))
                        images_inserted += 1
                        
                except Exception as e:
                    log_warning(f"Failed to insert image {img_info['filename']}: {e}")
        
        # Save output PDF
        doc.save(output_path)
        doc.close()
        
        log_info(f"✓ Reinserted {images_inserted} images")
        log_info(f"✓ Output saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        log_error(f"Failed to reinsert images: {e}")
        raise


def validate_manifest(manifest_path: Path) -> bool:
    """
    Validate that an image manifest is properly formatted.
    
    Args:
        manifest_path: Path to manifest.json file
        
    Returns:
        True if manifest is valid, False otherwise
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Check required fields
        required_fields = ["source_pdf", "pages", "total_images"]
        for field in required_fields:
            if field not in manifest:
                log_error(f"Manifest missing required field: {field}")
                return False
        
        # Check that image files exist
        img_dir = manifest_path.parent
        missing_files = []
        
        for page_info in manifest["pages"].values():
            for img_info in page_info["images"]:
                img_path = img_dir / img_info["filename"]
                if not img_path.exists():
                    missing_files.append(img_info["filename"])
        
        if missing_files:
            log_warning(f"Missing image files: {missing_files[:5]}...")  # Show first 5
            return False
        
        return True
        
    except Exception as e:
        log_error(f"Failed to validate manifest: {e}")
        return False


def get_manifest_info(manifest_path: Path) -> Dict:
    """
    Get summary information from an image manifest.
    
    Args:
        manifest_path: Path to manifest.json file
        
    Returns:
        Dictionary with manifest summary
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        pages_with_images = sum(1 for page_info in manifest["pages"].values() 
                              if page_info["image_count"] > 0)
        
        total_size = 0
        img_dir = manifest_path.parent
        
        for page_info in manifest["pages"].values():
            for img_info in page_info["images"]:
                total_size += img_info.get("size_bytes", 0)
        
        return {
            "total_images": manifest["total_images"],
            "pages_with_images": pages_with_images,
            "total_pages": len(manifest["pages"]),
            "total_size_mb": total_size / (1024 * 1024),
            "source_pdf": manifest["source_pdf"]
        }
        
    except Exception as e:
        log_error(f"Failed to get manifest info: {e}")
        return {}


def print_manifest_info(manifest_path: Path):
    """Print formatted manifest information."""
    info = get_manifest_info(manifest_path)
    
    if not info:
        console.print("[red]Could not read manifest information[/red]")
        return
    
    console.print(f"\n[bold]Image Manifest Info:[/bold]")
    console.print(f"  Total images: [yellow]{info['total_images']}[/yellow]")
    console.print(f"  Pages with images: [blue]{info['pages_with_images']}/{info['total_pages']}[/blue]")
    console.print(f"  Total size: [green]{info['total_size_mb']:.1f} MB[/green]")
    console.print(f"  Source PDF: [cyan]{info['source_pdf']}[/cyan]")
    console.print()


def check_page_count_compatibility(pdf_path: Path, manifest_path: Path, page_shift: int = 0) -> bool:
    """
    Check if PDF and manifest have compatible page counts.
    
    Args:
        pdf_path: Path to target PDF
        manifest_path: Path to image manifest
        page_shift: Page shift to account for
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        doc = fitz.open(pdf_path)
        target_pages = len(doc)
        doc.close()
        
        # Get highest page number from manifest
        max_source_page = max(int(p) for p in manifest["pages"].keys())
        max_target_page = max_source_page + page_shift
        
        if max_target_page > target_pages:
            log_warning(f"Page count mismatch: manifest needs page {max_target_page}, "
                       f"but PDF only has {target_pages} pages")
            return False
        
        return True
        
    except Exception as e:
        log_error(f"Failed to check page compatibility: {e}")
        return False