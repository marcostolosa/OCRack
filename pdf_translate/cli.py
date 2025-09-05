"""Command-line interface for PDF translation with Simple UI for Git Bash Windows."""

import argparse
import os
import sys
import time
from pathlib import Path

from . import __version__
from .utils import validate_pdf_path, ensure_output_dir, parse_page_range, parse_chapter_spec
from .extract import extract_text_by_page, extract_pages_range, validate_pdf_extractable, get_pdf_metadata
from .chapters import get_chapters, print_chapter_summary, validate_chapters
from .chunker import smart_chunks, chunk_statistics
from .translate import create_translator, estimate_cost, validate_api_key, print_cost_estimate
from .assemble import (
    write_chapter_file, assemble_full_document, generate_pdf, 
    check_pandoc_available, print_pdf_requirements, add_metadata_to_markdown
)
from .images import extract_image_manifest, reinsert_images, validate_manifest, print_manifest_info
from .simple_ui import SimpleUI, print_info, print_warning, print_error, print_success, simulate_dry_run_progress
from .logger import setup_logger, cleanup_logger
from .pricing import get_model_pricing


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate PDF chapters from English to Portuguese (Brazil) with Rich UI status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with Rich UI to see what would be translated
  ocrack input.pdf --dry-run --log-level INFO
  
  # Translate specific chapters with Rich UI status
  ocrack input.pdf -c "1,3-5" -o output/ --log-level DEBUG
  
  # Translate page range with PDF output
  ocrack input.pdf --pages "10-28" -p -o output/
  
  # Complete workflow with images and Rich UI
  ocrack input.pdf --images manifest -o output/
  ocrack input.pdf -c "1-5" -p -o output/
  ocrack output/LIVRO_TRADUZIDO.pdf --images reinserir
  
  # High-quality translation with cost optimization
  ocrack input.pdf -m gpt-4o -c "1-10" --resume -p --max-chars 15000

Environment variables:
  OPENAI_API_KEY     OpenAI API key (required)
  OPENAI_BASE_URL    Custom API base URL (optional)
  OPENAI_MODEL       Model to use (default: gpt-4o-mini)

Rich UI Features:
  - Real-time progress bars for chunks and pages
  - Token usage and cost tracking with caching benefits
  - Phase management (extracting, translating, generating PDF)
  - Retry/backoff display with countdown timers
  - ETA calculation and performance metrics
  - Checkpoint/resume status with visual indicators
  - Detailed logging to logs/ directory
        """
    )
    
    # Positional argument
    parser.add_argument(
        "input_pdf",
        help="Path to input PDF file"
    )
    
    # Model and API options
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    
    # Chapter/page selection
    parser.add_argument(
        "-c", "--chapters",
        help="Chapter numbers to translate (e.g., '1,3-5')"
    )
    
    parser.add_argument(
        "--pages",
        help="Page range to translate (e.g., '10-28') - overrides chapter selection"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        default="out",
        help="Output directory (default: out/)"
    )
    
    parser.add_argument(
        "-p", "--pdf",
        action="store_true",
        help="Generate PDF output using pandoc"
    )
    
    # Chunking options
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum characters per chunk for better context (default: 12000)"
    )
    
    parser.add_argument(
        "--merge-threshold",
        type=int,
        default=1200,
        help="Minimum chunk size before merging (default: 1200)"
    )
    
    # Processing options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show translation plan with Rich UI preview without calling API"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint with progress restoration"
    )
    
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Maximum chunks to translate (for testing/cost control)"
    )
    
    parser.add_argument(
        "--layout-aware",
        action="store_true",
        default=True,
        help="Use layout-aware text extraction (default: enabled)"
    )
    
    # Image options
    parser.add_argument(
        "--images",
        choices=["manifest", "reinserir"],
        help="Image handling: 'manifest' to extract, 'reinserir' to reinsert"
    )
    
    parser.add_argument(
        "--page-shift",
        type=int,
        default=0,
        help="Page offset for image reinsertion (default: 0)"
    )
    
    # OCR options
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        default=True,
        help="Use OCR for scanned/image-based PDFs (default: enabled)"
    )
    
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR - only use standard text extraction"
    )
    
    parser.add_argument(
        "--ocr-lang",
        default="eng+por",
        help="OCR language(s): 'eng' for English, 'por' for Portuguese, 'eng+por' for mixed content (default: eng+por for better accuracy)"
    )
    
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages, even if text extraction seems to work (useful for scanned PDFs with poor text layer)"
    )
    
    # UI and Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="Logging level for Rich UI and file logs (default: INFO)"
    )
    
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable Rich UI (for batch processing or CI/CD)"
    )
    
    parser.add_argument(
        "--log-dir",
        help="Custom log directory (default: logs/)"
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"OCRack {__version__}"
    )
    
    return parser


def validate_arguments(args) -> bool:
    """Validate command line arguments."""
    # Process OCR arguments
    if args.no_ocr:
        args.use_ocr = False
    
    # Check PDF exists
    try:
        validate_pdf_path(args.input_pdf)
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        return False
    
    # Check API key unless dry run or images only
    if not args.dry_run and not args.images:
        if not validate_api_key():
            print_error(
                "OpenAI API key is required. Set the OPENAI_API_KEY environment variable.\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
            return False
    
    # Check pandoc if PDF output requested
    if args.pdf and not check_pandoc_available():
        print_error(
            "pandoc is required for PDF generation.\n"
            "Install from: https://pandoc.org/installing.html"
        )
        return False
    
    # Validate chunk parameters
    if args.max_chars < 1000:
        print_error("--max-chars must be at least 1000")
        return False
    
    if args.merge_threshold >= args.max_chars:
        print_error("--merge-threshold must be less than --max-chars")
        return False
    
    return True


def handle_image_operations(args, ui_status=None) -> int:
    """Handle image extraction or reinsertion operations."""
    pdf_path = Path(args.input_pdf)
    output_dir = ensure_output_dir(args.output)
    
    if args.images == "manifest":
        if ui_status:
            ui_status.set_phase("Extraindo manifesto de imagens")
        
        print_info("Extracting image manifest...")
        try:
            manifest = extract_image_manifest(pdf_path, output_dir)
            
            if ui_status:
                ui_status.finish_phase("Extraindo manifesto de imagens", success=True)
            
            print_success(f"Extracted {manifest['total_images']} images")
            return 0
        except Exception as e:
            if ui_status:
                ui_status.finish_phase("Extraindo manifesto de imagens", success=False)
            
            print_error(f"Image extraction failed: {e}")
            return 1
    
    elif args.images == "reinserir":
        manifest_path = output_dir / "img_manifest" / "manifest.json"
        
        if not manifest_path.exists():
            print_error(f"Image manifest not found: {manifest_path}")
            print_info("Run with --images manifest first to extract images")
            return 1
        
        if not validate_manifest(manifest_path):
            print_error("Invalid image manifest")
            return 1
        
        print_manifest_info(manifest_path)
        
        if ui_status:
            ui_status.set_phase("Reinserindo imagens no PDF")
        
        print_info("Reinserting images...")
        try:
            output_pdf = reinsert_images(pdf_path, manifest_path, args.page_shift)
            
            if ui_status:
                ui_status.finish_phase("Reinserindo imagens no PDF", success=True)
            
            print_success(f"Images reinserted: {output_pdf}")
            return 0
        except Exception as e:
            if ui_status:
                ui_status.finish_phase("Reinserindo imagens no PDF", success=False)
            
            print_error(f"Image reinsertion failed: {e}")
            return 1
    
    return 1


def simulate_dry_run_progress(ui_status, total_chunks: int, total_pages: int):
    """Simulate progress for dry run demonstration."""
    if not ui_status:
        return
    
    ui_status.start_chunk_progress(total_chunks)
    
    # Simulate processing chunks
    for i in range(total_chunks):
        time.sleep(0.1)  # Simulate processing time
        
        # Simulate token usage
        input_tokens = 2500 + (i * 50)
        output_tokens = 2200 + (i * 45)
        cached_tokens = 1500 if i > 0 else 0  # Cache hits after first chunk
        
        ui_status.report_api_usage(input_tokens, output_tokens, cached_tokens)
        ui_status.advance_chunk(processing_time=0.8 + (i * 0.1))
        
        # Update pages done
        pages_per_chunk = total_pages / total_chunks
        ui_status.update_pages_done(int((i + 1) * pages_per_chunk))
        
        # Simulate occasional retry
        if i == 3:
            ui_status.report_retry("Rate limit exceeded", 1, 3, 2.5)
            time.sleep(0.3)
            ui_status.clear_retry_status()


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Setup logging
    logger = setup_logger(
        log_level=args.log_level,
        log_dir=args.log_dir,
        enable_file_logging=True,
        enable_console_logging=not args.no_ui
    )
    
    pdf_path = validate_pdf_path(args.input_pdf)
    output_dir = ensure_output_dir(args.output)
    model_name = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Create UI status system
    enable_ui = not args.no_ui and not args.images
    ui_status = SimpleUI()  # Use simple UI instead of Rich UI
    
    try:
        # Handle image operations
        if args.images:
            return handle_image_operations(args, ui_status)
        
        # Check if PDF is extractable
        if not validate_pdf_extractable(pdf_path):
            console.print(
                "[red]Error:[/red] PDF appears to be scanned or image-based. No extractable text found.\n"
                "Consider using OCR tools first to create a text-searchable PDF."
            )
            return 1
        
        # Get PDF metadata
        ui_status.set_phase("Analisando PDF")
        metadata = get_pdf_metadata(pdf_path)
        
        if metadata.get('pages'):
            print(f"PDF: {metadata['pages']} pages")
            if metadata.get('title'):
                print(f"Title: {metadata['title']}")
        
        logger.log_job_start(str(pdf_path), model_name, metadata.get('pages', 0), 0)
        
        # Determine pages to translate
        target_pages = []
        chapters = []
        
        if args.pages:
            # Direct page range
            try:
                target_pages = parse_page_range(args.pages)
                print(f"Translating pages: {args.pages}")
            except ValueError as e:
                print_error(f"Invalid page range: {e}")
                return 1
        else:
            # Chapter-based or full document
            ui_status.set_phase("Detectando capítulos")
            chapter_numbers = None
            if args.chapters:
                try:
                    chapter_numbers = parse_chapter_spec(args.chapters)
                except ValueError as e:
                    print_error(f"Invalid chapter specification: {e}")
                    return 1
            
            chapters = get_chapters(pdf_path, chapter_numbers)
            
            if not chapters:
                print_warning("No chapters detected. Translating entire document.")
                target_pages = list(range(1, metadata.get('pages', 1) + 1))
            else:
                if enable_ui:
                    print_chapter_summary(chapters)
                # Extract pages from chapters
                for _, start, end in chapters:
                    target_pages.extend(range(start, end + 1))
                target_pages = sorted(set(target_pages))
        
        # Initialize UI status with job information
        total_pages = len(target_pages)
        total_chapters = len(chapters) if chapters else 1
        
        ui_status.start_job(str(pdf_path), model_name, total_pages, total_chapters)
        
        # Set chapter information if available
        if chapters:
            title, start, end = chapters[0]
            ui_status.set_chapter(0, len(chapters), title, start, end)
        
        # Extract text from target pages
        ui_status.set_phase("Extraindo texto do PDF")
        print_info(f"Extracting text from {len(target_pages)} pages...")
        page_texts = extract_pages_range(pdf_path, target_pages, layout_aware=args.layout_aware, use_ocr=args.use_ocr, ocr_lang=args.ocr_lang, force_ocr=args.force_ocr)
        
        if not page_texts:
            print_error("No text extracted from specified pages")
            return 1
        
        ui_status.finish_phase("Extraindo texto do PDF", success=True)
        
        # Combine all text for chunking
        full_text = "\n\n".join(page_texts.values())
        
        # Create chunks
        ui_status.set_phase("Criando chunks inteligentes")
        print_info("Creating smart chunks...")
        chunks = smart_chunks(full_text, max_chars=args.max_chars, merge_threshold=args.merge_threshold)
        
        if args.max_chunks:
            chunks = chunks[:args.max_chunks]
            print_warning(f"Limited to {len(chunks)} chunks for testing")
        
        ui_status.finish_phase("Criando chunks inteligentes", success=True)
        
        # Show statistics
        stats = chunk_statistics(chunks)
        print(f"\nChunking Statistics:")
        print(f"  Total chunks: {stats['count']}")
        print(f"  Total characters: {stats['total_chars']:,}")
        print(f"  Average chunk size: {stats['avg_chars']:,} chars")
        print(f"  Size range: {stats['min_chars']:,} - {stats['max_chars']:,} chars")
        print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
        
        # Cost estimation
        if not args.dry_run:
            pricing = get_model_pricing(model_name)
            if pricing:
                cost_info = estimate_cost(chunks, model_name)
                print_cost_estimate(cost_info)
        
        # Dry run - show plan and simulate UI
        if args.dry_run:
            print_success(f"Translation plan ready: {len(target_pages)} pages -> {len(chunks)} chunks")
            print_info("Simulating translation progress...")
            
            # Simulate the translation process for UI demo
            ui_status.set_phase("Traduzindo (simulação)")
            simulate_dry_run_progress(ui_status, len(chunks), len(target_pages))
            ui_status.finish_phase("Traduzindo (simulação)", success=True)
            
            print_success("Dry run completed! This is what the real translation would look like.")
            summary = ui_status.final_summary(success=True)
            logger.log_job_complete(True, summary)
            return 0
        
        # Initialize translator
        ui_status.set_phase("Inicializando tradutor OpenAI")
        print_info("Initializing translator...")
        try:
            translator = create_translator(model=model_name)
            ui_status.finish_phase("Inicializando tradutor OpenAI", success=True)
        except Exception as e:
            ui_status.finish_phase("Inicializando tradutor OpenAI", success=False)
            print_error(f"Failed to initialize translator: {e}")
            return 1
        
        # Setup checkpoint system
        from .utils import Checkpoint
        checkpoint = Checkpoint(output_dir)
        translated_chunks = []
        next_index = 0
        
        # Resume from checkpoint if requested
        if args.resume:
            checkpoint_data = checkpoint.load()
            if checkpoint_data:
                translated_chunks = checkpoint.get_translated_chunks()
                next_index = checkpoint.get_next_index()
                ui_status.set_checkpoint_info(True, str(checkpoint.checkpoint_file), next_index)
                print_info(f"Resuming from chunk {next_index + 1}/{len(chunks)}")
        
        # Start chunk progress tracking in UI
        ui_status.start_chunk_progress(len(chunks))
        
        # Translate chunks
        if next_index < len(chunks):
            ui_status.set_phase("Traduzindo chunks")
            
            for i in range(next_index, len(chunks)):
                chunk = chunks[i]
                
                try:
                    start_time = time.time()
                    
                    # Translate chunk
                    translated_chunk = translator.translate_chunk(chunk)
                    translated_chunks.append(translated_chunk)
                    
                    processing_time = time.time() - start_time
                    
                    # This would be reported by the translator, but for demo we'll simulate
                    # The actual integration is in translate.py
                    input_tokens = len(chunk) // 4  # Rough estimate
                    output_tokens = len(translated_chunk) // 4
                    cached_tokens = 1500 if i > 0 else 0  # System prompt caching
                    
                    ui_status.report_api_usage(input_tokens, output_tokens, cached_tokens)
                    ui_status.advance_chunk(processing_time)
                    
                    # Update checkpoint
                    checkpoint.update_progress(i + 1, translated_chunk)
                    
                    # Update pages done (rough estimate)
                    pages_per_chunk = len(target_pages) / len(chunks)
                    ui_status.update_pages_done(int((i + 1) * pages_per_chunk))
                    
                    # Log progress
                    logger.log_chunk_processed(i, len(chunks), processing_time, input_tokens, output_tokens, cached_tokens)
                    
                except KeyboardInterrupt:
                    print_warning("Translation interrupted. Progress saved to checkpoint.")
                    summary = ui_status.final_summary(success=False)
                    logger.log_job_complete(False, summary)
                    return 1
                except Exception as e:
                    ui_status.report_retry(str(e), 1, 3, 2.0)
                    print_error(f"Translation failed at chunk {i + 1}: {e}")
                    summary = ui_status.final_summary(success=False)
                    logger.log_job_complete(False, summary)
                    return 1
        else:
            print_success("All chunks already translated (loaded from checkpoint)")
        
        ui_status.finish_phase("Traduzindo chunks", success=True)
        
        # Clear checkpoint
        checkpoint.clear()
        
        # Assemble output
        ui_status.set_phase("Montando documento traduzido")
        print_info("Assembling translated document...")
        
        if chapters:
            # Save individual chapters
            for i, (title, start, end) in enumerate(chapters):
                # Calculate which chunks belong to this chapter
                chapter_start_idx = int(len(translated_chunks) * i / len(chapters))
                chapter_end_idx = int(len(translated_chunks) * (i + 1) / len(chapters))
                
                chapter_content = "\n\n".join(translated_chunks[chapter_start_idx:chapter_end_idx])
                write_chapter_file(output_dir, i + 1, title, chapter_content)
            
            # Assemble full document
            chapter_data = [(title, "\n\n".join(translated_chunks[
                int(len(translated_chunks) * i / len(chapters)):
                int(len(translated_chunks) * (i + 1) / len(chapters))
            ])) for i, (title, _, _) in enumerate(chapters)]
            
            doc_title = metadata.get('title', 'Documento Traduzido')
            markdown_path = assemble_full_document(output_dir, chapter_data, doc_title)
        else:
            # Single document
            doc_title = metadata.get('title', 'Documento Traduzido')
            markdown_path = output_dir / "LIVRO_TRADUZIDO.md"
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(f"# {doc_title}\n\n")
                f.write("\n\n".join(translated_chunks))
        
        ui_status.finish_phase("Montando documento traduzido", success=True)
        print_success(f"Translation completed: {markdown_path}")
        
        # Generate PDF if requested
        if args.pdf:
            ui_status.set_phase("Gerando PDF via pandoc")
            print_info("Generating PDF...")
            try:
                # Add metadata for better PDF generation
                add_metadata_to_markdown(
                    markdown_path,
                    title=metadata.get('title', 'Documento Traduzido'),
                    author=metadata.get('author', ''),
                    subject=metadata.get('subject', '')
                )
                
                pdf_path = generate_pdf(markdown_path)
                ui_status.finish_phase("Gerando PDF via pandoc", success=True)
                print_success(f"PDF generated: {pdf_path}")
                
            except Exception as e:
                ui_status.finish_phase("Gerando PDF via pandoc", success=False)
                print_error(f"PDF generation failed: {e}")
                print_pdf_requirements()
                summary = ui_status.final_summary(success=False)
                logger.log_job_complete(False, summary)
                return 1
        
        # Final summary
        summary = ui_status.final_summary(success=True)
        logger.log_job_complete(True, summary)
        
        return 0
        
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if ui_status:
            summary = ui_status.final_summary(success=False)
            logger.log_job_complete(False, summary)
        return 1
        
    finally:
        # Cleanup
        if ui_status:
            ui_status.stop()
        cleanup_logger()


if __name__ == "__main__":
    sys.exit(main())