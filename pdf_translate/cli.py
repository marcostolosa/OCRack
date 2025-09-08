"Command-line interface for the PDF Translation Engine."

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from . import __version__
from .assemble import add_metadata_to_markdown, assemble_full_document, generate_pdf
from .chapters import get_chapters
from .chunker import smart_chunks
from .extract import extract_pages_range, get_pdf_metadata
from .images import extract_image_manifest
from .logger import cleanup_logger, setup_logger
from .status import StatusManager
from .translate import TranslationClient
from .utils import ensure_output_dir, parse_page_range, validate_pdf_path
from .pricing import get_model_pricing, calculate_cost, format_cost_estimate

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="An AI-powered engine for translating technical PDF documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument("-p", "--pages", required=True, help="Specific page range to translate (e.g., '10-28').")
    parser.add_argument("-o", "--output", default="out", help="Output directory (default: out/).")
    parser.add_argument("-m", "--model", help=f"OpenAI model (default: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}).")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Concurrent workers for translation (default: 8).")
    parser.add_argument("--no-images", action="store_true", help="Disable image extraction.")
    parser.add_argument("--version", action="version", version=f"OCRack {__version__}")
    return parser

def main_process(args):
    """The main logic of the application."""
    with StatusManager() as status:
        status.update("Initializing...")
        pdf_path = Path(args.input_pdf)
        output_dir = ensure_output_dir(args.output)
        model_name = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            status.log_error("OPENAI_API_KEY environment variable not set.")
            return 1

        status.update(f"Analyzing PDF: {pdf_path.name}")
        metadata = get_pdf_metadata(pdf_path)
        target_pages = parse_page_range(args.pages)

        status.update("Extracting text...")
        page_texts = extract_pages_range(pdf_path, target_pages)

        image_manifest = extract_image_manifest(pdf_path, output_dir, target_pages) if not args.no_images else None
        if image_manifest:
            status.log_info(f"Extracted {image_manifest.get('total_images', 0)} images.")

        chunks_by_page = {pn: smart_chunks(text) for pn, text in page_texts.items()}
        all_chunks_with_ref = [(pn, i, chunk) for pn, chunks in chunks_by_page.items() for i, chunk in enumerate(chunks)]

        translator = TranslationClient(model=model_name, api_key=api_key)
        translated_pages = {pn: [""] * len(chunks) for pn, chunks in chunks_by_page.items()}

        # Track token usage and costs
        total_input_tokens = 0
        total_output_tokens = 0 
        total_cached_tokens = 0
        total_requests = 0

        status.update("Translating chunks concurrently...")
        with Progress(SpinnerColumn(), TextColumn("[green]Translating..."), BarColumn(), TextColumn("{task.completed}/{task.total}")) as progress:
            task = progress.add_task("Chunks", total=len(all_chunks_with_ref))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_chunk_ref = {executor.submit(translator.translate_chunk, chunk): (pn, i) for pn, i, chunk in all_chunks_with_ref}
                for future in as_completed(future_to_chunk_ref):
                    pn, i = future_to_chunk_ref[future]
                    try:
                        translated_text, token_usage = future.result()
                        translated_pages[pn][i] = translated_text
                        
                        # Accumulate token usage
                        total_input_tokens += token_usage.get("input", 0)
                        total_output_tokens += token_usage.get("output", 0)
                        total_cached_tokens += token_usage.get("cached", 0)
                        total_requests += 1
                        
                    except Exception as e:
                        status.log_error(f"Page {pn}, Chunk {i+1} failed: {e}")
                        original_chunk = chunks_by_page[pn][i]
                        translated_pages[pn][i] = original_chunk
                    progress.update(task, advance=1)

        status.update("Assembling final document...")
        
        # Now that GPT returns HTML, we need to assemble HTML content instead of markdown
        final_html_content = ""
        for pn in sorted(translated_pages.keys()):
            page_html_chunks = translated_pages[pn]
            # Join HTML chunks with proper spacing
            page_html = "\n".join(page_html_chunks)
            final_html_content += page_html + "\n"

        doc_title = metadata.get('title', 'Translated Document')
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename_stem = f"{pdf_path.stem}_translated_{timestamp_str}"
        
        # Wrap with document title
        final_html_content = f"<h1>{doc_title}</h1>\n{final_html_content}"
        
        status.update("Generating final PDF...")
        
        # Use new HTML-based PDF generation
        images_dir = output_dir / "img_manifest" if image_manifest else None
        pdf_output_path = output_dir / f"{output_filename_stem}.pdf"
        
        from .assemble import generate_pdf_from_html
        pdf_output_path = generate_pdf_from_html(final_html_content, pdf_output_path, images_dir)
        
        # Calculate and display cost information
        pricing = get_model_pricing(model_name)
        total_cost = calculate_cost(total_input_tokens, total_output_tokens, total_cached_tokens, pricing)
        
        # Display cost summary
        status.log_success(f"SUCCESS! PDF gerado: {pdf_output_path}")
        status.log_info("")
        status.log_info("💰 CUSTO DA TRADUÇÃO:")
        status.log_info(f"   Modelo: {model_name}")
        status.log_info(f"   Total de requisições: {total_requests:,}")
        status.log_info(f"   Tokens (Input): {total_input_tokens:,}")
        status.log_info(f"   Tokens (Output): {total_output_tokens:,}")
        if total_cached_tokens > 0:
            cached_pct = (total_cached_tokens / total_input_tokens) * 100 if total_input_tokens > 0 else 0
            status.log_info(f"   Tokens (Cached): {total_cached_tokens:,} ({cached_pct:.1f}% desconto)")
        status.log_info(f"   💵 CUSTO TOTAL: ${total_cost:.4f} USD")
        
        if pricing and pricing.supports_caching and total_cached_tokens > 0:
            status.log_info("   ✨ Prompt caching ativo - economia significativa!")

        return 0

def main():
    """CLI Entry point."""
    parser = create_parser()
    args = parser.parse_args()
    try:
        return main_process(args)
    except Exception as e:
        print(f"[bold red]A critical error occurred: {e}[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
