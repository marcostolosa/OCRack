# OCRack

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?logo=openai&logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Advanced PDF translation tool with superior OCR capabilities and intelligent document processing.

## Features

- **OCR Engine**: PaddleOCR v3.2.0 for maximum text extraction accuracy
- **Translation**: GPT-4/4o-mini with optimized prompts for technical content
- **Chapter Detection**: Automatic extraction via PDF bookmarks or heuristic patterns
- **Smart Chunking**: Optimized block sizing for better context and cost efficiency
- **Checkpoint System**: Resume functionality for large documents
- **Image Support**: Extract and reinsert images at original positions
- **Output Formats**: PDF/HTML generation via Pandoc with LaTeX fallback
- **Cost Optimization**: Automatic prompt caching (50% discount) and token control
- **Resilience**: Exponential backoff retry, rate limiting, intelligent ETA
- **Telemetry**: Comprehensive logging and performance metrics

## Interface

Real-time status display with progress tracking:
- PDF name, LLM model, processing timestamps
- Chapter/page progress with completion percentages  
- Chunk progress bars with ETA and processing speed
- Token usage and cost estimation with caching savings
- Current operation status and error handling
- Checkpoint/resume state for interrupted sessions

## Installation

```bash
git clone https://github.com/ocrack/ocrack.git
cd ocrack
pip install -e .
```

Verify installation:
```bash
ocrack --version
```

### API Configuration

```bash
export OPENAI_API_KEY='your-key-here'
```

### PDF Generation (Optional)

For PDF output, install Pandoc and LaTeX:

**Ubuntu/Debian:**
```bash
sudo apt install pandoc texlive-xetex
```

**macOS:**
```bash
brew install pandoc mactex
```

**Windows:**
```bash
choco install pandoc miktex
```

## Usage

### Basic Translation

```bash
# Translate entire document
ocrack document.pdf -o output/

# Specific chapters
ocrack book.pdf -c "1,3-5" -o output/

# Page range
ocrack paper.pdf --pages "10-25" -o output/
```

### OCR and Advanced Options

```bash
# Force OCR on all pages
ocrack scanned_doc.pdf --force-ocr -o output/

# With PDF output
ocrack document.pdf -c "1-5" -p -o output/

# Custom model and chunking
ocrack large_doc.pdf -m gpt-4o --max-chars 8000 --resume -o output/
```

### Image Workflow

```bash
# Extract images
ocrack illustrated_book.pdf --images manifest -o output/

# Translate with images
ocrack illustrated_book.pdf -c "1-3" -p -o output/

# Reinsert images
ocrack output/LIVRO_TRADUZIDO.pdf --images reinserir
```

### Development and Testing

```bash
# Dry run (no API calls)
ocrack document.pdf --dry-run

# Limited chunks for testing
ocrack test.pdf --max-chunks 3 -o test_output/

# Debug logging
ocrack document.pdf --log-level DEBUG
```

## Command Line Options

```
usage: ocrack [-h] [-m MODEL] [-c CHAPTERS] [--pages PAGES] [-o OUTPUT] [-p]
              [--max-chars MAX_CHARS] [--merge-threshold MERGE_THRESHOLD]
              [--dry-run] [--resume] [--max-chunks MAX_CHUNKS]
              [--layout-aware] [--images {manifest,reinserir}]
              [--page-shift PAGE_SHIFT] [--use-ocr] [--no-ocr]
              [--ocr-lang OCR_LANG] [--force-ocr]
              [--log-level {DEBUG,INFO,WARN,ERROR}] [--no-ui]
              [--log-dir LOG_DIR] [--version]
              input_pdf

Options:
  -m MODEL              Model: gpt-4o, gpt-4o-mini (default)
  -c CHAPTERS           Chapters: "1,3-5,8"
  --pages PAGES         Page range: "10-25" (overrides chapter detection)
  -o OUTPUT             Output directory (default: out/)
  -p, --pdf             Generate PDF output via Pandoc
  --max-chars N         Max characters per chunk (default: 12000)
  --merge-threshold N   Merge small chunks threshold (default: 1200)
  --dry-run             Show translation plan without API calls
  --resume              Resume from checkpoint file
  --max-chunks N        Limit chunks for testing
  --layout-aware        Use layout-aware text extraction (default: enabled)
  --images ACTION       Extract images (manifest) or reinsert (reinserir)
  --page-shift N        Page offset for image reinsertion (default: 0)
  --use-ocr             Use OCR for scanned PDFs (default: enabled)
  --no-ocr              Disable OCR - only standard text extraction
  --ocr-lang LANG       OCR language: 'eng', 'por', 'eng+por' (default: eng+por)
  --force-ocr           Force OCR on all pages (higher accuracy)
  --log-level LEVEL     Logging: DEBUG, INFO, WARN, ERROR (default: INFO)
  --no-ui              Disable Rich UI (machine-readable output)
  --log-dir DIR         Custom log directory (default: logs/)
```

## Configuration

Environment variables:
- `OPENAI_API_KEY`: Required API key
- `OPENAI_MODEL`: Default model (default: gpt-4o-mini)
- `OPENAI_BASE_URL`: Custom API endpoint (optional)

## Output Structure

```
output/
├── LIVRO_TRADUZIDO.md          # Translated markdown
├── LIVRO_TRADUZIDO.html        # HTML fallback
├── LIVRO_TRADUZIDO.pdf         # PDF output (if -p flag)
├── img_manifest/               # Extracted images
│   ├── manifest.json          # Image metadata
│   └── page_*.png            # Image files
├── checkpoint.json            # Resume data
└── logs/                     # Session logs
    └── translate_YYYYMMDD_HHMMSS.log
```

## Logging

Structured logging with performance metrics:
- Session duration and processing speed
- Token usage and cost breakdown
- Error tracking and retry attempts
- Chunk-level processing times
- API response codes and latency

## Requirements

- Python 3.8+
- OpenAI API key
- 2GB+ RAM for OCR processing
- Internet connection for API calls
- Pandoc + LaTeX for PDF generation (optional)

## License

MIT License