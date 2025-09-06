# OCRack - Advanced PDF Translation Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)](https://openai.com/)
[![Rich UI](https://img.shields.io/badge/UI-Rich%20Terminal-purple.svg)](https://github.com/Textualize/rich)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional-grade PDF translation system with OCR capabilities, intelligent chunking, and automated image extraction/reinsertion.

![](assets/demo.png)

## Features

- **Default Behavior**: Automatic image extraction + translation + PDF generation
- **Smart Chunking**: Intelligent text segmentation for optimal translation context
- **Image Processing**: Automatic extraction and reinsertion with precise positioning
- **Cost Optimization**: Real-time cost tracking with OpenAI API caching support
- **Rich UI**: Professional terminal interface with detailed progress tracking
- **Robust Pipeline**: Error handling, retries, and checkpoint recovery

## Quick Start

### Installation
```bash
git clone https://github.com/marcostolosa/OCRack.git
cd OCRack
pip install -e .
```

#### Dependencies
```bash
# Core dependencies (installed automatically)
pip install -r requirements.txt

# External programs (install separately)
# - Pandoc: https://pandoc.org/installing.html
# - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
```

### Basic Usage
```bash
# Default: Extract images + translate + generate PDF
ocrack document.pdf -p "234-235"

# Skip image extraction
ocrack document.pdf -p "234-235" --no-ocr

# Terminal output only (no PDF)
ocrack document.pdf -p "234-235" --cli
```

## Command Reference

### Core Commands
```bash
# Page range translation
ocrack input.pdf --pages "10-28"

# Chapter-based translation  
ocrack input.pdf -c "1,3-5"

# High-quality model
ocrack input.pdf --pages "10-28" -m gpt-4o

# Cost control
ocrack input.pdf -c "1-10" --max-chunks 50
```

### Flags
| Flag | Description |
|------|-------------|
| `--pages "X-Y"` | Translate specific page range |
| `--no-ocr` | Disable image extraction |
| `--cli` | Terminal output (skip PDF generation) |
| `-c "1,3-5"` | Translate specific chapters |
| `-m MODEL` | OpenAI model (default: gpt-4o-mini) |
| `-o DIR` | Output directory |

## Architecture

### Pipeline
1. **PDF Analysis** - Document structure and metadata extraction
2. **Image Extraction** - Page-specific image harvesting with coordinates
3. **Text Extraction** - Layout-aware text processing with OCR fallback
4. **Smart Chunking** - Context-preserving text segmentation
5. **AI Translation** - OpenAI GPT-powered EN→PT-BR translation
6. **Document Assembly** - Markdown compilation with metadata
7. **PDF Generation** - Python-based PDF creation with image reinsertion

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"  # optional
export OPENAI_BASE_URL="..."       # optional
```

### File Structure
```
project/
├── out/                       # Default output directory
│   ├── LIVRO_TRADUZIDO.md     # Translated markdown
│   ├── LIVRO_TRADUZIDO.pdf    # Final PDF with images
│   └── img_manifest/          # Extracted images
│       ├── manifest.json      # Image metadata
│       └── page_XXX_img_XX.png
└── logs/
    └── translate_YYYYMMDD_HHMMSS.log
```

## Dependencies

### Python Libraries
- **Core**: `openai>=1.0.0`, `rich>=13.0.0`, `PyMuPDF>=1.23.0`
- **PDF**: `pdfplumber>=0.7.0`, `PyPDF2>=3.0.0`, `markdown-pdf`
- **OCR**: `paddleocr>=2.8.0`, `paddlepaddle>=2.6.0`, `pytesseract>=0.3.10`
- **Images**: `Pillow>=9.0.0`, `opencv-python>=4.8.0`

### External Programs  
- **Pandoc**: Fallback PDF generation (https://pandoc.org)
- **PaddleOCR**: Primary OCR engine (auto-installed)

## Advanced Usage

### Batch Processing
```bash
# Multiple page ranges
for pages in "1-50" "51-100" "101-150"; do
    ocrack document.pdf --pages "$pages" -o "batch_$pages"
done
```

### Cost (gpt-4o-mini)
- Input: $0.15/1K tokens, Output: $0.60/1K tokens  
- Typical: $0.001-0.01 per page
- Cache savings: Up to 50% on repeated content

## License

MIT License