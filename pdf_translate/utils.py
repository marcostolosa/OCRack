"""Utility functions for PDF translation."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

import platformdirs
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Temporarily use simple print for Windows compatibility
class SimpleConsole:
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        
console = SimpleConsole()


def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = re.sub(r'[^\w\s-]', '', text.strip().lower())
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]


class ETACalculator:
    """Calculate ETA based on moving average of processing times."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.start_time = None
    
    def start_chunk(self):
        """Mark the start of processing a chunk."""
        self.start_time = time.time()
    
    def end_chunk(self):
        """Mark the end of processing a chunk and record the time."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            self.start_time = None
    
    def get_eta(self, remaining_chunks: int) -> Optional[float]:
        """Get estimated time remaining in seconds."""
        if not self.times:
            return None
        
        avg_time = sum(self.times) / len(self.times)
        return avg_time * remaining_chunks


class Checkpoint:
    """Handle checkpoint save/resume functionality."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.data = {}
    
    def load(self) -> Dict[str, Any]:
        """Load checkpoint data."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"Checkpoint loaded: resuming from chunk {self.data.get('next_index', 0)}")
                return self.data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load checkpoint: {e}")
        
        return {}
    
    def save(self, data: Dict[str, Any]):
        """Save checkpoint data."""
        self.data.update(data)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error: Could not save checkpoint: {e}")
    
    def get_next_index(self) -> int:
        """Get the next chunk index to process."""
        return self.data.get('next_index', 0)
    
    def get_translated_chunks(self) -> List[str]:
        """Get already translated chunks."""
        return self.data.get('translated_chunks', [])
    
    def update_progress(self, next_index: int, translated_chunk: str):
        """Update progress and save checkpoint."""
        if 'translated_chunks' not in self.data:
            self.data['translated_chunks'] = []
        
        self.data['translated_chunks'].append(translated_chunk)
        self.data['next_index'] = next_index
        self.save(self.data)
    
    def clear(self):
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("SUCCESS: Checkpoint cleared")


def get_cache_dir() -> Path:
    """Get platform-appropriate cache directory."""
    return Path(platformdirs.user_cache_dir("ocrack"))


def format_time(seconds: Optional[float]) -> str:
    """Format seconds as human-readable time."""
    if seconds is None:
        return "unknown"
    
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def parse_page_range(page_range: str) -> List[int]:
    """Parse page range string like '1-3,5,7-9' into list of page numbers."""
    pages = []
    
    for part in page_range.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    
    return sorted(set(pages))


def parse_chapter_spec(chapter_spec: str) -> List[int]:
    """Parse chapter specification like '1,3-5' into list of chapter numbers."""
    chapters = []
    
    for part in chapter_spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            chapters.extend(range(start, end + 1))
        else:
            chapters.append(int(part))
    
    return sorted(set(chapters))


def create_progress_bar(description: str = "Processing"):
    """Create a rich progress bar for the translation process."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=True
    )


def log_info(message: str):
    """Log an info message."""
    print(f"INFO {message}")


def log_success(message: str):
    """Log a success message."""
    print(f"SUCCESS {message}")


def log_warning(message: str):
    """Log a warning message."""
    print(f"WARNING {message}")


def log_error(message: str):
    """Log an error message."""
    print(f"ERROR {message}")


def validate_pdf_path(pdf_path: str) -> Path:
    """Validate PDF file path exists and is readable."""
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    return path


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path