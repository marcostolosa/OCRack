"""Simple UI for Windows Git Bash - no Rich, just spinner and same-line updates."""

import sys
import time
import threading
from typing import Optional


class SimpleSpinner:
    """Simple spinner that works in Git Bash Windows."""
    
    def __init__(self):
        self.spinning = False
        self.spinner_thread = None
        self.current_message = ""
        self.spinner_chars = ['|', '/', '-', '\\']
        self.spinner_index = 0
        
    def start(self, message: str = "Working..."):
        """Start the spinner with a message."""
        self.current_message = message
        self.spinning = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        
    def update_message(self, message: str):
        """Update the spinner message."""
        self.current_message = message
        
    def stop(self, final_message: str = "Done"):
        """Stop the spinner and show final message."""
        self.spinning = False
        if self.spinner_thread:
            self.spinner_thread.join()
        # Clear line and show final message
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.write(f"{final_message}\n")
        sys.stdout.flush()
        
    def _spin(self):
        """Internal spinning loop."""
        while self.spinning:
            # Clear line and write spinner + message
            spinner_char = self.spinner_chars[self.spinner_index]
            sys.stdout.write('\r' + ' ' * 80 + '\r')  # Clear line
            sys.stdout.write(f"{spinner_char} {self.current_message}")
            sys.stdout.flush()
            
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
            time.sleep(0.1)


class SimpleUI:
    """Simple UI replacement for Rich - works perfectly in Git Bash Windows."""
    
    def __init__(self):
        self.spinner = SimpleSpinner()
        self.current_phase = ""
        self.total_chunks = 0
        self.completed_chunks = 0
        self.total_pages = 0
        self.completed_pages = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.job_started = False
        
    def start(self):
        """Start the UI."""
        pass  # No setup needed
        
    def stop(self, success: bool = True):
        """Stop the UI."""
        if success:
            self.spinner.stop("Translation completed successfully!")
        else:
            self.spinner.stop("Translation failed")
            
    def start_job(self, pdf_path: str, model: str, total_pages: int, total_chunks: int = 0):
        """Start a translation job."""
        self.job_started = True
        self.total_pages = total_pages
        self.total_chunks = total_chunks
        print_info(f"Starting translation job")
        print_info(f"PDF: {pdf_path}")
        print_info(f"Model: {model}")
        print_info(f"Total pages: {total_pages}")
        if total_chunks > 0:
            print_info(f"Total chunks: {total_chunks}")
            
    def final_summary(self, success: bool = True) -> dict:
        """Generate final summary."""
        summary = {
            "success": success,
            "total_chunks": self.completed_chunks,
            "total_pages": self.completed_pages,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }
        
        if success:
            print_success(f"Translation completed: {self.completed_chunks} chunks, {self.total_tokens:,} tokens, ${self.total_cost:.4f}")
        else:
            print_error(f"Translation failed after {self.completed_chunks} chunks")
            
        return summary
            
    def set_phase(self, phase: str):
        """Set current phase."""
        self.current_phase = phase
        self._update_display()
        
    def set_totals(self, chunks: int, pages: int):
        """Set total counts."""
        self.total_chunks = chunks
        self.total_pages = pages
        self._update_display()
        
    def start_chunk_progress(self, total_chunks: int):
        """Initialize chunk progress tracking."""
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self._update_display()
        
    def increment_chunk(self):
        """Increment completed chunks."""
        self.completed_chunks += 1
        self._update_display()
        
    def advance_chunk(self, processing_time: float = 0):
        """Advance chunk progress by one."""
        self.completed_chunks += 1
        self._update_display()
        
    def increment_page(self):
        """Increment completed pages."""
        self.completed_pages += 1
        self._update_display()
        
    def update_pages_done(self, done_pages: int):
        """Update number of pages completed."""
        self.completed_pages = done_pages
        self._update_display()
        
    def report_api_usage(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """Report API token usage."""
        self.total_tokens += input_tokens + output_tokens
        # Simple cost estimation (gpt-4o-mini)
        cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000
        self.total_cost += cost
        self._update_display()
        
    def report_retry(self, error: str, attempt: int, max_attempts: int, delay: float):
        """Report retry attempt."""
        message = f"Retrying ({attempt}/{max_attempts}) - {error[:30]}... waiting {delay:.1f}s"
        self.spinner.update_message(message)
        
    def clear_retry_status(self):
        """Clear retry status."""
        self._update_display()
        
    def finish_phase(self, phase_name: str = "", success: bool = True):
        """Finish current phase."""
        if phase_name:
            if success:
                self.current_phase = f"{phase_name} - Concluído"
            else:
                self.current_phase = f"{phase_name} - Falhou"
        self._update_display()
        
    def _update_display(self):
        """Update the spinner message with current progress."""
        if not hasattr(self, 'spinner'):
            return
            
        # Build progress message
        parts = []
        
        if self.current_phase:
            parts.append(self.current_phase)
            
        if self.total_chunks > 0:
            chunk_progress = f"Chunks: {self.completed_chunks}/{self.total_chunks}"
            parts.append(chunk_progress)
            
        if self.total_pages > 0:
            page_progress = f"Pages: {self.completed_pages}/{self.total_pages}"
            parts.append(page_progress)
            
        if self.total_tokens > 0:
            token_info = f"Tokens: {self.total_tokens:,}"
            parts.append(token_info)
            
        if self.total_cost > 0:
            cost_info = f"Cost: ${self.total_cost:.4f}"
            parts.append(cost_info)
            
        message = " | ".join(parts) if parts else "Working..."
        
        if self.spinner.spinning:
            self.spinner.update_message(message)
        else:
            self.spinner.start(message)


def print_info(message: str):
    """Print info message (replaces console.print)."""
    print(f"INFO: {message}")
    

def print_warning(message: str):
    """Print warning message."""
    print(f"WARNING: {message}")
    

def print_error(message: str):
    """Print error message."""
    print(f"ERROR: {message}")


def print_success(message: str):
    """Print success message."""
    print(f"SUCCESS: {message}")


# Simulate dry run progress for testing
def simulate_dry_run_progress(ui: SimpleUI, total_chunks: int, total_pages: int):
    """Simulate translation progress for dry run."""
    ui.set_totals(total_chunks, total_pages)
    ui.set_phase("Simulando tradução")
    
    for i in range(total_chunks):
        time.sleep(0.2)  # Simulate work
        ui.increment_chunk()
        ui.report_api_usage(500, 300, 100)  # Simulate API usage
        
        # Simulate page progress
        if i % max(1, total_chunks // total_pages) == 0 and ui.completed_pages < total_pages:
            ui.increment_page()
    
    ui.set_phase("Simulação concluída")
    time.sleep(0.5)