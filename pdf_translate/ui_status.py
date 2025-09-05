"""Rich-based UI status system for PDF translation with real-time progress tracking."""

import os
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Deque, Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress, TaskID, TextColumn, BarColumn, 
    MofNCompleteColumn, TimeRemainingColumn, SpinnerColumn
)
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.align import Align

from .pricing import get_model_pricing, calculate_cost


class UIStatus:
    """
    Rich-based status display for PDF translation with comprehensive progress tracking.
    
    Features:
    - Real-time progress bars for chunks and pages
    - Token usage and cost tracking with live updates
    - Phase management (extracting, translating, generating PDF, etc.)
    - Retry/backoff display with countdown
    - ETA calculation with moving average
    - Checkpoint/resume status
    - Interactive keyboard shortcuts
    - Graceful cancellation handling
    """
    
    def __init__(self, log_level: str = "INFO", enable_ui: bool = True):
        """
        Initialize the UI status system.
        
        Args:
            log_level: Logging level (INFO, DEBUG, WARN, ERROR)
            enable_ui: Whether to show Rich UI (False for tests/batch jobs)
        """
        self.enable_ui = enable_ui and os.getenv('TERM') != 'dumb'
        self.log_level = log_level.upper()
        self.console = Console()
        
        # Job information
        self.pdf_path: Optional[str] = None
        self.model_name: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.total_pages: int = 0
        self.total_chapters: int = 0
        self.done_pages: int = 0
        
        # Chapter tracking
        self.current_chapter_idx: int = 0
        self.current_chapter_title: str = ""
        self.current_chapter_pages: tuple = (0, 0)
        
        # Chunk progress
        self.total_chunks: int = 0
        self.done_chunks: int = 0
        self.chunk_progress: Optional[Progress] = None
        self.chunk_task: Optional[TaskID] = None
        
        # Page progress  
        self.page_progress: Optional[Progress] = None
        self.page_task: Optional[TaskID] = None
        
        # Token tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_requests: int = 0
        self.failed_requests: int = 0
        
        # Timing and ETA
        self.chunk_times: Deque[float] = deque(maxlen=10)  # Last 10 chunk times
        self.eta_seconds: Optional[float] = None
        
        # Phase management
        self.current_phase: str = "Inicializando"
        self.phase_start_time: Optional[datetime] = None
        self.completed_phases: Dict[str, float] = {}  # phase_name -> duration_seconds
        
        # Retry tracking
        self.last_error: str = ""
        self.retry_attempt: int = 0
        self.retry_max: int = 0
        self.retry_wait_seconds: float = 0
        self.retry_countdown_start: Optional[float] = None
        
        # Checkpoint info
        self.resume_active: bool = False
        self.checkpoint_path: str = ""
        self.skipped_chunks: int = 0
        
        # UI components
        self.layout: Optional[Layout] = None
        self.live: Optional[Live] = None
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
        
        # Keyboard interrupt handling
        self._original_sigint_handler = None
        self._canceling = False
        
        # Status flags
        self._started = False
        self._finished = False
        
    def start_job(self, pdf_path: str, model: str, total_pages: int, total_chapters: int):
        """
        Start a new translation job.
        
        Args:
            pdf_path: Path to the PDF being translated
            model: OpenAI model name
            total_pages: Total number of pages to translate
            total_chapters: Total number of chapters detected
        """
        self.pdf_path = pdf_path
        self.model_name = model
        self.total_pages = total_pages
        self.total_chapters = total_chapters
        self.start_time = datetime.now()
        self._started = True
        
        if not self.enable_ui:
            return
            
        # Setup keyboard interrupt handler
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._handle_keyboard_interrupt)
        
        # Create progress bars
        self.chunk_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Chunks"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False
        )
        
        self.page_progress = Progress(
            TextColumn("[bold green]Páginas"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            expand=False
        )
        
        # Initialize layout
        self._create_layout()
        
        # Start live display
        self.live = Live(self.layout, console=self.console, refresh_per_second=2)
        self.live.start()
        
        # Start update thread for countdown timers
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
    def set_chapter(self, idx: int, total: int, title: str, page_start: int, page_end: int):
        """
        Set current chapter information.
        
        Args:
            idx: Current chapter index (0-based)
            total: Total number of chapters
            title: Chapter title
            page_start: Starting page number
            page_end: Ending page number
        """
        self.current_chapter_idx = idx
        self.total_chapters = total
        self.current_chapter_title = title
        self.current_chapter_pages = (page_start, page_end)
        
    def update_pages_done(self, done_pages: int):
        """Update number of pages completed."""
        self.done_pages = done_pages
        if self.page_task is not None:
            self.page_progress.update(self.page_task, completed=done_pages)
            
    def start_chunk_progress(self, total_chunks: int):
        """
        Initialize chunk progress tracking.
        
        Args:
            total_chunks: Total number of chunks to process
        """
        self.total_chunks = total_chunks
        self.done_chunks = 0
        
        if self.enable_ui and self.chunk_progress:
            self.chunk_task = self.chunk_progress.add_task(
                "Traduzindo chunks", 
                total=total_chunks
            )
            
        if self.enable_ui and self.page_progress:
            self.page_task = self.page_progress.add_task(
                "Páginas processadas",
                total=self.total_pages
            )
            
    def advance_chunk(self, processing_time: float = 0):
        """
        Advance chunk progress by one.
        
        Args:
            processing_time: Time taken to process this chunk (for ETA calculation)
        """
        self.done_chunks += 1
        
        if processing_time > 0:
            self.chunk_times.append(processing_time)
            self._update_eta()
            
        if self.chunk_task is not None:
            self.chunk_progress.update(self.chunk_task, advance=1)
            
    def _update_eta(self):
        """Update ETA based on recent chunk processing times."""
        if not self.chunk_times or self.done_chunks >= self.total_chunks:
            self.eta_seconds = None
            return
            
        avg_time = sum(self.chunk_times) / len(self.chunk_times)
        remaining_chunks = self.total_chunks - self.done_chunks
        self.eta_seconds = avg_time * remaining_chunks
        
    def report_api_usage(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """
        Report API token usage.
        
        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens generated  
            cached_tokens: Cached input tokens (50% discount)
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens
        self.total_requests += 1
        
    def report_retry(self, error_msg: str, attempt: int, max_attempts: int, wait_seconds: float):
        """
        Report a retry attempt with backoff.
        
        Args:
            error_msg: Error message that caused the retry
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            wait_seconds: Seconds to wait before retry
        """
        self.last_error = error_msg
        self.retry_attempt = attempt
        self.retry_max = max_attempts
        self.retry_wait_seconds = wait_seconds
        self.retry_countdown_start = time.time()
        
        if attempt > max_attempts:
            self.failed_requests += 1
            
    def clear_retry_status(self):
        """Clear retry status after successful request."""
        self.last_error = ""
        self.retry_attempt = 0
        self.retry_countdown_start = None
        
    def set_phase(self, phase_name: str):
        """
        Set current processing phase.
        
        Args:
            phase_name: Name of the phase (e.g., "Extraindo texto", "Traduzindo")
        """
        # Complete previous phase
        if self.current_phase != "Inicializando" and self.phase_start_time:
            duration = (datetime.now() - self.phase_start_time).total_seconds()
            self.completed_phases[self.current_phase] = duration
            
        self.current_phase = phase_name
        self.phase_start_time = datetime.now()
        
    def finish_phase(self, phase_name: str, success: bool = True):
        """
        Mark a phase as completed.
        
        Args:
            phase_name: Name of the completed phase
            success: Whether the phase completed successfully
        """
        if self.phase_start_time:
            duration = (datetime.now() - self.phase_start_time).total_seconds()
            status_suffix = " ✓" if success else " ✗"
            self.completed_phases[phase_name + status_suffix] = duration
            
    def set_checkpoint_info(self, resume_active: bool, checkpoint_path: str = "", skipped_chunks: int = 0):
        """
        Set checkpoint/resume information.
        
        Args:
            resume_active: Whether resume mode is active
            checkpoint_path: Path to checkpoint file
            skipped_chunks: Number of chunks skipped due to checkpoint
        """
        self.resume_active = resume_active
        self.checkpoint_path = checkpoint_path
        self.skipped_chunks = skipped_chunks
        
    def _create_layout(self):
        """Create the Rich layout structure."""
        self.layout = Layout()
        
        # Split into header, body, footer
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into left and right columns
        self.layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Split left column
        self.layout["left"].split_column(
            Layout(name="chapter_info", size=4),
            Layout(name="progress", size=8),
            Layout(name="tokens", size=6)
        )
        
        # Split right column
        self.layout["right"].split_column(
            Layout(name="phase", size=4),
            Layout(name="retries", size=4),
            Layout(name="checkpoint", size=4)
        )
        
    def _get_header_panel(self) -> Panel:
        """Generate header panel with job information."""
        if not self.pdf_path:
            return Panel("Inicializando...", style="bold blue")
            
        filename = Path(self.pdf_path).name
        start_time = self.start_time.strftime("%H:%M:%S") if self.start_time else "?"
        
        header_text = f"📄 [bold]{filename}[/bold] | 🤖 {self.model_name} | ⏰ Iniciado às {start_time}"
        
        return Panel(header_text, style="bold blue")
        
    def _get_chapter_panel(self) -> Panel:
        """Generate chapter/pages information panel."""
        if self.total_chapters > 0:
            chapter_text = f"📖 Capítulo {self.current_chapter_idx + 1}/{self.total_chapters}\n"
            chapter_text += f"   [italic]{self.current_chapter_title}[/italic]\n"
            chapter_text += f"   📄 Páginas {self.current_chapter_pages[0]}-{self.current_chapter_pages[1]}"
        else:
            chapter_text = f"📄 Páginas {self.done_pages}/{self.total_pages} processadas"
            
        return Panel(chapter_text, title="Capítulo/Páginas", style="green")
        
    def _get_progress_panel(self) -> Panel:
        """Generate progress bars panel."""
        if not self.chunk_progress or not self.page_progress:
            return Panel("Aguardando início...", title="Progresso")
            
        # Create a combined renderable with both progress bars
        progress_group = []
        
        if self.chunk_task is not None:
            progress_group.append(self.chunk_progress)
            
        if self.page_task is not None:
            progress_group.append(self.page_progress)
            
        # Add ETA info
        eta_text = ""
        if self.eta_seconds and self.eta_seconds > 0:
            if self.eta_seconds < 60:
                eta_text = f"\n⏱️  ETA: {self.eta_seconds:.0f}s"
            elif self.eta_seconds < 3600:
                eta_text = f"\n⏱️  ETA: {self.eta_seconds/60:.1f}min"
            else:
                eta_text = f"\n⏱️  ETA: {self.eta_seconds/3600:.1f}h"
                
        # Add speed info
        speed_text = ""
        if len(self.chunk_times) >= 3:
            avg_time = sum(self.chunk_times) / len(self.chunk_times)
            chunks_per_min = 60 / avg_time if avg_time > 0 else 0
            speed_text = f"\n⚡ Velocidade: {chunks_per_min:.1f} chunks/min"
            
        combined_text = Text()
        if progress_group:
            combined_text.append("\n")
        combined_text.append(eta_text + speed_text)
            
        return Panel(
            *progress_group, 
            combined_text,
            title="Progresso",
            style="yellow"
        )
        
    def _get_tokens_panel(self) -> Panel:
        """Generate tokens and cost information panel."""
        # Calculate cost
        pricing = get_model_pricing(self.model_name) if self.model_name else None
        cost = calculate_cost(
            self.total_input_tokens,
            self.total_output_tokens, 
            self.total_cached_tokens,
            pricing
        ) if pricing else 0
        
        tokens_text = f"📥 Input: [cyan]{self.total_input_tokens:,}[/cyan]\n"
        tokens_text += f"📤 Output: [yellow]{self.total_output_tokens:,}[/yellow]\n"
        
        if self.total_cached_tokens > 0:
            savings_pct = (self.total_cached_tokens / (self.total_input_tokens + self.total_cached_tokens)) * 100
            tokens_text += f"🎯 Cached: [green]{self.total_cached_tokens:,}[/green] ({savings_pct:.1f}% off!)\n"
            
        tokens_text += f"💰 Custo: [green]${cost:.4f}[/green]\n"
        tokens_text += f"📡 Requests: {self.total_requests}"
        
        if self.failed_requests > 0:
            tokens_text += f" ([red]{self.failed_requests} falhas[/red])"
            
        return Panel(tokens_text, title="Tokens & Custo", style="cyan")
        
    def _get_phase_panel(self) -> Panel:
        """Generate current phase panel."""
        phase_text = f"🔄 [bold]{self.current_phase}[/bold]\n"
        
        # Show completed phases
        if self.completed_phases:
            phase_text += "\n✅ Concluídas:\n"
            for phase_name, duration in list(self.completed_phases.items())[-3:]:  # Last 3
                phase_text += f"   {phase_name}: {duration:.1f}s\n"
                
        return Panel(phase_text.rstrip(), title="Estado Atual", style="blue")
        
    def _get_retry_panel(self) -> Panel:
        """Generate retry/backoff information panel."""
        if not self.last_error:
            return Panel("✅ Sem erros", title="Retries & Backoff", style="green")
            
        retry_text = f"❌ [red]{self.last_error[:50]}...[/red]\n"
        retry_text += f"🔄 Tentativa {self.retry_attempt}/{self.retry_max}\n"
        
        # Show countdown if waiting
        if self.retry_countdown_start:
            elapsed = time.time() - self.retry_countdown_start
            remaining = max(0, self.retry_wait_seconds - elapsed)
            if remaining > 0:
                retry_text += f"⏳ Aguardando: {remaining:.1f}s"
            else:
                retry_text += "🚀 Tentando novamente..."
                
        return Panel(retry_text, title="Retries & Backoff", style="red")
        
    def _get_checkpoint_panel(self) -> Panel:
        """Generate checkpoint information panel."""
        if not self.resume_active:
            return Panel("💾 Checkpoint desabilitado", title="Checkpoint", style="dim")
            
        checkpoint_text = f"📁 [green]Resume ativo[/green]\n"
        if self.checkpoint_path:
            filename = Path(self.checkpoint_path).name
            checkpoint_text += f"📄 {filename}\n"
        if self.skipped_chunks > 0:
            checkpoint_text += f"⏭️  {self.skipped_chunks} chunks pulados"
            
        return Panel(checkpoint_text, title="Checkpoint", style="green")
        
    def _get_footer_panel(self) -> Panel:
        """Generate footer with keyboard shortcuts."""
        footer_text = "⌨️  [dim]Ctrl+C: Cancelar com segurança | Q: Sair | Logs salvos em logs/[/dim]"
        return Panel(footer_text, style="dim")
        
    def _update_layout(self):
        """Update all layout panels."""
        if not self.layout:
            return
            
        try:
            self.layout["header"].update(self._get_header_panel())
            self.layout["chapter_info"].update(self._get_chapter_panel())
            self.layout["progress"].update(self._get_progress_panel())
            self.layout["tokens"].update(self._get_tokens_panel())
            self.layout["phase"].update(self._get_phase_panel())
            self.layout["retries"].update(self._get_retry_panel())
            self.layout["checkpoint"].update(self._get_checkpoint_panel())
            self.layout["footer"].update(self._get_footer_panel())
        except Exception:
            # Silently handle layout update errors (e.g., terminal resize)
            pass
            
    def _update_loop(self):
        """Background thread for updating dynamic content."""
        while not self._stop_event.is_set():
            if self.enable_ui and self.layout:
                self._update_layout()
            time.sleep(0.5)  # Update every 500ms
            
    def _handle_keyboard_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if self._canceling:
            # Force exit on second Ctrl+C
            if self.live:
                self.live.stop()
            os._exit(1)
            
        self._canceling = True
        if self.enable_ui:
            self.set_phase("🛑 Cancelando com segurança...")
            self.console.print("\n[yellow]Cancelando com segurança... (Ctrl+C novamente para forçar saída)[/yellow]")
            
        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        raise KeyboardInterrupt()
        
    def final_summary(self, success: bool = True) -> Dict[str, Any]:
        """
        Generate final summary and stop UI.
        
        Args:
            success: Whether the job completed successfully
            
        Returns:
            Dictionary with comprehensive job metrics
        """
        self._finished = True
        self._stop_event.set()
        
        # Calculate total time
        total_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        # Complete current phase
        if self.current_phase and self.phase_start_time:
            phase_duration = (datetime.now() - self.phase_start_time).total_seconds()
            phase_status = " ✓" if success else " ✗"
            self.completed_phases[self.current_phase + phase_status] = phase_duration
            
        # Calculate final cost
        pricing = get_model_pricing(self.model_name) if self.model_name else None
        final_cost = calculate_cost(
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cached_tokens,
            pricing
        ) if pricing else 0
        
        # Create summary
        summary = {
            # Job info
            "success": success,
            "pdf_path": self.pdf_path,
            "model": self.model_name,
            "start_time": self.start_time,
            "total_duration_seconds": total_time,
            
            # Processing stats
            "total_pages": self.total_pages,
            "pages_completed": self.done_pages,
            "total_chapters": self.total_chapters,
            "total_chunks": self.total_chunks,
            "chunks_completed": self.done_chunks,
            
            # Token usage
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "final_cost": final_cost,
            
            # Request stats
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(1, self.total_requests),
            
            # Performance
            "avg_chunk_time": sum(self.chunk_times) / len(self.chunk_times) if self.chunk_times else 0,
            "chunks_per_minute": len(self.chunk_times) * 60 / total_time if total_time > 0 and self.chunk_times else 0,
            
            # Phases
            "completed_phases": dict(self.completed_phases),
            
            # Resume info
            "resumed_from_checkpoint": self.resume_active,
            "skipped_chunks": self.skipped_chunks,
        }
        
        # Stop UI
        if self.enable_ui:
            if self.live:
                # Show final summary table
                self._show_final_table(summary)
                time.sleep(2)  # Let user see the summary
                self.live.stop()
                
        # Join update thread
        if self._update_thread:
            self._update_thread.join(timeout=1)
            
        return summary
        
    def _show_final_table(self, summary: Dict[str, Any]):
        """Display final summary as a Rich table."""
        table = Table(title="📊 Resumo Final da Tradução", show_header=True)
        table.add_column("Métrica", style="cyan", no_wrap=True)
        table.add_column("Valor", style="green")
        
        # Status
        status_emoji = "✅" if summary["success"] else "❌"
        table.add_row("Status", f"{status_emoji} {'Concluído' if summary['success'] else 'Falhou'}")
        
        # Time
        duration = summary["total_duration_seconds"]
        if duration < 60:
            duration_str = f"{duration:.1f}s"
        elif duration < 3600:
            duration_str = f"{duration/60:.1f}min"
        else:
            duration_str = f"{duration/3600:.1f}h"
        table.add_row("Tempo Total", duration_str)
        
        # Processing
        table.add_row("Páginas", f"{summary['pages_completed']}/{summary['total_pages']}")
        table.add_row("Chunks", f"{summary['chunks_completed']}/{summary['total_chunks']}")
        
        # Tokens
        table.add_row("Tokens (Input)", f"{summary['total_input_tokens']:,}")
        table.add_row("Tokens (Output)", f"{summary['total_output_tokens']:,}")
        if summary['total_cached_tokens'] > 0:
            table.add_row("Tokens (Cached)", f"{summary['total_cached_tokens']:,} 🎯")
        
        # Cost
        table.add_row("Custo Total", f"${summary['final_cost']:.4f}")
        
        # Performance  
        table.add_row("Velocidade", f"{summary['chunks_per_minute']:.1f} chunks/min")
        
        # Success rate
        table.add_row("Taxa de Sucesso", f"{summary['success_rate']:.1%}")
        
        if summary.get('resumed_from_checkpoint'):
            table.add_row("Checkpoint", f"✅ {summary['skipped_chunks']} chunks pulados")
            
        self.layout["body"].update(Align.center(table))
        
    def stop(self):
        """Stop the UI system gracefully."""
        if not self._finished:
            self.final_summary(success=False)
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        success = exc_type is None
        self.final_summary(success=success)
        
        # Re-raise KeyboardInterrupt but not other exceptions during cleanup
        if exc_type is KeyboardInterrupt:
            return False  # Let the KeyboardInterrupt propagate
        return True  # Suppress other exceptions during cleanup


# Convenience function for simple usage
def create_ui_status(log_level: str = "INFO", enable_ui: bool = True) -> UIStatus:
    """
    Create a UIStatus instance with standard configuration.
    
    Args:
        log_level: Logging level
        enable_ui: Whether to enable Rich UI
        
    Returns:
        Configured UIStatus instance
    """
    return UIStatus(log_level=log_level, enable_ui=enable_ui)