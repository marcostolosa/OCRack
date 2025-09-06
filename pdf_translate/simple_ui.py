"""Enhanced UI with Rich spinners and beautiful progress indicators."""

import sys
import time
import threading
from typing import Optional
from rich.console import Console
from rich.spinner import Spinner
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import os

# Force UTF-8 encoding on Windows
if os.name == 'nt':
    import locale
    os.environ['PYTHONIOENCODING'] = 'utf-8'


class RichSpinner:
    """Rich spinner with beautiful animations and colors."""
    
    def __init__(self):
        # Force UTF-8 console on Windows
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.spinning = False
        self.live = None
        self.current_message = ""
        self.spinner = Spinner("dots", style="cyan")
        self.progress_info = {}
        
    def start(self, message: str = "Working..."):
        """Start the Rich spinner with a beautiful message."""
        self.current_message = message
        self.spinning = True
        
        if self.live is None:
            self.live = Live(self._render(), console=self.console, refresh_per_second=10)
            self.live.start()
        
    def update_message(self, message: str, **progress_info):
        """Update the spinner message with optional progress info."""
        self.current_message = message
        self.progress_info.update(progress_info)
        
        if self.live:
            self.live.update(self._render())
        
    def stop(self, final_message: str = "Done"):
        """Stop the Rich spinner and show final message."""
        self.spinning = False
        
        if self.live:
            self.live.stop()
            self.live = None
            
        # Show final message (Windows compatible - remove Unicode characters)
        clean_message = final_message.replace("✓", ">").replace("✗", "X")
        self.console.print(f"[green]SUCCESS:[/green] {clean_message}")
        
    def _render(self):
        """Render the current spinner state with Rich formatting."""
        # Create a table for organized display
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        
        # Add spinner and main message  
        spinner_text = Text("*", style="cyan")  # Simple spinner character
        message_text = Text(self.current_message, style="white")
        table.add_row(spinner_text, message_text)
        
        # Add progress information if available
        if self.progress_info:
            progress_parts = []
            if 'chunks' in self.progress_info:
                progress_parts.append(f"[blue]Chunks: {self.progress_info['chunks']}[/blue]")
            if 'pages' in self.progress_info:
                progress_parts.append(f"[green]Pages: {self.progress_info['pages']}[/green]")
            if 'tokens' in self.progress_info:
                progress_parts.append(f"[yellow]Tokens: {self.progress_info['tokens']:,}[/yellow]")
            if 'cost' in self.progress_info:
                progress_parts.append(f"[magenta]Cost: ${self.progress_info['cost']:.4f}[/magenta]")
                
            if progress_parts:
                progress_text = Text.from_markup(" | ".join(progress_parts))
                table.add_row("", progress_text)
        
        return table


class SimpleUI:
    """Enhanced UI with Rich spinners and beautiful progress indicators."""
    
    def __init__(self):
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.spinner = RichSpinner()
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
        """Start a translation job with modern, elegant status display."""
        self.job_started = True
        self.total_pages = total_pages
        self.total_chunks = total_chunks
        
        # Modern header with elegant styling
        self.console.print()
        self.console.rule("[bold cyan]OCRack Translation Engine[/bold cyan]", style="cyan")
        
        # Create elegant job information panel
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text
        
        # Job details in organized panels
        pdf_info = f"[bold]Arquivo:[/bold] {pdf_path.split('/')[-1]}\n[dim]Caminho: {pdf_path}[/dim]"
        model_info = f"[bold]Modelo IA:[/bold] [green]{model}[/green]\n[dim]Engine: OpenAI GPT[/dim]"
        pages_info = f"[bold]Páginas:[/bold] [yellow]{total_pages}[/yellow]\n[dim]Para tradução[/dim]"
        chunks_info = f"[bold]Chunks:[/bold] [magenta]{total_chunks if total_chunks > 0 else 'Calculando...'}[/magenta]\n[dim]Processamento inteligente[/dim]"
        
        panels = [
            Panel(pdf_info, title="[cyan]Documento[/cyan]", border_style="cyan"),
            Panel(model_info, title="[green]IA[/green]", border_style="green"),
            Panel(pages_info, title="[yellow]Escopo[/yellow]", border_style="yellow"),
            Panel(chunks_info, title="[magenta]Processamento[/magenta]", border_style="magenta")
        ]
        
        self.console.print(Columns(panels, equal=True, expand=True))
        self.console.print()
        
        # Start processing with descriptive message
        self.spinner.start("Iniciando pipeline de tradução...")
            
    def final_summary(self, success: bool = True) -> dict:
        """Generate detailed final summary with elegant Rich formatting."""
        summary = {
            "success": success,
            "total_chunks": self.completed_chunks,
            "total_pages": self.completed_pages,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }
        
        self.console.print()
        
        if success:
            # Success summary with detailed breakdown
            self.console.rule("[bold green]Tradução Concluída com Sucesso[/bold green]", style="green")
            
            from rich.panel import Panel
            from rich.columns import Columns
            
            # Create summary panels
            completion_info = f"[bold]Chunks:[/bold] [green]{self.completed_chunks}[/green]\n[dim]Blocos traduzidos[/dim]"
            pages_info = f"[bold]Páginas:[/bold] [yellow]{self.completed_pages}[/yellow]\n[dim]Processadas com sucesso[/dim]"
            tokens_info = f"[bold]Tokens:[/bold] [blue]{self.total_tokens:,}[/blue]\n[dim]Processados pela IA[/dim]"
            cost_info = f"[bold]Custo Total:[/bold] [green]${self.total_cost:.4f}[/green]\n[dim]Investimento na tradução[/dim]"
            
            summary_panels = [
                Panel(completion_info, title="[green]Conclusão[/green]", border_style="green"),
                Panel(pages_info, title="[yellow]Progresso[/yellow]", border_style="yellow"),
                Panel(tokens_info, title="[blue]Processamento[/blue]", border_style="blue"),
                Panel(cost_info, title="[green]Financeiro[/green]", border_style="green")
            ]
            
            self.console.print(Columns(summary_panels, equal=True, expand=True))
            self.console.print()
            self.console.print("[bold green]SUCCESS[/bold green] | Documento traduzido e PDF gerado com sucesso!", style="green")
            
        else:
            # Error summary
            self.console.rule("[bold red]Tradução Falhada[/bold red]", style="red")
            self.console.print(f"[bold red]ERROR[/bold red] | Falha após processar {self.completed_chunks} chunks", style="red")
            if self.total_cost > 0:
                self.console.print(f"[dim]Custo parcial: ${self.total_cost:.4f}[/dim]")
            
        return summary
            
    def set_phase(self, phase: str):
        """Set current phase with detailed, user-friendly descriptions."""
        # Map internal phase names to detailed user descriptions
        phase_descriptions = {
            "Analisando PDF": "ANALYSIS | Analisando estrutura do documento PDF...",
            "Detectando capítulos": "CHAPTERS | Detectando capítulos e estrutura do documento...",
            "Extraindo manifesto de imagens": "IMAGES | Extraindo imagens das páginas selecionadas...",
            "Extraindo texto do PDF": "EXTRACT | Extraindo e processando texto das páginas...",
            "Criando chunks inteligentes": "CHUNKING | Dividindo texto em chunks para tradução otimizada...",
            "Inicializando tradutor OpenAI": "AI-SETUP | Conectando com OpenAI GPT para tradução...",
            "Traduzindo chunks": "TRANSLATE | Traduzindo conteúdo (EN→PT-BR) com IA...",
            "Montando documento traduzido": "ASSEMBLY | Montando documento traduzido final...",
            "Gerando PDF via pandoc": "PDF-GEN | Gerando PDF final com formatação e imagens..."
        }
        
        # Use detailed description if available, otherwise use original phase name
        detailed_phase = phase_descriptions.get(phase, f"PROCESS | {phase}...")
        self.current_phase = detailed_phase
        
        # Update spinner with the detailed phase description
        if hasattr(self, 'spinner') and self.spinner.spinning:
            self.spinner.update_message(detailed_phase)
        
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
        """Report detailed API token usage with cost breakdown."""
        self.total_tokens += input_tokens + output_tokens
        
        # Detailed cost calculation for gpt-4o-mini (prices per 1K tokens)
        input_rate = 0.00015  # $0.15 per 1K input tokens  
        output_rate = 0.0006  # $0.60 per 1K output tokens
        
        input_cost = (input_tokens * input_rate) / 1000
        output_cost = (output_tokens * output_rate) / 1000
        total_chunk_cost = input_cost + output_cost
        
        # Cache savings (cached tokens don't cost anything)
        cache_savings = (cached_tokens * input_rate) / 1000 if cached_tokens > 0 else 0
        
        self.total_cost += total_chunk_cost
        
        # Show detailed cost information
        if cached_tokens > 0:
            cache_percent = (cached_tokens / (input_tokens + cached_tokens)) * 100
            self.console.print(f"[dim]COST | Chunk: ${total_chunk_cost:.6f} (Input: ${input_cost:.6f} + Output: ${output_cost:.6f}) | Cache savings: ${cache_savings:.6f} ({cache_percent:.1f}% off!)[/dim]")
        else:
            self.console.print(f"[dim]COST | Chunk: ${total_chunk_cost:.6f} (Input: ${input_cost:.6f} + Output: ${output_cost:.6f})[/dim]")
            
        self._update_display()
        
    def report_retry(self, error: str, attempt: int, max_attempts: int, delay: float):
        """Report retry attempt with Rich formatting."""
        message = f"Tentativa {attempt}/{max_attempts} - {error[:30]}... aguardando {delay:.1f}s"
        self.spinner.update_message(message)
        
        # Also print a warning for the retry
        print_warning(f"Retry {attempt}/{max_attempts}: {error} (waiting {delay:.1f}s)")
        
    def clear_retry_status(self):
        """Clear retry status."""
        self._update_display()
        
    def finish_phase(self, phase_name: str = "", success: bool = True):
        """Finish current phase with Rich formatting."""
        if phase_name:
            if success:
                self.current_phase = f"{phase_name} - Concluido"
            else:
                self.current_phase = f"{phase_name} - Falhou"
        self._update_display()
        
    def _update_display(self):
        """Update the spinner with current progress using Rich formatting."""
        if not hasattr(self, 'spinner'):
            return
            
        # Build progress info dictionary
        progress_info = {}
        
        if self.total_chunks > 0:
            progress_info['chunks'] = f"{self.completed_chunks}/{self.total_chunks}"
            
        if self.total_pages > 0:
            progress_info['pages'] = f"{self.completed_pages}/{self.total_pages}"
            
        if self.total_tokens > 0:
            progress_info['tokens'] = self.total_tokens
            
        if self.total_cost > 0:
            progress_info['cost'] = self.total_cost
            
        message = self.current_phase if self.current_phase else "Processing..."
        
        if self.spinner.spinning:
            self.spinner.update_message(message, **progress_info)
        else:
            self.spinner.start(message)
            if progress_info:
                self.spinner.update_message(message, **progress_info)


# Global console for consistent Rich formatting
_console = Console(force_terminal=True, legacy_windows=False)

def print_info(message: str):
    """Print info message with Rich formatting."""
    _console.print(f"[blue]INFO:[/blue] {message}")
    

def print_warning(message: str):
    """Print warning message with Rich formatting."""
    _console.print(f"[yellow]WARNING:[/yellow] {message}")
    

def print_error(message: str):
    """Print error message with Rich formatting."""
    _console.print(f"[red]ERROR:[/red] {message}")


def print_success(message: str):
    """Print success message with Rich formatting."""
    _console.print(f"[green]SUCCESS:[/green] {message}")


# Simulate dry run progress for testing
def simulate_dry_run_progress(ui: SimpleUI, total_chunks: int, total_pages: int):
    """Simulate translation progress for dry run with beautiful animations."""
    ui.set_totals(total_chunks, total_pages)
    ui.set_phase("Simulando traducao")
    
    for i in range(total_chunks):
        time.sleep(0.3)  # Slightly slower for better visual effect
        ui.increment_chunk()
        ui.report_api_usage(500, 300, 100)  # Simulate API usage
        
        # Update phase with current chunk
        ui.set_phase(f"Processando chunk {i+1}/{total_chunks}")
        
        # Simulate page progress
        if i % max(1, total_chunks // total_pages) == 0 and ui.completed_pages < total_pages:
            ui.increment_page()
    
    ui.set_phase("Simulacao concluida")
    time.sleep(0.8)