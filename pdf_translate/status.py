"""New status reporting system using rich.status and rich.console."""

import sys
from rich.console import Console
from rich.status import Status
from rich.text import Text

class StatusManager:
    """A context manager for handling a persistent status spinner."""

    def __init__(self):
        self._console = Console(stderr=True, force_terminal=True)
        self._status = Status("", console=self._console)

    def __enter__(self):
        self._status.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._status.stop()
        if exc_type:
            self.log_error(f"Processo finalizado com erro: {exc_val}")
        else:
            self.log_success("Processo concluído com sucesso.")

    def update(self, message: str, style: str = "bold cyan"):
        """Update the status spinner text."""
        self._status.update(Text(message, style=style))

    def log_info(self, message: str):
        """Print an informational message."""
        self._console.print(f"[bold blue]INFO[/bold blue]    | {message}")

    def log_warning(self, message: str):
        """Print a warning message."""
        self._console.print(f"[bold yellow]WARNING[/bold yellow] | {message}")

    def log_error(self, message: str):
        """Print an error message."""
        self._console.print(f"[bold red]ERROR[/bold red]   | {message}")

    def log_success(self, message: str):
        """Print a success message."""
        self._console.print(f"[bold green]SUCCESS[/bold green] | {message}")

    def print_table(self, title: str, data: dict):
        """Prints a simple key-value table."""
        from rich.table import Table
        table = Table(title=title, show_header=False, box=None)
        table.add_column(style="cyan")
        table.add_column()
        for key, value in data.items():
            table.add_row(f"{key}:", str(value))
        self._console.print(table)
