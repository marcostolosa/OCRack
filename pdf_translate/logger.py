"""Logging infrastructure for PDF translation with file and console output."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler


class TranslationLogger:
    """
    Centralized logging system for PDF translation.
    
    Features:
    - File logging with rotation
    - Rich console logging with colors
    - Structured log messages
    - Performance metrics logging
    - Error tracking and reporting
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the logging system.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARN, ERROR)
            log_dir: Directory for log files (default: logs/)
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.console = Console()
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(exist_ok=True)
        
        # Generate unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"translate_{timestamp}.log"
        self.log_file_path = self.log_dir / self.log_filename
        
        # Set up loggers
        self._setup_main_logger(max_file_size, backup_count)
        self._setup_performance_logger()
        self._setup_error_logger()
        
        # Track session metrics
        self.session_start = datetime.now()
        self.total_warnings = 0
        self.total_errors = 0
        self.api_calls = 0
        self.failed_api_calls = 0
        
    def _setup_main_logger(self, max_file_size: int, backup_count: int):
        """Set up the main application logger."""
        self.logger = logging.getLogger("pdf_translate")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        if self.enable_file_logging:
            file_handler = RotatingFileHandler(
                self.log_file_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler with Rich
        if self.enable_console_logging and os.getenv('TERM') != 'dumb':
            console_handler = RichHandler(
                console=self.console,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=self.log_level <= logging.DEBUG
            )
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)
            
    def _setup_performance_logger(self):
        """Set up performance metrics logger."""
        self.perf_logger = logging.getLogger("pdf_translate.performance")
        self.perf_logger.setLevel(logging.INFO)
        
        if self.enable_file_logging:
            perf_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            perf_handler = logging.FileHandler(perf_file, encoding='utf-8')
            perf_formatter = logging.Formatter(
                '%(asctime)s | PERF | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            perf_handler.setFormatter(perf_formatter)
            self.perf_logger.addHandler(perf_handler)
            
    def _setup_error_logger(self):
        """Set up error tracking logger."""
        self.error_logger = logging.getLogger("pdf_translate.errors")
        self.error_logger.setLevel(logging.WARNING)
        
        if self.enable_file_logging:
            error_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            error_handler = logging.FileHandler(error_file, encoding='utf-8')
            error_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s\n%(pathname)s:%(lineno)d in %(funcName)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            error_handler.setFormatter(error_formatter)
            self.error_logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.total_warnings += 1
        self.logger.warning(message, **kwargs)
        self.error_logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.total_errors += 1
        self.logger.error(message, **kwargs)
        self.error_logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical error message."""
        self.total_errors += 1
        self.logger.critical(message, **kwargs)
        self.error_logger.critical(message, **kwargs)
        
    def log_job_start(self, pdf_path: str, model: str, total_pages: int, total_chunks: int):
        """Log the start of a translation job."""
        self.info(f"=== TRANSLATION JOB STARTED ===")
        self.info(f"PDF: {pdf_path}")
        self.info(f"Model: {model}")
        self.info(f"Total pages: {total_pages}")
        self.info(f"Total chunks: {total_chunks}")
        self.info(f"Log file: {self.log_file_path}")
        
    def log_job_complete(self, success: bool, summary: dict):
        """Log job completion with summary."""
        status = "COMPLETED" if success else "FAILED"
        self.info(f"=== TRANSLATION JOB {status} ===")
        
        # Log summary metrics
        duration = summary.get('total_duration_seconds', 0)
        self.info(f"Duration: {duration:.1f}s")
        self.info(f"Pages: {summary.get('pages_completed', 0)}/{summary.get('total_pages', 0)}")
        self.info(f"Chunks: {summary.get('chunks_completed', 0)}/{summary.get('total_chunks', 0)}")
        self.info(f"Total tokens: {summary.get('total_tokens', 0):,}")
        self.info(f"Final cost: ${summary.get('final_cost', 0):.4f}")
        self.info(f"API calls: {summary.get('total_requests', 0)} ({summary.get('failed_requests', 0)} failed)")
        
        # Log to performance logger
        self._log_performance_summary(summary)
        
    def log_phase_start(self, phase_name: str):
        """Log the start of a processing phase."""
        self.info(f"Phase started: {phase_name}")
        
    def log_phase_complete(self, phase_name: str, duration: float, success: bool = True):
        """Log the completion of a processing phase."""
        status = "completed" if success else "failed"
        self.info(f"Phase {status}: {phase_name} ({duration:.1f}s)")
        self.perf_logger.info(f"PHASE | {phase_name} | {duration:.3f}s | {'SUCCESS' if success else 'FAILED'}")
        
    def log_chunk_processed(self, chunk_idx: int, total_chunks: int, processing_time: float, 
                           input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """Log processing of a single chunk."""
        self.debug(f"Chunk {chunk_idx + 1}/{total_chunks} processed in {processing_time:.2f}s")
        self.perf_logger.info(
            f"CHUNK | {chunk_idx + 1}/{total_chunks} | "
            f"time={processing_time:.3f}s | "
            f"tokens_in={input_tokens} | "
            f"tokens_out={output_tokens} | "
            f"cached={cached_tokens}"
        )
        
    def log_api_call(self, success: bool, attempt: int, response_time: float, 
                    input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0):
        """Log an API call."""
        self.api_calls += 1
        if not success:
            self.failed_api_calls += 1
            
        status = "SUCCESS" if success else "FAILED"
        self.debug(f"API call {status} (attempt {attempt}) in {response_time:.2f}s")
        
        self.perf_logger.info(
            f"API | {status} | attempt={attempt} | "
            f"response_time={response_time:.3f}s | "
            f"tokens_in={input_tokens} | "
            f"tokens_out={output_tokens} | "
            f"cached={cached_tokens}"
        )
        
    def log_retry_attempt(self, error_msg: str, attempt: int, max_attempts: int, 
                         wait_time: float):
        """Log a retry attempt."""
        self.warning(f"Retry {attempt}/{max_attempts}: {error_msg} (waiting {wait_time:.1f}s)")
        
    def log_checkpoint_save(self, chunk_idx: int, total_chunks: int):
        """Log checkpoint save."""
        self.debug(f"Checkpoint saved: {chunk_idx}/{total_chunks} chunks completed")
        
    def log_checkpoint_load(self, resumed_chunk: int, skipped_chunks: int):
        """Log checkpoint load."""
        self.info(f"Resuming from checkpoint: skipping {skipped_chunks} chunks, starting at chunk {resumed_chunk}")
        
    def log_cost_update(self, total_cost: float, input_tokens: int, output_tokens: int, 
                       cached_tokens: int, model: str):
        """Log cost information update."""
        cache_savings = cached_tokens / (input_tokens + cached_tokens) * 100 if (input_tokens + cached_tokens) > 0 else 0
        
        self.debug(f"Cost update: ${total_cost:.4f} | Model: {model}")
        if cached_tokens > 0:
            self.debug(f"Cache savings: {cache_savings:.1f}% ({cached_tokens:,} cached tokens)")
            
    def _log_performance_summary(self, summary: dict):
        """Log detailed performance summary."""
        duration = summary.get('total_duration_seconds', 0)
        chunks_per_min = summary.get('chunks_per_minute', 0)
        success_rate = summary.get('success_rate', 0)
        
        self.perf_logger.info("=== PERFORMANCE SUMMARY ===")
        self.perf_logger.info(f"Total duration: {duration:.1f}s")
        self.perf_logger.info(f"Chunks per minute: {chunks_per_min:.2f}")
        self.perf_logger.info(f"API success rate: {success_rate:.1%}")
        self.perf_logger.info(f"Total warnings: {self.total_warnings}")
        self.perf_logger.info(f"Total errors: {self.total_errors}")
        
        # Log phase breakdown
        phases = summary.get('completed_phases', {})
        for phase_name, phase_duration in phases.items():
            percentage = (phase_duration / duration * 100) if duration > 0 else 0
            self.perf_logger.info(f"Phase '{phase_name}': {phase_duration:.1f}s ({percentage:.1f}%)")
            
    def get_session_summary(self) -> dict:
        """Get summary of logging session."""
        duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "session_duration": duration,
            "log_file": str(self.log_file_path),
            "total_warnings": self.total_warnings,
            "total_errors": self.total_errors,
            "api_calls": self.api_calls,
            "failed_api_calls": self.failed_api_calls,
            "log_level": logging.getLevelName(self.log_level)
        }
        
    def close(self):
        """Close all logging handlers."""
        # Log session summary
        summary = self.get_session_summary()
        self.info("=== LOGGING SESSION ENDED ===")
        self.info(f"Session duration: {summary['session_duration']:.1f}s")
        self.info(f"Warnings: {summary['total_warnings']}, Errors: {summary['total_errors']}")
        self.info(f"Log file: {summary['log_file']}")
        
        # Close all handlers
        for logger in [self.logger, self.perf_logger, self.error_logger]:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


# Global logger instance
_logger_instance: Optional[TranslationLogger] = None


def get_logger() -> Optional[TranslationLogger]:
    """Get the global logger instance."""
    return _logger_instance


def setup_logger(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> TranslationLogger:
    """
    Set up the global logger instance.
    
    Args:
        log_level: Logging level
        log_dir: Log directory
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        
    Returns:
        Configured logger instance
    """
    global _logger_instance
    
    _logger_instance = TranslationLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging
    )
    
    return _logger_instance


def cleanup_logger():
    """Clean up the global logger instance."""
    global _logger_instance
    
    if _logger_instance:
        _logger_instance.close()
        _logger_instance = None


# Convenience functions for common logging operations
def log_info(message: str, **kwargs):
    """Log an info message."""
    if _logger_instance:
        _logger_instance.info(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log a debug message.""" 
    if _logger_instance:
        _logger_instance.debug(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log a warning message."""
    if _logger_instance:
        _logger_instance.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log an error message."""
    if _logger_instance:
        _logger_instance.error(message, **kwargs)