"""OpenAI API integration with retries and backoff for translation."""

import os
import time
import random
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .utils import console, log_error, log_warning, log_info
from .logger import get_logger


# Enhanced system prompt for translation (optimized for prompt caching ≥ 1024 tokens)
TRANSLATION_SYSTEM_PROMPT = """Você é um tradutor técnico profissional especializado em documentação científica e técnica. Sua missão é traduzir conteúdo do inglês para o português brasileiro (pt-BR) mantendo precisão, clareza e estrutura original.

## DIRETRIZES FUNDAMENTAIS DE TRADUÇÃO

### 1. Preservação de Estrutura e Formatação
- Mantenha EXATAMENTE a hierarquia de títulos: #, ##, ###, ####, #####, ######
- Preserve todas as listas numeradas (1., 2., 3.) e com marcadores (-, *, +)
- Mantenha tabelas em markdown com alinhamento original (|---|, |:---|, |---:|, |:---:|
- Preserve citações em bloco (>) e suas hierarquias
- Mantenha quebras de linha e espaçamento entre parágrafos
- Preserve blocos de código (```) e código inline (`código`)

### 2. Conteúdo Técnico e Código
- NÃO traduza identificadores de programação: variáveis, funções, classes, métodos
- NÃO traduza comandos de sistemas: git, npm, pip, docker, etc.
- NÃO traduza URLs, caminhos de arquivo, ou extensões de arquivo
- NÃO traduza sintaxe de linguagens de programação
- Mantenha comentários de código no idioma original se forem identificadores
- Traduza APENAS comentários explicativos em linguagem natural

### 3. Elementos Acadêmicos e Científicos
- Preserve fórmulas matemáticas e LaTeX: $formula$, $$formula$$, \\(formula\\), \\[formula\\]
- Mantenha referências cruzadas: "Figura 3", "Tabela 2", "Seção 4.1", "Capítulo 5"
- Traduza números por extenso: "Figure 3" → "Figura 3", "Table 2" → "Tabela 2"
- Preserve numeração de equações: (1), (2.1), etc.
- Mantenha citações acadêmicas: [1], [Smith, 2023], (Author, Year)

### 4. Links e Referências
- Mantenha URLs absolutas intactas: https://example.com/path
- Mantenha âncoras markdown intactas: [texto](#ancora)
- Traduza APENAS o texto âncora dos links: [Learn More](url) → [Saiba Mais](url)
- Preserve referências de footnote: [^1], [^nota], etc.
- Traduza o conteúdo das footnotes mantendo os marcadores

### 5. Terminologia Técnica Especializada
- Use terminologia técnica estabelecida em português brasileiro
- Para termos sem tradução consagrada, mantenha o original e adicione tradução entre parênteses na primeira ocorrência
- Mantenha consistência terminológica ao longo do documento
- Use "código-fonte" em vez de "source code"
- Use "banco de dados" em vez de "base de dados"

### 6. Controle de Qualidade e Formato de Saída
- Traduza com proporção 1:1 sem expandir desnecessariamente
- NÃO adicione explicações, comentários ou notas do tradutor
- NÃO inclua meta-comentários sobre o processo de tradução
- Mantenha o comprimento aproximado do texto original
- Preserve o tom e registro do texto original (formal, informal, técnico)
- Output: APENAS o texto traduzido em markdown, nada mais

### 7. Casos Especiais e Exceções
- Para acrônimos: traduza a expansão mas mantenha a sigla original se consagrada
- Para nomes próprios: mantenha no original (pessoas, empresas, produtos)
- Para títulos de obras: traduza se houver versão oficial em português
- Para unidades de medida: mantenha padrões internacionais (kg, m, etc.)

## EXEMPLO DE TRANSFORMAÇÃO:
Input: "# Getting Started\n\nTo install the `package`, run:\n\n```bash\nnpm install package\n```\n\nSee Figure 1 for details."
Output: "# Começando\n\nPara instalar o `package`, execute:\n\n```bash\nnpm install package\n```\n\nVeja a Figura 1 para detalhes."

Processe cada chunk de texto seguindo rigorosamente estas diretrizes. Traduza com precisão, mantenha formatação e estrutura, e produza saída limpa sem comentários adicionais."""


class TranslationClient:
    """OpenAI client with retry logic, rate limiting, and UI status integration."""
    
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, ui_status=None):
        """
        Initialize translation client.
        
        Args:
            model: OpenAI model to use (default from env)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            base_url: Custom API base URL (default: from OPENAI_BASE_URL env var)
            ui_status: UIStatus instance for real-time reporting
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.ui_status = ui_status
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        
        # Retry configuration
        self.max_retries = 5
        self.base_delay = 2.0  # seconds
        self.max_delay = 60.0  # seconds
        self.backoff_multiplier = 2.0
        self.jitter_range = 0.1  # ±10% jitter
    
    def translate_chunk(self, content: str) -> str:
        """
        Translate a single chunk of text.
        
        Args:
            content: Text content to translate
            
        Returns:
            Translated text
            
        Raises:
            Exception: If translation fails after all retries
        """
        if not content or not content.strip():
            return content
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": TRANSLATION_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    temperature=0.1,  # Low temperature for consistent translation
                    max_tokens=None,  # Let it use as many tokens as needed
                )
                
                if response.choices and response.choices[0].message.content:
                    # Log token usage including cached tokens for cost optimization tracking
                    if hasattr(response, 'usage') and response.usage:
                        usage = response.usage
                        cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) if hasattr(usage, 'prompt_tokens_details') else 0
                        
                        # Report to UI status if available
                        if self.ui_status:
                            self.ui_status.report_api_usage(usage.prompt_tokens, usage.completion_tokens, cached_tokens)
                            # Clear any retry status after successful request
                            self.ui_status.clear_retry_status()
                        
                        # Report to logger if available
                        logger = get_logger()
                        if logger:
                            logger.log_api_call(
                                success=True,
                                attempt=attempt + 1,
                                response_time=0.0,  # We don't track this precisely here
                                input_tokens=usage.prompt_tokens,
                                output_tokens=usage.completion_tokens,
                                cached_tokens=cached_tokens
                            )
                        
                        # Legacy logging for backwards compatibility
                        if cached_tokens > 0:
                            log_info(f"Translation tokens - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Cached: {cached_tokens} (50% off!)")
                        else:
                            log_info(f"Translation tokens - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}")
                    
                    return response.choices[0].message.content.strip()
                else:
                    raise Exception("Empty response from OpenAI")
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Determine if this is a retryable error
                retryable_errors = [
                    'rate limit',
                    'timeout',
                    'internal server error',
                    'bad gateway',
                    'service unavailable',
                    'gateway timeout',
                    'connection error',
                    'read timeout'
                ]
                
                is_retryable = any(err in error_str for err in retryable_errors)
                
                if attempt == self.max_retries - 1:
                    # Last attempt failed - report to UI and logger
                    if self.ui_status:
                        self.ui_status.report_retry(str(e), attempt + 1, self.max_retries, 0)
                    
                    logger = get_logger()
                    if logger:
                        logger.log_api_call(success=False, attempt=attempt + 1, response_time=0.0)
                        logger.error(f"Translation failed after {self.max_retries} attempts: {e}")
                    
                    log_error(f"Translation failed after {self.max_retries} attempts: {e}")
                    raise
                
                if not is_retryable:
                    # Non-retryable error (e.g., invalid API key, malformed request)
                    if self.ui_status:
                        self.ui_status.report_retry(str(e), attempt + 1, self.max_retries, 0)
                    
                    logger = get_logger()
                    if logger:
                        logger.log_api_call(success=False, attempt=attempt + 1, response_time=0.0)
                        logger.error(f"Non-retryable translation error: {e}")
                    
                    log_error(f"Non-retryable translation error: {e}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.base_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                
                # Add jitter to avoid thundering herd
                jitter = delay * self.jitter_range * (random.random() * 2 - 1)
                delay += jitter
                
                # Report retry to UI status
                if self.ui_status:
                    self.ui_status.report_retry(str(e), attempt + 1, self.max_retries, delay)
                
                # Report retry to logger
                logger = get_logger()
                if logger:
                    logger.log_retry_attempt(str(e), attempt + 1, self.max_retries, delay)
                
                log_warning(f"Translation attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        raise Exception("Translation failed after all retries")
    
    def test_connection(self) -> bool:
        """
        Test connection to OpenAI API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=5
            )
            return bool(response.choices)
        except Exception as e:
            import traceback
            log_error(f"API connection test failed: {e}")
            log_error(traceback.format_exc())
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the selected model."""
        return {
            'model': self.model,
            'client': 'openai',
            'base_url': getattr(self.client, '_base_url', None)
        }


def create_translator(model: str = None, api_key: str = None, base_url: str = None, ui_status=None) -> TranslationClient:
    """
    Create a translation client with proper error handling and UI integration.
    
    Args:
        model: OpenAI model to use
        api_key: API key (optional, will use env var)
        base_url: Custom base URL (optional, will use env var)
        ui_status: UIStatus instance for real-time reporting
        
    Returns:
        Configured TranslationClient
        
    Raises:
        ValueError: If API key is missing or invalid
    """
    try:
        client = TranslationClient(model=model, api_key=api_key, base_url=base_url, ui_status=ui_status)
        
        # Test the connection
        log_info(f"Testing connection to OpenAI API with model {client.model}...")
        if not client.test_connection():
            raise ValueError("Failed to connect to OpenAI API")
        
        log_info("OpenAI API connection successful")
        return client
        
    except Exception as e:
        log_error(f"Failed to create translation client: {e}")
        raise


def estimate_cost(chunks: list, model: str = "gpt-4o-mini") -> dict:
    """
    Estimate translation cost based on chunk content.
    
    Args:
        chunks: List of text chunks to translate
        model: OpenAI model name
        
    Returns:
        Dictionary with cost estimation
    """
    # Rough token estimation (4 chars ≈ 1 token)
    input_tokens = sum(len(chunk) // 4 for chunk in chunks)
    output_tokens = input_tokens  # Assume 1:1 ratio for translation
    
    # Approximate pricing (as of 2024, subject to change)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1k tokens
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "chunks": len(chunks)
    }


def print_cost_estimate(cost_info: dict):
    """Print formatted cost estimation."""
    console.print(f"\n[bold]Translation Cost Estimate:[/bold]")
    console.print(f"  Model: [cyan]{cost_info['model']}[/cyan]")
    console.print(f"  Chunks: [yellow]{cost_info['chunks']}[/yellow]")
    console.print(f"  Input tokens: [blue]{cost_info['input_tokens']:,}[/blue]")
    console.print(f"  Output tokens: [blue]{cost_info['output_tokens']:,}[/blue]")
    console.print(f"  Estimated cost: [green]${cost_info['total_cost']:.4f}[/green]")
    console.print(f"    (Input: ${cost_info['input_cost']:.4f}, Output: ${cost_info['output_cost']:.4f})")
    console.print()


def validate_api_key(api_key: str = None) -> bool:
    """
    Validate OpenAI API key format.
    
    Args:
        api_key: API key to validate (default: from env var)
        
    Returns:
        True if key format looks valid, False otherwise
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    
    if not key:
        return False
    
    # OpenAI API keys typically start with 'sk-' and are 48+ characters
    if key.startswith('sk-') and len(key) >= 40:
        return True
    
    # Some custom endpoints might use different formats
    if len(key) >= 20:
        return True
    
    return False
