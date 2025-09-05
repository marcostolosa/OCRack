"""Cost calculation and pricing models for OpenAI API usage."""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for an OpenAI model."""
    
    model_name: str
    price_per_1k_input: float      # Price per 1K input tokens
    price_per_1k_output: float     # Price per 1K output tokens
    price_per_1k_cached_input: float  # Price per 1K cached input tokens (usually 50% off)
    supports_caching: bool = True   # Whether model supports prompt caching
    

# Updated pricing as of 2024 (prices in USD per 1K tokens)
# NOTE: These prices change frequently - update as needed
MODEL_PRICING = {
    # GPT-4o models
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        price_per_1k_input=0.005,
        price_per_1k_output=0.015,
        price_per_1k_cached_input=0.0025,  # 50% off input
        supports_caching=True
    ),
    
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini", 
        price_per_1k_input=0.00015,
        price_per_1k_output=0.0006,
        price_per_1k_cached_input=0.000075,  # 50% off input
        supports_caching=True
    ),
    
    # GPT-4 models
    "gpt-4": ModelPricing(
        model_name="gpt-4",
        price_per_1k_input=0.03,
        price_per_1k_output=0.06,
        price_per_1k_cached_input=0.015,
        supports_caching=False
    ),
    
    "gpt-4-turbo": ModelPricing(
        model_name="gpt-4-turbo",
        price_per_1k_input=0.01,
        price_per_1k_output=0.03,
        price_per_1k_cached_input=0.005,
        supports_caching=True
    ),
    
    "gpt-4-turbo-preview": ModelPricing(
        model_name="gpt-4-turbo-preview",
        price_per_1k_input=0.01,
        price_per_1k_output=0.03,
        price_per_1k_cached_input=0.005,
        supports_caching=True
    ),
    
    # GPT-3.5 models
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        price_per_1k_input=0.0005,
        price_per_1k_output=0.0015,
        price_per_1k_cached_input=0.00025,
        supports_caching=False
    ),
    
    "gpt-3.5-turbo-16k": ModelPricing(
        model_name="gpt-3.5-turbo-16k",
        price_per_1k_input=0.003,
        price_per_1k_output=0.004,
        price_per_1k_cached_input=0.0015,
        supports_caching=False
    ),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """
    Get pricing information for a specific model.
    
    Args:
        model_name: Name of the OpenAI model
        
    Returns:
        ModelPricing object if model is known, None otherwise
    """
    # Handle versioned model names (e.g., "gpt-4o-2024-08-06")
    base_model = model_name.lower()
    
    # Try exact match first
    if base_model in MODEL_PRICING:
        return MODEL_PRICING[base_model]
    
    # Try to match base model name for versioned models
    for known_model in MODEL_PRICING:
        if base_model.startswith(known_model):
            return MODEL_PRICING[known_model]
    
    # Default to gpt-4o-mini for unknown models
    return MODEL_PRICING.get("gpt-4o-mini")


def calculate_cost(
    input_tokens: int,
    output_tokens: int, 
    cached_tokens: int,
    pricing: Optional[ModelPricing]
) -> float:
    """
    Calculate total cost for API usage.
    
    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cached_tokens: Number of cached input tokens (get 50% discount)
        pricing: Model pricing information
        
    Returns:
        Total cost in USD
    """
    if not pricing:
        return 0.0
    
    # Regular input tokens (full price)
    regular_input_tokens = max(0, input_tokens - cached_tokens)
    
    # Calculate costs
    regular_input_cost = (regular_input_tokens / 1000) * pricing.price_per_1k_input
    cached_input_cost = (cached_tokens / 1000) * pricing.price_per_1k_cached_input
    output_cost = (output_tokens / 1000) * pricing.price_per_1k_output
    
    return regular_input_cost + cached_input_cost + output_cost


def estimate_cost_for_text(
    text: str,
    model_name: str,
    assume_cached_ratio: float = 0.3
) -> Dict[str, float]:
    """
    Estimate cost for translating a given text.
    
    Args:
        text: Text to be translated
        model_name: OpenAI model name
        assume_cached_ratio: Assumed ratio of tokens that will be cached (0.0-1.0)
        
    Returns:
        Dictionary with cost breakdown
    """
    pricing = get_model_pricing(model_name)
    if not pricing:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_cost": 0.0,
            "cost_breakdown": {}
        }
    
    # Rough token estimation (4 chars ≈ 1 token)
    estimated_input_tokens = len(text) // 4
    estimated_output_tokens = int(estimated_input_tokens * 1.1)  # Translation might be slightly longer
    
    # Account for system prompt tokens (our enhanced prompt is ~1500 tokens)
    system_prompt_tokens = 1500
    total_input_tokens = estimated_input_tokens + system_prompt_tokens
    
    # Estimate cached tokens
    if pricing.supports_caching:
        # System prompt is always cached after first request
        cached_tokens = int(system_prompt_tokens + estimated_input_tokens * assume_cached_ratio)
    else:
        cached_tokens = 0
    
    total_cost = calculate_cost(
        total_input_tokens,
        estimated_output_tokens,
        cached_tokens,
        pricing
    )
    
    # Cost breakdown
    regular_input_tokens = total_input_tokens - cached_tokens
    cost_breakdown = {
        "input_cost": (regular_input_tokens / 1000) * pricing.price_per_1k_input,
        "cached_cost": (cached_tokens / 1000) * pricing.price_per_1k_cached_input,
        "output_cost": (estimated_output_tokens / 1000) * pricing.price_per_1k_output,
    }
    
    return {
        "input_tokens": total_input_tokens,
        "output_tokens": estimated_output_tokens,
        "cached_tokens": cached_tokens,
        "total_cost": total_cost,
        "cost_breakdown": cost_breakdown,
        "model": model_name,
        "supports_caching": pricing.supports_caching
    }


def get_cost_per_page_estimate(model_name: str, words_per_page: int = 300) -> Dict[str, float]:
    """
    Get estimated cost per page for different content densities.
    
    Args:
        model_name: OpenAI model name
        words_per_page: Average words per page
        
    Returns:
        Dictionary with cost estimates per page
    """
    pricing = get_model_pricing(model_name)
    if not pricing:
        return {}
    
    # Estimate tokens per page (roughly 4 chars per token, 5 chars per word)
    chars_per_page = words_per_page * 5  # Average word length
    tokens_per_page = chars_per_page // 4
    
    estimates = {}
    
    # Different scenarios
    scenarios = {
        "light": (tokens_per_page * 0.8, 0.4),    # Light text, 40% cached
        "normal": (tokens_per_page * 1.0, 0.3),   # Normal text, 30% cached  
        "dense": (tokens_per_page * 1.3, 0.2),    # Dense technical text, 20% cached
        "very_dense": (tokens_per_page * 1.8, 0.1) # Very dense with code/math, 10% cached
    }
    
    for scenario_name, (input_toks, cached_ratio) in scenarios.items():
        output_toks = int(input_toks * 1.1)  # Slight expansion
        cached_toks = int(input_toks * cached_ratio) if pricing.supports_caching else 0
        
        # Add system prompt tokens
        total_input = int(input_toks) + 1500  # Our system prompt
        if pricing.supports_caching:
            cached_toks += 1500  # System prompt is cached
        
        cost = calculate_cost(total_input, output_toks, cached_toks, pricing)
        estimates[scenario_name] = cost
    
    return estimates


def format_cost_estimate(estimate: Dict[str, float]) -> str:
    """
    Format a cost estimate as a human-readable string.
    
    Args:
        estimate: Cost estimate dictionary
        
    Returns:
        Formatted string with cost breakdown
    """
    if not estimate:
        return "Custo não disponível"
    
    output = f"Modelo: {estimate['model']}\n"
    output += f"Tokens (Input): {estimate['input_tokens']:,}\n"
    output += f"Tokens (Output): {estimate['output_tokens']:,}\n"
    
    if estimate['cached_tokens'] > 0:
        savings_pct = (estimate['cached_tokens'] / estimate['input_tokens']) * 100
        output += f"Tokens (Cached): {estimate['cached_tokens']:,} ({savings_pct:.1f}% desconto!)\n"
    
    output += f"Custo Total: ${estimate['total_cost']:.4f}\n"
    
    if 'cost_breakdown' in estimate:
        breakdown = estimate['cost_breakdown']
        output += "\nDetalhamento:\n"
        output += f"  Input: ${breakdown['input_cost']:.4f}\n"
        if breakdown['cached_cost'] > 0:
            output += f"  Cached: ${breakdown['cached_cost']:.4f}\n"
        output += f"  Output: ${breakdown['output_cost']:.4f}\n"
    
    if estimate.get('supports_caching'):
        output += "\n💡 Este modelo suporta prompt caching (50% desconto)!"
    
    return output


def get_all_models() -> Dict[str, ModelPricing]:
    """
    Get all available model pricing information.
    
    Returns:
        Dictionary mapping model names to pricing info
    """
    return MODEL_PRICING.copy()


def get_cheapest_model() -> Tuple[str, ModelPricing]:
    """
    Get the cheapest model for translation.
    
    Returns:
        Tuple of (model_name, pricing_info) for cheapest model
    """
    cheapest_model = None
    cheapest_cost = float('inf')
    
    # Calculate cost for a standard 1000-token translation
    test_input_tokens = 1000
    test_output_tokens = 1100
    test_cached_tokens = 300  # Assume some caching
    
    for model_name, pricing in MODEL_PRICING.items():
        cost = calculate_cost(test_input_tokens, test_output_tokens, test_cached_tokens, pricing)
        if cost < cheapest_cost:
            cheapest_cost = cost
            cheapest_model = model_name
    
    return cheapest_model, MODEL_PRICING[cheapest_model]


def get_recommended_model(budget_per_page: float = 0.01) -> Optional[Tuple[str, ModelPricing]]:
    """
    Get recommended model based on budget constraints.
    
    Args:
        budget_per_page: Maximum budget per page in USD
        
    Returns:
        Tuple of (model_name, pricing_info) or None if no model fits budget
    """
    # Test with typical page (300 words)
    for quality_order in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
        if quality_order not in MODEL_PRICING:
            continue
            
        pricing = MODEL_PRICING[quality_order]
        estimates = get_cost_per_page_estimate(quality_order, words_per_page=300)
        
        # Use "normal" scenario cost
        normal_cost = estimates.get("normal", float('inf'))
        
        if normal_cost <= budget_per_page:
            return quality_order, pricing
    
    return None