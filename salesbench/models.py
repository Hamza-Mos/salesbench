"""Model registry for SalesBench benchmarking.

Provides a registry of supported frontier models with context window configurations,
plus utilities for parsing model specifications.

Temperature settings (following tau-bench):
- Seller agent: 0.0 (deterministic for reproducibility)
- Buyer simulator: 0.0 (deterministic for reproducibility)
"""

from dataclasses import dataclass
from typing import Optional

# Default temperatures for reproducibility (following tau-bench)
DEFAULT_SELLER_TEMPERATURE = 0.0
DEFAULT_BUYER_TEMPERATURE = 0.0

# Default buyer model (cheap and fast)
DEFAULT_BUYER_MODEL = "openai/gpt-4o-mini"


@dataclass
class ModelConfig:
    """Configuration for a supported model."""

    context_window: int  # Total context window in tokens
    output_token_limit: int  # Max output tokens
    provider: str
    compression_threshold: float = 0.8
    reserved_system_tokens: int = 2000

    @property
    def available_context(self) -> int:
        """Tokens available for conversation history."""
        return self.context_window - self.reserved_system_tokens - self.output_token_limit

    @property
    def compression_trigger(self) -> int:
        """Token count at which to trigger compression."""
        return int(self.available_context * self.compression_threshold)


# Supported frontier models registry (Jan 2026)
SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # ============= OpenAI (Frontier) =============
    # GPT-5.2 series (latest - 400K context)
    "gpt-5.2": ModelConfig(400_000, 128_000, "openai"),
    "gpt-5.2-mini": ModelConfig(400_000, 128_000, "openai"),
    "gpt-5.2-chat": ModelConfig(128_000, 16_384, "openai"),  # Instant mode
    # GPT-5 series
    "gpt-5": ModelConfig(400_000, 128_000, "openai"),
    "gpt-5-thinking": ModelConfig(196_000, 32_000, "openai"),
    # O-series reasoning (latest)
    "o4-mini": ModelConfig(200_000, 100_000, "openai"),
    "o3": ModelConfig(200_000, 100_000, "openai"),
    "o3-pro": ModelConfig(200_000, 100_000, "openai"),
    # GPT-4o (kept for buyer model - cost efficient)
    "gpt-4o": ModelConfig(128_000, 16_384, "openai"),
    "gpt-4o-mini": ModelConfig(128_000, 16_384, "openai"),
    # ============= Anthropic (Frontier) =============
    # Claude 4.5 series (latest)
    "claude-opus-4-5-20251101": ModelConfig(200_000, 64_000, "anthropic"),
    "claude-sonnet-4-5-20250929": ModelConfig(200_000, 64_000, "anthropic"),
    "claude-haiku-4-5-20251001": ModelConfig(200_000, 64_000, "anthropic"),
    # ============= Google Gemini (Frontier) =============
    # Gemini 3 series (latest)
    "gemini-3-pro": ModelConfig(1_048_576, 65_536, "google"),
    "gemini-3-flash": ModelConfig(1_048_576, 65_536, "google"),
    # Gemini 2.5 series
    "gemini-2.5-pro": ModelConfig(1_048_576, 65_536, "google"),
    "gemini-2.5-flash": ModelConfig(1_048_576, 65_536, "google"),
    # ============= xAI Grok (Frontier) =============
    # Grok 4 series (latest - 2M context)
    "grok-4-1-fast": ModelConfig(2_000_000, 8_000, "xai"),
    # ============= Open Source via OpenRouter/Together (Frontier) =============
    # DeepSeek (latest)
    "deepseek-v3.2": ModelConfig(128_000, 8_192, "openrouter"),
    "deepseek-v3.2-speciale": ModelConfig(128_000, 8_192, "openrouter"),
    "deepseek-r1": ModelConfig(128_000, 8_192, "openrouter"),
    # Llama 3.3 (latest stable)
    "llama-3.3-70b-instruct": ModelConfig(131_000, 4_096, "openrouter"),
    # Qwen3 (latest)
    "qwen3-coder-480b": ModelConfig(256_000, 32_768, "openrouter"),
    "qwen3-235b-a22b": ModelConfig(262_000, 16_384, "openrouter"),
    # GLM (latest)
    "glm-4.6": ModelConfig(200_000, 8_192, "openrouter"),
}

# Build KNOWN_MODELS from SUPPORTED_MODELS for backward compatibility
KNOWN_MODELS: dict[str, str] = {
    model: config.provider for model, config in SUPPORTED_MODELS.items()
}

# Models to benchmark by default (representative set)
# Format: provider/model
DEFAULT_BENCHMARK_MODELS: list[str] = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-5-haiku-20241022",
    "google/gemini-1.5-pro",
    "google/gemini-1.5-flash",
]

# Provider aliases for common shortcuts
PROVIDER_ALIASES: dict[str, str] = {
    "oai": "openai",
    "ant": "anthropic",
    "claude": "anthropic",
    "gpt": "openai",
    "gemini": "google",
}


def get_model_config(model: str) -> ModelConfig:
    """Get config for a supported model.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-opus-4-5-20251101").

    Returns:
        ModelConfig for the model.

    Raises:
        ValueError: If model is not in SUPPORTED_MODELS.
    """
    if model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(f"Unsupported model: '{model}'.\nSupported models:\n{supported}")
    return SUPPORTED_MODELS[model]


def is_supported_model(model: str) -> bool:
    """Check if model is in supported list."""
    return model in SUPPORTED_MODELS


@dataclass
class ModelSpec:
    """Parsed model specification with configuration."""

    model: str
    provider: str
    config: Optional[ModelConfig] = None  # None for backward compatibility

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def context_window(self) -> int:
        """Get context window size."""
        if self.config:
            return self.config.context_window
        return 128_000  # Default fallback

    @property
    def compression_trigger(self) -> int:
        """Get compression trigger threshold."""
        if self.config:
            return self.config.compression_trigger
        return 87_000  # Default fallback


def parse_model_spec(spec: str, strict: bool = True) -> ModelSpec:
    """Parse a model specification string, validates against SUPPORTED_MODELS.

    Format: provider/model (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929")

    For OpenRouter models with slashes: "openrouter/meta-llama/llama-3.1-70b"

    Args:
        spec: Model specification string in provider/model format.
        strict: If True, raises ValueError for unsupported models.

    Returns:
        ModelSpec with model, provider, and config.

    Raises:
        ValueError: If format is invalid or (if strict) model not supported.
    """
    spec = spec.strip()

    if "/" not in spec:
        # Check if it's a known model (backward compatibility)
        if spec in KNOWN_MODELS:
            config = SUPPORTED_MODELS.get(spec)
            return ModelSpec(model=spec, provider=KNOWN_MODELS[spec], config=config)
        if strict:
            supported = ", ".join(sorted(SUPPORTED_MODELS.keys()))
            raise ValueError(f"Unsupported model: '{spec}'.\nSupported models:\n{supported}")
        raise ValueError(
            f"Invalid model format: '{spec}'. Use 'provider/model' format "
            f"(e.g., 'openai/gpt-4o', 'anthropic/claude-sonnet-4-5-20250929')"
        )

    parts = spec.split("/", 1)
    provider = parts[0].lower()
    model = parts[1]

    # Handle provider aliases
    provider = PROVIDER_ALIASES.get(provider, provider)

    # Validate against SUPPORTED_MODELS
    if strict and model not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS.keys()))
        raise ValueError(f"Unsupported model: '{model}'.\nSupported models:\n{supported}")

    config = SUPPORTED_MODELS.get(model)
    return ModelSpec(model=model, provider=provider, config=config)


def parse_model_list(specs: str) -> list[ModelSpec]:
    """Parse a comma-separated list of model specifications.

    Args:
        specs: Comma-separated model specs (e.g., "openai/gpt-4o,anthropic/claude-sonnet-4-20250514")

    Returns:
        List of ModelSpec objects.
    """
    models = []
    for spec in specs.split(","):
        spec = spec.strip()
        if spec:
            models.append(parse_model_spec(spec))
    return models


def list_known_models() -> dict[str, list[str]]:
    """List known models grouped by provider.

    Returns:
        Dictionary mapping provider names to lists of model names.
    """
    by_provider: dict[str, list[str]] = {}
    for model, provider in KNOWN_MODELS.items():
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(model)
    return by_provider


def get_default_benchmark_models() -> list[ModelSpec]:
    """Get the default set of models to benchmark.

    Returns:
        List of ModelSpec for default benchmark models.
    """
    return [parse_model_spec(m) for m in DEFAULT_BENCHMARK_MODELS]


def get_all_known_models() -> list[ModelSpec]:
    """Get all known models as ModelSpecs.

    Returns:
        List of all ModelSpec in the registry.
    """
    return [ModelSpec(model=m, provider=p) for m, p in KNOWN_MODELS.items()]
