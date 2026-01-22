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
    input_price_per_million: Optional[float] = None  # $ per 1M input tokens
    output_price_per_million: Optional[float] = None  # $ per 1M output tokens
    compaction_keep_recent: int = 10  # Recent messages to keep verbatim during compaction

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
    "gpt-5.2": ModelConfig(
        400_000, 128_000, "openai",
        input_price_per_million=1.75, output_price_per_million=14.00,
        compaction_keep_recent=15,
    ),
    "gpt-5-mini": ModelConfig(
        400_000, 128_000, "openai",
        input_price_per_million=0.25, output_price_per_million=2.00,
        compaction_keep_recent=8,
    ),
    "gpt-5-nano": ModelConfig(
        128_000, 16_384, "openai",
        input_price_per_million=0.05, output_price_per_million=0.40,
        compaction_keep_recent=6,
    ),
    # GPT-5 series
    "gpt-5": ModelConfig(
        400_000, 128_000, "openai",
        input_price_per_million=1.25, output_price_per_million=10.00,
        compaction_keep_recent=15,
    ),
    # O-series reasoning (latest)
    "o4-mini": ModelConfig(
        200_000, 100_000, "openai",
        input_price_per_million=1.10, output_price_per_million=4.40,
        compaction_keep_recent=8,
    ),
    "o3": ModelConfig(
        200_000, 100_000, "openai",
        input_price_per_million=2.00, output_price_per_million=8.00,
        compaction_keep_recent=12,
    ),
    "o3-pro": ModelConfig(
        200_000, 100_000, "openai",
        input_price_per_million=20.00, output_price_per_million=80.00,
        compaction_keep_recent=15,
    ),
    # GPT-4o (kept for buyer model - cost efficient)
    "gpt-4o": ModelConfig(
        128_000, 16_384, "openai",
        input_price_per_million=2.50, output_price_per_million=10.00,
        compaction_keep_recent=10,
    ),
    "gpt-4o-mini": ModelConfig(
        128_000, 16_384, "openai",
        input_price_per_million=0.15, output_price_per_million=0.60,
        compaction_keep_recent=6,
    ),
    # ============= Anthropic (Frontier) =============
    # Claude 4.5 series (latest)
    "claude-opus-4-5-20251101": ModelConfig(
        200_000, 64_000, "anthropic",
        input_price_per_million=5.00, output_price_per_million=25.00,
        compaction_keep_recent=15,
    ),
    "claude-sonnet-4-5-20250929": ModelConfig(
        200_000, 64_000, "anthropic",
        input_price_per_million=3.00, output_price_per_million=15.00,
        compaction_keep_recent=12,
    ),
    "claude-haiku-4-5-20251001": ModelConfig(
        200_000, 64_000, "anthropic",
        input_price_per_million=1.00, output_price_per_million=5.00,
        compaction_keep_recent=8,
    ),
    # ============= Google Gemini (Frontier) =============
    # Gemini 3 series (latest - in preview)
    "gemini-3-pro-preview": ModelConfig(
        1_048_576, 65_536, "google",
        input_price_per_million=2.00, output_price_per_million=12.00,
        compaction_keep_recent=15,
    ),
    "gemini-3-flash-preview": ModelConfig(
        1_048_576, 65_536, "google",
        input_price_per_million=0.50, output_price_per_million=3.00,
        compaction_keep_recent=10,
    ),
    # Gemini 2.5 series
    "gemini-2.5-pro": ModelConfig(
        1_048_576, 65_536, "google",
        input_price_per_million=1.25, output_price_per_million=10.00,
        compaction_keep_recent=15,
    ),
    "gemini-2.5-flash": ModelConfig(
        1_048_576, 65_536, "google",
        input_price_per_million=0.30, output_price_per_million=2.50,
        compaction_keep_recent=10,
    ),
    # ============= xAI Grok (Frontier) =============
    # Grok 4 series (latest - 2M context)
    "grok-4-1-fast": ModelConfig(
        2_000_000, 8_000, "xai",
        input_price_per_million=0.20, output_price_per_million=0.50,
        compaction_keep_recent=15,
    ),
    # ============= Open Source via OpenRouter/Together (Frontier) =============
    # DeepSeek (latest)
    "deepseek-v3.2": ModelConfig(
        128_000, 8_192, "openrouter",
        input_price_per_million=0.28, output_price_per_million=0.42,
        compaction_keep_recent=10,
    ),
    "deepseek-v3.2-speciale": ModelConfig(
        128_000, 8_192, "openrouter",
        input_price_per_million=0.28, output_price_per_million=0.42,
        compaction_keep_recent=10,
    ),
    "deepseek-r1": ModelConfig(
        128_000, 8_192, "openrouter",
        input_price_per_million=0.55, output_price_per_million=2.19,
        compaction_keep_recent=10,
    ),
    # Llama 3.3 (latest stable)
    "llama-3.3-70b-instruct": ModelConfig(
        131_000, 4_096, "openrouter",
        input_price_per_million=0.10, output_price_per_million=0.32,
        compaction_keep_recent=10,
    ),
    # Qwen3 (latest)
    "qwen3-coder-480b": ModelConfig(
        256_000, 32_768, "openrouter",
        input_price_per_million=0.22, output_price_per_million=0.95,
        compaction_keep_recent=12,
    ),
    "qwen3-235b-a22b": ModelConfig(
        262_000, 16_384, "openrouter",
        input_price_per_million=0.45, output_price_per_million=3.50,
        compaction_keep_recent=12,
    ),
    # GLM (latest)
    "glm-4.6": ModelConfig(
        200_000, 8_192, "openrouter",
        input_price_per_million=0.35, output_price_per_million=1.55,
        compaction_keep_recent=10,
    ),
}

# Build KNOWN_MODELS from SUPPORTED_MODELS for backward compatibility
KNOWN_MODELS: dict[str, str] = {
    model: config.provider for model, config in SUPPORTED_MODELS.items()
}

# Models to benchmark by default (representative frontier set)
# Format: provider/model
DEFAULT_BENCHMARK_MODELS: list[str] = [
    "openai/gpt-5.2",
    "anthropic/claude-opus-4-5-20251101",
    "google/gemini-3-pro-preview",
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
