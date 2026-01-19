"""LLM client abstraction for SalesBench.

Supports multiple providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- OpenRouter (100+ models)
- xAI (Grok)
- Together AI (Llama, Mixtral)
- Google (Gemini)
"""

from salesbench.llm.client import (
    LLMClient,
    LLMResponse,
    LLMProvider,
    OpenAIClient,
    AnthropicClient,
    OpenRouterClient,
    XAIClient,
    TogetherClient,
    GoogleClient,
    MockLLMClient,
    create_client,
    get_api_key_for_provider,
    detect_available_provider,
    DEFAULT_MODELS,
    API_KEY_ENV_VARS,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMProvider",
    "OpenAIClient",
    "AnthropicClient",
    "OpenRouterClient",
    "XAIClient",
    "TogetherClient",
    "GoogleClient",
    "MockLLMClient",
    "create_client",
    "get_api_key_for_provider",
    "detect_available_provider",
    "DEFAULT_MODELS",
    "API_KEY_ENV_VARS",
]
