"""LLM client abstraction with multi-provider support.

Supported providers:
- OpenAI (GPT-4o, GPT-4o-mini, etc.)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
- OpenRouter (access to 100+ models)
- xAI (Grok)
- Together AI (open source models)
- Google (Gemini)
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    XAI = "xai"
    TOGETHER = "together"
    GOOGLE = "google"
    PRIMEINTELLECT = "primeintellect"
    MOCK = "mock"


# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.OPENROUTER: "openai/gpt-4o-mini",
    LLMProvider.XAI: "grok-beta",
    LLMProvider.TOGETHER: "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    LLMProvider.GOOGLE: "gemini-1.5-flash",
    LLMProvider.PRIMEINTELLECT: "deepseek-ai/DeepSeek-R1",
    LLMProvider.MOCK: "mock-model",
}

# Environment variable names for API keys
API_KEY_ENV_VARS = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    LLMProvider.XAI: "XAI_API_KEY",
    LLMProvider.TOGETHER: "TOGETHER_API_KEY",
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
    LLMProvider.PRIMEINTELLECT: "PRIMEINTELLECT_API_KEY",
}


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str
    provider: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "provider": self.provider,
        }


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    provider: LLMProvider

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            messages: List of message dicts with role and content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Optional response format (e.g., {"type": "json_object"}).

        Returns:
            LLMResponse with the completion.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4o, GPT-4o-mini, etc.)."""

    provider = LLMProvider.OPENAI

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
            provider="openai",
        )

    def get_model_name(self) -> str:
        return self.model


class AnthropicClient(LLMClient):
    """Anthropic API client (Claude 3.5 Sonnet, Claude 3 Opus, etc.)."""

    provider = LLMProvider.ANTHROPIC

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install: pip install anthropic")
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Extract system message if present
        system_message = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_message:
            kwargs["system"] = system_message

        response = client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "stop",
            provider="anthropic",
        )

    def get_model_name(self) -> str:
        return self.model


class OpenRouterClient(LLMClient):
    """OpenRouter API client (access to 100+ models via unified API)."""

    provider = LLMProvider.OPENROUTER

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model or self.model,
            usage=usage,
            finish_reason=response.choices[0].finish_reason or "stop",
            provider="openrouter",
        )

    def get_model_name(self) -> str:
        return self.model


class XAIClient(LLMClient):
    """xAI API client (Grok)."""

    provider = LLMProvider.XAI

    def __init__(
        self,
        model: str = "grok-beta",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("XAI_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError("xAI API key required. Set XAI_API_KEY env var or pass api_key.")

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.x.ai/v1",
                )
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model or self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=response.choices[0].finish_reason or "stop",
            provider="xai",
        )

    def get_model_name(self) -> str:
        return self.model


class TogetherClient(LLMClient):
    """Together AI API client (Llama, Mixtral, etc.)."""

    provider = LLMProvider.TOGETHER

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError(
                "Together AI API key required. Set TOGETHER_API_KEY env var or pass api_key."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.together.xyz/v1",
                )
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model or self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=response.choices[0].finish_reason or "stop",
            provider="together",
        )

    def get_model_name(self) -> str:
        return self.model


class PrimeIntellectClient(LLMClient):
    """Prime Intellect API client (DeepSeek, open-source models).

    Prime Intellect provides OpenAI-compatible API access to open-source models.
    """

    provider = LLMProvider.PRIMEINTELLECT

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-R1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("PRIMEINTELLECT_API_KEY")
        self._base_url = base_url or os.environ.get(
            "PRIMEINTELLECT_BASE_URL", "https://inference.primeintellect.ai/v1"
        )
        self._client = None

        if not self._api_key:
            raise ValueError(
                "Prime Intellect API key required. Set PRIMEINTELLECT_API_KEY env var or pass api_key."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model or self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=response.choices[0].finish_reason or "stop",
            provider="primeintellect",
        )

    def get_model_name(self) -> str:
        return self.model


class GoogleClient(LLMClient):
    """Google AI API client (Gemini)."""

    provider = LLMProvider.GOOGLE

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None

        if not self._api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var or pass api_key.")

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. Install: pip install google-generativeai"
                )
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert messages to Gemini format
        history = []
        current_message = None

        for msg in messages:
            if msg["role"] == "system":
                # Prepend system message to first user message
                current_message = msg["content"] + "\n\n"
            elif msg["role"] == "user":
                content = (current_message or "") + msg["content"]
                current_message = None
                history.append({"role": "user", "parts": [content]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [msg["content"]]})

        # Get the last user message as the prompt
        if history and history[-1]["role"] == "user":
            prompt = history[-1]["parts"][0]
            chat_history = history[:-1]
        else:
            prompt = "Continue"
            chat_history = history

        try:
            import google.generativeai as genai

            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            if response_format and response_format.get("type") == "json_object":
                generation_config.response_mime_type = "application/json"

            chat = client.start_chat(history=chat_history)
            response = chat.send_message(prompt, generation_config=generation_config)

            return LLMResponse(
                content=response.text,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count
                    if hasattr(response, "usage_metadata")
                    else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count
                    if hasattr(response, "usage_metadata")
                    else 0,
                    "total_tokens": response.usage_metadata.total_token_count
                    if hasattr(response, "usage_metadata")
                    else 0,
                },
                finish_reason="stop",
                provider="google",
            )
        except Exception as e:
            raise RuntimeError(f"Google API call failed: {e}")

    def get_model_name(self) -> str:
        return self.model


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    provider = LLMProvider.MOCK

    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = responses or ['{"decision": "REJECT_PLAN", "reason": "Mock response"}']
        self._call_count = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict[str, str]] = None,
    ) -> LLMResponse:
        response_idx = self._call_count % len(self.responses)
        self._call_count += 1

        return LLMResponse(
            content=self.responses[response_idx],
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            finish_reason="stop",
            provider="mock",
        )

    def get_model_name(self) -> str:
        return "mock-model"


# Provider registry
PROVIDERS = {
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.ANTHROPIC: AnthropicClient,
    LLMProvider.OPENROUTER: OpenRouterClient,
    LLMProvider.XAI: XAIClient,
    LLMProvider.TOGETHER: TogetherClient,
    LLMProvider.GOOGLE: GoogleClient,
    LLMProvider.PRIMEINTELLECT: PrimeIntellectClient,
    LLMProvider.MOCK: MockLLMClient,
}


def create_client(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Create an LLM client for the specified provider.

    Args:
        provider: Provider name (openai, anthropic, openrouter, xai, together, google, mock).
        model: Model name (provider-specific). Uses default if not specified.
        api_key: API key for the provider. Uses env var if not specified.
        **kwargs: Additional provider-specific arguments.

    Returns:
        LLMClient instance.

    Raises:
        ValueError: If provider is unknown.

    Examples:
        # OpenAI
        client = create_client("openai", model="gpt-4o")

        # Anthropic
        client = create_client("anthropic", model="claude-3-5-sonnet-20241022")

        # OpenRouter (access to many models)
        client = create_client("openrouter", model="anthropic/claude-3.5-sonnet")

        # xAI Grok
        client = create_client("xai", model="grok-beta")

        # Together AI
        client = create_client("together", model="meta-llama/Llama-3.1-70B-Instruct-Turbo")

        # Google Gemini
        client = create_client("google", model="gemini-1.5-pro")
    """
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        valid = [p.value for p in LLMProvider]
        raise ValueError(f"Unknown provider: {provider}. Valid providers: {valid}")

    client_class = PROVIDERS[provider_enum]
    default_model = DEFAULT_MODELS[provider_enum]

    if provider_enum == LLMProvider.MOCK:
        return client_class(responses=kwargs.get("responses"))

    return client_class(
        model=model or default_model,
        api_key=api_key,
    )


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """Get the API key environment variable for a provider.

    Args:
        provider: Provider name.

    Returns:
        API key from environment, or None if not set.
    """
    try:
        provider_enum = LLMProvider(provider.lower())
        env_var = API_KEY_ENV_VARS.get(provider_enum)
        if env_var:
            return os.environ.get(env_var)
    except ValueError:
        pass
    return None


def detect_available_provider() -> Optional[str]:
    """Detect which LLM provider has an API key configured.

    Returns:
        First available provider name, or None if none configured.
    """
    # Check in order of preference
    priority_order = [
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.OPENROUTER,
        LLMProvider.XAI,
        LLMProvider.TOGETHER,
        LLMProvider.GOOGLE,
        LLMProvider.PRIMEINTELLECT,
    ]

    for provider in priority_order:
        env_var = API_KEY_ENV_VARS.get(provider)
        if env_var and os.environ.get(env_var):
            return provider.value

    return None
