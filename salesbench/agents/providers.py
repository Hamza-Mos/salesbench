"""LLM provider abstractions for tool calling.

Implements the Strategy pattern to eliminate duplicate code between
OpenAI-style and Anthropic-style API calls.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from salesbench.core.types import ToolCall

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


def _to_api_name(tool_name: str) -> str:
    """Convert tool name to OpenAI-compatible format (dots to double underscores)."""
    return tool_name.replace(".", "__")


def _from_api_name(api_name: str) -> str:
    """Convert OpenAI API name back to tool name (double underscores to dots)."""
    return api_name.replace("__", ".")


class ToolCallingProvider(ABC):
    """Abstract base for LLM providers with tool calling."""

    def __init__(self, client, model: str, api_key: Optional[str] = None):
        """Initialize provider.

        Args:
            client: LLM client wrapper.
            model: Model name.
            api_key: Optional API key.
        """
        self._client = client
        self._model = model
        self._api_key = api_key

    @abstractmethod
    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int]:
        """Make LLM call with tool support.

        Args:
            messages: Conversation messages.
            tools: Tool schemas.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (message, tool_calls, input_tokens, output_tokens).
        """
        pass

    @abstractmethod
    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt when guardrails triggered (propose_plan without message).

        Args:
            messages: Original conversation messages.
            tool_calls: Tool calls that were made (includes propose_plan).
            system_prompt: System prompt to use.
            temperature: Sampling temperature.

        Returns:
            Tuple of (spoken_message, input_tokens, output_tokens).
        """
        pass


class OpenAIProvider(ToolCallingProvider):
    """OpenAI-style tool calling (also xAI, Together, OpenRouter)."""

    def _get_raw_client(self):
        """Get the underlying OpenAI client."""
        if openai is None:
            raise ImportError("openai package required for OpenAI-style tool calling")

        if hasattr(self._client, "_client") and isinstance(self._client._client, openai.OpenAI):
            return self._client._client
        else:
            return openai.OpenAI(
                api_key=self._api_key,
                base_url=getattr(self._client, "_base_url", None),
            )

    def _get_api_params(self, model_name: str, max_tokens: int) -> dict:
        """Get API parameters based on model capabilities."""
        uses_new_api = any(x in model_name for x in ["gpt-5", "gpt-4.1", "o1", "o3"])
        if uses_new_api:
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int]:
        """Make LLM call with tool support."""
        raw_client = self._get_raw_client()
        model_name = self._client.get_model_name()

        api_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": "auto",
            **self._get_api_params(model_name, max_tokens),
        }

        response = raw_client.chat.completions.create(**api_params)

        # Extract token counts
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Parse message and tool calls
        tool_calls = []
        response_message = response.choices[0].message
        text_content = response_message.content

        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    tool_calls.append(
                        ToolCall(
                            tool_name=_from_api_name(tc.function.name),
                            arguments=args,
                            call_id=tc.id,
                        )
                    )
                except json.JSONDecodeError:
                    continue

        return text_content, tool_calls, input_tokens, output_tokens

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt to get spoken message for propose_plan."""
        propose_call = next(
            (tc for tc in tool_calls if tc.tool_name == "calling.propose_plan"), None
        )
        if not propose_call:
            return None, 0, 0

        args = propose_call.arguments or {}
        plan_id = args.get("plan_id", "insurance plan")
        premium = args.get("monthly_premium", "")
        coverage = args.get("coverage_amount", "")

        reprompt_message = (
            f"You just called propose_plan for {plan_id} "
            f"(${premium}/month, ${coverage} coverage) but you didn't include "
            "any spoken message to the buyer. The buyer cannot see tool calls - "
            "they only hear what you say.\n\n"
            "Please provide ONLY your spoken pitch (1-3 sentences) that presents "
            "this offer to the buyer. Do not include any tool calls or explanations."
        )

        reprompt_messages = messages + [{"role": "user", "content": reprompt_message}]

        raw_client = self._get_raw_client()
        model_name = self._client.get_model_name()

        api_params = {
            "model": model_name,
            "messages": reprompt_messages,
            "temperature": temperature,
            **self._get_api_params(model_name, 256),
        }

        response = raw_client.chat.completions.create(**api_params)

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        text_content = response.choices[0].message.content
        if text_content and text_content.strip():
            return text_content.strip(), input_tokens, output_tokens
        return None, input_tokens, output_tokens


class AnthropicProvider(ToolCallingProvider):
    """Anthropic-style tool calling."""

    def _get_raw_client(self):
        """Get the underlying Anthropic client."""
        if anthropic is None:
            raise ImportError("anthropic package required for Anthropic API")

        if hasattr(self._client, "_client") and isinstance(self._client._client, anthropic.Anthropic):
            return self._client._client
        else:
            return anthropic.Anthropic(api_key=self._api_key)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int]:
        """Make LLM call with tool support."""
        from salesbench.core.protocol import get_all_tool_schemas

        raw_client = self._get_raw_client()

        # Convert tools to Anthropic format
        anthropic_tools = [
            {
                "name": _to_api_name(tool),
                "description": f"Call the {tool} tool",
                "input_schema": schema,
            }
            for tool, schema in get_all_tool_schemas().items()
        ]

        # Extract system message
        system_message = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_message = m["content"]
            else:
                chat_messages.append(m)

        response = raw_client.messages.create(
            model=self._client.get_model_name(),
            max_tokens=max_tokens,
            system=system_message or "",
            messages=chat_messages,
            tools=anthropic_tools,
        )

        # Extract token counts
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Parse text content and tool calls
        tool_calls = []
        text_content = None

        for content in response.content:
            if content.type == "text":
                text_content = content.text
            elif content.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_name=_from_api_name(content.name),
                        arguments=content.input,
                        call_id=content.id,
                    )
                )

        return text_content, tool_calls, input_tokens, output_tokens

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt to get spoken message for propose_plan."""
        propose_call = next(
            (tc for tc in tool_calls if tc.tool_name == "calling.propose_plan"), None
        )
        if not propose_call:
            return None, 0, 0

        args = propose_call.arguments or {}
        plan_id = args.get("plan_id", "insurance plan")
        premium = args.get("monthly_premium", "")
        coverage = args.get("coverage_amount", "")

        reprompt_message = (
            f"You just called propose_plan for {plan_id} "
            f"(${premium}/month, ${coverage} coverage) but you didn't include "
            "any spoken message to the buyer. The buyer cannot see tool calls - "
            "they only hear what you say.\n\n"
            "Please provide ONLY your spoken pitch (1-3 sentences) that presents "
            "this offer to the buyer. Do not include any tool calls or explanations."
        )

        raw_client = self._get_raw_client()

        # Filter out system message and add reprompt
        chat_messages = [m for m in messages if m["role"] != "system"]
        chat_messages.append({"role": "user", "content": reprompt_message})

        response = raw_client.messages.create(
            model=self._client.get_model_name(),
            max_tokens=256,
            system=system_prompt,
            messages=chat_messages,
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        for content in response.content:
            if content.type == "text" and content.text and content.text.strip():
                return content.text.strip(), input_tokens, output_tokens
        return None, input_tokens, output_tokens


class JSONProvider(ToolCallingProvider):
    """JSON-based tool calling for providers without native tool support."""

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int]:
        """Make LLM call using JSON-based tool calling."""
        response = self._client.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            content = json.loads(response.content)
            message = content.get("message")
            raw_calls = content.get("tool_calls", [])

            tool_calls = []
            for tc_data in raw_calls:
                if isinstance(tc_data, dict):
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc_data.get("tool_name", ""),
                            arguments=tc_data.get("arguments", {}),
                        )
                    )

            # Estimate tokens (JSON provider doesn't give us exact counts)
            input_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            output_tokens = len(response.content) // 4

            return message, tool_calls, input_tokens, output_tokens
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON tool response: {e}")
            return None, [], 0, 0

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
    ) -> tuple[Optional[str], int, int]:
        """JSON provider reprompt is not implemented - returns None."""
        return None, 0, 0


def get_provider(
    provider_name: str,
    client,
    model: str,
    api_key: Optional[str] = None,
) -> ToolCallingProvider:
    """Factory function to get the appropriate provider.

    Args:
        provider_name: Provider name (openai, anthropic, etc.).
        client: LLM client wrapper.
        model: Model name.
        api_key: Optional API key.

    Returns:
        ToolCallingProvider instance.
    """
    if provider_name in ["openai", "openrouter", "xai", "together"]:
        return OpenAIProvider(client, model, api_key)
    elif provider_name == "anthropic":
        return AnthropicProvider(client, model, api_key)
    else:
        return JSONProvider(client, model, api_key)
