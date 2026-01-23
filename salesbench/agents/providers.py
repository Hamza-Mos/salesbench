"""LLM provider abstractions for tool calling.

Implements the Strategy pattern to eliminate duplicate code between
OpenAI-style and Anthropic-style API calls.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

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
    ) -> tuple[Optional[str], list[ToolCall], int, int, Optional[Any]]:
        """Make LLM call with tool support.

        Args:
            messages: Conversation messages.
            tools: Tool schemas.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (message, tool_calls, input_tokens, output_tokens, raw_assistant_content).
            raw_assistant_content is provider-specific data (e.g., Gemini Content with thought_signature)
            that should be preserved for multi-turn conversations.
        """
        pass

    @abstractmethod
    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
        raw_assistant_content: Optional[Any] = None,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt when guardrails triggered (propose_plan without message).

        Args:
            messages: Original conversation messages.
            tool_calls: Tool calls that were made (includes propose_plan).
            system_prompt: System prompt to use.
            temperature: Sampling temperature.
            raw_assistant_content: Provider-specific content from the assistant's response
                                  (e.g., Gemini Content with thought_signature). Required
                                  for Gemini 3 to maintain conversation continuity.

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
    ) -> tuple[Optional[str], list[ToolCall], int, int, Optional[Any]]:
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

        return text_content, tool_calls, input_tokens, output_tokens, None

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
        raw_assistant_content: Optional[Any] = None,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt to get spoken message for propose_plan."""
        # Defensive: handle None or empty tool_calls
        if not tool_calls:
            return None, 0, 0

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
            **self._get_api_params(model_name, 4096),  # High enough for any model
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

        if hasattr(self._client, "_client") and isinstance(
            self._client._client, anthropic.Anthropic
        ):
            return self._client._client
        else:
            return anthropic.Anthropic(api_key=self._api_key)

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int, Optional[Any]]:
        """Make LLM call with tool support."""
        raw_client = self._get_raw_client()

        # Convert provided tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": _to_api_name(func["name"]),
                        "description": func.get("description", f"Call the {func['name']} tool"),
                        "input_schema": func.get("parameters", {}),
                    }
                )

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
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

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

        return text_content, tool_calls, input_tokens, output_tokens, None

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
        raw_assistant_content: Optional[Any] = None,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt to get spoken message for propose_plan."""
        # Defensive: handle None or empty tool_calls
        if not tool_calls:
            return None, 0, 0

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
            max_tokens=4096,  # High enough for any model
            temperature=temperature,
            system=system_prompt,
            messages=chat_messages,
        )

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        for content in response.content:
            if content.type == "text" and content.text and content.text.strip():
                return content.text.strip(), input_tokens, output_tokens
        return None, input_tokens, output_tokens


class GoogleProvider(ToolCallingProvider):
    """Google Gemini native tool calling using the new google-genai SDK."""

    def _get_client(self):
        """Get the Google GenAI client."""
        import os

        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package required. Install: pip install google-genai")

        # Get API key from parameter, client, or environment
        api_key = self._api_key
        if not api_key and hasattr(self._client, "_api_key"):
            api_key = self._client._api_key
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var or pass api_key.")

        return genai.Client(api_key=api_key)

    def _convert_enums_to_strings(self, schema: dict) -> dict:
        """Convert enum schemas for Gemini compatibility.

        Gemini requires:
        1. All enum types must be STRING (not integer/number)
        2. All enum values must be strings
        """
        import copy

        schema = copy.deepcopy(schema)

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if "enum" in prop_schema:
                    # Gemini requires enum type to be string
                    if prop_schema.get("type") in ("integer", "number"):
                        prop_schema["type"] = "string"
                    # Convert all enum values to strings
                    prop_schema["enum"] = [str(v) for v in prop_schema["enum"]]

        return schema

    def _convert_string_args_to_int(self, args: dict, tools: list[dict], tool_name: str) -> dict:
        """Convert string arguments back to integers based on original schema."""
        # Find the original tool schema
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                if func["name"] == tool_name:
                    props = func.get("parameters", {}).get("properties", {})
                    for arg_name, arg_value in args.items():
                        if arg_name in props:
                            prop_type = props[arg_name].get("type")
                            if prop_type == "integer" and isinstance(arg_value, str):
                                try:
                                    args[arg_name] = int(arg_value)
                                except ValueError:
                                    pass
                    break
        return args

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int, Optional[Any]]:
        """Make LLM call with native Gemini tool support."""
        from google.genai import types
        from google.genai.types import AutomaticFunctionCallingConfig

        client = self._get_client()

        # Convert OpenAI-style tools to Gemini function declarations
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                parameters = func.get("parameters", {})
                # Convert integer enums to strings for Gemini compatibility
                parameters = self._convert_enums_to_strings(parameters)
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=_to_api_name(func["name"]),
                        description=func.get("description", ""),
                        parameters=parameters,
                    )
                )

        # Create tool config
        gemini_tools = types.Tool(function_declarations=function_declarations)

        # Convert messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])])
                )
            elif msg["role"] == "assistant":
                # Check if we have stored Gemini content with thought_signature
                # This is critical for Gemini 3 which requires thought_signature preservation
                if gemini_content := msg.get("gemini_content"):
                    contents.append(gemini_content)
                else:
                    # Fallback for non-Gemini history or older messages
                    contents.append(
                        types.Content(
                            role="model", parts=[types.Part.from_text(text=msg["content"])]
                        )
                    )
            elif msg["role"] == "tool":
                # Handle tool results - Gemini expects function responses as user role
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.get("tool_name", "unknown"),
                                response={"result": msg["content"]},
                            )
                        ],
                    )
                )

        # Build generation config
        # Disable AFC (Automatic Function Calling) - Gemini's SDK has this enabled by default
        # which causes it to loop on function calls, interfering with our reprompt mechanism
        # Configure safety settings to allow sales/insurance content (BLOCK_ONLY_HIGH)
        #
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "tools": [gemini_tools],
            "automatic_function_calling": AutomaticFunctionCallingConfig(disable=True),
            "safety_settings": [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        }

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        # Retry loop for transient failures (safety filters, empty responses)
        MAX_RETRIES = 1
        input_tokens = 0
        output_tokens = 0
        response = None

        for attempt in range(MAX_RETRIES + 1):
            response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            # Extract token counts (accumulate across retries)
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata:
                input_tokens += getattr(usage_metadata, "prompt_token_count", 0) or 0
                output_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0

            # Check for blocked prompt
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, "block_reason", None)
                if block_reason:
                    logger.warning(f"[GEMINI] Prompt blocked: {block_reason}")
                    # Prompt blocked - no retry will help
                    return None, [], input_tokens, output_tokens, None

            # Check for empty candidates
            if not response.candidates:
                logger.warning("[GEMINI] No candidates in response (likely blocked)")
                if attempt < MAX_RETRIES:
                    logger.info(
                        f"[GEMINI] Retrying due to empty candidates (attempt {attempt + 1})"
                    )
                    continue
                return None, [], input_tokens, output_tokens, None

            # Check finish_reason
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            finish_reason_name = (
                getattr(finish_reason, "name", str(finish_reason)) if finish_reason else "UNKNOWN"
            )

            if finish_reason_name not in ("STOP", "MAX_TOKENS"):
                logger.warning(f"[GEMINI] Non-standard finish_reason: {finish_reason_name}")

                # For transient errors, retry - content is typically empty or unusable
                if (
                    finish_reason_name
                    in ("SAFETY", "RECITATION", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL")
                    and attempt < MAX_RETRIES
                ):
                    # Log safety ratings for debugging
                    safety_ratings = getattr(candidate, "safety_ratings", [])
                    for rating in safety_ratings:
                        if getattr(rating, "blocked", False):
                            logger.warning(
                                f"[GEMINI] Blocked by {getattr(rating, 'category', 'unknown')}: {getattr(rating, 'probability', 'unknown')}"
                            )
                    logger.info(
                        f"[GEMINI] Retrying due to {finish_reason_name} (attempt {attempt + 1})"
                    )
                    continue

                # For MALFORMED_FUNCTION_CALL after exhausting retries, return empty results
                # to avoid parsing a malformed response that could cause iteration errors
                if finish_reason_name == "MALFORMED_FUNCTION_CALL":
                    logger.warning(
                        f"[GEMINI] MALFORMED_FUNCTION_CALL after {MAX_RETRIES + 1} attempts, returning empty"
                    )
                    return None, [], input_tokens, output_tokens, None

            # Response looks valid, break out of retry loop
            break

        # Parse response for text and tool calls
        text_content = None
        tool_calls = []

        # Preserve raw Gemini content for thought_signature (Gemini 3 requirement)
        raw_gemini_content = None
        try:
            if response and response.candidates and response.candidates[0].content:
                raw_gemini_content = response.candidates[0].content
                parts = getattr(raw_gemini_content, "parts", None)
                if parts:
                    for part in parts:
                        # Check for text content
                        if hasattr(part, "text") and part.text:
                            text_content = part.text
                        # Check for function call
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            # Convert args to dict (defensive: handle various formats)
                            args = {}
                            fc_args = getattr(fc, "args", None)
                            if fc_args:
                                try:
                                    if hasattr(fc_args, "items"):
                                        args = dict(fc_args)
                                    elif isinstance(fc_args, dict):
                                        args = fc_args
                                    else:
                                        # Try to convert if it's some other iterable
                                        args = dict(fc_args)
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"[GEMINI] Failed to parse function args: {e}")
                                    args = {}
                            # Convert string args back to integers based on original schema
                            fc_name = getattr(fc, "name", "unknown")
                            args = self._convert_string_args_to_int(args, tools, fc_name)
                            tool_calls.append(
                                ToolCall(
                                    tool_name=_from_api_name(fc_name),
                                    arguments=args,
                                    call_id=f"gemini_{fc_name}_{len(tool_calls)}",
                                )
                            )
        except Exception as e:
            logger.warning(f"[GEMINI] Error parsing response: {e}")
            # Return what we have so far (defensive)

        # Log detailed info when response is empty
        if not text_content and not tool_calls:
            finish_reason_str = "UNKNOWN"
            if response and response.candidates:
                fr = getattr(response.candidates[0], "finish_reason", None)
                finish_reason_str = getattr(fr, "name", str(fr)) if fr else "UNKNOWN"
            logger.warning(
                f"[GEMINI] Empty response - finish_reason={finish_reason_str}, "
                f"candidates={len(response.candidates) if response and response.candidates else 0}, "
                f"parts={len(raw_gemini_content.parts) if raw_gemini_content and raw_gemini_content.parts else 0}"
            )

        return text_content, tool_calls, input_tokens, output_tokens, raw_gemini_content

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
        raw_assistant_content: Optional[Any] = None,
    ) -> tuple[Optional[str], int, int]:
        """Re-prompt to get spoken message for propose_plan."""
        from google.genai import types
        from google.genai.types import AutomaticFunctionCallingConfig

        logger.info("[GEMINI REPROMPT] Called for propose_plan without message")

        # Defensive: handle None or empty tool_calls
        if not tool_calls:
            logger.info("[GEMINI REPROMPT] No tool_calls provided")
            return None, 0, 0

        propose_call = next(
            (tc for tc in tool_calls if tc.tool_name == "calling.propose_plan"), None
        )
        if not propose_call:
            logger.info("[GEMINI REPROMPT] No propose_plan found in tool_calls")
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

        client = self._get_client()

        # Build contents for reprompt - preserve thought_signature from stored Gemini content
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])])
                )
            elif msg["role"] == "assistant":
                # Check if we have stored Gemini content with thought_signature
                if gemini_content := msg.get("gemini_content"):
                    contents.append(gemini_content)
                else:
                    # Fallback for non-Gemini history
                    contents.append(
                        types.Content(
                            role="model", parts=[types.Part.from_text(text=msg["content"])]
                        )
                    )

        # CRITICAL: Add the current assistant response (with thought_signature) before reprompt
        # This is the response that triggered the reprompt - it contains the tool calls
        # but no text. Gemini 3 needs to see this with its thought_signature intact.
        if raw_assistant_content is not None:
            logger.info("[GEMINI REPROMPT] Including assistant response with thought_signature")
            contents.append(raw_assistant_content)

        # Add reprompt message
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=reprompt_message)])
        )

        # Explicitly disable function calling during reprompt
        # This prevents UNEXPECTED_TOOL_CALL when Gemini tries to make tool calls
        # based on conversation history even though no tools are passed
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        )

        # Configure safety settings to be more permissive for sales content
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=4096,  # High enough for any model including thinking models
            system_instruction=system_prompt,
            tool_config=tool_config,
            # Disable AFC even though we have no tools - SDK may still log warnings
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
        )

        # Retry loop for reprompt - Gemini can return empty responses
        MAX_REPROMPT_RETRIES = 3
        total_in_tokens = 0
        total_out_tokens = 0

        for attempt in range(MAX_REPROMPT_RETRIES):
            response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            # Track tokens
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata:
                total_in_tokens += getattr(usage_metadata, "prompt_token_count", 0) or 0
                total_out_tokens += getattr(usage_metadata, "candidates_token_count", 0) or 0

            # Check for text in response
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text and part.text.strip():
                        logger.info(
                            f"[GEMINI REPROMPT] Got text on attempt {attempt + 1}: {part.text.strip()[:100]}..."
                        )
                        return part.text.strip(), total_in_tokens, total_out_tokens

                # Log if model returned function calls despite tool_config mode=NONE
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        logger.warning(
                            "[GEMINI REPROMPT] Model returned function call during reprompt (ignoring)"
                        )
                        break

            # Log why we're retrying
            finish_reason = None
            finish_reason_name = "UNKNOWN"
            if response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                finish_reason_name = (
                    getattr(finish_reason, "name", str(finish_reason))
                    if finish_reason
                    else "UNKNOWN"
                )
            logger.warning(
                f"[GEMINI REPROMPT] No text on attempt {attempt + 1}, finish_reason={finish_reason_name}"
            )

        logger.warning(f"[GEMINI REPROMPT] All {MAX_REPROMPT_RETRIES} attempts failed")
        return None, total_in_tokens, total_out_tokens


class JSONProvider(ToolCallingProvider):
    """JSON-based tool calling for providers without native tool support."""

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[str], list[ToolCall], int, int, Optional[Any]]:
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

            return message, tool_calls, input_tokens, output_tokens, None
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse JSON tool response: {e}")
            return None, [], 0, 0, None

    def reprompt_for_message(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        system_prompt: str,
        temperature: float,
        raw_assistant_content: Optional[Any] = None,
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
        provider_name: Provider name (openai, anthropic, google, etc.).
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
    elif provider_name == "google":
        return GoogleProvider(client, model, api_key)
    else:
        return JSONProvider(client, model, api_key)
