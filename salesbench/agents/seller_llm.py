"""LLM-based seller agent with multi-provider support.

Supports multiple LLM providers via the modular adapter:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- OpenRouter (100+ models)
- xAI (Grok)
- Together AI (Llama, Mixtral)
- Google (Gemini)
"""

import json
from typing import TYPE_CHECKING, Any, Optional, Tuple

from salesbench.agents.seller_base import (
    SellerAgent,
    SellerConfig,
    SellerObservation,
)
from salesbench.core.protocol import SellerAction, get_all_tool_schemas
from salesbench.core.types import ToolCall
from salesbench.llm import LLMClient, create_client, detect_available_provider

if TYPE_CHECKING:
    from salesbench.context.episode import EpisodeContext

SELLER_SYSTEM_PROMPT = """You are an AI insurance sales agent. Your goal is to sell insurance plans to leads.

## HOW YOU COMMUNICATE

You have two types of output:
1. **Message**: Free-form text that the buyer sees (your actual conversation)
2. **Tool calls**: Operational/analytical actions (CRM, analytics, etc.)

When in a call, your MESSAGE is what the buyer hears. Tool calls like `calling.propose_plan` are purely for ANALYTICS - they record the offer details but do NOT speak to the buyer. You must write your pitch in the message.

## CONVERSATION QUALITY (IMPORTANT)

This benchmark expects **long, natural back-and-forth**. Behave like a professional agent:
- Keep each spoken message **short (1–4 sentences)** and **conversational**.
- Ask **one good question at a time** and **wait for the buyer's reply**.
- If the buyer asks a question or raises an objection, **answer it directly first** (don’t jump to another offer).
- Do discovery early: confirm **goal**, **timeline**, **budget range**, **current coverage**, and **beneficiaries/dependents**.
- Do not repeat the same pitch verbatim. Adapt to what the buyer just said.
- Only present an offer after you have at least minimal qualification (unless the lead is clearly HOT).

## CRITICAL WORKFLOW - State Machine

You are in one of these states:
1. **NO_LEADS** → Search for leads with `crm.search_leads`
2. **HAVE_LEADS** → Call a lead with `calling.start_call(lead_id="...")`
3. **IN_CALL** → Send a message to pitch your offer, use `calling.propose_plan` to record it analytically

### STATE TRANSITIONS:
- NO_LEADS + search returns leads → HAVE_LEADS (next: start_call)
- HAVE_LEADS + start_call → IN_CALL (next: message + propose_plan)
- IN_CALL + call ends → HAVE_LEADS (call next lead) or NO_LEADS (search more)

## ⚠️ CRITICAL RULES - READ CAREFULLY

1. **NEVER search twice in a row.** If you just searched and got leads, your ONLY valid action is `calling.start_call`.
2. **Use the lead_id from search results.** The "Next Action" section shows you exactly which lead_id to use.
3. **If search returns 0 leads**, try a different temperature: hot → warm → lukewarm → cold.
4. **When pitching a plan**, write your pitch in the MESSAGE and call `propose_plan` to record the offer details.

## Available Tools

### CRM Tools
- `crm.search_leads`: Search leads by temperature (hot/warm/lukewarm/cold), income, age
- `crm.get_lead`: Get lead details
- `crm.update_lead`: Update notes
- `crm.log_call`: Log call outcome

### Calling Tools
- `calling.start_call`: Start call with a lead - REQUIRES lead_id from search results
- `calling.propose_plan`: **ANALYTICS ONLY** - Records the offer details. Does NOT speak to buyer. Use your message to pitch.
- `calling.end_call`: End current call

**IMPORTANT**: `calling.propose_plan` does not include a `pitch` argument. The buyer hears ONLY your normal assistant message. Always write your pitch as your message text, then call `calling.propose_plan` to record the structured offer.

**NOTE**: When a buyer ACCEPTS a plan, the call ends automatically. Do NOT call `end_call` after an acceptance - move directly to the next lead. Also, do NOT call a lead again after they've accepted - they're already a customer!

### Product Tools
- `products.list_plans`: List available plans (TERM, WHOLE, UL, VUL, LTC, DI)
- `products.quote_premium`: Calculate premium for plan/age/coverage

## Quick Reference

**Plan Selection:**
- Young families (25-40, dependents): TERM - affordable protection
- Higher income (40+): WHOLE - cash value + protection
- Single/no dependents: Smaller TERM or skip

**Pricing:** Monthly premium = 5-10% of monthly income max.

**next_step:** "close_now" (hot), "schedule_followup" (warm), "request_info" (cold)

## Example - Correct Workflow

```
Turn 1: crm.search_leads(temperature="hot")
        → Returns: lead_abc123, lead_xyz789
Turn 2: calling.start_call(lead_id="lead_abc123")
        → Call connected with John, age 35, $80k income, 2 dependents
Turn 3: MESSAGE: "Hi John—quick question: do you already have coverage through work, and about how much?"
Turn 4: MESSAGE: "Got it. And what monthly budget range feels comfortable if this protected your family?"
Turn 5: MESSAGE: "Thanks. Based on that, here’s a simple option that fits…"
        + calling.propose_plan(plan_id="TERM", monthly_premium=150, coverage_amount=500000, next_step="close_now")
Turn 6: [Based on buyer response] MESSAGE: "I hear you on price—if we adjust coverage slightly, would that help?"
        + another propose_plan or calling.end_call()
```

## WRONG - Do Not Do This

```
Turn 1: crm.search_leads(temperature="hot") → Returns leads
Turn 2: crm.search_leads(temperature="hot") ← WRONG! You already have leads!
```

```
Turn 3: calling.propose_plan(...) with NO MESSAGE ← WRONG! The buyer won't hear anything!
```

If you have leads, CALL THEM. When pitching, ALWAYS include a message."""


def _to_api_name(tool_name: str) -> str:
    """Convert tool name to OpenAI-compatible format (dots to double underscores)."""
    return tool_name.replace(".", "__")


def _from_api_name(api_name: str) -> str:
    """Convert OpenAI API name back to tool name (double underscores to dots)."""
    return api_name.replace("__", ".")


def _postprocess_tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
    """De-dupe tool calls and cap calling.propose_plan to one per action.

    Some models/providers occasionally emit repeated identical tool calls in a single
    response. This breaks benchmark conversations by spamming tool execution errors.
    We keep the first occurrence (stable) and drop duplicates. Additionally, we allow
    at most one `calling.propose_plan` per action, regardless of arguments.
    """

    seen: set[tuple[str, str]] = set()
    filtered: list[ToolCall] = []
    propose_plan_seen = False

    for tc in tool_calls:
        # Cap propose_plan to one per action (stable: keep first).
        if tc.tool_name == "calling.propose_plan":
            if propose_plan_seen:
                continue
            propose_plan_seen = True

        # De-dupe exact repeats (stable: keep first).
        try:
            args_key = json.dumps(tc.arguments or {}, sort_keys=True, default=str)
        except TypeError:
            args_key = str(tc.arguments)
        sig = (tc.tool_name, args_key)
        if sig in seen:
            continue
        seen.add(sig)
        filtered.append(tc)

    return filtered


def _has_propose_plan(tool_calls: list[ToolCall]) -> bool:
    """Check if any tool call is calling.propose_plan."""
    return any(tc.tool_name == "calling.propose_plan" for tc in tool_calls)


TOOLS_FOR_LLM = [
    {
        "type": "function",
        "function": {
            "name": _to_api_name(tool),
            "description": f"Call the {tool} tool",
            "parameters": schema,
        },
    }
    for tool, schema in get_all_tool_schemas().items()
]


class LLMSellerAgent(SellerAgent):
    """LLM-based seller agent with multi-provider support."""

    def __init__(
        self,
        config: Optional[SellerConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize LLM seller agent.

        Args:
            config: Agent configuration.
            provider: LLM provider (openai, anthropic, openrouter, xai, together, google).
                     Auto-detects from environment if not specified.
            model: Model name. Uses provider default if not specified.
            api_key: API key (uses env var if not provided).
        """
        super().__init__(config)

        # Auto-detect provider if not specified
        if provider is None:
            provider = detect_available_provider()
            if provider is None:
                raise ValueError(
                    "No LLM provider configured. Set one of these environment variables:\n"
                    "  - OPENAI_API_KEY (OpenAI)\n"
                    "  - ANTHROPIC_API_KEY (Anthropic)\n"
                    "  - OPENROUTER_API_KEY (OpenRouter)\n"
                    "  - XAI_API_KEY (xAI/Grok)\n"
                    "  - TOGETHER_API_KEY (Together AI)\n"
                    "  - GOOGLE_API_KEY (Google Gemini)"
                )

        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._client: Optional[LLMClient] = None
        self._conversation_history: list[dict] = []
        self._available_leads: list[dict] = []  # Synced from AnchoredState
        self._called_lead_ids: set[str] = set()  # Synced from AnchoredState
        self._accepted_lead_ids: set[str] = set()  # Synced from AnchoredState

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            self._client = create_client(
                provider=self.provider,
                model=self.model or self.config.model,
                api_key=self._api_key,
            )
        return self._client

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._total_api_cost = 0.0
        self._turn_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._conversation_history = []
        self._available_leads = []
        self._called_lead_ids = set()
        self._accepted_lead_ids = set()

    def _sync_from_anchored_state(self, episode_context: Optional["EpisodeContext"]) -> None:
        """Sync agent state from AnchoredState (single source of truth).

        The agent should READ from AnchoredState, not maintain parallel state.
        This ensures the agent always has accurate information about:
        - Which leads have been found
        - Which leads have been called
        - Which leads have accepted
        """
        if episode_context is None:
            return

        anchored = episode_context._anchored_state
        self._called_lead_ids = set(anchored.called_lead_ids)
        self._accepted_lead_ids = set(anchored.accepted_lead_ids)

        # Rebuild available leads from AnchoredState (uncalled, non-accepted)
        self._available_leads = [
            {
                "lead_id": lead.lead_id,
                "name": lead.name,
                "temperature": lead.temperature,
                "annual_income": lead.income,
            }
            for lead in anchored.found_leads.values()
            if lead.lead_id not in anchored.called_lead_ids
            and lead.lead_id not in anchored.accepted_lead_ids
        ]

    def act(
        self,
        observation: SellerObservation,
        tools: Optional[list] = None,
        episode_context: Optional["EpisodeContext"] = None,
    ) -> SellerAction:
        """Decide what to say and which tools to call using LLM.

        Args:
            observation: Current observation from environment.
            tools: Optional tool schemas (uses built-in if not provided).
            episode_context: Optional episode context for conversation history.
                            If provided, uses compressed episode history instead
                            of internal sliding window.

        Returns:
            SellerAction containing message and/or tool calls.
        """
        # Sync agent state from AnchoredState (single source of truth)
        self._sync_from_anchored_state(episode_context)

        # Build user message from observation
        user_message = self._build_user_message(observation)

        # Add to conversation history (for fallback when no episode_context)
        self._conversation_history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Get LLM response (returns message and tool_calls)
        message, tool_calls = self._call_llm(user_message, episode_context)

        # Add assistant response to history
        assistant_content = message or ""
        if tool_calls:
            assistant_content += (
                f"\n[Tool calls: {json.dumps([tc.to_dict() for tc in tool_calls])}]"
            )
        self._conversation_history.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        self._turn_count += 1
        return SellerAction(tool_calls=tool_calls, message=message)

    def _build_user_message(self, obs: SellerObservation) -> str:
        """Build user message from observation.

        Shows facts only - no directive language. The model decides based on context.
        The AnchoredState context block (injected separately) provides the authoritative
        state about leads, calls, and searches.
        """
        lines = [
            "## Current State",
            f"- Day {obs.current_day}/10, Hour {obs.current_hour}:00",
            f"- Remaining minutes today: {obs.remaining_minutes}",
            f"- Total calls made: {obs.total_calls}",
            f"- Accepts: {obs.total_accepts}, Rejects: {obs.total_rejects}",
            f"- DNC Violations: {obs.total_dnc_violations}",
        ]

        if obs.in_call:
            lines.extend(
                [
                    "",
                    f"## In Call: {obs.current_lead_id}",
                    f"- Duration: {obs.call_duration} minutes",
                    f"- Offers made this call: {obs.offers_this_call}",
                ]
            )

        # Show tool results (just facts, no tracking updates - that's handled by AnchoredState)
        if obs.last_tool_results:
            lines.extend(
                [
                    "",
                    "## Last Tool Results",
                ]
            )
            for result in obs.last_tool_results:
                if result.success:
                    # Format the result data
                    data_str = json.dumps(result.data, default=str)
                    if len(data_str) > 1000:
                        data_str = data_str[:1000] + "..."
                    lines.append(f"✓ {result.call_id}: {data_str}")
                else:
                    lines.append(f"✗ {result.call_id}: {result.error}")

        if obs.message:
            lines.extend(
                [
                    "",
                    "## System Message",
                    obs.message,
                ]
            )

        # Current status section - just facts
        lines.append("")
        uncalled_leads = self._available_leads  # Already filtered in _sync_from_anchored_state
        if obs.in_call:
            lines.append(
                f"STATUS: In call with {obs.current_lead_id}, {obs.offers_this_call} offers made"
            )
        elif uncalled_leads:
            lines.append(f"STATUS: {len(uncalled_leads)} uncalled leads available")
        else:
            lines.append("STATUS: No leads available")

        return "\n".join(lines)

    def _call_llm(
        self,
        user_message: str,
        episode_context: Optional["EpisodeContext"] = None,
    ) -> Tuple[Optional[str], list[ToolCall]]:
        """Call LLM API with function calling.

        Args:
            user_message: The current user message (observation).
            episode_context: Optional episode context for conversation history.

        Returns:
            Tuple of (message, tool_calls) where message is optional free-form text
            and tool_calls is a list of tool calls.
        """
        client = self._get_client()

        messages = [{"role": "system", "content": SELLER_SYSTEM_PROMPT}]

        # Use episode context if provided, otherwise fall back to sliding window
        if episode_context is not None:
            # Get compressed episode history from context
            context_messages = episode_context.get_seller_view()
            messages.extend(context_messages)
            # Add the current observation as the latest message
            messages.append({"role": "user", "content": user_message})
        else:
            # Fallback: use internal sliding window (last 20 messages)
            messages.extend(self._conversation_history[-20:])

        # Use tool calling for OpenAI-compatible APIs
        if self.provider in ["openai", "openrouter", "xai", "together"]:
            return self._call_with_tools(client, messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(client, messages)
        else:
            # Fallback to JSON-based tool calling
            return self._call_with_json(client, messages)

    def _call_with_tools(
        self, client: LLMClient, messages: list
    ) -> Tuple[Optional[str], list[ToolCall]]:
        """Call using OpenAI-style tool calling.

        Returns:
            Tuple of (message, tool_calls).
        """
        # We need to use the raw OpenAI client for tool calling
        import openai

        if hasattr(client, "_client") and isinstance(client._client, openai.OpenAI):
            raw_client = client._client
        else:
            raw_client = openai.OpenAI(
                api_key=self._api_key,
                base_url=getattr(client, "_base_url", None),
            )

        model_name = client.get_model_name()

        # Newer models (gpt-5.x, o1, o3, etc.) use max_completion_tokens
        uses_new_api = any(x in model_name for x in ["gpt-5", "gpt-4.1", "o1", "o3"])

        api_params = {
            "model": model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "tools": TOOLS_FOR_LLM,
            "tool_choice": "auto",  # Let model output text and/or tools
        }

        if uses_new_api:
            api_params["max_completion_tokens"] = self.config.max_tokens
        else:
            api_params["max_tokens"] = self.config.max_tokens

        response = raw_client.chat.completions.create(**api_params)

        # Track cost
        if response.usage:
            self._track_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                client.get_model_name(),
            )

        # Parse message and tool calls from response
        tool_calls = []
        response_message = response.choices[0].message

        # Get the text content (message to buyer)
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

        # If no output at all, use fallback
        if not tool_calls and not text_content:
            return None, self._fallback_action()

        # Post-process tool calls
        filtered_tool_calls = _postprocess_tool_calls(tool_calls)

        # GUARDRAIL: If propose_plan without message, re-prompt once
        if _has_propose_plan(filtered_tool_calls) and not (text_content and text_content.strip()):
            text_content = self._reprompt_for_message(client, messages, filtered_tool_calls)

        return text_content, filtered_tool_calls

    def _call_anthropic(
        self, client: LLMClient, messages: list
    ) -> Tuple[Optional[str], list[ToolCall]]:
        """Call Anthropic API with tool use.

        Returns:
            Tuple of (message, tool_calls).
        """
        import anthropic

        if hasattr(client, "_client") and isinstance(client._client, anthropic.Anthropic):
            raw_client = client._client
        else:
            raw_client = anthropic.Anthropic(api_key=self._api_key)

        # Convert tools to Anthropic format (Anthropic accepts dots in names)
        anthropic_tools = [
            {
                "name": _to_api_name(tool),
                "description": f"Call the {tool} tool",
                "input_schema": schema,
            }
            for tool, schema in get_all_tool_schemas().items()
        ]

        # Filter out system message
        chat_messages = [m for m in messages if m["role"] != "system"]

        response = raw_client.messages.create(
            model=client.get_model_name(),
            max_tokens=self.config.max_tokens,
            system=SELLER_SYSTEM_PROMPT,
            messages=chat_messages,
            tools=anthropic_tools,
        )

        # Track cost
        self._track_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            client.get_model_name(),
        )

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

        # If no output at all, use fallback
        if not tool_calls and not text_content:
            return None, self._fallback_action()

        # Post-process tool calls
        filtered_tool_calls = _postprocess_tool_calls(tool_calls)

        # GUARDRAIL: If propose_plan without message, re-prompt once
        if _has_propose_plan(filtered_tool_calls) and not (text_content and text_content.strip()):
            text_content = self._reprompt_for_message_anthropic(
                client, messages, filtered_tool_calls
            )

        return text_content, filtered_tool_calls

    def _call_with_json(
        self, client: LLMClient, messages: list
    ) -> Tuple[Optional[str], list[ToolCall]]:
        """Call using JSON-based tool calling (for providers without native tool support).

        Returns:
            Tuple of (message, tool_calls).
        """
        response = client.complete(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
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

            if tool_calls or message:
                return message, _postprocess_tool_calls(tool_calls)
        except (json.JSONDecodeError, KeyError):
            pass

        return None, self._fallback_action()

    def _reprompt_for_message(
        self,
        client: LLMClient,
        messages: list,
        tool_calls: list[ToolCall],
    ) -> Optional[str]:
        """Re-prompt LLM to provide spoken message for propose_plan.

        When the model calls propose_plan without any text content, we make
        a second call (no tools) asking for just the spoken pitch.

        Args:
            client: The LLM client.
            messages: Original conversation messages.
            tool_calls: The tool calls that were made (includes propose_plan).

        Returns:
            The spoken message text, or None if still missing.
        """
        import openai

        # Extract propose_plan details for context
        propose_call = next(
            (tc for tc in tool_calls if tc.tool_name == "calling.propose_plan"), None
        )
        if not propose_call:
            return None

        args = propose_call.arguments or {}
        plan_id = args.get("plan_id", "insurance plan")
        premium = args.get("monthly_premium", "")
        coverage = args.get("coverage_amount", "")

        # Build a focused re-prompt
        reprompt_message = (
            f"You just called propose_plan for {plan_id} "
            f"(${premium}/month, ${coverage} coverage) but you didn't include "
            "any spoken message to the buyer. The buyer cannot see tool calls - "
            "they only hear what you say.\n\n"
            "Please provide ONLY your spoken pitch (1-3 sentences) that presents "
            "this offer to the buyer. Do not include any tool calls or explanations."
        )

        # Make a second call without tools
        reprompt_messages = messages + [{"role": "user", "content": reprompt_message}]

        if hasattr(client, "_client") and isinstance(client._client, openai.OpenAI):
            raw_client = client._client
        else:
            raw_client = openai.OpenAI(
                api_key=self._api_key,
                base_url=getattr(client, "_base_url", None),
            )

        model_name = client.get_model_name()
        uses_new_api = any(x in model_name for x in ["gpt-5", "gpt-4.1", "o1", "o3"])

        api_params = {
            "model": model_name,
            "messages": reprompt_messages,
            "temperature": self.config.temperature,
        }

        if uses_new_api:
            api_params["max_completion_tokens"] = 256
        else:
            api_params["max_tokens"] = 256

        response = raw_client.chat.completions.create(**api_params)

        # Track cost
        if response.usage:
            self._track_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                client.get_model_name(),
            )

        text_content = response.choices[0].message.content
        if text_content and text_content.strip():
            return text_content.strip()
        return None

    def _reprompt_for_message_anthropic(
        self,
        client: LLMClient,
        messages: list,
        tool_calls: list[ToolCall],
    ) -> Optional[str]:
        """Re-prompt Anthropic LLM to provide spoken message for propose_plan.

        Args:
            client: The Anthropic client.
            messages: Original conversation messages.
            tool_calls: The tool calls that were made (includes propose_plan).

        Returns:
            The spoken message text, or None if still missing.
        """
        import anthropic

        # Extract propose_plan details for context
        propose_call = next(
            (tc for tc in tool_calls if tc.tool_name == "calling.propose_plan"), None
        )
        if not propose_call:
            return None

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

        if hasattr(client, "_client") and isinstance(client._client, anthropic.Anthropic):
            raw_client = client._client
        else:
            raw_client = anthropic.Anthropic(api_key=self._api_key)

        # Filter out system message and add reprompt
        chat_messages = [m for m in messages if m["role"] != "system"]
        chat_messages.append({"role": "user", "content": reprompt_message})

        response = raw_client.messages.create(
            model=client.get_model_name(),
            max_tokens=256,
            system=SELLER_SYSTEM_PROMPT,
            messages=chat_messages,
        )

        # Track cost
        self._track_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            client.get_model_name(),
        )

        # Extract text content
        for content in response.content:
            if content.type == "text" and content.text and content.text.strip():
                return content.text.strip()
        return None

    def _fallback_action(self) -> list[ToolCall]:
        """Raise error when LLM returns nothing - no silent fallbacks."""
        raise RuntimeError("LLM returned no output (no message and no tool calls)")


class StreamingLLMSellerAgent(LLMSellerAgent):
    """LLM seller with streaming support for real-time feedback."""

    def __init__(
        self,
        config: Optional[SellerConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        on_token: Optional[callable] = None,
    ):
        """Initialize streaming LLM seller.

        Args:
            config: Agent configuration.
            provider: LLM provider.
            model: Model name.
            api_key: API key.
            on_token: Callback for each streamed token.
        """
        super().__init__(config, provider, model, api_key)
        self.on_token = on_token or (lambda x: None)

    def _call_with_tools(
        self, client: LLMClient, messages: list
    ) -> Tuple[Optional[str], list[ToolCall]]:
        """Call with streaming for OpenAI-compatible APIs.

        Returns:
            Tuple of (message, tool_calls).
        """
        import openai

        if hasattr(client, "_client") and isinstance(client._client, openai.OpenAI):
            raw_client = client._client
        else:
            raw_client = openai.OpenAI(api_key=self._api_key)

        model_name = client.get_model_name()

        # Newer models (gpt-5.x, o1, o3, etc.) use max_completion_tokens
        uses_new_api = any(x in model_name for x in ["gpt-5", "gpt-4.1", "o1", "o3"])

        api_params = {
            "model": model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "tools": TOOLS_FOR_LLM,
            "tool_choice": "auto",  # Let model output text and/or tools
            "stream": True,
        }

        if uses_new_api:
            api_params["max_completion_tokens"] = self.config.max_tokens
        else:
            api_params["max_tokens"] = self.config.max_tokens

        stream = raw_client.chat.completions.create(**api_params)

        # Collect streamed content
        tool_calls_data = {}
        text_content = ""

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Collect text content
            if delta.content:
                text_content += delta.content
                self.on_token(delta.content)

            # Collect tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index not in tool_calls_data:
                        tool_calls_data[tc.index] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[tc.index]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[tc.index]["arguments"] += tc.function.arguments

        # Parse collected tool calls
        tool_calls = []
        for _, data in sorted(tool_calls_data.items()):
            try:
                args = json.loads(data["arguments"])
                tool_calls.append(
                    ToolCall(
                        tool_name=_from_api_name(data["name"]),
                        arguments=args,
                        call_id=data["id"],
                    )
                )
            except json.JSONDecodeError:
                continue

        # If no output at all, use fallback
        if not tool_calls and not text_content:
            return None, self._fallback_action()

        # Post-process tool calls
        filtered_tool_calls = _postprocess_tool_calls(tool_calls)

        # GUARDRAIL: If propose_plan without message, re-prompt once
        if _has_propose_plan(filtered_tool_calls) and not (text_content and text_content.strip()):
            text_content = self._reprompt_for_message(client, messages, filtered_tool_calls)

        return text_content or None, filtered_tool_calls


class ReActSellerAgent(LLMSellerAgent):
    """ReAct-style seller agent with explicit reasoning.

    Uses a Reason-Act loop where the agent explicitly reasons
    before each action.
    """

    def _build_user_message(self, obs: SellerObservation) -> str:
        """Build ReAct-style user message."""
        base_message = super()._build_user_message(obs)

        react_prompt = """
## ReAct Instructions

Before deciding on your response, explicitly reason about:
1. **Observation**: What do you observe from the current state?
2. **Thought**: What strategy should you use? Why?
3. **Action**: What message will you send and/or what tools will you call?

Format your response as:
{
    "observation": "What I see...",
    "thought": "My reasoning...",
    "action": {
        "message": "Your message to the buyer (if in a call)",
        "tool_calls": [...]
    }
}
"""
        return base_message + react_prompt

    def _call_llm(
        self,
        user_message: str,
        episode_context: Optional["EpisodeContext"] = None,
    ) -> Tuple[Optional[str], list[ToolCall]]:
        """Call LLM with ReAct-style prompting.

        Args:
            user_message: The current user message (observation).
            episode_context: Optional episode context for conversation history.

        Returns:
            Tuple of (message, tool_calls).
        """
        client = self._get_client()

        messages = [{"role": "system", "content": SELLER_SYSTEM_PROMPT}]

        # Use episode context if provided, otherwise fall back to sliding window
        if episode_context is not None:
            context_messages = episode_context.get_seller_view()
            messages.extend(context_messages)
            messages.append({"role": "user", "content": user_message})
        else:
            messages.extend(self._conversation_history[-20:])

        response = client.complete(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            content = json.loads(response.content)
            # Store reasoning for debugging
            self._last_reasoning = {
                "observation": content.get("observation", ""),
                "thought": content.get("thought", ""),
            }

            # Extract message and tool calls
            action = content.get("action", content)
            message = action.get("message")
            raw_calls = action.get("tool_calls", [])

            tool_calls = []
            for tc_data in raw_calls:
                if isinstance(tc_data, dict):
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc_data.get("tool_name", ""),
                            arguments=tc_data.get("arguments", {}),
                        )
                    )

            if tool_calls or message:
                return message, _postprocess_tool_calls(tool_calls)
        except (json.JSONDecodeError, KeyError):
            pass

        return None, self._fallback_action()

    def get_stats(self) -> dict[str, Any]:
        """Include last reasoning in stats."""
        stats = super().get_stats()
        if hasattr(self, "_last_reasoning"):
            stats["last_reasoning"] = self._last_reasoning
        return stats
