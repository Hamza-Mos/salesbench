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

**IMPORTANT**: Always include a `pitch` parameter when calling `calling.propose_plan`. This is what you SAY to the buyer. Make it conversational and persuasive based on their situation.

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
Turn 3: MESSAGE: "Hi John! I'm calling about some great term life options for your family..."
        + calling.propose_plan(plan_id="TERM", monthly_premium=150, coverage_amount=500000, next_step="close_now")
Turn 4: [Based on buyer response] MESSAGE: "I understand your concern about the premium..."
        + calling.end_call() or another propose_plan
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
        self._available_leads: list[dict] = []  # Track leads found but not yet called
        self._called_lead_ids: set[str] = set()  # Track which leads we've called
        self._accepted_lead_ids: set[str] = set()  # Track leads who accepted (don't call again)
        self._exhausted_temps: set[str] = set()  # Temperature tiers that returned 0 leads
        self._last_search_temp: Optional[str] = None  # Last temperature searched

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
        self._exhausted_temps = set()
        self._last_search_temp = None

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
        # Build user message from observation (also updates lead tracking)
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

        # HARD CONSTRAINT: Intercept redundant search_leads when we have uncalled leads
        # This prevents models from looping on search forever
        uncalled_leads = [
            lead
            for lead in self._available_leads
            if lead.get("lead_id") not in self._called_lead_ids
            and lead.get("lead_id") not in self._accepted_lead_ids
        ]

        if not observation.in_call:
            corrected_calls = []
            for tc in tool_calls:
                if tc.tool_name == "crm.search_leads":
                    if uncalled_leads:
                        # Replace search with a call to the first available lead
                        lead_id = uncalled_leads[0].get("lead_id", "")
                        corrected_calls.append(
                            ToolCall(
                                tool_name="calling.start_call",
                                arguments={"lead_id": lead_id},
                                call_id=tc.call_id,
                            )
                        )
                    else:
                        # Check if searching exhausted temperature - redirect to next tier
                        search_temp = tc.arguments.get("temperature", "hot")
                        if search_temp in self._exhausted_temps:
                            temp_order = ["hot", "warm", "lukewarm", "cold"]
                            next_temp = None
                            for temp in temp_order:
                                if temp not in self._exhausted_temps:
                                    next_temp = temp
                                    break
                            if next_temp:
                                # Redirect to next available tier
                                corrected_calls.append(
                                    ToolCall(
                                        tool_name="crm.search_leads",
                                        arguments={**tc.arguments, "temperature": next_temp},
                                        call_id=tc.call_id,
                                    )
                                )
                            # else: all temps exhausted, let it fail naturally
                            else:
                                corrected_calls.append(tc)
                        else:
                            corrected_calls.append(tc)
                else:
                    corrected_calls.append(tc)
            tool_calls = corrected_calls

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
        """Build user message from observation."""
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
                    "## Active Call",
                    f"- Lead: {obs.current_lead_id}",
                    f"- Duration: {obs.call_duration} minutes",
                    f"- Offers presented: {obs.offers_this_call}",
                ]
            )

        # Process tool results and update lead tracking
        if obs.last_tool_results:
            lines.extend(
                [
                    "",
                    "## Last Tool Results",
                ]
            )
            for result in obs.last_tool_results:
                if result.success:
                    # Special handling for search results - update available leads
                    if result.data and "leads" in result.data:
                        leads = result.data["leads"]
                        searched_temp = result.data.get("filters_applied", {}).get("temperature")
                        if leads:
                            # Found leads - clear this temp from exhausted
                            if searched_temp:
                                self._exhausted_temps.discard(searched_temp)
                            # Update available leads list (exclude already called AND accepted leads)
                            self._available_leads = [
                                lead
                                for lead in leads
                                if lead.get("lead_id") not in self._called_lead_ids
                                and lead.get("lead_id") not in self._accepted_lead_ids
                            ]
                            # Show all found, but mark accepted ones
                            lines.append(f"✓ Found {len(leads)} leads:")
                            for lead in leads:
                                lead_id = lead.get("lead_id", "unknown")
                                name = lead.get("name", "Unknown")
                                temp = lead.get("temperature", "?")
                                income = lead.get("annual_income", 0)
                                risk = lead.get("risk_class", "?")
                                dependents = lead.get("num_dependents", 0)
                                status = (
                                    " [ALREADY ACCEPTED - SKIP]"
                                    if lead_id in self._accepted_lead_ids
                                    else ""
                                )
                                lines.append(
                                    f"  - {lead_id}: {name} ({temp}, ${income:,}/yr, {dependents} dependents, {risk} risk){status}"
                                )
                        else:
                            # No leads found - mark this temperature as exhausted
                            if searched_temp:
                                self._exhausted_temps.add(searched_temp)
                            lines.append(
                                f"✓ Search returned 0 leads for '{searched_temp or 'this filter'}' - try next temperature tier."
                            )
                    # Track when we start a call
                    elif result.data and result.data.get("call_started"):
                        lead_id = result.data.get("lead_id", "")
                        if lead_id:
                            self._called_lead_ids.add(lead_id)
                            # Remove from available leads
                            self._available_leads = [
                                lead
                                for lead in self._available_leads
                                if lead.get("lead_id") != lead_id
                            ]
                        data_str = json.dumps(result.data, default=str)
                        lines.append(f"✓ {result.call_id}: {data_str}")
                    # Track when a lead accepts (don't call them again!)
                    elif result.data and result.data.get("decision") == "accept_plan":
                        # Mark this lead as accepted
                        if obs.current_lead_id:
                            self._accepted_lead_ids.add(obs.current_lead_id)
                        data_str = json.dumps(result.data, default=str)
                        if len(data_str) > 1000:
                            data_str = data_str[:1000] + "..."
                        lines.append(f"✓ {result.call_id}: {data_str}")
                        lines.append("  ⚠️ Lead ACCEPTED - call auto-ended, move to next lead!")
                    else:
                        # For other results, show full data (reasonably sized)
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

        # Show available leads if we have any (persisted from previous searches)
        # Exclude both called and accepted leads
        uncalled_leads = [
            lead
            for lead in self._available_leads
            if lead.get("lead_id") not in self._called_lead_ids
            and lead.get("lead_id") not in self._accepted_lead_ids
        ]

        # Add explicit next-action guidance based on state
        lines.append("")
        if obs.in_call:
            lines.append(
                "## Next Action: You are in a call. Use `calling.propose_plan` to make an offer, or `calling.end_call` to end."
            )
        elif uncalled_leads:
            # We have uncalled leads - strongly guide to call one
            first_lead = uncalled_leads[0]
            first_lead_id = first_lead.get("lead_id", "")
            lines.append(f"## Available Leads ({len(uncalled_leads)} uncalled):")
            for lead in uncalled_leads[:5]:  # Show up to 5
                lead_id = lead.get("lead_id", "unknown")
                name = lead.get("name", "Unknown")
                lines.append(f"  - {lead_id}: {name}")
            lines.append("")
            lines.append(
                f'## Next Action: CALL A LEAD NOW using `calling.start_call(lead_id="{first_lead_id}")`'
            )
            lines.append("DO NOT SEARCH AGAIN. You already have leads to call.")
        else:
            # Show exhausted temps and suggest next tier
            temp_order = ["hot", "warm", "lukewarm", "cold"]
            next_temp = None
            for temp in temp_order:
                if temp not in self._exhausted_temps:
                    next_temp = temp
                    break

            if self._exhausted_temps:
                lines.append(
                    f"## Exhausted temperature tiers: {', '.join(sorted(self._exhausted_temps))}"
                )

            if next_temp:
                lines.append(
                    f'## Next Action: Search for leads with `crm.search_leads(temperature="{next_temp}")`, then call them.'
                )
            else:
                # All temperatures exhausted
                lines.append(
                    "## All lead temperature tiers have been searched. Episode should end."
                )

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

        return text_content, tool_calls

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

        return text_content, tool_calls

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
                return message, tool_calls
        except (json.JSONDecodeError, KeyError):
            pass

        return None, self._fallback_action()

    def _fallback_action(self) -> list[ToolCall]:
        """Generate a smart fallback action based on current state."""
        # If we have available leads, call one instead of searching again
        # Exclude accepted leads - they're already customers!
        uncalled_leads = [
            lead
            for lead in self._available_leads
            if lead.get("lead_id") not in self._called_lead_ids
            and lead.get("lead_id") not in self._accepted_lead_ids
        ]
        if uncalled_leads:
            lead_id = uncalled_leads[0].get("lead_id", "")
            return [
                ToolCall(
                    tool_name="calling.start_call",
                    arguments={"lead_id": lead_id},
                )
            ]
        # Otherwise, search for leads - use next available temperature tier
        temp_order = ["hot", "warm", "lukewarm", "cold"]
        next_temp = "hot"
        for temp in temp_order:
            if temp not in self._exhausted_temps:
                next_temp = temp
                break
        return [
            ToolCall(
                tool_name="crm.search_leads",
                arguments={"temperature": next_temp, "limit": 10},
            )
        ]


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

        return text_content or None, tool_calls


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
                return message, tool_calls
        except (json.JSONDecodeError, KeyError):
            pass

        return None, self._fallback_action()

    def get_stats(self) -> dict[str, Any]:
        """Include last reasoning in stats."""
        stats = super().get_stats()
        if hasattr(self, "_last_reasoning"):
            stats["last_reasoning"] = self._last_reasoning
        return stats
