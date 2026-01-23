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
import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from salesbench.agents.providers import ToolCallingProvider, _to_api_name, get_provider
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

SELLER_SYSTEM_PROMPT = """You are Alex, an AI insurance sales agent. Your goal is to sell insurance plans to leads.

## HOW YOU COMMUNICATE

You have two types of output:
1. **Message**: Free-form text that the buyer sees (your actual conversation)
2. **Tool calls**: Operational/analytical actions (CRM, analytics, etc.)

When in a call, your MESSAGE is what the buyer hears. Tool calls like `calling.propose_plan` are purely for ANALYTICS - they record the offer details but do NOT speak to the buyer. You must write your pitch in the message.

## CONVERSATION QUALITY (IMPORTANT)

This benchmark expects **long, natural back-and-forth**. Behave like a professional agent:
- Keep each spoken message **short (1–4 sentences)** and **conversational**.
- Ask **one good question at a time** and **wait for the buyer's reply**.
- If the buyer asks a question or raises an objection, **answer it directly first** (don't jump to another offer).
- Do discovery early: confirm **goal**, **timeline**, **budget range**, **current coverage**, and **beneficiaries/dependents**.
- Do not repeat the same pitch verbatim. Adapt to what the buyer just said.
- Only present an offer after you have at least minimal qualification (unless the lead is clearly HOT).

## CRITICAL WORKFLOW - State Machine

You are in one of these states:
1. **NO_LEADS** → Search for leads with `crm.search_leads`
2. **HAVE_LEADS** → Call a lead with `calling.start_call(lead_id="...")`
3. **IN_CALL** → Pitch, handle objections, use `calling.propose_plan` to record offers
4. **SALE_CLOSED** → Buyer accepted! Call `calling.end_call` immediately

### STATE TRANSITIONS:
- NO_LEADS + search returns leads → HAVE_LEADS
- HAVE_LEADS + start_call → IN_CALL
- IN_CALL + buyer ACCEPTS → SALE_CLOSED (MUST call end_call)
- IN_CALL + buyer rejects → IN_CALL (may present another offer)
- IN_CALL + buyer hangs up or requests DNC → HAVE_LEADS or NO_LEADS
- SALE_CLOSED + end_call → HAVE_LEADS or NO_LEADS

## ⚠️ CRITICAL RULES - READ CAREFULLY

1. **NEVER search twice in a row.** If you just searched and got leads, your ONLY valid action is `calling.start_call`.
2. **Use the lead_id from search results.** The "Next Action" section shows you exactly which lead_id to use.
3. **If search returns 0 leads**, try a different temperature: hot → warm → lukewarm → cold.
4. **When pitching a plan**, write your pitch in the MESSAGE and call `propose_plan` to record the offer details.
5. **When buyer ACCEPTS**: The tool result will say "Buyer ACCEPTED the plan!" - your ONLY next action is `calling.end_call`. Do NOT propose again. Do NOT continue pitching. The sale is DONE.

## ⚠️ STALL DETECTION - MANDATORY ACTION REQUIRED

If you see a "STALL DETECTED" warning, you MUST take action immediately:
- **Call `calling.end_call(reason='lead_not_ready')`** - This is NOT optional
- Do NOT continue the conversation
- Do NOT ask more questions
- Do NOT repeat your pitch

**Signs you should end the call EVEN WITHOUT a stall warning:**
- Buyer keeps asking the same generic questions in a loop (e.g., "what do most people choose?")
- Buyer deflects 3+ times without engaging with specifics
- Buyer says "I'm not ready to decide" or similar multiple times
- You've explained the same concept 2+ times already
- Conversation is going in circles with no progress

**CORRECT response to stall:**
```
MESSAGE: "I understand you need more time. I'll note your interest and you can reach out when ready."
+ calling.end_call(reason='lead_not_ready')
```

**WRONG response to stall:**
```
MESSAGE: "Let me explain one more time what most people choose..."  ← NEVER DO THIS
```

Time is limited. Move on to the next lead rather than spinning wheels.

## Available Tools

### CRM Tools
- `crm.search_leads`: Search leads with filters:
  - `temperature`: hot/warm/lukewarm/cold/hostile (try different temps to find more leads)
  - `min_income`: Minimum annual income (e.g., 50000, 100000)
  - `max_age`: Maximum age (e.g., 40, 55)
  - `limit`: Max results (default 10)
  TIP: Vary filters between searches to find different prospects.
- `crm.get_lead`: Get lead details
- `crm.update_lead`: Update notes
- `crm.log_call`: Log call outcome

### Calling Tools
- `calling.start_call`: Start call with a lead - REQUIRES lead_id from search results
- `calling.propose_plan`: **ANALYTICS ONLY** - Records the offer details. Does NOT speak to buyer. Use your message to pitch.
- `calling.end_call`: End current call

**IMPORTANT**: `calling.propose_plan` does not include a `pitch` argument. The buyer hears ONLY your normal assistant message. Always write your pitch as your message text, then call `calling.propose_plan` to record the structured offer.

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

## Example - Complete Sale Flow

```
Turn 1: crm.search_leads(temperature="hot")
        → Returns: lead_abc123, lead_xyz789
Turn 2: calling.start_call(lead_id="lead_abc123")
        → Call connected with John, age 35, $80k income, 2 dependents
Turn 3: MESSAGE: "Hi John—quick question: do you already have coverage through work, and about how much?"
Turn 4: MESSAGE: "Got it. And what monthly budget range feels comfortable if this protected your family?"
Turn 5: MESSAGE: "Thanks. Based on that, here's a simple option that fits…"
        + calling.propose_plan(plan_id="TERM", monthly_premium=150, coverage_amount=500000, next_step="close_now")
        → Tool result: "Buyer ACCEPTED the plan! End the call with calling.end_call to finalize."
Turn 6: MESSAGE: "Great, I'll get the paperwork started. Thanks John!"
        + calling.end_call(reason="sale_completed")
        → DONE. Move to next lead.
```

**Handling rejections vs. acceptance:**
- IF rejection: Try different angle/price + propose_plan (up to 3 attempts)
- IF acceptance: Thank them + calling.end_call (REQUIRED - do NOT propose again)

## ⚠️ CALL STATE AWARENESS

Before using call-related tools, check your state:
- **Can only use `calling.propose_plan`** when IN_CALL (after start_call, before end_call)
- **Can only use `calling.end_call`** when IN_CALL
- **After `calling.end_call`** succeeds, you are NO LONGER in a call

The observation includes `in_call: true/false` - always check this before calling tools.

Common mistakes to avoid:
- ❌ Calling end_call after already ending the call
- ❌ Calling propose_plan after end_call without starting a new call
- ❌ Calling end_call multiple times in the same turn

## WRONG - Do Not Do This

```
Turn 1: crm.search_leads(temperature="hot") → Returns leads
Turn 2: crm.search_leads(temperature="hot") ← WRONG! You already have leads!
```

```
Turn 3: calling.propose_plan(...) with NO MESSAGE ← WRONG! The buyer won't hear anything!
```

If you have leads, CALL THEM. When pitching, ALWAYS include a message."""


def _postprocess_tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
    """De-dupe tool calls and cap certain calls to one per action.

    Some models/providers occasionally emit repeated identical tool calls in a single
    response. This breaks benchmark conversations by spamming tool execution errors.
    We keep the first occurrence (stable) and drop duplicates. Additionally, we allow
    at most one `calling.propose_plan` and one `calling.end_call` per action.
    """
    # Defensive: handle None or non-list inputs
    if not tool_calls:
        return []

    seen: set[tuple[str, str]] = set()
    filtered: list[ToolCall] = []
    propose_plan_seen = False
    end_call_seen = False

    for tc in tool_calls:
        # Cap propose_plan to one per action (stable: keep first).
        if tc.tool_name == "calling.propose_plan":
            if propose_plan_seen:
                continue
            propose_plan_seen = True

        # Cap end_call to one per action (stable: keep first).
        if tc.tool_name == "calling.end_call":
            if end_call_seen:
                continue
            end_call_seen = True

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
    if not tool_calls:
        return False
    return any(tc.tool_name == "calling.propose_plan" for tc in tool_calls)


# Build tool schemas for LLM (OpenAI format)
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
        self._provider_impl: Optional[ToolCallingProvider] = None

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            self._client = create_client(
                provider=self.provider,
                model=self.model or self.config.model,
                api_key=self._api_key,
            )
        return self._client

    def _get_provider_impl(self) -> ToolCallingProvider:
        """Get or create the provider implementation."""
        if self._provider_impl is None:
            client = self._get_client()
            self._provider_impl = get_provider(
                self.provider,
                client,
                self.model or self.config.model,
                self._api_key,
            )
        return self._provider_impl

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self._total_api_cost = 0.0
        self._turn_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

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
            episode_context: Episode context for conversation history (required).

        Returns:
            SellerAction containing message and/or tool calls.

        Raises:
            ValueError: If episode_context is not provided.
        """
        if episode_context is None:
            raise ValueError("episode_context is required - benchmark always provides it")

        # Build user message from observation
        user_message = self._build_user_message(observation, episode_context)

        # Get LLM response (returns message, tool_calls, and raw_llm_content)
        message, tool_calls, raw_llm_content = self._call_llm(user_message, episode_context)

        self._turn_count += 1
        return SellerAction(tool_calls=tool_calls, message=message, raw_llm_content=raw_llm_content)

    def _build_user_message(self, obs: SellerObservation, episode_context: "EpisodeContext") -> str:
        """Build user message from observation.

        Shows facts only - no directive language. The model decides based on context.
        The AnchoredState context block (injected separately) provides the authoritative
        state about leads, calls, and searches.
        """
        lines = [
            "## Current State",
            f"- Time elapsed: {obs.elapsed_hours}h {obs.elapsed_minutes}m",
            f"- Remaining minutes: {obs.remaining_minutes}",
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

        # Current status section - get uncalled leads from AnchoredState (single source of truth)
        lines.append("")
        uncalled_leads = episode_context._anchored_state.get_uncalled_leads()
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
        episode_context: "EpisodeContext",
    ) -> tuple[Optional[str], list[ToolCall], Optional[object]]:
        """Call LLM API with function calling.

        Args:
            user_message: The current user message (observation).
            episode_context: Episode context for conversation history (required).

        Returns:
            Tuple of (message, tool_calls, raw_llm_content) where message is optional
            free-form text, tool_calls is a list of tool calls, and raw_llm_content
            is provider-specific data (e.g., Gemini Content with thought_signature).
        """
        client = self._get_client()
        provider_impl = self._get_provider_impl()

        messages = [{"role": "system", "content": SELLER_SYSTEM_PROMPT}]

        # Trigger compaction if needed before getting seller view
        episode_context.trigger_seller_compaction_sync()

        # Get compressed episode history from context
        context_messages = episode_context.get_seller_view()
        messages.extend(context_messages)
        # Add the current observation as the latest message
        messages.append({"role": "user", "content": user_message})

        # Use provider abstraction for the call
        text_content, tool_calls, in_tokens, out_tokens, raw_llm_content = (
            provider_impl.call_with_tools(
                messages=messages,
                tools=TOOLS_FOR_LLM,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )

        # Track cost
        self._track_cost(in_tokens, out_tokens, client.get_model_name())

        # If no output at all, use fallback
        if not tool_calls and not text_content:
            return None, self._fallback_action(), None

        # Post-process tool calls
        filtered_tool_calls = _postprocess_tool_calls(tool_calls)

        # GUARDRAIL: If propose_plan without message, re-prompt once
        if _has_propose_plan(filtered_tool_calls) and not (text_content and text_content.strip()):
            logger.info("[SELLER] Triggering reprompt for propose_plan without message")
            # Pass raw_llm_content so Gemini 3 can see the assistant's response with thought_signature
            reprompt_text, reprompt_in, reprompt_out = provider_impl.reprompt_for_message(
                messages,
                filtered_tool_calls,
                SELLER_SYSTEM_PROMPT,
                self.config.temperature,
                raw_assistant_content=raw_llm_content,
            )
            self._track_cost(reprompt_in, reprompt_out, client.get_model_name())
            if reprompt_text:
                logger.info(f"[SELLER] Reprompt succeeded, got: {reprompt_text[:100]}...")
                text_content = reprompt_text
            else:
                logger.warning("[SELLER] Reprompt returned no text")

        return text_content, filtered_tool_calls, raw_llm_content

    def _fallback_action(self) -> list[ToolCall]:
        """Raise error when LLM returns nothing - no silent fallbacks."""
        raise RuntimeError("LLM returned no output (no message and no tool calls)")
