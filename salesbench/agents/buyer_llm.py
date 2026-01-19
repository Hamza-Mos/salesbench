"""LLM-based buyer simulator.

The buyer simulator is invoked ONLY by calling.propose_plan and returns
ONLY structured decisions (ACCEPT_PLAN, REJECT_PLAN, END_CALL).

Supports multiple LLM providers via the modular adapter:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- OpenRouter (100+ models)
- xAI (Grok)
- Together AI (Llama, Mixtral)
- Google (Gemini)
"""

import json
from typing import Callable, Optional

from salesbench.core.types import (
    BuyerDecision,
    BuyerResponseData,
    CallSession,
    PlanOffer,
)
from salesbench.envs.sales_mvp.personas import Persona
from salesbench.llm import LLMClient, create_client, detect_available_provider

# Type alias for the buyer simulator function
# Args: persona, offer, session, seller_pitch (optional)
BuyerSimulatorFn = Callable[[Persona, PlanOffer, CallSession, Optional[str]], BuyerResponseData]


BUYER_SYSTEM_PROMPT = """You are simulating a potential insurance buyer in a cold-call scenario.
You will receive information about your persona (who you are) and an insurance offer.

Your task is to decide whether to:
1. ACCEPT_PLAN - Accept the offered insurance plan
2. REJECT_PLAN - Reject this specific offer but stay on the call
3. END_CALL - End the call entirely

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{
    "decision": "ACCEPT_PLAN" | "REJECT_PLAN" | "END_CALL",
    "reason": "Brief explanation (1-2 sentences)",
    "dialogue": "What you actually say to the salesperson (1-3 sentences, in first person, conversational)"
}

The "dialogue" field is what you verbally say to the salesperson. Make it natural and human:
- For ACCEPT_PLAN: Express agreement, maybe mention what convinced you
- For REJECT_PLAN: Voice your specific objection based on your persona's style
- For END_CALL: Say goodbye or express frustration, depending on your mood

Base your decision on:
- Your persona's financial situation (can you afford this?)
- Your persona's trust level and interest in insurance
- Your persona's patience (how many offers have been presented?)
- Whether the coverage meets your needs
- Your objection style
- Your life situation and triggers

Guidelines for realistic behavior:
- HOT leads: Very interested, ready to buy if price is reasonable
- WARM leads: Interested but need convincing, will accept good offers
- LUKEWARM leads: Skeptical, need excellent value to accept
- COLD leads: Not interested, rarely accept unless exceptional deal
- HOSTILE leads: Do not want to be called, will end call quickly

DO NOT:
- Accept offers you cannot afford
- Be unrealistically easy or difficult
- Ignore your persona's characteristics
- Accept on the first offer if you're a cold/hostile lead
"""


def create_buyer_prompt(
    persona: Persona,
    offer: PlanOffer,
    session: CallSession,
    seller_pitch: Optional[str] = None,
) -> str:
    """Create the prompt for the buyer LLM.

    Args:
        persona: The buyer persona.
        offer: The plan being offered.
        session: Current call session.
        seller_pitch: What the salesperson said when presenting the offer.

    Returns:
        Formatted prompt string.
    """
    # Calculate affordability
    monthly_income = persona.annual_income / 12
    max_affordable = monthly_income * persona.hidden.close_threshold
    is_affordable = offer.monthly_premium <= max_affordable

    prompt = f"""## Your Persona
Name: {persona.name}
Age: {persona.age}
Job: {persona.job}
Annual Income: ${persona.annual_income:,}
Monthly Income: ${monthly_income:,.2f}
Household: {persona.household_size} people ({persona.num_dependents} dependents)
Has Spouse: {"Yes" if persona.has_spouse else "No"}

## Your Internal State (your true feelings)
- Trust in insurance salespeople: {persona.hidden.trust:.0%}
- Interest in buying insurance: {persona.hidden.interest:.0%}
- Patience remaining: {persona.hidden.patience:.0%}
- Maximum you'd pay monthly: ${max_affordable:.2f} ({persona.hidden.close_threshold:.1%} of income)

## Your Objection Style
You tend to object in a {persona.objection_style.value} manner.

## Life Trigger (why you might need insurance)
{persona.trigger}

## Current Temperature
You are a {persona.temperature.value.upper()} lead.

## Call Context
- Offers presented so far: {len(session.offers_presented)}
- Call duration: {session.duration_minutes} minutes

## The Offer Being Presented
- Plan Type: {offer.plan_id.value}
- Monthly Premium: ${offer.monthly_premium:.2f}
- Coverage Amount: ${offer.coverage_amount:,.0f}
- Proposed Next Step: {offer.next_step.value}
{"- Term: " + str(offer.term_years) + " years" if offer.term_years else ""}

## Affordability Check
This offer is {"AFFORDABLE" if is_affordable else "NOT AFFORDABLE"} for you.
(Premium ${offer.monthly_premium:.2f} vs your max ${max_affordable:.2f})

## What the Salesperson Said
{seller_pitch if seller_pitch else "(The salesperson presented the offer without additional commentary.)"}

Based on all this information, what is your decision?
Remember to respond with ONLY the JSON object."""

    return prompt


class LLMBuyerSimulator:
    """LLM-based buyer simulator with multi-provider support."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
    ):
        """Initialize the buyer simulator.

        Args:
            provider: LLM provider (openai, anthropic, openrouter, xai, together, google).
            model: Model name. Uses provider default if not specified.
            temperature: Sampling temperature (lower = more deterministic).
            api_key: API key (uses env var if not provided).

        Raises:
            ValueError: If no API key is configured for the provider.
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._client: Optional[LLMClient] = None

    def _get_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            self._client = create_client(
                provider=self.provider,
                model=self.model,
                api_key=self._api_key,
            )
        return self._client

    def __call__(
        self,
        persona: Persona,
        offer: PlanOffer,
        session: CallSession,
        seller_pitch: Optional[str] = None,
    ) -> BuyerResponseData:
        """Simulate buyer decision using LLM.

        Args:
            persona: The buyer persona.
            offer: The plan being offered.
            session: Current call session.
            seller_pitch: What the salesperson said when presenting the offer.

        Returns:
            BuyerResponseData with decision and reason.

        Raises:
            RuntimeError: If LLM call fails.
        """
        prompt = create_buyer_prompt(persona, offer, session, seller_pitch)

        try:
            client = self._get_client()
            response = client.complete(
                messages=[
                    {"role": "system", "content": BUYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=250,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.content)

            decision = BuyerDecision(result["decision"].lower())
            reason = result.get("reason", "")
            dialogue = result.get("dialogue", "")

            return BuyerResponseData(decision=decision, reason=reason, dialogue=dialogue)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {e}")
        except KeyError as e:
            raise RuntimeError(f"LLM response missing required field: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM buyer simulation failed: {e}")


def create_buyer_simulator(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> BuyerSimulatorFn:
    """Create an LLM-based buyer simulator function.

    Args:
        provider: LLM provider. Auto-detects from environment if not specified.
        model: Model name. Uses provider default if not specified.
        temperature: LLM temperature.
        api_key: API key for the provider.

    Returns:
        Buyer simulator function.

    Raises:
        ValueError: If no provider is specified and none can be auto-detected.

    Examples:
        # Auto-detect provider from environment
        simulator = create_buyer_simulator()

        # Use specific provider
        simulator = create_buyer_simulator(provider="anthropic", model="claude-3-5-sonnet-20241022")

        # Use OpenRouter for any model
        simulator = create_buyer_simulator(provider="openrouter", model="meta-llama/llama-3.1-70b-instruct")
    """
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

    return LLMBuyerSimulator(
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
