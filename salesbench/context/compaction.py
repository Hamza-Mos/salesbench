"""LLM-based context compaction for conversation history.

Provides compaction prompts and factory functions for creating LLM-based
compaction functions for both seller and buyer agents.
"""

import logging
from typing import Awaitable, Callable

from salesbench.llm.client import create_client

logger = logging.getLogger(__name__)


SELLER_COMPACTION_PROMPT = """You are summarizing a sales agent's conversation history.
The agent needs to remember key facts to work effectively.

Messages to summarize:
{messages}

Create a MEMORY SUMMARY. Include:
- Conversations with each lead (discussed, objections, offers)
- Lead responses and decisions
- Key facts learned from tools (not raw data)
- Patterns (what works, what doesn't)

Format as concise bullets organized by lead. Skip raw tool output."""


BUYER_COMPACTION_PROMPT = """You are summarizing a sales conversation from the buyer's perspective.
You need to remember key facts to behave consistently.

Conversation:
{dialogue}

Create a MEMORY SUMMARY. Include:
- Offers received (plan, price, coverage) and your response
- Budget/affordability statements you made
- Objections and concerns you raised
- Trust signals about the seller
- Personal details you shared

Format as concise bullets. Skip filler."""


def create_seller_compaction_fn(
    provider: str,
    model: str,
) -> Callable[[str], Awaitable[str]]:
    """Create a compaction function for seller context.

    Uses the specified LLM to summarize older messages into a memory summary.

    Args:
        provider: LLM provider (openai, anthropic, etc.).
        model: Model name.

    Returns:
        Async function that takes message text and returns a summary.
    """
    client = create_client(provider=provider, model=model)

    async def compact(text: str) -> str:
        prompt = SELLER_COMPACTION_PROMPT.format(messages=text)
        response = client.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.0,  # Deterministic for reproducibility
        )
        logger.debug(f"[Seller Compaction] Generated summary: {len(response.content)} chars")
        return response.content

    return compact


def create_buyer_compaction_fn(
    provider: str,
    model: str,
) -> Callable[[str], Awaitable[str]]:
    """Create a compaction function for buyer context.

    Uses the specified LLM to summarize older dialogue into a memory summary.

    Args:
        provider: LLM provider (openai, anthropic, etc.).
        model: Model name.

    Returns:
        Async function that takes dialogue text and returns a summary.
    """
    client = create_client(provider=provider, model=model)

    async def compact(text: str) -> str:
        prompt = BUYER_COMPACTION_PROMPT.format(dialogue=text)
        response = client.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.0,  # Deterministic for reproducibility
        )
        logger.debug(f"[Buyer Compaction] Generated summary: {len(response.content)} chars")
        return response.content

    return compact
