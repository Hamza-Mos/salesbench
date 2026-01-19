"""Simple summary-based compaction."""

from typing import List, Optional, Callable
from salesbench.context.compaction.base import Compactor
from salesbench.context.buffers import Message


class SimpleSummaryCompactor(Compactor):
    """Compacts messages by summarizing older ones.

    Keeps recent messages intact and replaces older messages
    with a summary.
    """

    def __init__(
        self,
        keep_recent: int = 10,
        summary_fn: Optional[Callable[[List[Message]], str]] = None,
    ):
        """Initialize the compactor.

        Args:
            keep_recent: Number of recent messages to keep intact.
            summary_fn: Optional function to generate summaries.
                       Takes list of messages, returns summary string.
        """
        self.keep_recent = keep_recent
        self.summary_fn = summary_fn or self._default_summary

    def compact(self, messages: List[Message]) -> List[Message]:
        """Compact messages by summarizing older ones."""
        if len(messages) <= self.keep_recent:
            return messages

        # Split into old and recent
        old_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]

        # Generate summary of old messages
        summary = self.summary_fn(old_messages)

        # Create summary message
        summary_message = Message(
            role="system",
            content=f"[Conversation summary]: {summary}",
            priority=5,  # Medium priority
        )

        return [summary_message] + recent_messages

    def should_compact(self, messages: List[Message], max_tokens: int) -> bool:
        """Check if compaction is needed based on token count."""
        total_tokens = sum(m.token_estimate() for m in messages)
        # Compact when at 80% capacity
        return total_tokens > max_tokens * 0.8

    def _default_summary(self, messages: List[Message]) -> str:
        """Default summary: count messages by type."""
        role_counts = {}
        for m in messages:
            role_counts[m.role] = role_counts.get(m.role, 0) + 1

        parts = [f"{count} {role} messages" for role, count in role_counts.items()]
        return f"Previous conversation contained: {', '.join(parts)}"


class LLMSummaryCompactor(SimpleSummaryCompactor):
    """Compactor that uses an LLM to generate summaries."""

    def __init__(
        self,
        llm_client,
        keep_recent: int = 10,
        summary_prompt: Optional[str] = None,
    ):
        """Initialize with an LLM client.

        Args:
            llm_client: LLM client for generating summaries.
            keep_recent: Number of recent messages to keep.
            summary_prompt: Custom prompt for summarization.
        """
        self.llm_client = llm_client
        self.summary_prompt = summary_prompt or self._default_prompt()
        super().__init__(keep_recent=keep_recent, summary_fn=self._llm_summary)

    def _default_prompt(self) -> str:
        return """Summarize the following conversation concisely.
Focus on:
- Key decisions made
- Important information shared
- Current state of the interaction

Keep the summary under 200 words."""

    def _llm_summary(self, messages: List[Message]) -> str:
        """Generate summary using LLM."""
        # Format messages for the prompt
        formatted = []
        for m in messages:
            formatted.append(f"{m.role.upper()}: {m.content[:500]}")

        conversation = "\n".join(formatted)

        try:
            response = self.llm_client.complete(
                messages=[
                    {"role": "system", "content": self.summary_prompt},
                    {"role": "user", "content": conversation},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            return response.content
        except Exception as e:
            # Fallback to default summary on error
            return self._default_summary(messages)

    def _default_summary(self, messages: List[Message]) -> str:
        """Fallback summary."""
        return f"[{len(messages)} previous messages in conversation]"
