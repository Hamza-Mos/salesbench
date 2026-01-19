"""Context management for LLM interactions.

Manages conversation context and message history for both
buyer simulator and seller agent interactions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationContext:
    """Context for a single conversation/call."""

    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 50

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role (system, user, assistant).
            content: Message content.
            **metadata: Additional metadata.
        """
        self.messages.append(Message(role=role, content=content, metadata=metadata))

        # Trim if too long
        if len(self.messages) > self.max_messages:
            # Keep system prompt context but trim middle messages
            self.messages = self.messages[-self.max_messages :]

    def to_api_messages(self) -> list[dict[str, str]]:
        """Convert to format for API calls.

        Returns:
            List of message dicts with role and content.
        """
        result = [{"role": "system", "content": self.system_prompt}]
        result.extend(msg.to_dict() for msg in self.messages)
        return result

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []


class ContextManager:
    """Manages multiple conversation contexts."""

    def __init__(self):
        """Initialize the context manager."""
        self._contexts: dict[str, ConversationContext] = {}

    def create_context(
        self,
        context_id: str,
        system_prompt: str,
        max_messages: int = 50,
    ) -> ConversationContext:
        """Create a new conversation context.

        Args:
            context_id: Unique identifier for the context.
            system_prompt: System prompt for the conversation.
            max_messages: Maximum messages to retain.

        Returns:
            The created ConversationContext.
        """
        context = ConversationContext(
            system_prompt=system_prompt,
            max_messages=max_messages,
        )
        self._contexts[context_id] = context
        return context

    def get_context(self, context_id: str) -> Optional[ConversationContext]:
        """Get an existing context.

        Args:
            context_id: Context identifier.

        Returns:
            The ConversationContext if found, None otherwise.
        """
        return self._contexts.get(context_id)

    def delete_context(self, context_id: str) -> bool:
        """Delete a context.

        Args:
            context_id: Context identifier.

        Returns:
            True if deleted, False if not found.
        """
        if context_id in self._contexts:
            del self._contexts[context_id]
            return True
        return False

    def clear_all(self) -> None:
        """Clear all contexts."""
        self._contexts.clear()
