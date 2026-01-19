"""Context serialization for checkpointing and debugging."""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from salesbench.context.buffers import Message, MessageBuffer


class ContextSerializer(ABC):
    """Abstract base class for context serializers."""

    @abstractmethod
    def serialize(self, messages: List["Message"]) -> str:
        """Serialize messages to string."""
        pass

    @abstractmethod
    def deserialize(self, data: str) -> List["Message"]:
        """Deserialize string to messages."""
        pass

    def serialize_buffer(self, buffer: "MessageBuffer") -> str:
        """Serialize a message buffer."""
        return self.serialize(buffer.get_messages())


class JSONContextSerializer(ContextSerializer):
    """JSON-based context serializer.

    Produces human-readable JSON with full message details.
    """

    def __init__(self, indent: int = 2, include_metadata: bool = True):
        self.indent = indent
        self.include_metadata = include_metadata

    def serialize(self, messages: List["Message"]) -> str:
        data = []
        for msg in messages:
            entry = {
                "role": msg.role,
                "content": msg.content,
                "priority": msg.priority,
                "timestamp": msg.timestamp,
            }
            if msg.name:
                entry["name"] = msg.name
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if self.include_metadata and msg.metadata:
                entry["metadata"] = msg.metadata
            data.append(entry)

        return json.dumps(data, indent=self.indent)

    def deserialize(self, data: str) -> List["Message"]:
        from salesbench.context.buffers import Message

        entries = json.loads(data)
        messages = []

        for entry in entries:
            msg = Message(
                role=entry["role"],
                content=entry["content"],
                name=entry.get("name"),
                tool_call_id=entry.get("tool_call_id"),
                metadata=entry.get("metadata", {}),
                priority=entry.get("priority", 0),
                timestamp=entry.get("timestamp", 0),
            )
            messages.append(msg)

        return messages


class CompactContextSerializer(ContextSerializer):
    """Compact serializer for efficient storage.

    Produces minimal JSON with abbreviated keys.
    """

    # Key abbreviations
    KEY_MAP = {
        "role": "r",
        "content": "c",
        "name": "n",
        "tool_call_id": "t",
        "priority": "p",
        "timestamp": "ts",
    }

    REVERSE_KEY_MAP = {v: k for k, v in KEY_MAP.items()}

    def serialize(self, messages: List["Message"]) -> str:
        data = []
        for msg in messages:
            entry = {
                "r": msg.role,
                "c": msg.content,
            }
            if msg.name:
                entry["n"] = msg.name
            if msg.tool_call_id:
                entry["t"] = msg.tool_call_id
            if msg.priority:
                entry["p"] = msg.priority
            if msg.timestamp:
                entry["ts"] = msg.timestamp
            data.append(entry)

        return json.dumps(data, separators=(",", ":"))

    def deserialize(self, data: str) -> List["Message"]:
        from salesbench.context.buffers import Message

        entries = json.loads(data)
        messages = []

        for entry in entries:
            msg = Message(
                role=entry["r"],
                content=entry["c"],
                name=entry.get("n"),
                tool_call_id=entry.get("t"),
                priority=entry.get("p", 0),
                timestamp=entry.get("ts", 0),
            )
            messages.append(msg)

        return messages


class DebugContextSerializer(ContextSerializer):
    """Debug serializer with verbose, human-readable output.

    Useful for logging and debugging context issues.
    """

    def __init__(self, max_content_length: int = 200):
        self.max_content_length = max_content_length

    def serialize(self, messages: List["Message"]) -> str:
        lines = []
        lines.append(f"=== Context ({len(messages)} messages) ===")
        lines.append(f"Generated at: {datetime.utcnow().isoformat()}")
        lines.append("")

        total_tokens = 0
        for i, msg in enumerate(messages):
            tokens = msg.token_estimate()
            total_tokens += tokens

            content = msg.content
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."

            lines.append(f"[{i}] {msg.role.upper()} (p={msg.priority}, ~{tokens} tokens)")
            if msg.name:
                lines.append(f"    Tool: {msg.name}")
            lines.append(f"    {content}")
            lines.append("")

        lines.append(f"=== Total: ~{total_tokens} tokens ===")

        return "\n".join(lines)

    def deserialize(self, data: str) -> List["Message"]:
        # Debug format is not meant to be deserialized
        raise NotImplementedError("Debug format cannot be deserialized")


class CheckpointSerializer:
    """Serializer for full context checkpoints.

    Includes buffer state, policy settings, and metadata.
    """

    def __init__(self, base_serializer: Optional[ContextSerializer] = None):
        self.base_serializer = base_serializer or JSONContextSerializer()

    def create_checkpoint(
        self,
        buffer: "MessageBuffer",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Create a checkpoint of the context state.

        Args:
            buffer: Message buffer to checkpoint.
            metadata: Optional metadata to include.

        Returns:
            Checkpoint dict that can be serialized.
        """
        messages = buffer.get_messages()

        checkpoint = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "message_count": len(messages),
            "token_count": buffer.token_count(),
            "messages": json.loads(self.base_serializer.serialize(messages)),
        }

        if metadata:
            checkpoint["metadata"] = metadata

        return checkpoint

    def restore_checkpoint(
        self,
        checkpoint: dict,
        buffer: "MessageBuffer",
    ) -> None:
        """Restore a checkpoint to a buffer.

        Args:
            checkpoint: Checkpoint dict.
            buffer: Buffer to restore to.
        """
        buffer.clear()

        messages_json = json.dumps(checkpoint["messages"])
        messages = self.base_serializer.deserialize(messages_json)

        for msg in messages:
            buffer.add(msg)

    def to_json(self, checkpoint: dict) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(checkpoint, indent=2)

    def from_json(self, data: str) -> dict:
        """Deserialize checkpoint from JSON string."""
        return json.loads(data)
