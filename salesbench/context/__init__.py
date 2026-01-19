"""Context management for LLM calls.

This module provides comprehensive context management including:
- Message buffers (sliding window, priority-based, key-event tracking)
- Compaction strategies (summary-based, key-events, tool results)
- Context policies for different scenarios
- Serialization for checkpointing and debugging
"""

# Buffers
from salesbench.context.buffers import (
    KeyEventBuffer,
    Message,
    MessageBuffer,
    PriorityBuffer,
    SlidingWindowBuffer,
)

# Compaction
from salesbench.context.compaction import (
    Compactor,
    KeyEventsCompactor,
    SimpleSummaryCompactor,
)
from salesbench.context.compaction.key_events import ToolResultCompactor
from salesbench.context.compaction.simple_summary import LLMSummaryCompactor
from salesbench.context.manager import ContextManager, ConversationContext

# Policies
from salesbench.context.policies import (
    AggressiveCompactionPolicy,
    CallContextPolicy,
    ContextBudget,
    ContextPolicy,
    DefaultContextPolicy,
    PreserveRecentPolicy,
)

# Serializers
from salesbench.context.serializers import (
    CheckpointSerializer,
    CompactContextSerializer,
    ContextSerializer,
    DebugContextSerializer,
    JSONContextSerializer,
)

__all__ = [
    # Manager
    "ContextManager",
    "ConversationContext",
    # Buffers
    "Message",
    "MessageBuffer",
    "SlidingWindowBuffer",
    "PriorityBuffer",
    "KeyEventBuffer",
    # Compaction
    "Compactor",
    "SimpleSummaryCompactor",
    "LLMSummaryCompactor",
    "KeyEventsCompactor",
    "ToolResultCompactor",
    # Policies
    "ContextBudget",
    "ContextPolicy",
    "DefaultContextPolicy",
    "AggressiveCompactionPolicy",
    "PreserveRecentPolicy",
    "CallContextPolicy",
    # Serializers
    "ContextSerializer",
    "JSONContextSerializer",
    "CompactContextSerializer",
    "DebugContextSerializer",
    "CheckpointSerializer",
]
