"""Context management for LLM calls.

This module provides comprehensive context management including:
- Message buffers (sliding window, priority-based, key-event tracking)
- Compaction strategies (summary-based, key-events, tool results)
- Context policies for different scenarios
- Serialization for checkpointing and debugging
"""

from salesbench.context.manager import ContextManager, ConversationContext

# Buffers
from salesbench.context.buffers import (
    Message,
    MessageBuffer,
    SlidingWindowBuffer,
    PriorityBuffer,
    KeyEventBuffer,
)

# Compaction
from salesbench.context.compaction import (
    Compactor,
    SimpleSummaryCompactor,
    KeyEventsCompactor,
)
from salesbench.context.compaction.simple_summary import LLMSummaryCompactor
from salesbench.context.compaction.key_events import ToolResultCompactor

# Policies
from salesbench.context.policies import (
    ContextBudget,
    ContextPolicy,
    DefaultContextPolicy,
    AggressiveCompactionPolicy,
    PreserveRecentPolicy,
    CallContextPolicy,
)

# Serializers
from salesbench.context.serializers import (
    ContextSerializer,
    JSONContextSerializer,
    CompactContextSerializer,
    DebugContextSerializer,
    CheckpointSerializer,
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
