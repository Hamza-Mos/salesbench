"""Context management for LLM calls.

Provides episode context management with:
- Anchored state (critical info that survives compaction)
- Observation masking (efficient token reduction)
- Model-aware compression triggers
"""

from salesbench.context.buffers import (
    KeyEventBuffer,
    Message,
    MessageBuffer,
    SimpleCompactBuffer,
    SlidingWindowBuffer,
)
from salesbench.context.episode import (
    AnchoredState,
    BuyerContextManager,
    DecisionRecord,
    EpisodeContext,
    LeadConversation,
    LeadSummary,
    SellerContextManager,
)

__all__ = [
    # Episode context
    "EpisodeContext",
    "LeadConversation",
    "AnchoredState",
    "LeadSummary",
    "DecisionRecord",
    "SellerContextManager",
    "BuyerContextManager",
    # Buffers
    "Message",
    "MessageBuffer",
    "SlidingWindowBuffer",
    "KeyEventBuffer",
    "SimpleCompactBuffer",
]
