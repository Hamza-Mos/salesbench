"""Context management for LLM calls.

Provides episode context management with:
- Anchored state (critical info that survives compaction)
- LLM-based compaction (model summarizes older messages)
- Model-aware compression triggers
"""

from salesbench.context.anchored_state import (
    AnchoredState,
    DecisionRecord,
    LeadSummary,
)
from salesbench.context.buffers import (
    LLMCompactBuffer,
    Message,
    MessageBuffer,
)
from salesbench.context.compaction import (
    BUYER_COMPACTION_PROMPT,
    SELLER_COMPACTION_PROMPT,
    create_buyer_compaction_fn,
    create_seller_compaction_fn,
)
from salesbench.context.episode import (
    BuyerContextManager,
    EpisodeContext,
    LeadConversation,
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
    "LLMCompactBuffer",
    # Compaction
    "SELLER_COMPACTION_PROMPT",
    "BUYER_COMPACTION_PROMPT",
    "create_seller_compaction_fn",
    "create_buyer_compaction_fn",
]
