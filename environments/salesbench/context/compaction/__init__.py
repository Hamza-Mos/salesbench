"""Context compaction strategies.

Compaction reduces context size while preserving important information:
- SimpleSummaryCompactor: Summarizes older messages
- LLMSummaryCompactor: Uses LLM to generate intelligent summaries
- KeyEventsCompactor: Preserves key events, drops routine messages
- ToolResultCompactor: Optimizes tool result storage
"""

from salesbench.context.compaction.base import Compactor
from salesbench.context.compaction.simple_summary import (
    SimpleSummaryCompactor,
    LLMSummaryCompactor,
)
from salesbench.context.compaction.key_events import (
    KeyEventsCompactor,
    ToolResultCompactor,
)

__all__ = [
    "Compactor",
    "SimpleSummaryCompactor",
    "LLMSummaryCompactor",
    "KeyEventsCompactor",
    "ToolResultCompactor",
]
