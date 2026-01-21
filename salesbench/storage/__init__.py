"""Storage layer for SalesBench.

Provides multiple storage backends:
- JSONResultsWriter: Simple JSON file storage (default)
- SupabaseWriter: Optional cloud database integration
"""

from salesbench.storage.json_writer import JSONResultsWriter
from salesbench.storage.supabase_writer import (
    BatchWriter,
    SupabaseConfig,
    SupabaseWriter,
)

__all__ = [
    "JSONResultsWriter",
    "SupabaseWriter",
    "SupabaseConfig",
    "BatchWriter",
]
