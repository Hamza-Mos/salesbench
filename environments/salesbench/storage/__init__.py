"""Storage layer for SalesBench.

Optional outer-layer integration for Supabase batch writes.
"""

from salesbench.storage.supabase_writer import (
    SupabaseWriter,
    SupabaseConfig,
    BatchWriter,
)

__all__ = [
    "SupabaseWriter",
    "SupabaseConfig",
    "BatchWriter",
]
