"""Storage layer for SalesBench.

Optional outer-layer integration for Supabase batch writes.
"""

from salesbench.storage.supabase_writer import (
    BatchWriter,
    SupabaseConfig,
    SupabaseWriter,
)

__all__ = [
    "SupabaseWriter",
    "SupabaseConfig",
    "BatchWriter",
]
