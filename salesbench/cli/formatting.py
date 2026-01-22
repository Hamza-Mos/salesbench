"""Output formatting utilities for CLI commands.

Consolidates json/table output logic to eliminate repeated format checks.
"""

import json
from typing import Any


class OutputFormatter:
    """Handles consistent output formatting across CLI commands."""

    def __init__(self, format_type: str = "table"):
        """Initialize formatter.

        Args:
            format_type: Output format - 'json' or 'table'.
        """
        self.format_type = format_type

    @property
    def is_json(self) -> bool:
        """Check if output should be JSON."""
        return self.format_type == "json"

    def output(self, data: Any, table_fn: callable) -> None:
        """Output data in the configured format.

        Args:
            data: Data to output (used directly for JSON).
            table_fn: Function to call for table output (no args).
        """
        if self.is_json:
            print(json.dumps(data, indent=2))
        else:
            table_fn()

    def print_json(self, data: Any) -> None:
        """Print data as JSON."""
        print(json.dumps(data, indent=2))

    def print_header(self, title: str, width: int = 80) -> None:
        """Print a formatted header for table output."""
        if not self.is_json:
            print(title)
            print("=" * width)

    def print_separator(self, width: int = 80, char: str = "-") -> None:
        """Print a separator line for table output."""
        if not self.is_json:
            print(char * width)

    def print_section(self, title: str, width: int = 40) -> None:
        """Print a section header for table output."""
        if not self.is_json:
            print(f"\n{title}")
            print("-" * width)
