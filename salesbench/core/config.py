"""Configuration dataclasses for SalesBench."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BudgetConfig:
    """Time system configuration.

    Time is the only natural constraint - agents work until time runs out.

    Time Model Options:
        - "action": Fixed costs per action (default)
        - "token": Time based on token consumption

    Both metrics are always tracked regardless of which model is active.
    """

    total_hours: int = 80  # Total simulated hours (e.g., 80 = 10 days × 8 hrs)
    minutes_per_hour: int = 60  # Minutes per hour

    # Time model configuration
    time_model: Literal["action", "token"] = "action"
    conversation_turn_cost: float = 2.0  # Minutes per conversation turn during call
    tokens_per_minute: float = 150.0  # For token-based time calculation

    # Action time costs (minutes)
    start_call_cost: float = 1.0  # Time to initiate a call
    search_cost: float = 1.0  # Time for CRM search
    propose_plan_cost: float = 4.0  # Time to present offer + get response

    # Stall detection threshold
    max_turns_without_tool_call: int = 30  # Turns without tool call before warning

    @property
    def total_minutes(self) -> int:
        """Total simulated minutes in the episode."""
        return self.total_hours * self.minutes_per_hour


@dataclass
class PersonaGenerationConfig:
    """Configuration for persona generation."""

    # Temperature distribution (must sum to 1.0)
    hot_probability: float = 0.03
    warm_probability: float = 0.12
    lukewarm_probability: float = 0.35
    cold_probability: float = 0.40
    hostile_probability: float = 0.10

    # Age distribution
    min_age: int = 25
    max_age: int = 70
    mean_age: int = 42

    # Income distribution (annual, USD)
    min_income: int = 30000
    max_income: int = 500000
    median_income: int = 75000

    # Household distribution
    max_dependents: int = 5
    spouse_probability: float = 0.55

    def validate(self) -> None:
        """Validate configuration."""
        total_prob = (
            self.hot_probability
            + self.warm_probability
            + self.lukewarm_probability
            + self.cold_probability
            + self.hostile_probability
        )
        if abs(total_prob - 1.0) > 0.001:
            raise ValueError(f"Temperature probabilities must sum to 1.0, got {total_prob}")


@dataclass
class SalesBenchConfig:
    """Main configuration for SalesBench episodes.

    This is the episode-level configuration embedded in BenchmarkConfig.
    Model configuration (seller_model, buyer_model) lives at the BenchmarkConfig level.
    """

    # Random seed for reproducibility
    seed: int = 42

    # Number of leads to generate
    num_leads: int = 100

    # Sub-configurations
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    persona_generation: PersonaGenerationConfig = field(default_factory=PersonaGenerationConfig)

    # Debug options
    debug: bool = False
    log_tool_calls: bool = False

    def validate(self) -> None:
        """Validate all configuration."""
        self.persona_generation.validate()

        if self.num_leads < 1:
            raise ValueError("num_leads must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "seed": self.seed,
            "num_leads": self.num_leads,
            "budget": {
                "total_hours": self.budget.total_hours,
                "time_model": self.budget.time_model,
                "conversation_turn_cost": self.budget.conversation_turn_cost,
                "tokens_per_minute": self.budget.tokens_per_minute,
                "start_call_cost": self.budget.start_call_cost,
                "search_cost": self.budget.search_cost,
                "propose_plan_cost": self.budget.propose_plan_cost,
                "max_turns_without_tool_call": self.budget.max_turns_without_tool_call,
            },
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SalesBenchConfig":
        """Create from dictionary."""
        config = cls(
            seed=data.get("seed", 42),
            num_leads=data.get("num_leads", 100),
            debug=data.get("debug", False),
        )

        if "budget" in data:
            budget_data = data["budget"]
            # Use dataclass defaults by referencing config.budget values
            config.budget = BudgetConfig(
                total_hours=budget_data.get("total_hours", config.budget.total_hours),
                time_model=budget_data.get("time_model", config.budget.time_model),
                conversation_turn_cost=budget_data.get("conversation_turn_cost", config.budget.conversation_turn_cost),
                tokens_per_minute=budget_data.get("tokens_per_minute", config.budget.tokens_per_minute),
                start_call_cost=budget_data.get("start_call_cost", config.budget.start_call_cost),
                search_cost=budget_data.get("search_cost", config.budget.search_cost),
                propose_plan_cost=budget_data.get("propose_plan_cost", config.budget.propose_plan_cost),
                max_turns_without_tool_call=budget_data.get("max_turns_without_tool_call", config.budget.max_turns_without_tool_call),
            )

        return config
