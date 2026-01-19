"""Configuration dataclasses for SalesBench."""

from dataclasses import dataclass, field


@dataclass
class BudgetConfig:
    """Budget limits for the simulation."""

    # Time budgets
    total_days: int = 10  # Total simulated business days
    hours_per_day: int = 8  # Working hours per day (9 AM - 5 PM)
    minutes_per_hour: int = 60  # Minutes per hour

    # Call budgets
    max_calls_per_day: int = 50  # Max calls seller can make per day
    max_call_duration_minutes: int = 30  # Max duration per call
    max_offers_per_call: int = 3  # Max offers per call

    # Tool budgets
    max_tool_calls_per_turn: int = 10  # Max tool calls in one turn

    @property
    def total_minutes(self) -> int:
        """Total simulated minutes in the episode."""
        return self.total_days * self.hours_per_day * self.minutes_per_hour

    @property
    def minutes_per_day(self) -> int:
        """Working minutes per day."""
        return self.hours_per_day * self.minutes_per_hour


@dataclass
class ScoringConfig:
    """Scoring weights and parameters."""

    # Primary rewards
    accept_reward: float = 100.0  # Base reward for accepted plan
    close_now_bonus: float = 50.0  # Bonus for immediate close
    schedule_followup_bonus: float = 20.0  # Bonus for scheduled followup

    # Penalties
    reject_penalty: float = -5.0  # Penalty for rejected offer
    end_call_penalty: float = -10.0  # Penalty for buyer ending call
    dnc_penalty: float = -200.0  # Penalty for Do Not Call violation

    # Efficiency bonuses
    time_efficiency_weight: float = 0.1  # Reward for completing faster
    cost_efficiency_weight: float = 0.05  # Reward for fewer LLM calls

    # Premium-based rewards
    premium_multiplier: float = 0.5  # Multiply monthly premium for reward

    # Bounds
    min_score: float = -1000.0
    max_score: float = 10000.0


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
    """Main configuration for SalesBench."""

    # Random seed for reproducibility
    seed: int = 42

    # Number of leads to generate
    num_leads: int = 100

    # Sub-configurations
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    persona_generation: PersonaGenerationConfig = field(default_factory=PersonaGenerationConfig)

    # LLM configuration
    buyer_model: str = "gpt-4o-mini"  # Model for buyer simulator
    buyer_temperature: float = 0.3  # Lower temperature for deterministic decisions

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
                "total_days": self.budget.total_days,
                "hours_per_day": self.budget.hours_per_day,
                "max_calls_per_day": self.budget.max_calls_per_day,
                "max_call_duration_minutes": self.budget.max_call_duration_minutes,
            },
            "scoring": {
                "accept_reward": self.scoring.accept_reward,
                "close_now_bonus": self.scoring.close_now_bonus,
                "reject_penalty": self.scoring.reject_penalty,
                "dnc_penalty": self.scoring.dnc_penalty,
            },
            "buyer_model": self.buyer_model,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SalesBenchConfig":
        """Create from dictionary."""
        config = cls(
            seed=data.get("seed", 42),
            num_leads=data.get("num_leads", 100),
            debug=data.get("debug", False),
            buyer_model=data.get("buyer_model", "gpt-4o-mini"),
        )

        if "budget" in data:
            budget_data = data["budget"]
            config.budget = BudgetConfig(
                total_days=budget_data.get("total_days", 10),
                hours_per_day=budget_data.get("hours_per_day", 8),
                max_calls_per_day=budget_data.get("max_calls_per_day", 50),
                max_call_duration_minutes=budget_data.get("max_call_duration_minutes", 30),
            )

        if "scoring" in data:
            scoring_data = data["scoring"]
            config.scoring = ScoringConfig(
                accept_reward=scoring_data.get("accept_reward", 100.0),
                close_now_bonus=scoring_data.get("close_now_bonus", 50.0),
                reject_penalty=scoring_data.get("reject_penalty", -5.0),
                dnc_penalty=scoring_data.get("dnc_penalty", -200.0),
            )

        return config
