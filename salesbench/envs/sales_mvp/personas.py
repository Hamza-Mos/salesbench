"""Seeded persona generation for SalesBench.

Generates 100 reproducible leads per episode with realistic distributions.
"""

import random
from dataclasses import dataclass
from typing import Any, Optional

from salesbench.core.config import PersonaGenerationConfig
from salesbench.core.types import (
    LeadID,
    LeadTemperature,
    ObjectionStyle,
    RiskClass,
    generate_lead_id,
)

# Archetype definitions with parameter ranges
ARCHETYPES = [
    {
        "name": "Young Professional",
        "age_range": (25, 35),
        "income_range": (50000, 120000),
        "jobs": [
            "Software Engineer",
            "Marketing Manager",
            "Financial Analyst",
            "Consultant",
            "Product Manager",
        ],
        "triggers": ["new job", "first home purchase", "getting married"],
        "objection_styles": [ObjectionStyle.QUESTIONING, ObjectionStyle.INDIRECT],
        "risk_tolerance": "high",
        "base_trust": 0.4,
        "base_interest": 0.3,
    },
    {
        "name": "New Parent",
        "age_range": (28, 42),
        "income_range": (60000, 150000),
        "jobs": ["Teacher", "Nurse", "Engineer", "Accountant", "Manager"],
        "triggers": ["new baby", "growing family", "child education planning"],
        "objection_styles": [ObjectionStyle.PRICE_FOCUSED, ObjectionStyle.INDIRECT],
        "risk_tolerance": "medium",
        "base_trust": 0.5,
        "base_interest": 0.6,
    },
    {
        "name": "Mid-Career Professional",
        "age_range": (35, 50),
        "income_range": (80000, 200000),
        "jobs": ["Director", "Senior Manager", "Attorney", "Physician", "Business Owner"],
        "triggers": ["career advancement", "estate planning", "retirement concerns"],
        "objection_styles": [ObjectionStyle.DIRECT, ObjectionStyle.QUESTIONING],
        "risk_tolerance": "medium",
        "base_trust": 0.5,
        "base_interest": 0.4,
    },
    {
        "name": "Pre-Retiree",
        "age_range": (50, 65),
        "income_range": (100000, 300000),
        "jobs": [
            "Executive",
            "Senior Partner",
            "Retired Executive",
            "Business Owner",
            "Consultant",
        ],
        "triggers": ["retirement planning", "legacy planning", "health concerns"],
        "objection_styles": [ObjectionStyle.DIRECT, ObjectionStyle.PRICE_FOCUSED],
        "risk_tolerance": "low",
        "base_trust": 0.6,
        "base_interest": 0.5,
    },
    {
        "name": "Small Business Owner",
        "age_range": (30, 55),
        "income_range": (75000, 250000),
        "jobs": [
            "Restaurant Owner",
            "Contractor",
            "Retail Owner",
            "Freelance Consultant",
            "Agency Owner",
        ],
        "triggers": ["business protection", "key person insurance", "buy-sell agreement"],
        "objection_styles": [ObjectionStyle.PRICE_FOCUSED, ObjectionStyle.DIRECT],
        "risk_tolerance": "medium",
        "base_trust": 0.4,
        "base_interest": 0.5,
    },
    {
        "name": "Healthcare Worker",
        "age_range": (25, 55),
        "income_range": (45000, 180000),
        "jobs": [
            "Nurse",
            "Physician Assistant",
            "Medical Technician",
            "Physical Therapist",
            "Pharmacist",
        ],
        "triggers": ["disability concerns", "income protection", "family security"],
        "objection_styles": [ObjectionStyle.QUESTIONING, ObjectionStyle.INDIRECT],
        "risk_tolerance": "low",
        "base_trust": 0.6,
        "base_interest": 0.5,
    },
    {
        "name": "Blue Collar Worker",
        "age_range": (25, 55),
        "income_range": (35000, 80000),
        "jobs": ["Construction Worker", "Electrician", "Plumber", "Mechanic", "Factory Worker"],
        "triggers": ["workplace injury concern", "family protection", "mortgage"],
        "objection_styles": [ObjectionStyle.PRICE_FOCUSED, ObjectionStyle.TRUST_ISSUES],
        "risk_tolerance": "low",
        "base_trust": 0.3,
        "base_interest": 0.4,
    },
    {
        "name": "High Net Worth",
        "age_range": (40, 65),
        "income_range": (250000, 500000),
        "jobs": ["CEO", "Investment Banker", "Surgeon", "Law Partner", "Tech Founder"],
        "triggers": ["estate tax planning", "wealth transfer", "charitable giving"],
        "objection_styles": [ObjectionStyle.DIRECT, ObjectionStyle.QUESTIONING],
        "risk_tolerance": "high",
        "base_trust": 0.5,
        "base_interest": 0.4,
    },
    {
        "name": "Single Parent",
        "age_range": (28, 50),
        "income_range": (40000, 100000),
        "jobs": [
            "Teacher",
            "Administrative Assistant",
            "Sales Representative",
            "Social Worker",
            "Nurse",
        ],
        "triggers": ["sole provider anxiety", "child guardianship", "college savings"],
        "objection_styles": [ObjectionStyle.PRICE_FOCUSED, ObjectionStyle.INDIRECT],
        "risk_tolerance": "low",
        "base_trust": 0.5,
        "base_interest": 0.7,
    },
    {
        "name": "Skeptic",
        "age_range": (30, 60),
        "income_range": (50000, 150000),
        "jobs": ["Engineer", "Scientist", "Professor", "Analyst", "Researcher"],
        "triggers": ["data-driven decision", "logical argument", "comparison shopping"],
        "objection_styles": [ObjectionStyle.QUESTIONING, ObjectionStyle.TRUST_ISSUES],
        "risk_tolerance": "medium",
        "base_trust": 0.2,
        "base_interest": 0.2,
    },
]

# First names for persona generation
FIRST_NAMES = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "William",
    "Barbara",
    "David",
    "Elizabeth",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Charles",
    "Karen",
    "Christopher",
    "Nancy",
    "Daniel",
    "Lisa",
    "Matthew",
    "Betty",
    "Anthony",
    "Margaret",
    "Mark",
    "Sandra",
    "Donald",
    "Ashley",
    "Steven",
    "Kimberly",
    "Paul",
    "Emily",
    "Andrew",
    "Donna",
    "Joshua",
    "Michelle",
    "Kevin",
    "Dorothy",
    "Brian",
    "Carol",
    "George",
    "Amanda",
    "Edward",
    "Melissa",
    "Ronald",
    "Deborah",
    "Timothy",
    "Stephanie",
    "Jason",
    "Rebecca",
    "Jeffrey",
    "Sharon",
    "Ryan",
    "Laura",
    "Jacob",
    "Cynthia",
    "Gary",
    "Kathleen",
    "Nicholas",
    "Amy",
    "Eric",
    "Angela",
    "Jonathan",
    "Shirley",
    "Stephen",
    "Anna",
    "Larry",
    "Brenda",
    "Justin",
    "Pamela",
    "Scott",
    "Emma",
    "Brandon",
    "Nicole",
    "Benjamin",
    "Helen",
    "Samuel",
    "Samantha",
    "Raymond",
    "Katherine",
    "Gregory",
    "Christine",
    "Frank",
    "Debra",
    "Alexander",
    "Rachel",
    "Patrick",
    "Carolyn",
    "Jack",
    "Janet",
    "Dennis",
    "Catherine",
]

LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
    "Wright",
    "Scott",
    "Torres",
    "Nguyen",
    "Hill",
    "Flores",
    "Green",
    "Adams",
    "Nelson",
    "Baker",
    "Hall",
    "Rivera",
    "Campbell",
    "Mitchell",
    "Carter",
    "Roberts",
    "Chen",
    "Kim",
    "Patel",
    "Singh",
    "Kumar",
    "Cohen",
]


@dataclass
class HiddenState:
    """Hidden persona state not visible to seller."""

    trust: float  # 0.0-1.0, affects acceptance
    interest: float  # 0.0-1.0, affects engagement
    patience: float  # 0.0-1.0, how long before hanging up
    dnc_risk: float  # 0.0-1.0, probability of requesting DNC
    close_threshold: float  # Premium-to-income ratio they'll accept

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trust": self.trust,
            "interest": self.interest,
            "patience": self.patience,
            "dnc_risk": self.dnc_risk,
            "close_threshold": self.close_threshold,
        }


@dataclass
class Persona:
    """A generated persona representing a potential insurance buyer."""

    # Public fields (visible to seller)
    lead_id: LeadID
    name: str
    age: int
    job: str
    annual_income: int
    household_size: int
    has_spouse: bool
    num_dependents: int
    trigger: str  # Life event that might motivate purchase
    objection_style: ObjectionStyle
    temperature: LeadTemperature
    risk_class: RiskClass

    # Hidden state (not visible to seller)
    hidden: HiddenState

    # CRM metadata
    notes: str = ""
    call_count: int = 0
    last_contact_day: Optional[int] = None
    on_dnc_list: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        """Convert to dictionary with only public fields."""
        return {
            "lead_id": self.lead_id,
            "name": self.name,
            "age": self.age,
            "job": self.job,
            "annual_income": self.annual_income,
            "household_size": self.household_size,
            "has_spouse": self.has_spouse,
            "num_dependents": self.num_dependents,
            "trigger": self.trigger,
            "objection_style": self.objection_style.value,
            "temperature": self.temperature.value,
            "risk_class": self.risk_class.value,
            "notes": self.notes,
            "call_count": self.call_count,
            "last_contact_day": self.last_contact_day,
            "on_dnc_list": self.on_dnc_list,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to dictionary with all fields (for debugging)."""
        result = self.to_public_dict()
        result["hidden"] = self.hidden.to_dict()
        return result

    def monthly_income(self) -> float:
        """Calculate monthly income."""
        return self.annual_income / 12

    def affordable_premium(self) -> float:
        """Calculate maximum affordable monthly premium based on hidden threshold."""
        return self.monthly_income() * self.hidden.close_threshold


class PersonaGenerator:
    """Generates reproducible personas from a seed."""

    def __init__(self, seed: int, config: Optional[PersonaGenerationConfig] = None):
        self.seed = seed
        self.config = config or PersonaGenerationConfig()
        self.config.validate()
        self._rng = random.Random(seed)

    def _choose_temperature(self) -> LeadTemperature:
        """Choose temperature based on configured distribution."""
        r = self._rng.random()
        cumulative = 0.0

        cumulative += self.config.hot_probability
        if r < cumulative:
            return LeadTemperature.HOT

        cumulative += self.config.warm_probability
        if r < cumulative:
            return LeadTemperature.WARM

        cumulative += self.config.lukewarm_probability
        if r < cumulative:
            return LeadTemperature.LUKEWARM

        cumulative += self.config.cold_probability
        if r < cumulative:
            return LeadTemperature.COLD

        return LeadTemperature.HOSTILE

    def _choose_risk_class(self, age: int) -> RiskClass:
        """Choose risk class based on age and randomness."""
        # Older people tend to have worse risk classes
        age_factor = (age - 25) / 45  # 0.0 at 25, 1.0 at 70

        r = self._rng.random()
        adjusted_r = r + (age_factor * 0.3)

        if adjusted_r < 0.15:
            return RiskClass.PREFERRED_PLUS
        elif adjusted_r < 0.35:
            return RiskClass.PREFERRED
        elif adjusted_r < 0.60:
            return RiskClass.STANDARD_PLUS
        elif adjusted_r < 0.85:
            return RiskClass.STANDARD
        else:
            return RiskClass.SUBSTANDARD

    def _generate_hidden_state(
        self,
        temperature: LeadTemperature,
        archetype: dict,
    ) -> HiddenState:
        """Generate hidden state based on temperature and archetype."""
        base_trust = archetype["base_trust"]
        base_interest = archetype["base_interest"]

        # Adjust based on temperature
        temp_modifiers = {
            LeadTemperature.HOT: {"trust": 0.3, "interest": 0.4, "patience": 0.3, "dnc_risk": -0.2},
            LeadTemperature.WARM: {
                "trust": 0.15,
                "interest": 0.2,
                "patience": 0.15,
                "dnc_risk": -0.1,
            },
            LeadTemperature.LUKEWARM: {
                "trust": 0.0,
                "interest": 0.0,
                "patience": 0.0,
                "dnc_risk": 0.0,
            },
            LeadTemperature.COLD: {
                "trust": -0.15,
                "interest": -0.2,
                "patience": -0.1,
                "dnc_risk": 0.1,
            },
            LeadTemperature.HOSTILE: {
                "trust": -0.3,
                "interest": -0.3,
                "patience": -0.3,
                "dnc_risk": 0.3,
            },
        }

        mods = temp_modifiers[temperature]

        # Add randomness
        trust = max(0.0, min(1.0, base_trust + mods["trust"] + self._rng.gauss(0, 0.1)))
        interest = max(0.0, min(1.0, base_interest + mods["interest"] + self._rng.gauss(0, 0.1)))
        patience = max(0.1, min(1.0, 0.5 + mods["patience"] + self._rng.gauss(0, 0.15)))
        dnc_risk = max(0.0, min(0.8, 0.1 + mods["dnc_risk"] + self._rng.gauss(0, 0.05)))

        # Close threshold: what % of monthly income they'd pay for insurance
        # Higher interest = willing to pay more
        base_threshold = 0.03 + (interest * 0.07)  # 3-10% of monthly income
        close_threshold = max(0.01, min(0.15, base_threshold + self._rng.gauss(0, 0.01)))

        return HiddenState(
            trust=trust,
            interest=interest,
            patience=patience,
            dnc_risk=dnc_risk,
            close_threshold=close_threshold,
        )

    def generate_one(self) -> Persona:
        """Generate a single persona."""
        # Choose archetype
        archetype = self._rng.choice(ARCHETYPES)

        # Generate basic info
        age = self._rng.randint(*archetype["age_range"])
        income = self._rng.randint(*archetype["income_range"])
        job = self._rng.choice(archetype["jobs"])
        trigger = self._rng.choice(archetype["triggers"])

        # Name
        first_name = self._rng.choice(FIRST_NAMES)
        last_name = self._rng.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"

        # Household
        has_spouse = self._rng.random() < self.config.spouse_probability
        num_dependents = self._rng.randint(0, self.config.max_dependents)
        household_size = 1 + (1 if has_spouse else 0) + num_dependents

        # Characteristics
        temperature = self._choose_temperature()
        objection_style = self._rng.choice(archetype["objection_styles"])
        risk_class = self._choose_risk_class(age)

        # Hidden state
        hidden = self._generate_hidden_state(temperature, archetype)

        return Persona(
            lead_id=generate_lead_id(),
            name=name,
            age=age,
            job=job,
            annual_income=income,
            household_size=household_size,
            has_spouse=has_spouse,
            num_dependents=num_dependents,
            trigger=trigger,
            objection_style=objection_style,
            temperature=temperature,
            risk_class=risk_class,
            hidden=hidden,
        )

    def generate_batch(self, n: int) -> list[Persona]:
        """Generate n personas."""
        return [self.generate_one() for _ in range(n)]

    def reset(self) -> None:
        """Reset the RNG to initial seed state."""
        self._rng = random.Random(self.seed)
