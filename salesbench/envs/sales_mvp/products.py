"""Insurance product catalog with deterministic pricing.

Products:
- TERM: Term life insurance (10, 15, 20, 30 year terms)
- WHOLE: Whole life insurance
- UL: Universal life insurance
- VUL: Variable universal life
- LTC: Long-term care insurance
- DI: Disability insurance
"""

from dataclasses import dataclass
from typing import Any, Optional

from salesbench.core.types import CoverageTier, PlanType, RiskClass

# Coverage amounts by tier
COVERAGE_AMOUNTS = {
    CoverageTier.BASIC: 100_000,
    CoverageTier.STANDARD: 250_000,
    CoverageTier.ENHANCED: 500_000,
    CoverageTier.PREMIUM: 1_000_000,
    CoverageTier.ELITE: 2_000_000,
}

# Base rates per $1000 of coverage (monthly)
# Indexed by age band
BASE_RATES = {
    PlanType.TERM: {
        # Age bands: 25-34, 35-44, 45-54, 55-64, 65+
        "25-34": 0.08,
        "35-44": 0.12,
        "45-54": 0.25,
        "55-64": 0.55,
        "65+": 1.20,
    },
    PlanType.WHOLE: {
        "25-34": 0.85,
        "35-44": 1.10,
        "45-54": 1.50,
        "55-64": 2.20,
        "65+": 3.50,
    },
    PlanType.UL: {
        "25-34": 0.65,
        "35-44": 0.85,
        "45-54": 1.15,
        "55-64": 1.70,
        "65+": 2.80,
    },
    PlanType.VUL: {
        "25-34": 0.70,
        "35-44": 0.90,
        "45-54": 1.25,
        "55-64": 1.85,
        "65+": 3.00,
    },
    PlanType.LTC: {
        "25-34": 0.15,
        "35-44": 0.25,
        "45-54": 0.50,
        "55-64": 1.00,
        "65+": 2.00,
    },
    PlanType.DI: {
        "25-34": 1.50,
        "35-44": 1.80,
        "45-54": 2.20,
        "55-64": 2.80,
        "65+": 3.50,
    },
}

# Risk class multipliers
RISK_MULTIPLIERS = {
    RiskClass.PREFERRED_PLUS: 0.75,
    RiskClass.PREFERRED: 0.90,
    RiskClass.STANDARD_PLUS: 1.00,
    RiskClass.STANDARD: 1.15,
    RiskClass.SUBSTANDARD: 1.50,
}

# Term length multipliers (for TERM plans)
TERM_MULTIPLIERS = {
    10: 0.80,
    15: 0.90,
    20: 1.00,
    30: 1.25,
}


def get_age_band(age: int) -> str:
    """Get the age band for pricing."""
    if age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"


@dataclass
class InsuranceProduct:
    """An insurance product in the catalog."""

    plan_id: PlanType
    name: str
    description: str
    min_coverage: float
    max_coverage: float
    min_age: int
    max_age: int
    features: list[str]
    term_options: Optional[list[int]] = None  # For TERM plans

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "plan_id": self.plan_id.value,
            "name": self.name,
            "description": self.description,
            "min_coverage": self.min_coverage,
            "max_coverage": self.max_coverage,
            "min_age": self.min_age,
            "max_age": self.max_age,
            "features": self.features,
        }
        if self.term_options:
            result["term_options"] = self.term_options
        return result


class ProductCatalog:
    """Insurance product catalog with pricing."""

    def __init__(self):
        self._products = self._initialize_products()

    def _initialize_products(self) -> dict[PlanType, InsuranceProduct]:
        """Initialize the product catalog."""
        return {
            PlanType.TERM: InsuranceProduct(
                plan_id=PlanType.TERM,
                name="Term Life Insurance",
                description="Affordable coverage for a specific period. Ideal for temporary needs like mortgage protection or income replacement during working years.",
                min_coverage=50_000,
                max_coverage=5_000_000,
                min_age=18,
                max_age=75,
                features=[
                    "Lowest premium rates",
                    "Fixed premiums for term length",
                    "Convertible to permanent insurance",
                    "No cash value accumulation",
                ],
                term_options=[10, 15, 20, 30],
            ),
            PlanType.WHOLE: InsuranceProduct(
                plan_id=PlanType.WHOLE,
                name="Whole Life Insurance",
                description="Permanent coverage with guaranteed death benefit and cash value accumulation. Premiums remain level for life.",
                min_coverage=25_000,
                max_coverage=10_000_000,
                min_age=18,
                max_age=80,
                features=[
                    "Lifetime coverage guarantee",
                    "Fixed premiums never increase",
                    "Cash value grows tax-deferred",
                    "Can borrow against cash value",
                    "Dividend potential (participating policies)",
                ],
            ),
            PlanType.UL: InsuranceProduct(
                plan_id=PlanType.UL,
                name="Universal Life Insurance",
                description="Flexible permanent coverage with adjustable premiums and death benefits. Cash value earns interest based on current rates.",
                min_coverage=50_000,
                max_coverage=10_000_000,
                min_age=18,
                max_age=80,
                features=[
                    "Flexible premium payments",
                    "Adjustable death benefit",
                    "Cash value earns current interest rate",
                    "Tax-deferred growth",
                    "Access to cash value through loans",
                ],
            ),
            PlanType.VUL: InsuranceProduct(
                plan_id=PlanType.VUL,
                name="Variable Universal Life Insurance",
                description="Combines insurance protection with investment options. Cash value can be invested in various sub-accounts.",
                min_coverage=100_000,
                max_coverage=25_000_000,
                min_age=18,
                max_age=75,
                features=[
                    "Investment flexibility",
                    "Higher growth potential",
                    "Flexible premiums",
                    "Tax-advantaged investing",
                    "Market risk on cash value",
                ],
            ),
            PlanType.LTC: InsuranceProduct(
                plan_id=PlanType.LTC,
                name="Long-Term Care Insurance",
                description="Covers costs of long-term care services including nursing homes, assisted living, and home health care.",
                min_coverage=50_000,
                max_coverage=500_000,
                min_age=40,
                max_age=79,
                features=[
                    "Covers nursing home care",
                    "Home health care benefits",
                    "Assisted living coverage",
                    "Inflation protection options",
                    "Tax-qualified benefits",
                ],
            ),
            PlanType.DI: InsuranceProduct(
                plan_id=PlanType.DI,
                name="Disability Insurance",
                description="Replaces income if you become unable to work due to illness or injury. Protects your most valuable asset - your ability to earn.",
                min_coverage=1_000,  # Monthly benefit
                max_coverage=15_000,  # Monthly benefit
                min_age=18,
                max_age=60,
                features=[
                    "Income replacement up to 60-70%",
                    "Own-occupation coverage available",
                    "Benefit periods up to age 65",
                    "Waiting period options",
                    "Cost of living adjustments",
                ],
            ),
        }

    def list_products(self) -> list[dict[str, Any]]:
        """List all products in the catalog."""
        return [p.to_dict() for p in self._products.values()]

    def get_product(self, plan_id: PlanType) -> Optional[InsuranceProduct]:
        """Get a specific product."""
        return self._products.get(plan_id)

    def quote_premium(
        self,
        plan_id: PlanType,
        age: int,
        coverage_amount: float,
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        term_years: Optional[int] = None,
    ) -> dict[str, Any]:
        """Calculate monthly premium for a plan.

        Args:
            plan_id: The insurance plan type.
            age: Age of the insured.
            coverage_amount: Coverage amount in dollars.
            risk_class: Risk classification.
            term_years: Term length (for TERM plans).

        Returns:
            Dict with premium quote details.
        """
        product = self._products.get(plan_id)
        if not product:
            return {"error": f"Unknown plan: {plan_id}"}

        # Validate age
        if age < product.min_age or age > product.max_age:
            return {"error": f"Age {age} outside valid range {product.min_age}-{product.max_age}"}

        # Validate coverage
        if coverage_amount < product.min_coverage or coverage_amount > product.max_coverage:
            return {
                "error": f"Coverage ${coverage_amount:,.0f} outside valid range ${product.min_coverage:,.0f}-${product.max_coverage:,.0f}"
            }

        # For TERM, validate term_years
        if plan_id == PlanType.TERM:
            if term_years is None:
                term_years = 20  # Default
            if term_years not in TERM_MULTIPLIERS:
                return {"error": f"Invalid term length. Options: {list(TERM_MULTIPLIERS.keys())}"}

        # Calculate premium
        age_band = get_age_band(age)
        base_rate = BASE_RATES[plan_id][age_band]
        risk_mult = RISK_MULTIPLIERS[risk_class]

        # Base premium per $1000
        premium_per_thousand = base_rate * risk_mult

        # Apply term multiplier for TERM plans
        if plan_id == PlanType.TERM and term_years:
            premium_per_thousand *= TERM_MULTIPLIERS[term_years]

        # Calculate total monthly premium
        monthly_premium = (coverage_amount / 1000) * premium_per_thousand

        # Round to cents
        monthly_premium = round(monthly_premium, 2)

        result = {
            "plan_id": plan_id.value,
            "plan_name": product.name,
            "coverage_amount": coverage_amount,
            "monthly_premium": monthly_premium,
            "annual_premium": round(monthly_premium * 12, 2),
            "age": age,
            "risk_class": risk_class.value,
        }

        if plan_id == PlanType.TERM:
            result["term_years"] = term_years

        # Add cash value projection for permanent policies
        if plan_id in (PlanType.WHOLE, PlanType.UL, PlanType.VUL):
            # Simplified cash value projection (year 10)
            # Roughly 30-50% of premiums paid become cash value
            total_paid_10yr = monthly_premium * 12 * 10
            cv_ratio = 0.35 if plan_id == PlanType.WHOLE else 0.30
            result["projected_cash_value_year_10"] = round(total_paid_10yr * cv_ratio, 2)

        if plan_id == PlanType.DI:
            # DI uses monthly benefit instead of coverage
            result["monthly_benefit"] = coverage_amount
            result["benefit_period"] = "To age 65"
            result["waiting_period"] = "90 days"
            del result["coverage_amount"]

        if plan_id == PlanType.LTC:
            result["daily_benefit"] = round(coverage_amount / 1095, 2)  # 3-year pool
            result["benefit_period"] = "3 years"
            result["waiting_period"] = "90 days"

        return result

    def get_recommended_coverage(
        self,
        annual_income: int,
        age: int,
        has_dependents: bool,
        plan_id: PlanType,
    ) -> float:
        """Get recommended coverage amount based on profile."""
        product = self._products.get(plan_id)
        if not product:
            return 0

        if plan_id == PlanType.DI:
            # DI: typically 60% of monthly income
            monthly = annual_income / 12
            recommended = min(monthly * 0.6, product.max_coverage)
            return max(product.min_coverage, recommended)

        if plan_id == PlanType.LTC:
            # LTC: based on regional care costs (simplified)
            base = 150_000  # 3-year benefit
            if annual_income > 150_000:
                base = 250_000
            return min(base, product.max_coverage)

        # Life insurance: income replacement rule
        # Typical: 10-15x income, adjusted for age and dependents
        multiplier = 12 if has_dependents else 8
        if age > 50:
            multiplier = max(5, multiplier - 3)
        elif age > 60:
            multiplier = max(3, multiplier - 5)

        recommended = annual_income * multiplier
        recommended = max(product.min_coverage, min(recommended, product.max_coverage))

        return recommended
