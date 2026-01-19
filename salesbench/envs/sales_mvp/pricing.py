"""Detailed insurance pricing tables.

This module contains exact rate tables for all insurance products:
- Term Life (10, 15, 20, 30 year)
- Whole Life
- Universal Life (UL)
- Variable Universal Life (VUL)
- Long-Term Care (LTC)
- Disability Insurance (DI)

Rates are per $1,000 of coverage per month, indexed by:
- Age band
- Gender (optional)
- Risk class
- Product-specific factors
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from salesbench.core.types import PlanType, RiskClass


class Gender(str, Enum):
    """Gender for pricing (affects life insurance rates)."""

    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"  # Use average rates


class SmokingStatus(str, Enum):
    """Smoking status for underwriting."""

    NON_SMOKER = "non_smoker"
    SMOKER = "smoker"
    FORMER_SMOKER = "former_smoker"  # Quit 2+ years ago


class HealthTier(str, Enum):
    """Health classification for underwriting."""

    EXCELLENT = "excellent"  # No health issues
    GOOD = "good"  # Minor issues
    FAIR = "fair"  # Moderate issues
    POOR = "poor"  # Significant issues


# =============================================================================
# TERM LIFE INSURANCE RATES
# Base rates per $1,000 coverage per month (Non-smoker, Preferred Plus)
# =============================================================================

TERM_LIFE_BASE_RATES = {
    # 10-Year Term
    10: {
        (18, 24): {"male": 0.050, "female": 0.043},
        (25, 29): {"male": 0.053, "female": 0.046},
        (30, 34): {"male": 0.058, "female": 0.050},
        (35, 39): {"male": 0.073, "female": 0.063},
        (40, 44): {"male": 0.103, "female": 0.088},
        (45, 49): {"male": 0.158, "female": 0.135},
        (50, 54): {"male": 0.248, "female": 0.205},
        (55, 59): {"male": 0.398, "female": 0.323},
        (60, 64): {"male": 0.638, "female": 0.508},
        (65, 69): {"male": 1.048, "female": 0.838},
        (70, 75): {"male": 1.758, "female": 1.408},
    },
    # 15-Year Term
    15: {
        (18, 24): {"male": 0.058, "female": 0.050},
        (25, 29): {"male": 0.063, "female": 0.054},
        (30, 34): {"male": 0.070, "female": 0.060},
        (35, 39): {"male": 0.090, "female": 0.078},
        (40, 44): {"male": 0.128, "female": 0.108},
        (45, 49): {"male": 0.198, "female": 0.168},
        (50, 54): {"male": 0.318, "female": 0.263},
        (55, 59): {"male": 0.518, "female": 0.418},
        (60, 64): {"male": 0.858, "female": 0.688},
        (65, 69): {"male": 1.428, "female": 1.138},
        (70, 75): {"male": 2.408, "female": 1.928},
    },
    # 20-Year Term
    20: {
        (18, 24): {"male": 0.068, "female": 0.058},
        (25, 29): {"male": 0.075, "female": 0.065},
        (30, 34): {"male": 0.085, "female": 0.073},
        (35, 39): {"male": 0.110, "female": 0.095},
        (40, 44): {"male": 0.160, "female": 0.138},
        (45, 49): {"male": 0.253, "female": 0.213},
        (50, 54): {"male": 0.418, "female": 0.343},
        (55, 59): {"male": 0.698, "female": 0.558},
        (60, 64): {"male": 1.178, "female": 0.938},
        (65, 69): {"male": 1.998, "female": 1.598},
    },
    # 30-Year Term
    30: {
        (18, 24): {"male": 0.095, "female": 0.083},
        (25, 29): {"male": 0.108, "female": 0.093},
        (30, 34): {"male": 0.125, "female": 0.108},
        (35, 39): {"male": 0.168, "female": 0.145},
        (40, 44): {"male": 0.253, "female": 0.213},
        (45, 49): {"male": 0.408, "female": 0.338},
        (50, 54): {"male": 0.688, "female": 0.558},
        (55, 59): {"male": 1.158, "female": 0.928},
    },
}

# =============================================================================
# WHOLE LIFE INSURANCE RATES
# Base rates per $1,000 coverage per month (Non-smoker, Preferred)
# =============================================================================

WHOLE_LIFE_BASE_RATES = {
    (18, 24): {"male": 0.65, "female": 0.55},
    (25, 29): {"male": 0.75, "female": 0.63},
    (30, 34): {"male": 0.88, "female": 0.73},
    (35, 39): {"male": 1.05, "female": 0.88},
    (40, 44): {"male": 1.30, "female": 1.08},
    (45, 49): {"male": 1.63, "female": 1.35},
    (50, 54): {"male": 2.05, "female": 1.70},
    (55, 59): {"male": 2.60, "female": 2.15},
    (60, 64): {"male": 3.35, "female": 2.78},
    (65, 69): {"male": 4.40, "female": 3.65},
    (70, 75): {"male": 5.85, "female": 4.85},
    (76, 80): {"male": 7.90, "female": 6.55},
}

# =============================================================================
# UNIVERSAL LIFE INSURANCE RATES
# Base rates per $1,000 coverage per month
# =============================================================================

UNIVERSAL_LIFE_BASE_RATES = {
    (18, 24): {"male": 0.50, "female": 0.43},
    (25, 29): {"male": 0.58, "female": 0.50},
    (30, 34): {"male": 0.68, "female": 0.58},
    (35, 39): {"male": 0.83, "female": 0.70},
    (40, 44): {"male": 1.03, "female": 0.88},
    (45, 49): {"male": 1.30, "female": 1.10},
    (50, 54): {"male": 1.65, "female": 1.40},
    (55, 59): {"male": 2.13, "female": 1.80},
    (60, 64): {"male": 2.78, "female": 2.35},
    (65, 69): {"male": 3.68, "female": 3.10},
    (70, 75): {"male": 4.95, "female": 4.18},
    (76, 80): {"male": 6.75, "female": 5.70},
}

# =============================================================================
# VARIABLE UNIVERSAL LIFE RATES
# Base rates per $1,000 coverage per month (includes M&E charges)
# =============================================================================

VARIABLE_UNIVERSAL_LIFE_BASE_RATES = {
    (18, 24): {"male": 0.55, "female": 0.48},
    (25, 29): {"male": 0.63, "female": 0.55},
    (30, 34): {"male": 0.75, "female": 0.65},
    (35, 39): {"male": 0.93, "female": 0.80},
    (40, 44): {"male": 1.15, "female": 0.98},
    (45, 49): {"male": 1.48, "female": 1.25},
    (50, 54): {"male": 1.90, "female": 1.60},
    (55, 59): {"male": 2.48, "female": 2.08},
    (60, 64): {"male": 3.25, "female": 2.73},
    (65, 69): {"male": 4.33, "female": 3.63},
    (70, 75): {"male": 5.85, "female": 4.90},
}

# =============================================================================
# LONG-TERM CARE INSURANCE RATES
# Rates per $100 of monthly benefit
# =============================================================================

LTC_BASE_RATES = {
    # 3-year benefit period, 90-day elimination
    "3_year": {
        (40, 44): {"male": 1.80, "female": 2.70},
        (45, 49): {"male": 2.40, "female": 3.60},
        (50, 54): {"male": 3.30, "female": 4.95},
        (55, 59): {"male": 4.65, "female": 6.98},
        (60, 64): {"male": 6.75, "female": 10.13},
        (65, 69): {"male": 10.20, "female": 15.30},
        (70, 74): {"male": 15.90, "female": 23.85},
        (75, 79): {"male": 25.50, "female": 38.25},
    },
    # 5-year benefit period
    "5_year": {
        (40, 44): {"male": 2.25, "female": 3.38},
        (45, 49): {"male": 3.00, "female": 4.50},
        (50, 54): {"male": 4.13, "female": 6.19},
        (55, 59): {"male": 5.81, "female": 8.72},
        (60, 64): {"male": 8.44, "female": 12.66},
        (65, 69): {"male": 12.75, "female": 19.13},
        (70, 74): {"male": 19.88, "female": 29.81},
        (75, 79): {"male": 31.88, "female": 47.81},
    },
    # Unlimited benefit
    "unlimited": {
        (40, 44): {"male": 3.15, "female": 4.73},
        (45, 49): {"male": 4.20, "female": 6.30},
        (50, 54): {"male": 5.78, "female": 8.66},
        (55, 59): {"male": 8.14, "female": 12.21},
        (60, 64): {"male": 11.81, "female": 17.72},
        (65, 69): {"male": 17.85, "female": 26.78},
        (70, 74): {"male": 27.83, "female": 41.74},
    },
}

# =============================================================================
# DISABILITY INSURANCE RATES
# Rates per $100 of monthly benefit
# =============================================================================

DI_BASE_RATES = {
    # Own-occupation, to age 65
    "own_occ_65": {
        (18, 24): {"male": 1.80, "female": 2.52},
        (25, 29): {"male": 2.10, "female": 2.94},
        (30, 34): {"male": 2.55, "female": 3.57},
        (35, 39): {"male": 3.15, "female": 4.41},
        (40, 44): {"male": 3.90, "female": 5.46},
        (45, 49): {"male": 4.80, "female": 6.72},
        (50, 54): {"male": 5.85, "female": 8.19},
        (55, 59): {"male": 6.90, "female": 9.66},
    },
    # Any-occupation, to age 65
    "any_occ_65": {
        (18, 24): {"male": 1.35, "female": 1.89},
        (25, 29): {"male": 1.58, "female": 2.21},
        (30, 34): {"male": 1.91, "female": 2.68},
        (35, 39): {"male": 2.36, "female": 3.31},
        (40, 44): {"male": 2.93, "female": 4.10},
        (45, 49): {"male": 3.60, "female": 5.04},
        (50, 54): {"male": 4.39, "female": 6.14},
        (55, 59): {"male": 5.18, "female": 7.25},
    },
    # 5-year benefit period
    "5_year": {
        (18, 24): {"male": 1.20, "female": 1.68},
        (25, 29): {"male": 1.40, "female": 1.96},
        (30, 34): {"male": 1.70, "female": 2.38},
        (35, 39): {"male": 2.10, "female": 2.94},
        (40, 44): {"male": 2.60, "female": 3.64},
        (45, 49): {"male": 3.20, "female": 4.48},
        (50, 54): {"male": 3.90, "female": 5.46},
        (55, 59): {"male": 4.60, "female": 6.44},
    },
}

# =============================================================================
# MULTIPLIERS
# =============================================================================

# Risk class multipliers
RISK_CLASS_MULTIPLIERS = {
    RiskClass.PREFERRED_PLUS: 0.80,
    RiskClass.PREFERRED: 0.90,
    RiskClass.STANDARD_PLUS: 1.00,
    RiskClass.STANDARD: 1.20,
    RiskClass.SUBSTANDARD: 1.60,
}

# Smoking status multipliers
SMOKING_MULTIPLIERS = {
    SmokingStatus.NON_SMOKER: 1.00,
    SmokingStatus.FORMER_SMOKER: 1.25,
    SmokingStatus.SMOKER: 2.00,
}

# Health tier multipliers
HEALTH_MULTIPLIERS = {
    HealthTier.EXCELLENT: 0.90,
    HealthTier.GOOD: 1.00,
    HealthTier.FAIR: 1.30,
    HealthTier.POOR: 1.75,
}

# Occupation class multipliers (for DI)
OCCUPATION_MULTIPLIERS = {
    "professional": 0.85,  # Desk jobs, executives
    "white_collar": 1.00,  # Office workers
    "light_manual": 1.25,  # Light physical work
    "heavy_manual": 1.75,  # Construction, labor
    "hazardous": 2.50,  # High-risk occupations
}


def get_age_band(age: int, rate_table: dict) -> tuple[int, int]:
    """Find the age band for a given age in a rate table."""
    for band in rate_table.keys():
        if isinstance(band, tuple) and band[0] <= age <= band[1]:
            return band
    # Return highest band if age exceeds all bands
    bands = [b for b in rate_table.keys() if isinstance(b, tuple)]
    return max(bands, key=lambda x: x[1])


@dataclass
class PremiumQuote:
    """Detailed premium quote."""

    plan_type: PlanType
    base_rate: float
    final_rate: float
    monthly_premium: float
    annual_premium: float

    # Input factors
    age: int
    gender: Gender
    coverage_amount: float
    risk_class: RiskClass
    smoking_status: SmokingStatus

    # Applied multipliers
    multipliers: dict[str, float]

    # Product-specific
    term_years: Optional[int] = None
    benefit_period: Optional[str] = None
    waiting_period: Optional[str] = None
    projected_cash_value_10yr: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "plan_type": self.plan_type.value,
            "base_rate": self.base_rate,
            "final_rate": self.final_rate,
            "monthly_premium": self.monthly_premium,
            "annual_premium": self.annual_premium,
            "age": self.age,
            "gender": self.gender.value,
            "coverage_amount": self.coverage_amount,
            "risk_class": self.risk_class.value,
            "smoking_status": self.smoking_status.value,
            "multipliers": self.multipliers,
        }
        if self.term_years:
            result["term_years"] = self.term_years
        if self.benefit_period:
            result["benefit_period"] = self.benefit_period
        if self.waiting_period:
            result["waiting_period"] = self.waiting_period
        if self.projected_cash_value_10yr:
            result["projected_cash_value_10yr"] = self.projected_cash_value_10yr
        return result


class PricingEngine:
    """Engine for calculating insurance premiums."""

    def quote_term_life(
        self,
        age: int,
        gender: Gender,
        coverage: float,
        term_years: int,
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        smoking_status: SmokingStatus = SmokingStatus.NON_SMOKER,
    ) -> PremiumQuote:
        """Get quote for term life insurance."""
        if term_years not in TERM_LIFE_BASE_RATES:
            term_years = 20

        rate_table = TERM_LIFE_BASE_RATES[term_years]
        age_band = get_age_band(age, rate_table)
        gender_key = gender.value if gender != Gender.UNISEX else "male"

        base_rate = rate_table[age_band][gender_key]

        # Apply multipliers
        multipliers = {
            "risk_class": RISK_CLASS_MULTIPLIERS[risk_class],
            "smoking": SMOKING_MULTIPLIERS[smoking_status],
        }

        final_rate = base_rate
        for mult in multipliers.values():
            final_rate *= mult

        monthly = (coverage / 1000) * final_rate

        return PremiumQuote(
            plan_type=PlanType.TERM,
            base_rate=base_rate,
            final_rate=round(final_rate, 4),
            monthly_premium=round(monthly, 2),
            annual_premium=round(monthly * 12, 2),
            age=age,
            gender=gender,
            coverage_amount=coverage,
            risk_class=risk_class,
            smoking_status=smoking_status,
            multipliers=multipliers,
            term_years=term_years,
        )

    def quote_whole_life(
        self,
        age: int,
        gender: Gender,
        coverage: float,
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        smoking_status: SmokingStatus = SmokingStatus.NON_SMOKER,
    ) -> PremiumQuote:
        """Get quote for whole life insurance."""
        age_band = get_age_band(age, WHOLE_LIFE_BASE_RATES)
        gender_key = gender.value if gender != Gender.UNISEX else "male"

        base_rate = WHOLE_LIFE_BASE_RATES[age_band][gender_key]

        multipliers = {
            "risk_class": RISK_CLASS_MULTIPLIERS[risk_class],
            "smoking": SMOKING_MULTIPLIERS[smoking_status],
        }

        final_rate = base_rate
        for mult in multipliers.values():
            final_rate *= mult

        monthly = (coverage / 1000) * final_rate

        # Estimate cash value at year 10 (simplified)
        cash_value_10yr = monthly * 12 * 10 * 0.40

        return PremiumQuote(
            plan_type=PlanType.WHOLE,
            base_rate=base_rate,
            final_rate=round(final_rate, 4),
            monthly_premium=round(monthly, 2),
            annual_premium=round(monthly * 12, 2),
            age=age,
            gender=gender,
            coverage_amount=coverage,
            risk_class=risk_class,
            smoking_status=smoking_status,
            multipliers=multipliers,
            projected_cash_value_10yr=round(cash_value_10yr, 2),
        )

    def quote_ul(
        self,
        age: int,
        gender: Gender,
        coverage: float,
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        smoking_status: SmokingStatus = SmokingStatus.NON_SMOKER,
    ) -> PremiumQuote:
        """Get quote for universal life insurance."""
        age_band = get_age_band(age, UNIVERSAL_LIFE_BASE_RATES)
        gender_key = gender.value if gender != Gender.UNISEX else "male"

        base_rate = UNIVERSAL_LIFE_BASE_RATES[age_band][gender_key]

        multipliers = {
            "risk_class": RISK_CLASS_MULTIPLIERS[risk_class],
            "smoking": SMOKING_MULTIPLIERS[smoking_status],
        }

        final_rate = base_rate
        for mult in multipliers.values():
            final_rate *= mult

        monthly = (coverage / 1000) * final_rate
        cash_value_10yr = monthly * 12 * 10 * 0.35

        return PremiumQuote(
            plan_type=PlanType.UL,
            base_rate=base_rate,
            final_rate=round(final_rate, 4),
            monthly_premium=round(monthly, 2),
            annual_premium=round(monthly * 12, 2),
            age=age,
            gender=gender,
            coverage_amount=coverage,
            risk_class=risk_class,
            smoking_status=smoking_status,
            multipliers=multipliers,
            projected_cash_value_10yr=round(cash_value_10yr, 2),
        )

    def quote_vul(
        self,
        age: int,
        gender: Gender,
        coverage: float,
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        smoking_status: SmokingStatus = SmokingStatus.NON_SMOKER,
    ) -> PremiumQuote:
        """Get quote for variable universal life insurance."""
        age_band = get_age_band(age, VARIABLE_UNIVERSAL_LIFE_BASE_RATES)
        gender_key = gender.value if gender != Gender.UNISEX else "male"

        base_rate = VARIABLE_UNIVERSAL_LIFE_BASE_RATES[age_band][gender_key]

        multipliers = {
            "risk_class": RISK_CLASS_MULTIPLIERS[risk_class],
            "smoking": SMOKING_MULTIPLIERS[smoking_status],
        }

        final_rate = base_rate
        for mult in multipliers.values():
            final_rate *= mult

        monthly = (coverage / 1000) * final_rate
        # VUL cash value varies based on investment performance
        cash_value_10yr = monthly * 12 * 10 * 0.30  # Conservative estimate

        return PremiumQuote(
            plan_type=PlanType.VUL,
            base_rate=base_rate,
            final_rate=round(final_rate, 4),
            monthly_premium=round(monthly, 2),
            annual_premium=round(monthly * 12, 2),
            age=age,
            gender=gender,
            coverage_amount=coverage,
            risk_class=risk_class,
            smoking_status=smoking_status,
            multipliers=multipliers,
            projected_cash_value_10yr=round(cash_value_10yr, 2),
        )

    def quote_ltc(
        self,
        age: int,
        gender: Gender,
        monthly_benefit: float,
        benefit_period: str = "3_year",
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        health_tier: HealthTier = HealthTier.GOOD,
    ) -> PremiumQuote:
        """Get quote for long-term care insurance."""
        if benefit_period not in LTC_BASE_RATES:
            benefit_period = "3_year"

        rate_table = LTC_BASE_RATES[benefit_period]
        age_band = get_age_band(age, rate_table)
        gender_key = gender.value if gender != Gender.UNISEX else "male"

        base_rate = rate_table[age_band][gender_key]

        multipliers = {
            "risk_class": RISK_CLASS_MULTIPLIERS[risk_class],
            "health": HEALTH_MULTIPLIERS[health_tier],
        }

        final_rate = base_rate
        for mult in multipliers.values():
            final_rate *= mult

        monthly = (monthly_benefit / 100) * final_rate

        benefit_period_display = {
            "3_year": "3 years",
            "5_year": "5 years",
            "unlimited": "Unlimited",
        }

        return PremiumQuote(
            plan_type=PlanType.LTC,
            base_rate=base_rate,
            final_rate=round(final_rate, 4),
            monthly_premium=round(monthly, 2),
            annual_premium=round(monthly * 12, 2),
            age=age,
            gender=gender,
            coverage_amount=monthly_benefit,
            risk_class=risk_class,
            smoking_status=SmokingStatus.NON_SMOKER,
            multipliers=multipliers,
            benefit_period=benefit_period_display.get(benefit_period, benefit_period),
            waiting_period="90 days",
        )

    def quote_di(
        self,
        age: int,
        gender: Gender,
        monthly_benefit: float,
        benefit_type: str = "own_occ_65",
        occupation_class: str = "white_collar",
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
    ) -> PremiumQuote:
        """Get quote for disability insurance."""
        if benefit_type not in DI_BASE_RATES:
            benefit_type = "own_occ_65"

        rate_table = DI_BASE_RATES[benefit_type]
        age_band = get_age_band(age, rate_table)
        gender_key = gender.value if gender != Gender.UNISEX else "male"

        base_rate = rate_table[age_band][gender_key]

        multipliers = {
            "risk_class": RISK_CLASS_MULTIPLIERS[risk_class],
            "occupation": OCCUPATION_MULTIPLIERS.get(occupation_class, 1.0),
        }

        final_rate = base_rate
        for mult in multipliers.values():
            final_rate *= mult

        monthly = (monthly_benefit / 100) * final_rate

        benefit_type_display = {
            "own_occ_65": "Own-occupation to age 65",
            "any_occ_65": "Any-occupation to age 65",
            "5_year": "5-year benefit period",
        }

        return PremiumQuote(
            plan_type=PlanType.DI,
            base_rate=base_rate,
            final_rate=round(final_rate, 4),
            monthly_premium=round(monthly, 2),
            annual_premium=round(monthly * 12, 2),
            age=age,
            gender=gender,
            coverage_amount=monthly_benefit,
            risk_class=risk_class,
            smoking_status=SmokingStatus.NON_SMOKER,
            multipliers=multipliers,
            benefit_period=benefit_type_display.get(benefit_type, benefit_type),
            waiting_period="90 days",
        )

    def quote(
        self,
        plan_type: PlanType,
        age: int,
        gender: Gender,
        coverage: float,
        risk_class: RiskClass = RiskClass.STANDARD_PLUS,
        smoking_status: SmokingStatus = SmokingStatus.NON_SMOKER,
        **kwargs,
    ) -> PremiumQuote:
        """Get quote for any plan type."""
        if plan_type == PlanType.TERM:
            term_years = kwargs.get("term_years", 20)
            return self.quote_term_life(
                age, gender, coverage, term_years, risk_class, smoking_status
            )
        elif plan_type == PlanType.WHOLE:
            return self.quote_whole_life(age, gender, coverage, risk_class, smoking_status)
        elif plan_type == PlanType.UL:
            return self.quote_ul(age, gender, coverage, risk_class, smoking_status)
        elif plan_type == PlanType.VUL:
            return self.quote_vul(age, gender, coverage, risk_class, smoking_status)
        elif plan_type == PlanType.LTC:
            benefit_period = kwargs.get("benefit_period", "3_year")
            health_tier = kwargs.get("health_tier", HealthTier.GOOD)
            return self.quote_ltc(age, gender, coverage, benefit_period, risk_class, health_tier)
        elif plan_type == PlanType.DI:
            benefit_type = kwargs.get("benefit_type", "own_occ_65")
            occupation_class = kwargs.get("occupation_class", "white_collar")
            return self.quote_di(age, gender, coverage, benefit_type, occupation_class, risk_class)
        else:
            raise ValueError(f"Unknown plan type: {plan_type}")
