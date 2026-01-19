"""Structured offer schema with riders, underwriting, and coverage tiers.

This module provides detailed offer structures that go beyond basic
PlanOffer, including:
- Policy riders and endorsements
- Underwriting classifications
- Coverage tier configurations
- Payment options
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from salesbench.core.types import CoverageTier, NextStep, PlanType


class RiderType(str, Enum):
    """Available policy riders."""

    # Life Insurance Riders
    ACCIDENTAL_DEATH = "accidental_death"  # ADB - Double benefit for accidents
    WAIVER_OF_PREMIUM = "waiver_of_premium"  # WOP - Waives premium if disabled
    ACCELERATED_DEATH = "accelerated_death"  # ADB - Access benefit if terminal
    CHILD_TERM = "child_term"  # Covers children
    SPOUSE_TERM = "spouse_term"  # Covers spouse
    GUARANTEED_INSURABILITY = "guaranteed_insurability"  # GIO - Buy more later
    RETURN_OF_PREMIUM = "return_of_premium"  # ROP - Get premiums back
    CHRONIC_ILLNESS = "chronic_illness"  # Access benefit for chronic illness
    CRITICAL_ILLNESS = "critical_illness"  # Lump sum for critical illness

    # DI Riders
    COST_OF_LIVING = "cost_of_living"  # COLA - Inflation adjustment
    FUTURE_INCREASE = "future_increase"  # Buy more coverage later
    RESIDUAL_DISABILITY = "residual_disability"  # Partial disability benefits
    CATASTROPHIC_DISABILITY = "catastrophic_disability"  # Extra for severe disability

    # LTC Riders
    INFLATION_PROTECTION = "inflation_protection"  # 3% or 5% compound
    SHARED_CARE = "shared_care"  # Share benefits with spouse
    RESTORATION_OF_BENEFITS = "restoration_of_benefits"  # Restore used benefits
    RETURN_OF_PREMIUM_LTC = "return_of_premium_ltc"  # Get premiums back if unused


class PaymentFrequency(str, Enum):
    """Payment frequency options."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


class PaymentMethod(str, Enum):
    """Payment method options."""

    BANK_DRAFT = "bank_draft"  # ACH/EFT
    CREDIT_CARD = "credit_card"
    CHECK = "check"
    PAYROLL_DEDUCTION = "payroll_deduction"


class UnderwritingClass(str, Enum):
    """Underwriting classification."""

    SUPER_PREFERRED = "super_preferred"  # Best rates, perfect health
    PREFERRED_PLUS = "preferred_plus"  # Excellent health
    PREFERRED = "preferred"  # Very good health
    STANDARD_PLUS = "standard_plus"  # Good health, minor issues
    STANDARD = "standard"  # Average health
    SUBSTANDARD_A = "substandard_a"  # Minor health issues
    SUBSTANDARD_B = "substandard_b"  # Moderate health issues
    SUBSTANDARD_C = "substandard_c"  # Significant health issues
    DECLINE = "decline"  # Cannot underwrite


class UnderwritingMethod(str, Enum):
    """Underwriting method options."""

    FULL_UNDERWRITING = "full_underwriting"  # Exam + medical records
    SIMPLIFIED_ISSUE = "simplified_issue"  # Health questions only
    GUARANTEED_ISSUE = "guaranteed_issue"  # No health questions
    ACCELERATED_UNDERWRITING = "accelerated_underwriting"  # Digital, instant


@dataclass
class RiderConfig:
    """Configuration for a policy rider."""

    rider_type: RiderType
    name: str
    description: str
    monthly_cost: float  # Additional monthly premium
    benefit_amount: Optional[float] = None
    benefit_percentage: Optional[float] = None  # % of base benefit
    is_included: bool = False  # True if included in base policy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "rider_type": self.rider_type.value,
            "name": self.name,
            "description": self.description,
            "monthly_cost": self.monthly_cost,
            "is_included": self.is_included,
        }
        if self.benefit_amount:
            result["benefit_amount"] = self.benefit_amount
        if self.benefit_percentage:
            result["benefit_percentage"] = self.benefit_percentage
        return result


# Standard rider definitions
AVAILABLE_RIDERS = {
    RiderType.ACCIDENTAL_DEATH: RiderConfig(
        rider_type=RiderType.ACCIDENTAL_DEATH,
        name="Accidental Death Benefit",
        description="Pays an additional death benefit if death is caused by an accident",
        monthly_cost=0.0,  # Calculated based on coverage
        benefit_percentage=100.0,  # Usually equal to base benefit
    ),
    RiderType.WAIVER_OF_PREMIUM: RiderConfig(
        rider_type=RiderType.WAIVER_OF_PREMIUM,
        name="Waiver of Premium",
        description="Waives premium payments if you become totally disabled",
        monthly_cost=0.0,
    ),
    RiderType.ACCELERATED_DEATH: RiderConfig(
        rider_type=RiderType.ACCELERATED_DEATH,
        name="Accelerated Death Benefit",
        description="Access portion of death benefit if diagnosed with terminal illness",
        monthly_cost=0.0,
        benefit_percentage=50.0,
        is_included=True,  # Often included at no charge
    ),
    RiderType.CHILD_TERM: RiderConfig(
        rider_type=RiderType.CHILD_TERM,
        name="Children's Term Rider",
        description="Provides term coverage for all eligible children",
        monthly_cost=0.0,
        benefit_amount=10000.0,
    ),
    RiderType.SPOUSE_TERM: RiderConfig(
        rider_type=RiderType.SPOUSE_TERM,
        name="Spouse Term Rider",
        description="Provides term coverage for your spouse",
        monthly_cost=0.0,
        benefit_amount=50000.0,
    ),
    RiderType.GUARANTEED_INSURABILITY: RiderConfig(
        rider_type=RiderType.GUARANTEED_INSURABILITY,
        name="Guaranteed Insurability Option",
        description="Right to purchase additional coverage at specified dates without proof of insurability",
        monthly_cost=0.0,
    ),
    RiderType.RETURN_OF_PREMIUM: RiderConfig(
        rider_type=RiderType.RETURN_OF_PREMIUM,
        name="Return of Premium",
        description="Returns all premiums paid if you outlive the term",
        monthly_cost=0.0,  # Significant additional cost
    ),
    RiderType.CHRONIC_ILLNESS: RiderConfig(
        rider_type=RiderType.CHRONIC_ILLNESS,
        name="Chronic Illness Rider",
        description="Access death benefit early if diagnosed with a chronic illness",
        monthly_cost=0.0,
        benefit_percentage=25.0,
        is_included=True,
    ),
    RiderType.CRITICAL_ILLNESS: RiderConfig(
        rider_type=RiderType.CRITICAL_ILLNESS,
        name="Critical Illness Rider",
        description="Lump sum payment upon diagnosis of covered critical illness",
        monthly_cost=0.0,
        benefit_amount=25000.0,
    ),
    RiderType.COST_OF_LIVING: RiderConfig(
        rider_type=RiderType.COST_OF_LIVING,
        name="Cost of Living Adjustment",
        description="Increases benefits annually while on claim to keep pace with inflation",
        monthly_cost=0.0,
    ),
    RiderType.FUTURE_INCREASE: RiderConfig(
        rider_type=RiderType.FUTURE_INCREASE,
        name="Future Increase Option",
        description="Right to purchase additional coverage without proof of insurability",
        monthly_cost=0.0,
    ),
    RiderType.RESIDUAL_DISABILITY: RiderConfig(
        rider_type=RiderType.RESIDUAL_DISABILITY,
        name="Residual Disability",
        description="Provides partial benefits for partial disability",
        monthly_cost=0.0,
    ),
    RiderType.INFLATION_PROTECTION: RiderConfig(
        rider_type=RiderType.INFLATION_PROTECTION,
        name="Inflation Protection",
        description="Increases benefit pool annually by 3-5% compound",
        monthly_cost=0.0,
    ),
    RiderType.SHARED_CARE: RiderConfig(
        rider_type=RiderType.SHARED_CARE,
        name="Shared Care",
        description="Share benefits pool with spouse",
        monthly_cost=0.0,
    ),
}


@dataclass
class UnderwritingRequirements:
    """Underwriting requirements for a policy."""

    method: UnderwritingMethod
    requires_medical_exam: bool = False
    requires_blood_test: bool = False
    requires_medical_records: bool = False
    requires_phone_interview: bool = False
    health_questions_count: int = 0
    max_coverage_no_exam: float = 0.0
    decision_time_estimate: str = "2-4 weeks"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "requires_medical_exam": self.requires_medical_exam,
            "requires_blood_test": self.requires_blood_test,
            "requires_medical_records": self.requires_medical_records,
            "requires_phone_interview": self.requires_phone_interview,
            "health_questions_count": self.health_questions_count,
            "max_coverage_no_exam": self.max_coverage_no_exam,
            "decision_time_estimate": self.decision_time_estimate,
        }


# Standard underwriting requirements by method
UNDERWRITING_REQUIREMENTS = {
    UnderwritingMethod.FULL_UNDERWRITING: UnderwritingRequirements(
        method=UnderwritingMethod.FULL_UNDERWRITING,
        requires_medical_exam=True,
        requires_blood_test=True,
        requires_medical_records=True,
        requires_phone_interview=True,
        health_questions_count=50,
        max_coverage_no_exam=0,
        decision_time_estimate="2-6 weeks",
    ),
    UnderwritingMethod.SIMPLIFIED_ISSUE: UnderwritingRequirements(
        method=UnderwritingMethod.SIMPLIFIED_ISSUE,
        requires_medical_exam=False,
        requires_blood_test=False,
        requires_medical_records=False,
        requires_phone_interview=True,
        health_questions_count=15,
        max_coverage_no_exam=500000,
        decision_time_estimate="24-48 hours",
    ),
    UnderwritingMethod.GUARANTEED_ISSUE: UnderwritingRequirements(
        method=UnderwritingMethod.GUARANTEED_ISSUE,
        requires_medical_exam=False,
        requires_blood_test=False,
        requires_medical_records=False,
        requires_phone_interview=False,
        health_questions_count=0,
        max_coverage_no_exam=50000,
        decision_time_estimate="Immediate",
    ),
    UnderwritingMethod.ACCELERATED_UNDERWRITING: UnderwritingRequirements(
        method=UnderwritingMethod.ACCELERATED_UNDERWRITING,
        requires_medical_exam=False,
        requires_blood_test=False,
        requires_medical_records=False,
        requires_phone_interview=False,
        health_questions_count=10,
        max_coverage_no_exam=1000000,
        decision_time_estimate="Minutes to 24 hours",
    ),
}


@dataclass
class PaymentConfig:
    """Payment configuration for a policy."""

    frequency: PaymentFrequency
    method: PaymentMethod
    monthly_amount: float
    annual_amount: float
    first_payment_amount: float

    # Discounts
    annual_discount_pct: float = 2.0  # Typical discount for annual pay
    eft_discount_pct: float = 1.0  # Discount for bank draft

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frequency": self.frequency.value,
            "method": self.method.value,
            "monthly_amount": self.monthly_amount,
            "annual_amount": self.annual_amount,
            "first_payment_amount": self.first_payment_amount,
            "annual_discount_pct": self.annual_discount_pct,
            "eft_discount_pct": self.eft_discount_pct,
        }


@dataclass
class CoverageTierConfig:
    """Configuration for a coverage tier."""

    tier: CoverageTier
    coverage_amount: float
    name: str
    description: str
    is_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "coverage_amount": self.coverage_amount,
            "name": self.name,
            "description": self.description,
            "is_recommended": self.is_recommended,
        }


# Standard coverage tiers
STANDARD_COVERAGE_TIERS = {
    CoverageTier.BASIC: CoverageTierConfig(
        tier=CoverageTier.BASIC,
        coverage_amount=100_000,
        name="Basic Protection",
        description="Essential coverage for final expenses and short-term needs",
    ),
    CoverageTier.STANDARD: CoverageTierConfig(
        tier=CoverageTier.STANDARD,
        coverage_amount=250_000,
        name="Standard Protection",
        description="Covers mortgage, debts, and 3-5 years income replacement",
    ),
    CoverageTier.ENHANCED: CoverageTierConfig(
        tier=CoverageTier.ENHANCED,
        coverage_amount=500_000,
        name="Enhanced Protection",
        description="Comprehensive coverage for family security and education",
        is_recommended=True,
    ),
    CoverageTier.PREMIUM: CoverageTierConfig(
        tier=CoverageTier.PREMIUM,
        coverage_amount=1_000_000,
        name="Premium Protection",
        description="Full income replacement and legacy planning",
    ),
    CoverageTier.ELITE: CoverageTierConfig(
        tier=CoverageTier.ELITE,
        coverage_amount=2_000_000,
        name="Elite Protection",
        description="Maximum protection for high-income families",
    ),
}


@dataclass
class StructuredOffer:
    """Complete structured offer with all details."""

    # Base offer
    plan_type: PlanType
    coverage_amount: float
    monthly_premium: float
    annual_premium: float
    next_step: NextStep

    # Policy details
    term_years: Optional[int] = None
    coverage_tier: Optional[CoverageTier] = None

    # Riders
    included_riders: list[RiderConfig] = field(default_factory=list)
    optional_riders: list[RiderConfig] = field(default_factory=list)
    selected_riders: list[RiderType] = field(default_factory=list)

    # Underwriting
    underwriting_class: UnderwritingClass = UnderwritingClass.STANDARD
    underwriting_requirements: Optional[UnderwritingRequirements] = None

    # Payment
    payment_config: Optional[PaymentConfig] = None

    # Product-specific
    benefit_period: Optional[str] = None  # LTC/DI
    waiting_period: Optional[str] = None  # LTC/DI
    projected_cash_value_10yr: Optional[float] = None  # Permanent life

    # Metadata
    quote_valid_days: int = 30
    notes: Optional[str] = None

    def total_monthly_premium(self) -> float:
        """Calculate total monthly premium including riders."""
        base = self.monthly_premium
        rider_cost = sum(
            r.monthly_cost for r in self.included_riders if r.rider_type in self.selected_riders
        )
        return base + rider_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "plan_type": self.plan_type.value,
            "coverage_amount": self.coverage_amount,
            "monthly_premium": self.monthly_premium,
            "annual_premium": self.annual_premium,
            "next_step": self.next_step.value,
            "underwriting_class": self.underwriting_class.value,
            "quote_valid_days": self.quote_valid_days,
        }

        if self.term_years:
            result["term_years"] = self.term_years
        if self.coverage_tier:
            result["coverage_tier"] = self.coverage_tier.value
        if self.included_riders:
            result["included_riders"] = [r.to_dict() for r in self.included_riders]
        if self.optional_riders:
            result["optional_riders"] = [r.to_dict() for r in self.optional_riders]
        if self.selected_riders:
            result["selected_riders"] = [r.value for r in self.selected_riders]
        if self.underwriting_requirements:
            result["underwriting_requirements"] = self.underwriting_requirements.to_dict()
        if self.payment_config:
            result["payment_config"] = self.payment_config.to_dict()
        if self.benefit_period:
            result["benefit_period"] = self.benefit_period
        if self.waiting_period:
            result["waiting_period"] = self.waiting_period
        if self.projected_cash_value_10yr:
            result["projected_cash_value_10yr"] = self.projected_cash_value_10yr
        if self.notes:
            result["notes"] = self.notes

        result["total_monthly_premium"] = self.total_monthly_premium()

        return result


class OfferBuilder:
    """Builder for creating structured offers."""

    def __init__(self, plan_type: PlanType):
        """Initialize builder."""
        self.plan_type = plan_type
        self._coverage = 0.0
        self._premium = 0.0
        self._term_years = None
        self._next_step = NextStep.REQUEST_INFO
        self._underwriting_class = UnderwritingClass.STANDARD
        self._underwriting_method = UnderwritingMethod.SIMPLIFIED_ISSUE
        self._riders = []
        self._payment_frequency = PaymentFrequency.MONTHLY
        self._payment_method = PaymentMethod.BANK_DRAFT

    def with_coverage(self, amount: float, tier: Optional[CoverageTier] = None) -> "OfferBuilder":
        """Set coverage amount."""
        self._coverage = amount
        self._tier = tier
        return self

    def with_premium(self, monthly: float) -> "OfferBuilder":
        """Set premium."""
        self._premium = monthly
        return self

    def with_term(self, years: int) -> "OfferBuilder":
        """Set term length."""
        self._term_years = years
        return self

    def with_next_step(self, step: NextStep) -> "OfferBuilder":
        """Set next step."""
        self._next_step = step
        return self

    def with_underwriting(
        self,
        classification: UnderwritingClass,
        method: UnderwritingMethod,
    ) -> "OfferBuilder":
        """Set underwriting details."""
        self._underwriting_class = classification
        self._underwriting_method = method
        return self

    def with_rider(self, rider_type: RiderType) -> "OfferBuilder":
        """Add a rider."""
        self._riders.append(rider_type)
        return self

    def with_payment(
        self,
        frequency: PaymentFrequency,
        method: PaymentMethod,
    ) -> "OfferBuilder":
        """Set payment preferences."""
        self._payment_frequency = frequency
        self._payment_method = method
        return self

    def build(self) -> StructuredOffer:
        """Build the structured offer."""
        # Determine included vs optional riders based on plan type
        included = []
        optional = []

        life_plans = {PlanType.TERM, PlanType.WHOLE, PlanType.UL, PlanType.VUL}
        if self.plan_type in life_plans:
            included.append(AVAILABLE_RIDERS[RiderType.ACCELERATED_DEATH])
            included.append(AVAILABLE_RIDERS[RiderType.CHRONIC_ILLNESS])
            optional.extend(
                [
                    AVAILABLE_RIDERS[RiderType.ACCIDENTAL_DEATH],
                    AVAILABLE_RIDERS[RiderType.WAIVER_OF_PREMIUM],
                    AVAILABLE_RIDERS[RiderType.CHILD_TERM],
                    AVAILABLE_RIDERS[RiderType.GUARANTEED_INSURABILITY],
                ]
            )
            if self.plan_type == PlanType.TERM:
                optional.append(AVAILABLE_RIDERS[RiderType.RETURN_OF_PREMIUM])

        # Build payment config
        annual = self._premium * 12 * 0.98  # 2% annual discount
        first_payment = (
            self._premium if self._payment_frequency == PaymentFrequency.MONTHLY else annual
        )

        payment_config = PaymentConfig(
            frequency=self._payment_frequency,
            method=self._payment_method,
            monthly_amount=self._premium,
            annual_amount=round(annual, 2),
            first_payment_amount=first_payment,
        )

        return StructuredOffer(
            plan_type=self.plan_type,
            coverage_amount=self._coverage,
            monthly_premium=self._premium,
            annual_premium=round(self._premium * 12, 2),
            next_step=self._next_step,
            term_years=self._term_years,
            coverage_tier=getattr(self, "_tier", None),
            included_riders=included,
            optional_riders=optional,
            selected_riders=self._riders,
            underwriting_class=self._underwriting_class,
            underwriting_requirements=UNDERWRITING_REQUIREMENTS.get(self._underwriting_method),
            payment_config=payment_config,
        )
