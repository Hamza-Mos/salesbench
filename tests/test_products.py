"""Tests for product catalog and pricing."""

from salesbench.core.types import PlanType, RiskClass
from salesbench.envs.sales_mvp.products import (
    ProductCatalog,
    get_age_band,
)


class TestProductCatalog:
    """Tests for ProductCatalog."""

    def test_all_six_plan_types_exist(self, product_catalog: ProductCatalog):
        """Test that all 6 plan types are in catalog."""
        products = product_catalog.list_products()
        plan_ids = {p["plan_id"] for p in products}

        expected = {"TERM", "WHOLE", "UL", "VUL", "LTC", "DI"}
        assert plan_ids == expected

    def test_get_product_returns_correct_type(self, product_catalog: ProductCatalog):
        """Test getting individual products."""
        for plan_type in PlanType:
            product = product_catalog.get_product(plan_type)
            assert product is not None
            assert product.plan_id == plan_type

    def test_get_product_unknown_returns_none(self, product_catalog: ProductCatalog):
        """Test that unknown plan returns None."""
        # Create a mock invalid plan type
        result = product_catalog._products.get("INVALID")
        assert result is None

    def test_product_coverage_ranges_valid(self, product_catalog: ProductCatalog):
        """Test that all products have valid coverage ranges."""
        for plan_type in PlanType:
            product = product_catalog.get_product(plan_type)
            assert product.min_coverage > 0
            assert product.max_coverage > product.min_coverage

    def test_product_age_ranges_valid(self, product_catalog: ProductCatalog):
        """Test that all products have valid age ranges."""
        for plan_type in PlanType:
            product = product_catalog.get_product(plan_type)
            assert product.min_age >= 18
            assert product.max_age <= 85
            assert product.max_age > product.min_age

    def test_term_plan_has_term_options(self, product_catalog: ProductCatalog):
        """Test that TERM plan has term options."""
        term = product_catalog.get_product(PlanType.TERM)
        assert term.term_options is not None
        assert 10 in term.term_options
        assert 20 in term.term_options
        assert 30 in term.term_options


class TestPricing:
    """Tests for premium pricing calculations."""

    def test_quote_premium_basic(self, product_catalog: ProductCatalog):
        """Test basic premium quote."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        assert "error" not in quote
        assert quote["monthly_premium"] > 0
        assert quote["annual_premium"] == quote["monthly_premium"] * 12

    def test_quote_premium_is_deterministic(self, product_catalog: ProductCatalog):
        """Test that same inputs produce same premium."""
        params = {
            "plan_id": PlanType.TERM,
            "age": 35,
            "coverage_amount": 500000,
            "risk_class": RiskClass.STANDARD_PLUS,
            "term_years": 20,
        }

        quote1 = product_catalog.quote_premium(**params)
        quote2 = product_catalog.quote_premium(**params)

        assert quote1["monthly_premium"] == quote2["monthly_premium"]

    def test_premium_increases_with_age(self, product_catalog: ProductCatalog):
        """Test that premium increases with age."""
        young = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=30,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        old = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=55,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        assert old["monthly_premium"] > young["monthly_premium"]

    def test_premium_increases_with_coverage(self, product_catalog: ProductCatalog):
        """Test that premium increases with coverage amount."""
        low = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=100000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        high = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        assert high["monthly_premium"] > low["monthly_premium"]
        # Coverage 5x higher should mean premium roughly 5x higher
        ratio = high["monthly_premium"] / low["monthly_premium"]
        assert 4.5 < ratio < 5.5

    def test_risk_class_affects_premium(self, product_catalog: ProductCatalog):
        """Test that worse risk class means higher premium."""
        preferred = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.PREFERRED_PLUS,
            term_years=20,
        )

        standard = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD,
            term_years=20,
        )

        assert standard["monthly_premium"] > preferred["monthly_premium"]

    def test_term_length_affects_premium(self, product_catalog: ProductCatalog):
        """Test that longer term means higher premium."""
        short = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=10,
        )

        long = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=30,
        )

        assert long["monthly_premium"] > short["monthly_premium"]

    def test_quote_invalid_age_returns_error(self, product_catalog: ProductCatalog):
        """Test that invalid age returns error."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=90,  # Too old for TERM
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        assert "error" in quote

    def test_quote_invalid_coverage_returns_error(self, product_catalog: ProductCatalog):
        """Test that invalid coverage returns error."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=10,  # Too low
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        assert "error" in quote

    def test_quote_invalid_term_returns_error(self, product_catalog: ProductCatalog):
        """Test that invalid term length returns error."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=25,  # Invalid term
        )

        assert "error" in quote

    def test_whole_life_includes_cash_value(self, product_catalog: ProductCatalog):
        """Test that WHOLE life quote includes cash value projection."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.WHOLE,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
        )

        assert "projected_cash_value_year_10" in quote
        assert quote["projected_cash_value_year_10"] > 0

    def test_di_includes_benefit_info(self, product_catalog: ProductCatalog):
        """Test that DI quote includes benefit information."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.DI,
            age=35,
            coverage_amount=5000,  # Monthly benefit
            risk_class=RiskClass.STANDARD_PLUS,
        )

        assert "monthly_benefit" in quote
        assert "benefit_period" in quote
        assert "waiting_period" in quote

    def test_di_waiting_period_affects_premium(self, product_catalog: ProductCatalog):
        """Test that shorter waiting periods increase DI premium."""
        quote_90 = product_catalog.quote_premium(
            plan_id=PlanType.DI,
            age=35,
            coverage_amount=5000,
            risk_class=RiskClass.STANDARD_PLUS,
            waiting_period_days=90,
        )
        quote_30 = product_catalog.quote_premium(
            plan_id=PlanType.DI,
            age=35,
            coverage_amount=5000,
            risk_class=RiskClass.STANDARD_PLUS,
            waiting_period_days=30,
        )
        assert quote_30["monthly_premium"] > quote_90["monthly_premium"]
        assert "30 days" in quote_30["waiting_period"]

    def test_ltc_waiting_period_affects_premium(self, product_catalog: ProductCatalog):
        """Test that shorter waiting periods increase LTC premium."""
        quote_90 = product_catalog.quote_premium(
            plan_id=PlanType.LTC,
            age=55,
            coverage_amount=150000,
            risk_class=RiskClass.STANDARD,
            waiting_period_days=90,
        )
        quote_30 = product_catalog.quote_premium(
            plan_id=PlanType.LTC,
            age=55,
            coverage_amount=150000,
            risk_class=RiskClass.STANDARD,
            waiting_period_days=30,
        )
        assert quote_30["monthly_premium"] > quote_90["monthly_premium"]

    def test_waiting_period_rejected_for_life_insurance(self, product_catalog: ProductCatalog):
        """Test that waiting_period_days is rejected for non-DI/LTC plans."""
        result = product_catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD,
            waiting_period_days=30,
        )
        assert "error" in result
        assert "only applicable to DI and LTC" in result["error"]

    def test_invalid_waiting_period_returns_error(self, product_catalog: ProductCatalog):
        """Test that invalid waiting period returns error."""
        result = product_catalog.quote_premium(
            plan_id=PlanType.DI,
            age=35,
            coverage_amount=5000,
            risk_class=RiskClass.STANDARD_PLUS,
            waiting_period_days=45,  # Invalid
        )
        assert "error" in result
        assert "Invalid waiting period" in result["error"]

    def test_waiting_period_default_is_90_days(self, product_catalog: ProductCatalog):
        """Test that default waiting period is 90 days."""
        quote = product_catalog.quote_premium(
            plan_id=PlanType.DI,
            age=35,
            coverage_amount=5000,
            risk_class=RiskClass.STANDARD_PLUS,
        )
        assert quote["waiting_period"] == "90 days"


class TestAgeBand:
    """Tests for age band calculation."""

    def test_age_band_young(self):
        """Test age band for young ages."""
        assert get_age_band(25) == "25-34"
        assert get_age_band(30) == "25-34"
        assert get_age_band(34) == "25-34"

    def test_age_band_boundaries(self):
        """Test age band at boundaries."""
        assert get_age_band(34) == "25-34"
        assert get_age_band(35) == "35-44"
        assert get_age_band(44) == "35-44"
        assert get_age_band(45) == "45-54"

    def test_age_band_senior(self):
        """Test age band for seniors."""
        assert get_age_band(65) == "65+"
        assert get_age_band(75) == "65+"
        assert get_age_band(85) == "65+"


class TestRecommendedCoverage:
    """Tests for recommended coverage calculations."""

    def test_recommended_coverage_increases_with_income(self, product_catalog: ProductCatalog):
        """Test that recommended coverage increases with income."""
        low = product_catalog.get_recommended_coverage(
            annual_income=50000,
            age=35,
            has_dependents=True,
            plan_id=PlanType.TERM,
        )

        high = product_catalog.get_recommended_coverage(
            annual_income=150000,
            age=35,
            has_dependents=True,
            plan_id=PlanType.TERM,
        )

        assert high > low

    def test_recommended_coverage_higher_with_dependents(self, product_catalog: ProductCatalog):
        """Test that dependents increase recommended coverage."""
        no_deps = product_catalog.get_recommended_coverage(
            annual_income=100000,
            age=35,
            has_dependents=False,
            plan_id=PlanType.TERM,
        )

        with_deps = product_catalog.get_recommended_coverage(
            annual_income=100000,
            age=35,
            has_dependents=True,
            plan_id=PlanType.TERM,
        )

        assert with_deps > no_deps

    def test_di_coverage_is_income_percentage(self, product_catalog: ProductCatalog):
        """Test that DI coverage is based on income percentage."""
        coverage = product_catalog.get_recommended_coverage(
            annual_income=120000,
            age=35,
            has_dependents=True,
            plan_id=PlanType.DI,
        )

        monthly_income = 120000 / 12
        expected_max = monthly_income * 0.6

        assert coverage <= expected_max
