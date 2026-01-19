#!/usr/bin/env python3
"""End-to-end validation script for SalesBench.

This script validates that all SalesBench components work together correctly.
It runs through various scenarios to ensure production readiness.

Usage:
    python scripts/validate_e2e.py
    python scripts/validate_e2e.py --verbose
    python scripts/validate_e2e.py --quick  # Skip long-running tests
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, "environments")


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Optional[dict] = None


class SalesBenchValidator:
    """Validates SalesBench components and functionality."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[ValidationResult] = []

    def log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")

    def run_validation(self, name: str, func):
        """Run a validation function and record the result."""
        print(f"Running: {name}...", end=" ", flush=True)
        start = time.time()

        try:
            result = func()
            duration = time.time() - start

            if isinstance(result, tuple):
                passed, message = result
            else:
                passed, message = result, ""

            self.results.append(
                ValidationResult(
                    name=name,
                    passed=passed,
                    duration=duration,
                    message=message,
                )
            )

            status = "PASS" if passed else "FAIL"
            print(f"{status} ({duration:.2f}s)")
            if message and (not passed or self.verbose):
                print(f"    {message}")

        except Exception as e:
            duration = time.time() - start
            self.results.append(
                ValidationResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    message=str(e),
                )
            )
            print(f"ERROR ({duration:.2f}s)")
            print(f"    {e}")

    def validate_imports(self) -> tuple[bool, str]:
        """Validate that all required modules can be imported."""
        try:
            from salesbench import load_environment
            from salesbench.core.types import ToolCall, ToolResult
            from salesbench.envs.sales_mvp.personas import PersonaGenerator
            from salesbench.envs.sales_mvp.products import ProductCatalog
            from salesbench.envs.sales_mvp.verifiers.scoring import calculate_episode_score
            from salesbench.orchestrator.orchestrator import Orchestrator

            return True, "All imports successful"
        except ImportError as e:
            return False, f"Import failed: {e}"

    def validate_persona_generation(self) -> tuple[bool, str]:
        """Validate persona generation with seeding."""
        from salesbench.envs.sales_mvp.personas import PersonaGenerator

        gen1 = PersonaGenerator(seed=42)
        gen2 = PersonaGenerator(seed=42)

        personas1 = gen1.generate_batch(100)
        personas2 = gen2.generate_batch(100)

        # Check determinism
        for p1, p2 in zip(personas1, personas2):
            if p1.name != p2.name:
                return False, "Seeded generation is not deterministic"

        # Check hidden state bounds
        for p in personas1:
            if not (0 <= p.hidden.trust <= 1):
                return False, f"Trust out of bounds: {p.hidden.trust}"

        return True, "Generated 100 deterministic personas"

    def validate_product_pricing(self) -> tuple[bool, str]:
        """Validate product catalog and pricing."""
        from salesbench.core.types import PlanType, RiskClass
        from salesbench.envs.sales_mvp.products import ProductCatalog

        catalog = ProductCatalog()

        # Check all plans exist
        for plan_type in PlanType:
            product = catalog.get_product(plan_type)
            if product is None:
                return False, f"Missing plan: {plan_type}"

        # Test pricing
        quote = catalog.quote_premium(
            plan_id=PlanType.TERM,
            age=35,
            coverage_amount=500000,
            risk_class=RiskClass.STANDARD_PLUS,
            term_years=20,
        )

        if "error" in quote:
            return False, f"Pricing error: {quote['error']}"

        if quote["monthly_premium"] <= 0:
            return False, "Premium should be positive"

        return True, f"TERM 500K at 35yo = ${quote['monthly_premium']:.2f}/mo"

    def validate_environment_creation(self) -> tuple[bool, str]:
        """Validate environment can be created and reset."""
        from salesbench.core.config import BudgetConfig, SalesBenchConfig
        from salesbench.environment import SalesBenchEnvironment

        config = SalesBenchConfig(
            seed=42,
            num_leads=10,
            budget=BudgetConfig(total_days=10),
        )

        env = SalesBenchEnvironment(config=config)
        obs = env.reset()

        if "tools" not in obs:
            return False, "Observation missing tools"

        if "leads_summary" not in obs:
            return False, "Observation missing leads_summary"

        return True, f"Environment created with {obs['leads_summary']['total']} leads"

    def validate_tool_execution(self) -> tuple[bool, str]:
        """Validate basic tool execution."""
        from salesbench.core.config import BudgetConfig, SalesBenchConfig
        from salesbench.core.types import ToolCall
        from salesbench.environment import SalesBenchEnvironment

        config = SalesBenchConfig(seed=42, num_leads=10, budget=BudgetConfig(total_days=10))
        env = SalesBenchEnvironment(config=config)
        env.reset()

        # Mock buyer to avoid API calls
        def mock_buyer(lead, offer, session, pitch=None):
            from salesbench.core.types import BuyerDecision, BuyerResponseData

            return BuyerResponseData(decision=BuyerDecision.REJECT_PLAN, dialogue="No thanks")

        env.set_buyer_simulator(mock_buyer)

        # Test CRM search
        obs, _, _, info = env.step([ToolCall(tool_name="crm.search_leads", arguments={})])

        tool_results = info.get("tool_results", [])
        if not tool_results:
            return False, "No tool results returned"

        if not tool_results[0].get("success"):
            return False, f"Tool failed: {tool_results[0].get('error')}"

        return True, "CRM search executed successfully"

    def validate_call_flow(self) -> tuple[bool, str]:
        """Validate complete call flow."""
        from salesbench.core.config import BudgetConfig, SalesBenchConfig
        from salesbench.core.types import BuyerDecision, ToolCall
        from salesbench.environment import SalesBenchEnvironment

        config = SalesBenchConfig(seed=42, num_leads=10, budget=BudgetConfig(total_days=10))
        env = SalesBenchEnvironment(config=config)
        env.reset()

        # Track decisions
        decisions = []

        def tracking_buyer(lead, offer, session, pitch=None):
            from salesbench.core.types import BuyerResponseData

            # Accept first offer, reject second
            decision = (
                BuyerDecision.ACCEPT_PLAN if len(decisions) == 0 else BuyerDecision.REJECT_PLAN
            )
            decisions.append(decision)
            return BuyerResponseData(decision=decision, dialogue="OK")

        env.set_buyer_simulator(tracking_buyer)

        # Search leads
        _, _, _, info = env.step([ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})])
        lead_id = info["tool_results"][0]["data"]["leads"][0]["lead_id"]

        # Start call
        env.step([ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})])

        # Propose plan (with monthly_premium calculated)
        obs, reward, _, _ = env.step(
            [
                ToolCall(
                    tool_name="calling.propose_plan",
                    arguments={
                        "plan_id": "TERM",
                        "coverage_amount": 500000,
                        "monthly_premium": 60.0,  # Approx for 35yo, 500K TERM
                        "next_step": "close_now",
                    },
                )
            ]
        )

        if reward <= 0:
            return False, f"Expected positive reward for accept, got {reward}"

        # End call
        env.step([ToolCall(tool_name="calling.end_call", arguments={})])

        return True, f"Call flow completed with reward={reward:.2f}"

    def validate_scoring(self) -> tuple[bool, str]:
        """Validate scoring calculations."""
        from salesbench.core.types import NextStep
        from salesbench.envs.sales_mvp.verifiers.scoring import ScoringRubric

        rubric = ScoringRubric()

        # Test accept
        score = rubric.record_accept(monthly_premium=100, next_step=NextStep.CLOSE_NOW)
        if score <= 0:
            return False, "Accept should give positive score"

        # Test reject penalty
        penalty = rubric.record_reject()
        if penalty >= 0:
            return False, "Reject should give negative score"

        # Test DNC
        dnc_penalty = rubric.record_dnc_violation()
        if dnc_penalty >= -50:
            return False, "DNC should give significant penalty"

        return (
            True,
            f"Scoring validated: accept={score:.0f}, reject={penalty:.0f}, dnc={dnc_penalty:.0f}",
        )

    def validate_render(self) -> tuple[bool, str]:
        """Validate environment render methods."""
        from salesbench.core.config import BudgetConfig, SalesBenchConfig
        from salesbench.environment import SalesBenchEnvironment

        config = SalesBenchConfig(seed=42, num_leads=10, budget=BudgetConfig(total_days=10))
        env = SalesBenchEnvironment(config=config)
        env.reset()

        # Test JSON render
        json_out = env.render(mode="json")
        if not json_out:
            return False, "JSON render returned nothing"

        try:
            parsed = json.loads(json_out)
        except json.JSONDecodeError:
            return False, "JSON render output is not valid JSON"

        # Test text render
        text_out = env.render(mode="text")
        if not text_out or "SalesBench" not in text_out:
            return False, "Text render failed"

        return True, "Render modes (json, text) work correctly"

    def validate_verifiers_server_imports(self) -> tuple[bool, str]:
        """Validate verifiers server can be imported (not run)."""
        try:
            from salesbench.envs.sales_mvp.verifiers.server import create_app, verify_episode

            return True, "Verifiers server modules importable"
        except ImportError as e:
            # FastAPI is optional
            if "fastapi" in str(e).lower():
                return True, "Verifiers server importable (FastAPI optional)"
            return False, f"Import error: {e}"

    def run_all(self, quick: bool = False) -> bool:
        """Run all validations."""
        print("=" * 60)
        print("SalesBench End-to-End Validation")
        print("=" * 60)
        print()

        # Core validations
        self.run_validation("Import modules", self.validate_imports)
        self.run_validation("Persona generation", self.validate_persona_generation)
        self.run_validation("Product pricing", self.validate_product_pricing)
        self.run_validation("Environment creation", self.validate_environment_creation)
        self.run_validation("Tool execution", self.validate_tool_execution)
        self.run_validation("Call flow", self.validate_call_flow)
        self.run_validation("Scoring", self.validate_scoring)
        self.run_validation("Render methods", self.validate_render)
        self.run_validation("Verifiers server", self.validate_verifiers_server_imports)

        # Summary
        print()
        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"Results: {passed}/{total} validations passed")

        total_time = sum(r.duration for r in self.results)
        print(f"Total time: {total_time:.2f}s")

        if passed < total:
            print()
            print("Failed validations:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        print("=" * 60)

        return passed == total


def main():
    parser = argparse.ArgumentParser(description="SalesBench E2E Validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Skip long-running tests")
    args = parser.parse_args()

    validator = SalesBenchValidator(verbose=args.verbose)
    success = validator.run_all(quick=args.quick)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
