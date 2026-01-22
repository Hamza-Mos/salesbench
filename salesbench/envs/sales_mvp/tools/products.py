"""Product tools for the sales environment.

Tools:
- products.list_plans: List all available insurance plans
- products.get_plan: Get details about a specific plan
- products.quote_premium: Get a premium quote
"""

from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.types import PlanType, RiskClass, ToolResult
from salesbench.envs.sales_mvp.products import ProductCatalog

if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.state import EnvironmentState


class ProductTools:
    """Product tool implementations."""

    def __init__(self, state: "EnvironmentState", catalog: Optional[ProductCatalog] = None):
        self.state = state
        self.catalog = catalog or ProductCatalog()

    def list_plans(self) -> ToolResult:
        """List all available insurance plans.

        Returns:
            ToolResult with list of plans.
        """
        products = self.catalog.list_products()

        return ToolResult(
            call_id="",
            success=True,
            data={
                "plans": products,
                "total": len(products),
            },
        )

    def get_plan(self, plan_id: str) -> ToolResult:
        """Get detailed information about a plan.

        Args:
            plan_id: The plan ID (TERM, WHOLE, UL, VUL, LTC, DI).

        Returns:
            ToolResult with plan details.
        """
        try:
            plan_type = PlanType(plan_id)
        except ValueError:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid plan_id: {plan_id}. Valid: {[p.value for p in PlanType]}",
            )

        product = self.catalog.get_product(plan_type)
        if not product:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Plan not found: {plan_id}",
            )

        return ToolResult(
            call_id="",
            success=True,
            data={"plan": product.to_dict()},
        )

    def quote_premium(
        self,
        plan_id: str,
        age: int,
        coverage_amount: float,
        risk_class: Optional[str] = None,
        term_years: Optional[int] = None,
        waiting_period_days: Optional[int] = None,
    ) -> ToolResult:
        """Get a premium quote for a plan.

        Args:
            plan_id: The plan ID.
            age: Age of the insured.
            coverage_amount: Desired coverage amount.
            risk_class: Risk classification (optional).
            term_years: Term length for TERM plans (optional).

        Returns:
            ToolResult with premium quote.
        """
        try:
            plan_type = PlanType(plan_id)
        except ValueError:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid plan_id: {plan_id}. Valid: {[p.value for p in PlanType]}",
            )

        # Parse risk class
        risk = RiskClass.STANDARD_PLUS
        if risk_class:
            try:
                risk = RiskClass(risk_class)
            except ValueError:
                return ToolResult(
                    call_id="",
                    success=False,
                    error=f"Invalid risk_class: {risk_class}. Valid: {[r.value for r in RiskClass]}",
                )

        # Get quote
        quote = self.catalog.quote_premium(
            plan_id=plan_type,
            age=age,
            coverage_amount=coverage_amount,
            risk_class=risk,
            term_years=term_years,
            waiting_period_days=waiting_period_days,
        )

        if "error" in quote:
            return ToolResult(
                call_id="",
                success=False,
                error=quote["error"],
            )

        return ToolResult(
            call_id="",
            success=True,
            data={
                "quote": quote,
                "message": (
                    f"Quote ready: {quote['plan_name']} - "
                    f"${quote['monthly_premium']}/month for ${coverage_amount} coverage. "
                    "Use calling.propose_plan to present this offer to the buyer."
                ),
            },
        )

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a product tool.

        Args:
            tool_name: Full tool name (e.g., "products.list_plans").
            arguments: Tool arguments.

        Returns:
            ToolResult from the tool execution.
        """
        method_name = tool_name.replace("products.", "")

        if method_name == "list_plans":
            return self.list_plans()

        elif method_name == "get_plan":
            if "plan_id" not in arguments:
                return ToolResult(
                    call_id="",
                    success=False,
                    error="Missing required argument: plan_id",
                )
            return self.get_plan(arguments["plan_id"])

        elif method_name == "quote_premium":
            required = ["plan_id", "age", "coverage_amount"]
            missing = [r for r in required if r not in arguments]
            if missing:
                return ToolResult(
                    call_id="",
                    success=False,
                    error=f"Missing required arguments: {missing}",
                )
            return self.quote_premium(
                plan_id=arguments["plan_id"],
                age=arguments["age"],
                coverage_amount=arguments["coverage_amount"],
                risk_class=arguments.get("risk_class"),
                term_years=arguments.get("term_years"),
                waiting_period_days=arguments.get("waiting_period_days"),
            )

        else:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Unknown product tool: {tool_name}",
            )
