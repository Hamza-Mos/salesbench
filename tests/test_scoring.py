"""Tests for the revenue-based scoring system.

Score = Total Revenue (sum of monthly premiums from accepted plans).
"""

import pytest

from salesbench.envs.sales_mvp.verifiers.scoring import (
    RevenueMetrics,
    calculate_episode_revenue,
)


class TestRevenueMetrics:
    """Tests for RevenueMetrics dataclass."""

    def test_initial_values(self):
        """Test that metrics start at zero."""
        metrics = RevenueMetrics()
        assert metrics.total_revenue == 0.0
        assert metrics.num_accepts == 0
        assert metrics.num_rejects == 0
        assert metrics.num_end_calls == 0
        assert metrics.num_dnc_violations == 0

    def test_revenue_per_accept_with_no_accepts(self):
        """Test revenue_per_accept returns 0 when no accepts."""
        metrics = RevenueMetrics()
        assert metrics.revenue_per_accept == 0.0

    def test_revenue_per_accept_calculation(self):
        """Test revenue_per_accept is calculated correctly."""
        metrics = RevenueMetrics(total_revenue=300.0, num_accepts=2)
        assert metrics.revenue_per_accept == 150.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = RevenueMetrics(
            total_revenue=500.0,
            num_accepts=3,
            num_rejects=5,
            num_end_calls=2,
            num_dnc_violations=1,
        )
        result = metrics.to_dict()

        assert result["total_revenue"] == 500.0
        assert result["revenue_per_accept"] == pytest.approx(166.67, rel=0.01)
        assert result["counts"]["num_accepts"] == 3
        assert result["counts"]["num_rejects"] == 5
        assert result["counts"]["num_end_calls"] == 2
        assert result["counts"]["num_dnc_violations"] == 1


class TestCalculateEpisodeRevenue:
    """Tests for calculate_episode_revenue function."""

    def test_empty_state(self, sales_env):
        """Test with no calls made."""
        # Fresh environment with no calls
        metrics = calculate_episode_revenue(sales_env.state)

        assert metrics.total_revenue == 0.0
        assert metrics.num_accepts == 0
        assert metrics.num_rejects == 0
        assert metrics.num_end_calls == 0
        assert metrics.num_dnc_violations == 0

    def test_with_accepted_offer(self, sales_env, accepting_buyer_simulator):
        """Test revenue tracking with accepted offers."""
        from salesbench.core.types import ToolCall

        # Set up accepting buyer
        sales_env.set_buyer_simulator(accepting_buyer_simulator)

        # Get a lead
        leads = list(sales_env.state.leads.values())
        lead = leads[0]

        # Start call
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": str(lead.lead_id)},
            )
        )

        # Propose plan
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "monthly_premium": 150.0,
                    "coverage_amount": 500000,
                    "next_step": "close_now",
                },
            )
        )

        # End call to finalize (required for call_history)
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.end_call",
                arguments={},
            )
        )

        # Calculate revenue
        metrics = calculate_episode_revenue(sales_env.state)

        assert metrics.total_revenue == 150.0
        assert metrics.num_accepts == 1

    def test_with_rejected_offer(self, sales_env, mock_buyer_simulator):
        """Test that rejections don't add revenue."""
        from salesbench.core.types import ToolCall

        # Set up rejecting buyer
        sales_env.set_buyer_simulator(mock_buyer_simulator)

        # Get a lead
        leads = list(sales_env.state.leads.values())
        lead = leads[0]

        # Start call
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": str(lead.lead_id)},
            )
        )

        # Propose plan (will be rejected)
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "monthly_premium": 150.0,
                    "coverage_amount": 500000,
                    "next_step": "close_now",
                },
            )
        )

        # End call to finalize (required for call_history)
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.end_call",
                arguments={},
            )
        )

        # Calculate revenue
        metrics = calculate_episode_revenue(sales_env.state)

        assert metrics.total_revenue == 0.0  # No revenue from rejection
        assert metrics.num_accepts == 0
        assert metrics.num_rejects == 1

    def test_dnc_violations_tracked(self, sales_env):
        """Test that DNC violations are tracked but don't affect revenue."""
        from salesbench.core.types import LeadStatus, ToolCall

        # Get a lead and mark as DNC
        leads = list(sales_env.state.leads.values())
        lead = leads[0]
        lead.status = LeadStatus.DNC
        lead.on_dnc_list = True

        # Try to call (should record DNC violation)
        sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": str(lead.lead_id)},
            )
        )

        # Calculate revenue
        metrics = calculate_episode_revenue(sales_env.state)

        assert metrics.num_dnc_violations == 1
        assert metrics.total_revenue == 0.0  # DNC doesn't affect revenue

    def test_multiple_accepted_offers(self, sales_env, accepting_buyer_simulator):
        """Test cumulative revenue from multiple accepts."""
        from salesbench.core.types import ToolCall

        sales_env.set_buyer_simulator(accepting_buyer_simulator)
        leads = list(sales_env.state.leads.values())

        total_expected_revenue = 0.0

        # Make calls to first 3 leads
        for i, lead in enumerate(leads[:3]):
            premium = 100.0 + (i * 50.0)  # 100, 150, 200
            total_expected_revenue += premium

            # Start call
            sales_env.execute_tool(
                ToolCall(
                    tool_name="calling.start_call",
                    arguments={"lead_id": str(lead.lead_id)},
                )
            )

            # Propose plan (will be accepted)
            sales_env.execute_tool(
                ToolCall(
                    tool_name="calling.propose_plan",
                    arguments={
                        "plan_id": "TERM",
                        "monthly_premium": premium,
                        "coverage_amount": 500000,
                        "next_step": "close_now",
                    },
                )
            )

            # End call
            sales_env.execute_tool(
                ToolCall(
                    tool_name="calling.end_call",
                    arguments={},
                )
            )

        # Calculate revenue
        metrics = calculate_episode_revenue(sales_env.state)

        assert metrics.total_revenue == total_expected_revenue  # 100 + 150 + 200 = 450
        assert metrics.num_accepts == 3
        assert metrics.revenue_per_accept == 150.0  # 450 / 3


class TestScoringEdgeCases:
    """Tests for scoring edge cases."""

    def test_zero_premium_still_counts_as_accept(self):
        """Test that zero premium still counts as accept."""
        metrics = RevenueMetrics(total_revenue=0.0, num_accepts=1)
        assert metrics.num_accepts == 1
        assert metrics.revenue_per_accept == 0.0

    def test_dnc_does_not_affect_revenue_score(self):
        """Test that DNC violations don't reduce revenue (tracked separately)."""
        metrics = RevenueMetrics(
            total_revenue=500.0,
            num_accepts=3,
            num_dnc_violations=5,
        )
        # Revenue stays the same - DNC is just a metric
        assert metrics.total_revenue == 500.0

    def test_mixed_accepts_and_rejects(self):
        """Test metrics with mixed outcomes."""
        metrics = RevenueMetrics(
            total_revenue=300.0,
            num_accepts=2,
            num_rejects=5,
            num_end_calls=3,
        )
        # Revenue is only from accepts
        assert metrics.total_revenue == 300.0
        assert metrics.revenue_per_accept == 150.0
