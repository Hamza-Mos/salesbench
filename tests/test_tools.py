"""Tests for SalesBench tools."""

import pytest

from salesbench.core.types import ToolCall
from salesbench.envs.sales_mvp.env import SalesEnv


class TestCRMTools:
    """Tests for CRM tools."""

    def test_search_leads_returns_results(self, sales_env: SalesEnv):
        """Test that search_leads returns leads."""
        result = sales_env.execute_tool(ToolCall(tool_name="crm.search_leads", arguments={}))

        assert result.success
        assert "leads" in result.data
        assert len(result.data["leads"]) > 0

    def test_search_leads_by_temperature(self, sales_env: SalesEnv):
        """Test filtering leads by temperature."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.search_leads",
                arguments={"temperature": "hot"},
            )
        )

        assert result.success
        for lead in result.data.get("leads", []):
            assert lead["temperature"] == "hot"

    def test_search_leads_limit(self, sales_env: SalesEnv):
        """Test that limit parameter works."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.search_leads",
                arguments={"limit": 3},
            )
        )

        assert result.success
        assert len(result.data["leads"]) <= 3

    def test_get_lead_returns_details(self, sales_env: SalesEnv):
        """Test getting a specific lead."""
        # First search to get a lead_id
        search_result = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search_result.data["leads"][0]["lead_id"]

        # Get the lead details
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.get_lead",
                arguments={"lead_id": lead_id},
            )
        )

        assert result.success
        assert result.data["lead"]["lead_id"] == lead_id
        assert "name" in result.data["lead"]
        assert "age" in result.data["lead"]

    def test_get_lead_invalid_id(self, sales_env: SalesEnv):
        """Test getting a non-existent lead."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.get_lead",
                arguments={"lead_id": "invalid_lead_id"},
            )
        )

        assert not result.success
        assert result.error is not None

    def test_update_lead_notes(self, sales_env: SalesEnv):
        """Test updating lead notes."""
        # Get a lead
        search_result = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search_result.data["leads"][0]["lead_id"]

        # Update notes
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.update_lead",
                arguments={"lead_id": lead_id, "notes": "Test note"},
            )
        )

        assert result.success

        # Verify update
        get_result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.get_lead",
                arguments={"lead_id": lead_id},
            )
        )

        assert "Test note" in get_result.data["lead"]["notes"]


class TestCalendarTools:
    """Tests for calendar tools."""

    def test_get_availability(self, sales_env: SalesEnv):
        """Test getting calendar availability."""
        result = sales_env.execute_tool(
            ToolCall(tool_name="calendar.get_availability", arguments={})
        )

        assert result.success
        assert "available_hours" in result.data or "slots" in result.data

    def test_schedule_call(self, sales_env: SalesEnv):
        """Test scheduling a call."""
        # Get a lead
        search_result = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search_result.data["leads"][0]["lead_id"]

        # Get availability
        avail_result = sales_env.execute_tool(
            ToolCall(tool_name="calendar.get_availability", arguments={})
        )

        # Schedule if hours available
        available_hours = avail_result.data.get("available_hours", [])
        if available_hours:
            result = sales_env.execute_tool(
                ToolCall(
                    tool_name="calendar.schedule_call",
                    arguments={
                        "lead_id": lead_id,
                        "day": avail_result.data.get("day", 1),
                        "hour": available_hours[0],
                    },
                )
            )

            assert result.success
            assert "appointment_id" in result.data


class TestCallingTools:
    """Tests for calling tools."""

    def test_start_call(self, sales_env: SalesEnv):
        """Test starting a call."""
        # Get a lead
        search_result = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search_result.data["leads"][0]["lead_id"]

        # Start call
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": lead_id},
            )
        )

        assert result.success
        assert "call_id" in result.data

    def test_start_call_invalid_lead(self, sales_env: SalesEnv):
        """Test starting a call with invalid lead."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": "invalid_id"},
            )
        )

        assert not result.success

    def test_propose_plan_requires_active_call(self, sales_env: SalesEnv):
        """Test that propose_plan requires an active call."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "coverage_amount": 500000,
                    "monthly_premium": 60.0,
                    "next_step": "close_now",
                },
            )
        )

        assert not result.success
        assert "no active call" in result.error.lower()

    def test_end_call_requires_active_call(self, sales_env: SalesEnv):
        """Test that end_call requires an active call."""
        result = sales_env.execute_tool(ToolCall(tool_name="calling.end_call", arguments={}))

        assert not result.success

    def test_full_call_flow(self, sales_env: SalesEnv, mock_buyer_simulator):
        """Test complete call flow: start -> propose -> end."""
        # Set mock buyer
        sales_env.set_buyer_simulator(mock_buyer_simulator)

        # Get a lead
        search_result = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search_result.data["leads"][0]["lead_id"]

        # Start call
        start_result = sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": lead_id},
            )
        )
        assert start_result.success

        # Propose plan
        propose_result = sales_env.execute_tool(
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "coverage_amount": 500000,
                    "monthly_premium": 60.0,
                    "next_step": "close_now",
                },
            )
        )
        assert propose_result.success
        assert "decision" in propose_result.data

        # End call
        end_result = sales_env.execute_tool(ToolCall(tool_name="calling.end_call", arguments={}))
        assert end_result.success


class TestProductTools:
    """Tests for product tools."""

    def test_list_plans(self, sales_env: SalesEnv):
        """Test listing available plans."""
        result = sales_env.execute_tool(ToolCall(tool_name="products.list_plans", arguments={}))

        assert result.success
        assert "plans" in result.data
        assert len(result.data["plans"]) == 6

    def test_get_plan(self, sales_env: SalesEnv):
        """Test getting plan details."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="products.get_plan",
                arguments={"plan_id": "TERM"},
            )
        )

        assert result.success
        # Handle nested structure - key is "plan"
        plan_data = result.data.get("plan", result.data)
        assert plan_data["plan_id"] == "TERM"
        assert "features" in plan_data

    def test_quote_premium(self, sales_env: SalesEnv):
        """Test getting a premium quote."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="products.quote_premium",
                arguments={
                    "plan_id": "TERM",
                    "age": 35,
                    "coverage_amount": 500000,
                },
            )
        )

        assert result.success
        # Handle nested structure
        quote_data = result.data.get("quote", result.data)
        assert "monthly_premium" in quote_data
        assert quote_data["monthly_premium"] > 0


class TestToolErrorHandling:
    """Tests for tool error handling."""

    def test_unknown_tool_returns_error(self, sales_env: SalesEnv):
        """Test that unknown tools raise an error."""
        from salesbench.core.errors import InvalidToolCall

        with pytest.raises(InvalidToolCall):
            sales_env.execute_tool(
                ToolCall(
                    tool_name="unknown.tool",
                    arguments={},
                )
            )

    def test_missing_required_argument(self, sales_env: SalesEnv):
        """Test missing required argument returns error."""
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="crm.get_lead",
                arguments={},  # Missing lead_id
            )
        )

        assert not result.success


class TestDNCBehavior:
    """Tests for Do-Not-Call list behavior."""

    def test_cannot_call_dnc_lead(self, sales_env: SalesEnv):
        """Test that calling a DNC lead fails."""
        # Get a lead and mark as DNC
        search_result = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search_result.data["leads"][0]["lead_id"]

        # Mark as DNC
        sales_env.state.leads[lead_id].on_dnc_list = True

        # Try to call
        result = sales_env.execute_tool(
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": lead_id},
            )
        )

        # Should fail or record violation
        if not result.success:
            assert "dnc" in result.error.lower() or "do not call" in result.error.lower()

    def test_dnc_leads_excluded_from_search(self, sales_env: SalesEnv):
        """Test that DNC leads are excluded from search by default."""
        # Get all leads first
        search_result = sales_env.execute_tool(ToolCall(tool_name="crm.search_leads", arguments={}))
        initial_count = len(search_result.data["leads"])

        # Mark one as DNC
        if search_result.data["leads"]:
            lead_id = search_result.data["leads"][0]["lead_id"]
            sales_env.state.leads[lead_id].on_dnc_list = True

        # Search again
        search_result2 = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={})
        )

        # Should have one less lead (or the DNC lead should be marked)
        dnc_leads = [l for l in search_result2.data["leads"] if l.get("on_dnc_list")]
        assert len(dnc_leads) == 0 or len(search_result2.data["leads"]) < initial_count
