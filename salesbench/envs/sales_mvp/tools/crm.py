"""CRM tools for the sales environment.

Tools:
- crm.search_leads: Search for leads with filters
- crm.get_lead: Get details about a specific lead
- crm.update_lead: Update lead notes and status
- crm.log_call: Log a completed call
"""

from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.types import LeadID, ToolResult

if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.state import EnvironmentState


class CRMTools:
    """CRM tool implementations."""

    def __init__(self, state: "EnvironmentState"):
        self.state = state

    def search_leads(
        self,
        temperature: Optional[str] = None,
        min_income: Optional[int] = None,
        max_age: Optional[int] = None,
        limit: int = 10,
    ) -> ToolResult:
        """Search for leads matching criteria.

        Args:
            temperature: Filter by lead temperature (hot/warm/lukewarm/cold/hostile).
            min_income: Minimum annual income.
            max_age: Maximum age.
            limit: Maximum results to return.

        Returns:
            ToolResult with matching leads.
        """
        leads = self.state.search_leads(
            temperature=temperature,
            min_income=min_income,
            max_age=max_age,
            limit=limit,
        )

        return ToolResult(
            call_id="",  # Will be set by executor
            success=True,
            data={
                "leads": [lead.to_public_dict() for lead in leads],
                "total_found": len(leads),
                "filters_applied": {
                    k: v
                    for k, v in {
                        "temperature": temperature,
                        "min_income": min_income,
                        "max_age": max_age,
                    }.items()
                    if v is not None
                },
            },
        )

    def get_lead(self, lead_id: str) -> ToolResult:
        """Get detailed information about a lead.

        Args:
            lead_id: The lead ID to look up.

        Returns:
            ToolResult with lead details.
        """
        lead = self.state.get_lead(LeadID(lead_id))
        if not lead:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Lead not found: {lead_id}",
            )

        return ToolResult(
            call_id="",
            success=True,
            data={"lead": lead.to_public_dict()},
        )

    def update_lead(
        self,
        lead_id: str,
        notes: Optional[str] = None,
        temperature: Optional[str] = None,
    ) -> ToolResult:
        """Update lead information.

        Args:
            lead_id: The lead ID to update.
            notes: Notes to append.
            temperature: New temperature classification.

        Returns:
            ToolResult indicating success.
        """
        lead = self.state.get_lead(LeadID(lead_id))
        if not lead:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Lead not found: {lead_id}",
            )

        if notes:
            if lead.notes:
                lead.notes = f"{lead.notes}\n{notes}"
            else:
                lead.notes = notes

        if temperature:
            from salesbench.core.types import LeadTemperature

            try:
                lead.temperature = LeadTemperature(temperature)
            except ValueError:
                return ToolResult(
                    call_id="",
                    success=False,
                    error=f"Invalid temperature: {temperature}",
                )

        return ToolResult(
            call_id="",
            success=True,
            data={
                "lead_id": lead_id,
                "updated": True,
                "lead": lead.to_public_dict(),
            },
        )

    def log_call(
        self,
        lead_id: str,
        call_id: str,
        outcome: str,
        notes: Optional[str] = None,
    ) -> ToolResult:
        """Log a completed call.

        Args:
            lead_id: The lead that was called.
            call_id: The call ID.
            outcome: Call outcome (accepted/rejected/ended/no_answer).
            notes: Additional notes.

        Returns:
            ToolResult indicating success.
        """
        lead = self.state.get_lead(LeadID(lead_id))
        if not lead:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Lead not found: {lead_id}",
            )

        # Update lead's call count
        lead.call_count += 1
        lead.last_contact_day = self.state.time.current_day

        if notes:
            if lead.notes:
                lead.notes = f"{lead.notes}\nCall {call_id}: {notes}"
            else:
                lead.notes = f"Call {call_id}: {notes}"

        return ToolResult(
            call_id="",
            success=True,
            data={
                "logged": True,
                "lead_id": lead_id,
                "call_id": call_id,
                "outcome": outcome,
                "total_calls": lead.call_count,
            },
        )

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a CRM tool.

        Args:
            tool_name: Full tool name (e.g., "crm.search_leads").
            arguments: Tool arguments.

        Returns:
            ToolResult from the tool execution.
        """
        method_name = tool_name.replace("crm.", "")

        if method_name == "search_leads":
            return self.search_leads(
                temperature=arguments.get("temperature"),
                min_income=arguments.get("min_income"),
                max_age=arguments.get("max_age"),
                limit=arguments.get("limit", 10),
            )
        elif method_name == "get_lead":
            if "lead_id" not in arguments:
                return ToolResult(
                    call_id="",
                    success=False,
                    error="Missing required argument: lead_id",
                )
            return self.get_lead(arguments["lead_id"])
        elif method_name == "update_lead":
            if "lead_id" not in arguments:
                return ToolResult(
                    call_id="",
                    success=False,
                    error="Missing required argument: lead_id",
                )
            return self.update_lead(
                lead_id=arguments["lead_id"],
                notes=arguments.get("notes"),
                temperature=arguments.get("temperature"),
            )
        elif method_name == "log_call":
            required = ["lead_id", "call_id", "outcome"]
            missing = [r for r in required if r not in arguments]
            if missing:
                return ToolResult(
                    call_id="",
                    success=False,
                    error=f"Missing required arguments: {missing}",
                )
            return self.log_call(
                lead_id=arguments["lead_id"],
                call_id=arguments["call_id"],
                outcome=arguments["outcome"],
                notes=arguments.get("notes"),
            )
        else:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Unknown CRM tool: {tool_name}",
            )
