"""Main SalesEnv environment - the canonical state owner.

This is the central environment class that:
- Owns all state
- Routes tool calls to the appropriate handlers
- Manages the episode lifecycle
"""

from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.config import SalesBenchConfig
from salesbench.core.errors import (
    BudgetExceeded,
    EpisodeTerminated,
    InvalidToolCall,
)
from salesbench.core.protocol import SELLER_TOOLS
from salesbench.core.types import (
    ToolCall,
    ToolResult,
)
from salesbench.envs.sales_mvp.personas import PersonaGenerator
from salesbench.envs.sales_mvp.products import ProductCatalog
from salesbench.envs.sales_mvp.state import EnvironmentState
from salesbench.envs.sales_mvp.tools.calendar import CalendarTools
from salesbench.envs.sales_mvp.tools.calling import BuyerSimulatorFn, CallingTools
from salesbench.envs.sales_mvp.tools.crm import CRMTools
from salesbench.envs.sales_mvp.tools.products import ProductTools

if TYPE_CHECKING:
    from salesbench.context.episode import EpisodeContext


class SalesEnv:
    """Main sales environment.

    The SalesEnv is the canonical owner of all state. It:
    - Generates leads from a seed
    - Routes tool calls to appropriate handlers
    - Tracks time and budgets
    - Provides serialization for checkpointing
    """

    def __init__(self, config: Optional[SalesBenchConfig] = None):
        """Initialize the environment.

        Args:
            config: Configuration for the environment.
        """
        self.config = config or SalesBenchConfig()
        self.config.validate()

        # Initialize state
        self._state = EnvironmentState()

        # Initialize subsystems
        self._catalog = ProductCatalog()
        self._persona_generator = PersonaGenerator(
            seed=self.config.seed,
            config=self.config.persona_generation,
        )

        # Initialize tools
        self._crm_tools = CRMTools(self._state)
        self._calendar_tools = CalendarTools(self._state, self.config.budget)
        self._calling_tools = CallingTools(self._state, self.config.budget)
        self._product_tools = ProductTools(self._state, self._catalog)

        # Episode state
        self._initialized = False
        self._terminated = False
        self._termination_reason: Optional[str] = None

        # Store buyer simulator to preserve across resets
        self._buyer_simulator: Optional[BuyerSimulatorFn] = None

        # Store episode context for conversation history
        self._episode_context: Optional["EpisodeContext"] = None

    @property
    def state(self) -> EnvironmentState:
        """Get the current state."""
        return self._state

    @property
    def is_terminated(self) -> bool:
        """Check if the episode is terminated."""
        return self._terminated

    @property
    def termination_reason(self) -> Optional[str]:
        """Get the termination reason if terminated."""
        return self._termination_reason

    def reset(self) -> dict[str, Any]:
        """Reset the environment and generate leads.

        Returns:
            Initial observation dict.
        """
        # Reset state
        self._state = EnvironmentState()
        self._terminated = False
        self._termination_reason = None

        # Reset tools with new state
        self._crm_tools = CRMTools(self._state)
        self._calendar_tools = CalendarTools(self._state, self.config.budget)
        self._calling_tools = CallingTools(self._state, self.config.budget)
        self._product_tools = ProductTools(self._state, self._catalog)

        # Re-apply buyer simulator if it was set
        if self._buyer_simulator:
            self._calling_tools.set_buyer_simulator(self._buyer_simulator)

        # Re-apply episode context if it was set
        if self._episode_context:
            self._calling_tools.set_episode_context(self._episode_context)

        # Generate leads
        self._persona_generator.reset()
        leads = self._persona_generator.generate_batch(self.config.num_leads)
        for lead in leads:
            self._state.leads[lead.lead_id] = lead

        self._initialized = True

        return self._get_observation()

    def set_buyer_simulator(self, simulator: BuyerSimulatorFn) -> None:
        """Set a custom buyer simulator.

        Args:
            simulator: Callable that takes (Persona, PlanOffer, CallSession)
                      and returns BuyerResponseData.
        """
        # Store at env level to preserve across resets
        self._buyer_simulator = simulator
        self._calling_tools.set_buyer_simulator(simulator)

    def set_episode_context(self, context: "EpisodeContext") -> None:
        """Set the episode context for conversation history.

        The episode context is used to provide buyers with their
        negotiation history with the seller.

        Args:
            context: The episode context to use.
        """
        # Store at env level to preserve across resets
        self._episode_context = context
        self._calling_tools.set_episode_context(context)

    def get_buyer_conversational_response(self, seller_message: str) -> Optional[str]:
        """Get a conversational response from the buyer.

        This is called when the seller speaks while in a call.
        Returns a natural conversational response (not a decision).

        Args:
            seller_message: What the salesperson just said.

        Returns:
            The buyer's dialogue response, or None if not in a call.
        """
        return self._calling_tools.get_buyer_conversational_response(seller_message)

    @property
    def is_in_call(self) -> bool:
        """Check if there's an active call."""
        return self._state.active_call is not None

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            Result from the tool execution.

        Raises:
            EpisodeTerminated: If episode is already terminated.
            InvalidToolCall: If tool is invalid.
            BudgetExceeded: If tool call budget exceeded.
        """
        if not self._initialized:
            raise InvalidState("Environment not initialized. Call reset() first.")

        if self._terminated:
            raise EpisodeTerminated(self._termination_reason or "Episode ended")

        # Check tool call budget
        if self._state.tool_calls_this_turn >= self.config.budget.max_tool_calls_per_turn:
            raise BudgetExceeded(
                f"Tool call limit exceeded: {self.config.budget.max_tool_calls_per_turn}",
                budget_type="tool_calls_per_turn",
                limit=self.config.budget.max_tool_calls_per_turn,
                current=self._state.tool_calls_this_turn,
            )

        # Validate tool name
        if tool_call.tool_name not in SELLER_TOOLS:
            raise InvalidToolCall(
                f"Unknown tool: {tool_call.tool_name}",
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )

        # Route to appropriate handler
        result = self._route_tool_call(tool_call)
        result.call_id = tool_call.call_id

        # Update counters
        self._state.tool_calls_this_turn += 1
        self._state.total_tool_calls += 1

        # Check for episode termination
        self._check_termination()

        return result

    def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            List of results.
        """
        results = []
        for tc in tool_calls:
            try:
                result = self.execute_tool(tc)
                results.append(result)
            except (EpisodeTerminated, BudgetExceeded) as e:
                # Return error result and stop
                results.append(
                    ToolResult(
                        call_id=tc.call_id,
                        success=False,
                        error=str(e),
                    )
                )
                break
        return results

    def end_turn(self) -> dict[str, Any]:
        """End the current turn and return observation.

        Returns:
            Current observation dict.
        """
        self._state.reset_turn()

        # Check if day ended
        if self._state.time.is_end_of_day():
            self._state.reset_day()

        return self._get_observation()

    def _route_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Route a tool call to the appropriate handler."""
        tool_name = tool_call.tool_name
        args = tool_call.arguments

        if tool_name.startswith("crm."):
            return self._crm_tools.execute(tool_name, args)
        elif tool_name.startswith("calendar."):
            return self._calendar_tools.execute(tool_name, args)
        elif tool_name.startswith("calling."):
            return self._calling_tools.execute(tool_name, args)
        elif tool_name.startswith("products."):
            return self._product_tools.execute(tool_name, args)
        else:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                error=f"Unknown tool category: {tool_name}",
            )

    def _check_termination(self) -> None:
        """Check if the episode should terminate."""
        # Time-based termination
        if self._state.time.is_episode_ended(self.config.budget):
            self._terminated = True
            self._termination_reason = "Episode time ended (10 days)"
            return

        # All leads exhausted (all on DNC or already sold)
        active_leads = sum(1 for lead in self._state.leads.values() if not lead.on_dnc_list)
        if active_leads == 0:
            self._terminated = True
            self._termination_reason = "No more leads available"
            return

    def _get_observation(self) -> dict[str, Any]:
        """Get the current observation for the agent."""
        return {
            "time": self._state.time.to_dict(),
            "stats": self._state.stats.to_dict(),
            "has_active_call": self._state.active_call is not None,
            "active_call": (
                {
                    "call_id": self._state.active_call.call_id,
                    "lead_id": self._state.active_call.lead_id,
                    "duration_minutes": self._state.active_call.duration_minutes,
                    "offers_presented": len(self._state.active_call.offers_presented),
                }
                if self._state.active_call
                else None
            ),
            "leads_count": len(self._state.leads),
            "appointments_count": len(self._state.appointments),
            "terminated": self._terminated,
            "termination_reason": self._termination_reason,
        }

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all available tools.

        Returns:
            List of tool definitions with schemas.
        """
        from salesbench.core.protocol import get_all_tool_schemas

        schemas = get_all_tool_schemas()
        tools = []

        for tool_name, schema in schemas.items():
            tools.append(
                {
                    "name": tool_name,
                    "description": self._get_tool_description(tool_name),
                    "parameters": schema,
                }
            )

        return tools

    def _get_tool_description(self, tool_name: str) -> str:
        """Get human-readable description for a tool."""
        descriptions = {
            "crm.search_leads": "Search for leads with optional filters (temperature, income, age).",
            "crm.get_lead": "Get detailed information about a specific lead.",
            "crm.update_lead": "Update a lead's notes or temperature.",
            "crm.log_call": "Log a completed call with its outcome.",
            "calendar.get_availability": "Get available time slots for a given day.",
            "calendar.schedule_call": "Schedule a call with a lead.",
            "calling.start_call": "Start a call with a lead.",
            "calling.propose_plan": "Propose an insurance plan to the buyer (triggers buyer response).",
            "calling.end_call": "End the current call.",
            "products.list_plans": "List all available insurance plans.",
            "products.get_plan": "Get details about a specific insurance plan.",
            "products.quote_premium": "Get a premium quote for a plan based on age, coverage, and risk.",
        }
        return descriptions.get(tool_name, "No description available.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize environment state for checkpointing.

        Returns:
            Complete serialized state.
        """
        return {
            "config": self.config.to_dict(),
            "state": self._state.to_full_dict(),
            "terminated": self._terminated,
            "termination_reason": self._termination_reason,
            "initialized": self._initialized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SalesEnv":
        """Restore environment from serialized state.

        Args:
            data: Serialized state from to_dict().

        Returns:
            Restored SalesEnv instance.
        """
        config = SalesBenchConfig.from_dict(data["config"])
        env = cls(config)

        # Restore would require more complex deserialization
        # For now, just restore basic properties
        env._terminated = data.get("terminated", False)
        env._termination_reason = data.get("termination_reason")
        env._initialized = data.get("initialized", False)

        return env


# For backwards compatibility
from salesbench.core.errors import InvalidState
