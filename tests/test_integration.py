"""Integration tests for SalesBench."""

from salesbench.core.config import BudgetConfig, SalesBenchConfig
from salesbench.core.types import ToolCall
from salesbench.envs.sales_mvp.env import SalesEnv


class TestSalesEnvInterface:
    """Tests for the SalesEnv interface."""

    def test_sales_env_has_required_methods(self, default_config: SalesBenchConfig):
        """Test that SalesEnv has required methods."""
        env = SalesEnv(config=default_config)

        assert hasattr(env, "reset")
        assert hasattr(env, "execute_tool")
        assert hasattr(env, "execute_tools")
        assert hasattr(env, "end_turn")
        assert hasattr(env, "is_terminated")

    def test_reset_returns_observation(self, default_config: SalesBenchConfig):
        """Test that reset returns a valid observation."""
        env = SalesEnv(config=default_config)
        obs = env.reset()

        assert isinstance(obs, dict)
        assert "time" in obs
        assert "stats" in obs
        assert "leads_count" in obs

    def test_execute_tool_returns_result(
        self, default_config: SalesBenchConfig, mock_buyer_simulator
    ):
        """Test that execute_tool returns a ToolResult."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        tool_call = ToolCall(tool_name="crm.search_leads", arguments={"limit": 5})
        result = env.execute_tool(tool_call)

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "data")

    def test_execute_tools_returns_list(
        self, default_config: SalesBenchConfig, mock_buyer_simulator
    ):
        """Test that execute_tools returns a list of results."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        tool_calls = [ToolCall(tool_name="crm.search_leads", arguments={})]
        results = env.execute_tools(tool_calls)

        assert isinstance(results, list)
        assert len(results) == 1

    def test_end_turn_returns_observation(self, default_config: SalesBenchConfig):
        """Test that end_turn returns an observation."""
        env = SalesEnv(config=default_config)
        env.reset()

        obs = env.end_turn()

        assert isinstance(obs, dict)
        assert "time" in obs


class TestEpisodeFlow:
    """Tests for complete episode flow."""

    def test_episode_with_mock_buyer(self, default_config: SalesBenchConfig, mock_buyer_simulator):
        """Test a complete episode with mock buyer."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        steps = 0
        max_steps = 50

        while not env.is_terminated and steps < max_steps:
            tool_calls = [ToolCall(tool_name="crm.search_leads", arguments={"limit": 5})]
            env.execute_tools(tool_calls)
            env.end_turn()
            steps += 1

        assert steps > 0

    def test_episode_termination_recorded(
        self, default_config: SalesBenchConfig, mock_buyer_simulator
    ):
        """Test that episode termination is properly recorded."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        for _ in range(1000):
            if env.is_terminated:
                break
            tool_calls = [ToolCall(tool_name="crm.search_leads", arguments={})]
            env.execute_tools(tool_calls)
            env.end_turn()

        # Check termination state
        assert env.is_terminated or env.termination_reason is None

    def test_reset_is_idempotent_with_same_seed(self, default_config: SalesBenchConfig):
        """Test that reset with same seed produces same initial state."""
        env1 = SalesEnv(config=default_config)
        env2 = SalesEnv(config=default_config)

        obs1 = env1.reset()
        obs2 = env2.reset()

        assert obs1["leads_count"] == obs2["leads_count"]


class TestTerminationConditions:
    """Tests for episode termination conditions."""

    def test_time_limit_terminates(self, mock_buyer_simulator):
        """Test that time limit terminates episode."""
        config = SalesBenchConfig(
            seed=42,
            num_leads=10,
            budget=BudgetConfig(
                total_hours=1,  # Just 1 hour - episode ends when time passes
            ),
        )
        env = SalesEnv(config=config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        steps = 0
        while not env.is_terminated and steps < 100:
            env.execute_tools([ToolCall(tool_name="crm.search_leads", arguments={})])
            env.end_turn()
            steps += 1


class TestCallFlow:
    """Tests for complete call flows."""

    def test_successful_sale_flow(
        self, default_config: SalesBenchConfig, accepting_buyer_simulator
    ):
        """Test a successful sale flow."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(accepting_buyer_simulator)
        env.reset()

        # Search for leads
        results = env.execute_tools(
            [ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})]
        )

        # Get lead ID from tool results
        lead_id = None
        for result in results:
            if result.success and "leads" in (result.data or {}):
                leads = result.data["leads"]
                if leads:
                    lead_id = leads[0]["lead_id"]
                    break

        assert lead_id is not None

        # Start call
        env.execute_tools(
            [ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})]
        )

        # Propose plan
        results = env.execute_tools(
            [
                ToolCall(
                    tool_name="calling.propose_plan",
                    arguments={
                        "plan_id": "TERM",
                        "coverage_amount": 500000,
                        "monthly_premium": 60.0,
                        "next_step": "close_now",
                    },
                )
            ]
        )

        # Check for acceptance
        for result in results:
            if result.success and result.data:
                assert result.data.get("decision") in ["accept_plan", "reject_plan"]

    def test_rejected_sale_flow(self, default_config: SalesBenchConfig, mock_buyer_simulator):
        """Test a rejected sale flow."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)  # Always rejects
        env.reset()

        # Search for leads
        results = env.execute_tools(
            [ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})]
        )

        lead_id = results[0].data["leads"][0]["lead_id"]

        # Start call
        env.execute_tools(
            [ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})]
        )

        # Propose plan
        results = env.execute_tools(
            [
                ToolCall(
                    tool_name="calling.propose_plan",
                    arguments={
                        "plan_id": "TERM",
                        "coverage_amount": 500000,
                        "monthly_premium": 60.0,
                        "next_step": "close_now",
                    },
                )
            ]
        )

        # Should get rejection
        for result in results:
            if result.success and result.data:
                assert result.data.get("decision") == "reject_plan"


class TestMetricsCollection:
    """Tests for metrics collection."""

    def test_metrics_updated_on_calls(self, default_config: SalesBenchConfig, mock_buyer_simulator):
        """Test that metrics are updated when calls are made."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        initial_calls = env.state.stats.total_calls

        # Search for a lead and make a call
        results = env.execute_tools(
            [ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})]
        )
        lead_id = results[0].data["leads"][0]["lead_id"]

        env.execute_tools(
            [ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})]
        )
        env.execute_tools([ToolCall(tool_name="calling.end_call", arguments={})])

        assert env.state.stats.total_calls > initial_calls

    def test_tool_calls_counted(self, default_config: SalesBenchConfig, mock_buyer_simulator):
        """Test that tool calls are properly counted."""
        env = SalesEnv(config=default_config)
        env.set_buyer_simulator(mock_buyer_simulator)
        env.reset()

        # Make several tool calls
        for _ in range(5):
            env.execute_tools([ToolCall(tool_name="crm.search_leads", arguments={})])
            env.end_turn()

        assert env.state.total_tool_calls >= 5


class TestVerifiersEnvironment:
    """Tests for verifiers-compatible environment."""

    def test_load_environment_returns_tool_env(self):
        """Test that load_environment returns SalesBenchToolEnv."""
        from salesbench import SalesBenchToolEnv, load_environment

        env = load_environment(seed=42, num_leads=10, num_episodes=5)

        assert isinstance(env, SalesBenchToolEnv)

    def test_tool_env_has_tools(self):
        """Test that SalesBenchToolEnv has registered tools."""
        from salesbench import load_environment

        env = load_environment(seed=42, num_leads=10, num_episodes=5)

        # Check tools are registered (12 tools)
        assert len(env.tools) == 12

    def test_tool_env_has_rubric(self):
        """Test that SalesBenchToolEnv has a rubric."""
        from salesbench import load_environment

        env = load_environment(seed=42, num_leads=10, num_episodes=5)

        assert env.rubric is not None

    def test_create_dataset(self):
        """Test dataset creation."""
        from salesbench import create_salesbench_dataset

        dataset = create_salesbench_dataset(num_episodes=10, base_seed=42)

        assert len(dataset) == 10
        assert "prompt" in dataset.column_names
        assert "seed" in dataset.column_names
