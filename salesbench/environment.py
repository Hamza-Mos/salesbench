"""Verifiers-compatible environment for SalesBench.

This module provides a Prime Intellect Verifiers-compatible environment
for the SalesBench benchmark. It follows the verifiers interface for
seamless integration with the Environments Hub and Prime RL.

Usage:
    # Via verifiers
    import verifiers as vf
    env = vf.load_environment("salesbench", seed=42, num_leads=100)

    # Direct import
    from salesbench import load_environment
    env = load_environment(seed=42, num_leads=100)
"""

import json
import logging
import os
from typing import Optional

from datasets import Dataset

import verifiers as vf
from salesbench.core.config import BudgetConfig, SalesBenchConfig, ScoringConfig
from salesbench.envs.sales_mvp.env import SalesEnv
from salesbench.envs.sales_mvp.verifiers.scoring import (
    calculate_episode_score,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions (as standalone async functions for verifiers)
# =============================================================================
#
# These tools use explicit typing without 'from __future__ import annotations'
# to ensure type hints are evaluable by verifiers' schema generation.
# The 'env' parameter is skipped from the schema via add_tool(args_to_skip=["env"]).


async def crm_search_leads(
    env,
    temperature: str = None,
    min_income: int = None,
    max_age: int = None,
    limit: int = 10,
) -> str:
    """Search for leads matching criteria.

    Args:
        temperature: Filter by lead temperature (hot/warm/lukewarm/cold/hostile).
        min_income: Minimum annual income filter.
        max_age: Maximum age filter.
        limit: Maximum number of results to return.

    Returns:
        JSON string with matching leads.
    """
    result = env._crm_tools.search_leads(
        temperature=temperature,
        min_income=min_income,
        max_age=max_age,
        limit=limit,
    )
    return json.dumps(result.data if result.success else {"error": result.error})


async def crm_get_lead(env, lead_id: str) -> str:
    """Get detailed information about a specific lead.

    Args:
        lead_id: The ID of the lead to retrieve.

    Returns:
        JSON string with lead details.
    """
    result = env._crm_tools.get_lead(lead_id)
    return json.dumps(result.data if result.success else {"error": result.error})


async def crm_update_lead(
    env,
    lead_id: str,
    notes: str = None,
    temperature: str = None,
) -> str:
    """Update a lead's notes or temperature classification.

    Args:
        lead_id: The ID of the lead to update.
        notes: Notes to append to the lead.
        temperature: New temperature classification.

    Returns:
        JSON string with update result.
    """
    result = env._crm_tools.update_lead(lead_id, notes=notes, temperature=temperature)
    return json.dumps(result.data if result.success else {"error": result.error})


async def crm_log_call(
    env,
    lead_id: str,
    call_id: str,
    outcome: str,
    notes: str = None,
) -> str:
    """Log a completed call with its outcome.

    Args:
        lead_id: The ID of the lead that was called.
        call_id: The ID of the call.
        outcome: Call outcome (accepted/rejected/ended/no_answer).
        notes: Additional notes about the call.

    Returns:
        JSON string with log result.
    """
    result = env._crm_tools.log_call(lead_id, call_id, outcome, notes=notes)
    return json.dumps(result.data if result.success else {"error": result.error})


async def calendar_get_availability(env, day: int = None) -> str:
    """Get available time slots for scheduling calls.

    Args:
        day: Day number to check (1-10). Defaults to current day.

    Returns:
        JSON string with available time slots.
    """
    result = env._calendar_tools.get_availability(day=day)
    return json.dumps(result.data if result.success else {"error": result.error})


async def calendar_schedule_call(
    env,
    lead_id: str,
    day: int,
    hour: int,
) -> str:
    """Schedule a call with a lead.

    Args:
        lead_id: The ID of the lead to schedule with.
        day: Day number for the call (1-10).
        hour: Hour for the call (9-17).

    Returns:
        JSON string with scheduling result.
    """
    result = env._calendar_tools.schedule_call(lead_id, day, hour)
    return json.dumps(result.data if result.success else {"error": result.error})


async def calling_start_call(env, lead_id: str) -> str:
    """Start a call with a lead.

    Args:
        lead_id: The ID of the lead to call.

    Returns:
        JSON string with call session details.
    """
    result = env._calling_tools.start_call(lead_id)
    return json.dumps(result.data if result.success else {"error": result.error})


async def calling_propose_plan(
    env,
    plan_id: str,
    monthly_premium: float,
    coverage_amount: float,
    next_step: str,
    term_years: int = None,
) -> str:
    """Propose an insurance plan to the buyer and get their response.

    Args:
        plan_id: Type of plan (TERM, WHOLE, UL, VUL, LTC, DI).
        monthly_premium: Monthly premium amount.
        coverage_amount: Coverage amount.
        next_step: Proposed next step (schedule_followup, request_info, close_now).
        term_years: Term length for TERM plans.
    Returns:
        JSON string with buyer's decision and response.
    """
    result = env._calling_tools.propose_plan(
        plan_id=plan_id,
        monthly_premium=monthly_premium,
        coverage_amount=coverage_amount,
        next_step=next_step,
        term_years=term_years,
    )
    return json.dumps(result.data if result.success else {"error": result.error})


async def calling_end_call(env, reason: str = None) -> str:
    """End the current call.

    Args:
        reason: Reason for ending the call.

    Returns:
        JSON string with call summary.
    """
    result = env._calling_tools.end_call(reason=reason)
    return json.dumps(result.data if result.success else {"error": result.error})


async def products_list_plans(env) -> str:
    """List all available insurance plans.

    Returns:
        JSON string with list of available plans.
    """
    result = env._product_tools.list_plans()
    return json.dumps(result.data if result.success else {"error": result.error})


async def products_get_plan(env, plan_id: str) -> str:
    """Get detailed information about a specific insurance plan.

    Args:
        plan_id: The ID of the plan (TERM, WHOLE, UL, VUL, LTC, DI).

    Returns:
        JSON string with plan details.
    """
    result = env._product_tools.get_plan(plan_id)
    return json.dumps(result.data if result.success else {"error": result.error})


async def products_quote_premium(
    env,
    plan_id: str,
    age: int,
    coverage_amount: float,
    risk_class: str = None,
    term_years: int = None,
) -> str:
    """Get a premium quote for a plan based on age, coverage, and risk.

    Args:
        plan_id: The ID of the plan.
        age: Age of the insured.
        coverage_amount: Desired coverage amount.
        risk_class: Risk classification (default: standard_plus).
        term_years: Term length for TERM plans.

    Returns:
        JSON string with premium quote.
    """
    result = env._product_tools.quote_premium(
        plan_id=plan_id,
        age=age,
        coverage_amount=coverage_amount,
        risk_class=risk_class,
        term_years=term_years,
    )
    return json.dumps(result.data if result.success else {"error": result.error})


# Map tool names to functions
TOOL_FUNCTIONS = {
    "crm_search_leads": crm_search_leads,
    "crm_get_lead": crm_get_lead,
    "crm_update_lead": crm_update_lead,
    "crm_log_call": crm_log_call,
    "calendar_get_availability": calendar_get_availability,
    "calendar_schedule_call": calendar_schedule_call,
    "calling_start_call": calling_start_call,
    "calling_propose_plan": calling_propose_plan,
    "calling_end_call": calling_end_call,
    "products_list_plans": products_list_plans,
    "products_get_plan": products_get_plan,
    "products_quote_premium": products_quote_premium,
}


# =============================================================================
# Reward Functions for Rubric
# =============================================================================


async def episode_score(state: vf.State) -> float:
    """Calculate the final episode score.

    This reward function evaluates the agent's performance based on:
    - Accepted offers (primary reward)
    - Premium amounts (profit proxy)
    - Close vs followup bonuses
    - Efficiency (time and tool usage)
    - Penalties for rejections and DNC violations

    Args:
        state: The verifiers state containing the SalesEnv.

    Returns:
        Total episode score as a float.
    """
    sales_env: Optional[SalesEnv] = state.get("sales_env")
    if sales_env is None:
        return 0.0

    scoring_config = state.get("scoring_config", ScoringConfig())
    total_days = state.get("total_days", 10)

    components = calculate_episode_score(
        state=sales_env.state,
        config=scoring_config,
        total_days=total_days,
    )

    return components.total_score


async def acceptance_rate(state: vf.State) -> float:
    """Calculate the acceptance rate metric.

    Args:
        state: The verifiers state containing the SalesEnv.

    Returns:
        Acceptance rate as a float between 0 and 1.
    """
    sales_env: Optional[SalesEnv] = state.get("sales_env")
    if sales_env is None:
        return 0.0

    stats = sales_env.state.stats
    total_offers = stats.accepted_offers + stats.rejected_offers
    if total_offers == 0:
        return 0.0

    return stats.accepted_offers / total_offers


async def dnc_violation_count(state: vf.State) -> float:
    """Count DNC violations (lower is better, returned as negative).

    Args:
        state: The verifiers state containing the SalesEnv.

    Returns:
        Negative count of DNC violations.
    """
    sales_env: Optional[SalesEnv] = state.get("sales_env")
    if sales_env is None:
        return 0.0

    return -float(sales_env.state.stats.dnc_violations)


# =============================================================================
# SalesBench Verifiers Environment
# =============================================================================


class SalesBenchToolEnv(vf.StatefulToolEnv):
    """Verifiers-compatible SalesBench environment.

    This environment wraps the SalesBench simulation in a verifiers-compatible
    interface, enabling integration with Prime Intellect's Environments Hub
    and RL training infrastructure.

    The environment simulates a life insurance cold-calling scenario where
    an AI agent (seller) must navigate conversations with simulated buyers
    to sell insurance policies.
    """

    def __init__(
        self,
        dataset: Dataset,
        seed: int = 42,
        num_leads: int = 100,
        total_days: int = 10,
        max_calls_per_day: int = 50,
        buyer_model: str = "gpt-4o-mini",
        buyer_temperature: float = 0.3,
        max_turns: int = 100,
        **kwargs,
    ):
        """Initialize the SalesBench environment.

        Args:
            dataset: HuggingFace dataset with task configurations.
            seed: Base random seed for reproducibility.
            num_leads: Number of leads to generate per episode.
            total_days: Total simulated business days.
            max_calls_per_day: Maximum calls allowed per day.
            buyer_model: LLM model for buyer simulator.
            buyer_temperature: Temperature for buyer LLM.
            max_turns: Maximum turns per episode.
            **kwargs: Additional arguments passed to parent.
        """
        # Store configuration
        self.base_seed = seed
        self.num_leads = num_leads
        self.total_days = total_days
        self.max_calls_per_day = max_calls_per_day
        self.buyer_model = buyer_model
        self.buyer_temperature = buyer_temperature

        # Create rubric with scoring functions
        rubric = vf.Rubric(
            funcs=[episode_score, acceptance_rate, dnc_violation_count],
            weights=[1.0, 0.0, 0.0],  # Only episode_score contributes to total
        )

        # System prompt for the seller agent
        system_prompt = self._build_system_prompt()

        # Initialize without tools - we'll add them manually with skipped args
        super().__init__(
            dataset=dataset,
            tools=[],  # Empty - will add with args_to_skip
            max_turns=max_turns,
            rubric=rubric,
            system_prompt=system_prompt,
            **kwargs,
        )

        # Register each tool with 'env' argument skipped from schema
        for tool_func in TOOL_FUNCTIONS.values():
            self.add_tool(tool_func, args_to_skip=["env"])

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the seller agent."""
        return """You are an AI sales agent for a life insurance company. Your goal is to call leads, understand their needs, and sell appropriate insurance policies.

## Available Tools

You have access to the following tools:
- CRM tools: search_leads, get_lead, update_lead, log_call
- Calendar tools: get_availability, schedule_call
- Calling tools: start_call, propose_plan, end_call
- Product tools: list_plans, get_plan, quote_premium

## Workflow

1. Search for leads using crm_search_leads
2. Review lead details with crm_get_lead
3. Start a call with calling_start_call
4. Understand the buyer's needs through conversation
5. Use products_quote_premium to get accurate pricing
6. Present offers with calling_propose_plan
7. End calls appropriately with calling_end_call

## Important Rules

- Only call leads who are not on the Do Not Call (DNC) list
- Respect buyer decisions - don't be pushy
- Propose plans that fit the buyer's budget and needs
- Close deals when appropriate (close_now) or schedule followups

## Scoring

Your performance is measured by:
- Accepted offers (+100 points each)
- Premium amounts (bonus based on policy value)
- Close now vs followup bonuses
- Efficiency (fewer tool calls, faster completion)
- Penalties for rejections and DNC violations"""

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize SalesBench state for each rollout.

        This method is called at the start of each rollout to set up
        the SalesEnv instance with the appropriate configuration.

        Args:
            state: The verifiers state to populate.

        Returns:
            The state with SalesEnv initialized.
        """
        # Get seed from input or use base seed
        input_data = state.get("input", {})
        episode_seed = input_data.get("seed", self.base_seed)

        # Create budget config
        budget = BudgetConfig(
            total_days=self.total_days,
            max_calls_per_day=self.max_calls_per_day,
        )

        # Create SalesBench config
        config = SalesBenchConfig(
            seed=episode_seed,
            num_leads=self.num_leads,
            budget=budget,
            buyer_model=self.buyer_model,
        )

        # Create and initialize SalesEnv
        sales_env = SalesEnv(config)

        # Set up buyer simulator if API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            from salesbench.agents.buyer_llm import create_buyer_simulator

            simulator = create_buyer_simulator(
                model=self.buyer_model,
                temperature=self.buyer_temperature,
                api_key=api_key,
            )
            sales_env.set_buyer_simulator(simulator)

        # Reset environment to generate leads
        sales_env.reset()

        # Store in state
        state["sales_env"] = sales_env
        state["scoring_config"] = config.scoring
        state["total_days"] = self.total_days

        # Add initial context to prompt
        obs = sales_env._get_observation()
        context = f"""
## Current State
- Day: {obs['time']['current_day']}, Hour: {obs['time']['current_hour']}:00
- Available leads: {obs['leads_count']}
- Total days: {self.total_days}

Start by searching for leads and making calls."""

        # Append context to the prompt
        if state.get("prompt") and len(state["prompt"]) > 0:
            last_msg = state["prompt"][-1]
            if last_msg.get("role") == "user":
                last_msg["content"] = last_msg.get("content", "") + context

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject the SalesEnv instance into tool arguments.

        This method is called before each tool execution to inject
        the current SalesEnv state into the tool call.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Arguments provided by the model.
            messages: Current conversation messages.
            state: The verifiers state containing SalesEnv.
            **kwargs: Additional arguments.

        Returns:
            Updated tool arguments with env injected.
        """
        sales_env = state.get("sales_env")
        if sales_env is None:
            raise RuntimeError("SalesEnv not initialized in state")

        # Inject env into arguments
        tool_args["env"] = sales_env

        return tool_args

    @vf.stop
    async def episode_terminated(self, state: vf.State) -> bool:
        """Check if the SalesBench episode has terminated.

        Args:
            state: The verifiers state containing SalesEnv.

        Returns:
            True if the episode should end.
        """
        sales_env: Optional[SalesEnv] = state.get("sales_env")
        if sales_env is None:
            return True

        return sales_env.is_terminated


# =============================================================================
# Dataset Generation
# =============================================================================


def create_salesbench_dataset(
    num_episodes: int = 100,
    base_seed: int = 42,
) -> Dataset:
    """Create a dataset of SalesBench episodes.

    Each row represents a different episode with a unique seed,
    ensuring reproducible but varied scenarios.

    Args:
        num_episodes: Number of episodes in the dataset.
        base_seed: Base seed for generating episode seeds.

    Returns:
        HuggingFace Dataset with episode configurations.
    """
    episodes = []
    for i in range(num_episodes):
        episode_seed = base_seed + i
        episodes.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "You are starting a new sales shift. "
                            "Your goal is to call leads and sell life insurance policies. "
                            "Use the available tools to search for leads, make calls, "
                            "and close deals. Good luck!"
                        ),
                    }
                ],
                "seed": episode_seed,
                "info": json.dumps({"episode_index": i, "seed": episode_seed}),
            }
        )

    return Dataset.from_list(episodes)


# =============================================================================
# Load Environment Function (Verifiers Entry Point)
# =============================================================================


def load_environment(
    seed: int = 42,
    num_leads: int = 100,
    total_days: int = 10,
    max_calls_per_day: int = 50,
    buyer_model: str = "gpt-4o-mini",
    buyer_temperature: float = 0.3,
    num_episodes: int = 100,
    max_turns: int = 100,
) -> SalesBenchToolEnv:
    """Load the SalesBench environment.

    This is the main entry point for the verifiers framework.
    It creates a fully configured SalesBenchToolEnv instance.

    Args:
        seed: Base random seed for reproducibility.
        num_leads: Number of leads to generate per episode.
        total_days: Total simulated business days.
        max_calls_per_day: Maximum calls allowed per day.
        buyer_model: LLM model for buyer simulator.
        buyer_temperature: Temperature for buyer LLM.
        num_episodes: Number of episodes in the dataset.
        max_turns: Maximum turns per episode.

    Returns:
        Configured SalesBenchToolEnv instance.

    Example:
        # Via verifiers
        import verifiers as vf
        env = vf.load_environment("salesbench", seed=42)
        results = env.evaluate(client, model="gpt-4o")

        # Direct usage
        from salesbench import load_environment
        env = load_environment(seed=42, num_leads=50)
    """
    # Create dataset
    dataset = create_salesbench_dataset(
        num_episodes=num_episodes,
        base_seed=seed,
    )

    # Create environment
    env = SalesBenchToolEnv(
        dataset=dataset,
        seed=seed,
        num_leads=num_leads,
        total_days=total_days,
        max_calls_per_day=max_calls_per_day,
        buyer_model=buyer_model,
        buyer_temperature=buyer_temperature,
        max_turns=max_turns,
    )

    return env
