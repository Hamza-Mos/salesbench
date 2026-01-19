"""Tests for scoring rubric."""

import pytest

from salesbench.core.config import ScoringConfig
from salesbench.core.types import NextStep
from salesbench.envs.sales_mvp.verifiers.scoring import (
    ScoreComponents,
    ScoringRubric,
    calculate_episode_score,
)


class TestScoreComponents:
    """Tests for ScoreComponents dataclass."""

    def test_total_score_calculation(self):
        """Test that total score sums all components correctly."""
        components = ScoreComponents(
            accept_rewards=100.0,
            close_bonuses=50.0,
            followup_bonuses=25.0,
            premium_rewards=10.0,
            time_efficiency_bonus=5.0,
            cost_efficiency_bonus=5.0,
            reject_penalties=-15.0,
            end_call_penalties=-10.0,
            dnc_penalties=-200.0,
        )

        expected = 100 + 50 + 25 + 10 + 5 + 5 - 15 - 10 - 200
        assert components.total_score == expected

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all necessary fields."""
        components = ScoreComponents(num_accepts=3, num_rejects=2)
        d = components.to_dict()

        assert "total_score" in d
        assert "components" in d
        assert "counts" in d
        assert d["counts"]["num_accepts"] == 3
        assert d["counts"]["num_rejects"] == 2


class TestScoringRubric:
    """Tests for ScoringRubric."""

    @pytest.fixture
    def rubric(self):
        """Create a scoring rubric with default config."""
        return ScoringRubric()

    @pytest.fixture
    def custom_rubric(self):
        """Create a rubric with custom config."""
        config = ScoringConfig(
            accept_reward=200.0,
            reject_penalty=-10.0,
            dnc_penalty=-500.0,
        )
        return ScoringRubric(config)

    def test_record_accept_adds_reward(self, rubric: ScoringRubric):
        """Test that accepting a plan adds reward."""
        score_added = rubric.record_accept(
            monthly_premium=100.0,
            next_step=NextStep.CLOSE_NOW,
        )

        assert score_added > 0
        assert rubric.components.num_accepts == 1
        assert rubric.components.accept_rewards > 0

    def test_accept_with_close_now_bonus(self, rubric: ScoringRubric):
        """Test that close_now gives bonus."""
        close_now_score = rubric.record_accept(
            monthly_premium=100.0,
            next_step=NextStep.CLOSE_NOW,
        )

        rubric.reset()

        followup_score = rubric.record_accept(
            monthly_premium=100.0,
            next_step=NextStep.SCHEDULE_FOLLOWUP,
        )

        assert close_now_score > followup_score

    def test_accept_premium_bonus(self):
        """Test that higher premiums give higher bonuses."""
        rubric1 = ScoringRubric()
        rubric2 = ScoringRubric()

        rubric1.record_accept(monthly_premium=50.0, next_step=NextStep.CLOSE_NOW)
        rubric2.record_accept(monthly_premium=200.0, next_step=NextStep.CLOSE_NOW)

        assert rubric2.components.premium_rewards > rubric1.components.premium_rewards

    def test_record_reject_adds_penalty(self, rubric: ScoringRubric):
        """Test that rejection adds penalty."""
        score_added = rubric.record_reject()

        assert score_added < 0
        assert rubric.components.num_rejects == 1
        assert rubric.components.reject_penalties < 0

    def test_record_end_call_adds_penalty(self, rubric: ScoringRubric):
        """Test that buyer ending call adds penalty."""
        score_added = rubric.record_end_call()

        assert score_added < 0
        assert rubric.components.num_end_calls == 1

    def test_record_dnc_violation_adds_large_penalty(self, rubric: ScoringRubric):
        """Test that DNC violation adds large penalty."""
        score_added = rubric.record_dnc_violation()

        assert score_added < -100  # Should be significant
        assert rubric.components.num_dnc_violations == 1
        assert rubric.components.dnc_penalties < -100

    def test_custom_config_values(self, custom_rubric: ScoringRubric):
        """Test that custom config values are used."""
        # Accept with custom reward
        custom_rubric.record_accept(
            monthly_premium=100.0,
            next_step=NextStep.CLOSE_NOW,
        )
        assert custom_rubric.components.accept_rewards == 200.0

        # DNC with custom penalty
        custom_rubric.record_dnc_violation()
        assert custom_rubric.components.dnc_penalties == -500.0

    def test_reset_clears_state(self, rubric: ScoringRubric):
        """Test that reset clears all state."""
        rubric.record_accept(monthly_premium=100.0, next_step=NextStep.CLOSE_NOW)
        rubric.record_reject()

        rubric.reset()

        assert rubric.components.num_accepts == 0
        assert rubric.components.num_rejects == 0
        assert rubric.total_score == 0

    def test_efficiency_bonus_for_early_finish(self, rubric: ScoringRubric):
        """Test efficiency bonus when finishing early with accepts."""
        rubric.record_accept(monthly_premium=100.0, next_step=NextStep.CLOSE_NOW)

        bonus = rubric.calculate_efficiency_bonuses(
            days_used=5,
            total_days=10,
            tool_calls=100,
            max_tool_calls=500,
        )

        assert bonus > 0
        assert rubric.components.time_efficiency_bonus > 0

    def test_no_efficiency_bonus_without_accepts(self, rubric: ScoringRubric):
        """Test no efficiency bonus when no accepts."""
        bonus = rubric.calculate_efficiency_bonuses(
            days_used=5,
            total_days=10,
            tool_calls=100,
            max_tool_calls=500,
        )

        assert bonus == 0

    def test_bounded_score_min(self):
        """Test that score is bounded at minimum."""
        config = ScoringConfig(min_score=-1000, max_score=10000)
        rubric = ScoringRubric(config)

        # Add many penalties
        for _ in range(100):
            rubric.record_dnc_violation()

        bounded = rubric.get_bounded_score()
        assert bounded == -1000

    def test_bounded_score_max(self):
        """Test that score is bounded at maximum."""
        config = ScoringConfig(min_score=-1000, max_score=10000)
        rubric = ScoringRubric(config)

        # Add many rewards
        for _ in range(200):
            rubric.record_accept(monthly_premium=500.0, next_step=NextStep.CLOSE_NOW)

        bounded = rubric.get_bounded_score()
        assert bounded == 10000


class TestCalculateEpisodeScore:
    """Tests for calculate_episode_score function."""

    def test_empty_state_returns_zero(self, sales_env):
        """Test that empty state returns zero score."""
        # Reset and don't do anything
        sales_env.reset()

        components = calculate_episode_score(sales_env.state)

        assert components.total_score == 0
        assert components.num_accepts == 0

    def test_accepts_increase_score(self, sales_env, accepting_buyer_simulator):
        """Test that accepted offers increase score."""
        from salesbench.core.types import ToolCall

        sales_env.reset()
        sales_env.set_buyer_simulator(accepting_buyer_simulator)

        # Get a lead and make a call
        search = sales_env.execute_tool(
            ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})
        )
        lead_id = search.data["leads"][0]["lead_id"]

        # Start call
        sales_env.execute_tool(
            ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})
        )

        # Propose plan
        sales_env.execute_tool(
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

        # End call
        sales_env.execute_tool(ToolCall(tool_name="calling.end_call", arguments={}))

        # Calculate score
        components = calculate_episode_score(sales_env.state)

        assert components.num_accepts == 1
        assert components.accept_rewards > 0
        assert components.total_score > 0


class TestScoringEdgeCases:
    """Tests for scoring edge cases."""

    def test_multiple_offers_same_call(self):
        """Test scoring with multiple offers in same call."""
        rubric = ScoringRubric()
        rubric.record_reject()  # First offer rejected
        rubric.record_reject()  # Second offer rejected
        rubric.record_accept(  # Third offer accepted
            monthly_premium=150.0,
            next_step=NextStep.SCHEDULE_FOLLOWUP,
        )

        assert rubric.components.num_rejects == 2
        assert rubric.components.num_accepts == 1
        # Net score should still be positive if accept reward > 2 * reject penalty
        # Default: accept=100, reject=-5, so 100 - 10 = 90 + premium bonus

    def test_dnc_dominates_score(self):
        """Test that DNC violations severely impact score."""
        rubric = ScoringRubric()
        # Get some accepts
        rubric.record_accept(monthly_premium=100.0, next_step=NextStep.CLOSE_NOW)
        rubric.record_accept(monthly_premium=100.0, next_step=NextStep.CLOSE_NOW)

        score_before_dnc = rubric.total_score

        # Add DNC violation
        rubric.record_dnc_violation()

        # Score should be significantly lower
        assert rubric.total_score < score_before_dnc - 100

    def test_zero_premium_still_gets_base_reward(self):
        """Test that zero premium still gets base accept reward."""
        rubric = ScoringRubric()
        score = rubric.record_accept(
            monthly_premium=0.0,
            next_step=NextStep.CLOSE_NOW,
        )

        # Should at least get base accept + close bonus
        assert score > 0
        assert rubric.components.accept_rewards > 0
