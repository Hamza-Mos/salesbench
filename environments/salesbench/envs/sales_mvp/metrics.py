"""Metrics engine for SalesBench.

Computes and tracks comprehensive metrics for:
- Performance analysis
- Model comparison
- Training progress monitoring
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import statistics

if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.state import EnvironmentState


@dataclass
class CallMetrics:
    """Metrics related to calls."""

    total_calls: int = 0
    successful_calls: int = 0  # Calls with accept
    failed_calls: int = 0  # Calls with buyer end
    total_call_minutes: int = 0
    avg_call_duration: float = 0.0
    calls_per_day: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_call_minutes": self.total_call_minutes,
            "avg_call_duration": round(self.avg_call_duration, 2),
            "calls_per_day": round(self.calls_per_day, 2),
        }


@dataclass
class OfferMetrics:
    """Metrics related to offers."""

    total_offers: int = 0
    accepted_offers: int = 0
    rejected_offers: int = 0
    acceptance_rate: float = 0.0
    avg_offers_per_call: float = 0.0
    avg_premium_accepted: float = 0.0
    total_premium_accepted: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_offers": self.total_offers,
            "accepted_offers": self.accepted_offers,
            "rejected_offers": self.rejected_offers,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_offers_per_call": round(self.avg_offers_per_call, 2),
            "avg_premium_accepted": round(self.avg_premium_accepted, 2),
            "total_premium_accepted": round(self.total_premium_accepted, 2),
        }


@dataclass
class LeadMetrics:
    """Metrics related to leads."""

    total_leads: int = 0
    leads_contacted: int = 0
    leads_converted: int = 0
    leads_on_dnc: int = 0
    conversion_rate: float = 0.0
    contact_rate: float = 0.0
    dnc_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_leads": self.total_leads,
            "leads_contacted": self.leads_contacted,
            "leads_converted": self.leads_converted,
            "leads_on_dnc": self.leads_on_dnc,
            "conversion_rate": round(self.conversion_rate, 4),
            "contact_rate": round(self.contact_rate, 4),
            "dnc_rate": round(self.dnc_rate, 4),
        }


@dataclass
class EfficiencyMetrics:
    """Efficiency-related metrics."""

    total_tool_calls: int = 0
    tool_calls_per_accept: float = 0.0
    minutes_per_accept: float = 0.0
    days_used: int = 0
    time_efficiency: float = 0.0  # 0-1, higher is better

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "tool_calls_per_accept": round(self.tool_calls_per_accept, 2),
            "minutes_per_accept": round(self.minutes_per_accept, 2),
            "days_used": self.days_used,
            "time_efficiency": round(self.time_efficiency, 4),
        }


@dataclass
class TemperatureBreakdown:
    """Breakdown by lead temperature."""

    hot: dict[str, int] = field(default_factory=lambda: {"contacted": 0, "converted": 0})
    warm: dict[str, int] = field(default_factory=lambda: {"contacted": 0, "converted": 0})
    lukewarm: dict[str, int] = field(default_factory=lambda: {"contacted": 0, "converted": 0})
    cold: dict[str, int] = field(default_factory=lambda: {"contacted": 0, "converted": 0})
    hostile: dict[str, int] = field(default_factory=lambda: {"contacted": 0, "converted": 0})

    def to_dict(self) -> dict[str, Any]:
        return {
            "hot": self.hot,
            "warm": self.warm,
            "lukewarm": self.lukewarm,
            "cold": self.cold,
            "hostile": self.hostile,
        }


@dataclass
class EpisodeMetrics:
    """Complete metrics for an episode."""

    calls: CallMetrics = field(default_factory=CallMetrics)
    offers: OfferMetrics = field(default_factory=OfferMetrics)
    leads: LeadMetrics = field(default_factory=LeadMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    temperature_breakdown: TemperatureBreakdown = field(default_factory=TemperatureBreakdown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls.to_dict(),
            "offers": self.offers.to_dict(),
            "leads": self.leads.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "temperature_breakdown": self.temperature_breakdown.to_dict(),
        }


class MetricsEngine:
    """Computes comprehensive metrics from environment state."""

    def __init__(self, total_days: int = 10):
        """Initialize the metrics engine.

        Args:
            total_days: Total days available in episode.
        """
        self.total_days = total_days

    def compute(self, state: "EnvironmentState") -> EpisodeMetrics:
        """Compute all metrics from environment state.

        Args:
            state: Environment state to analyze.

        Returns:
            EpisodeMetrics with all computed metrics.
        """
        metrics = EpisodeMetrics()

        # Compute call metrics
        metrics.calls = self._compute_call_metrics(state)

        # Compute offer metrics
        metrics.offers = self._compute_offer_metrics(state)

        # Compute lead metrics
        metrics.leads = self._compute_lead_metrics(state)

        # Compute efficiency metrics
        metrics.efficiency = self._compute_efficiency_metrics(state, metrics)

        # Compute temperature breakdown
        metrics.temperature_breakdown = self._compute_temperature_breakdown(state)

        return metrics

    def _compute_call_metrics(self, state: "EnvironmentState") -> CallMetrics:
        """Compute call-related metrics."""
        from salesbench.core.types import BuyerDecision

        metrics = CallMetrics()
        metrics.total_calls = state.stats.total_calls
        metrics.total_call_minutes = state.stats.total_call_minutes

        # Count successful/failed calls
        for call in state.call_history:
            if call.outcome == BuyerDecision.ACCEPT_PLAN:
                metrics.successful_calls += 1
            elif call.outcome == BuyerDecision.END_CALL:
                metrics.failed_calls += 1

        # Calculate averages
        if metrics.total_calls > 0:
            metrics.avg_call_duration = metrics.total_call_minutes / metrics.total_calls

        days_used = max(1, state.time.current_day)
        metrics.calls_per_day = metrics.total_calls / days_used

        return metrics

    def _compute_offer_metrics(self, state: "EnvironmentState") -> OfferMetrics:
        """Compute offer-related metrics."""
        metrics = OfferMetrics()
        metrics.accepted_offers = state.stats.accepted_offers
        metrics.rejected_offers = state.stats.rejected_offers
        metrics.total_offers = metrics.accepted_offers + metrics.rejected_offers

        # Calculate acceptance rate
        if metrics.total_offers > 0:
            metrics.acceptance_rate = metrics.accepted_offers / metrics.total_offers

        # Calculate average offers per call
        if state.stats.total_calls > 0:
            metrics.avg_offers_per_call = metrics.total_offers / state.stats.total_calls

        # Calculate premium metrics from call history
        accepted_premiums = []
        for call in state.call_history:
            for i, response in enumerate(call.buyer_responses):
                from salesbench.core.types import BuyerDecision
                if response.decision == BuyerDecision.ACCEPT_PLAN:
                    if i < len(call.offers_presented):
                        premium = call.offers_presented[i].monthly_premium
                        accepted_premiums.append(premium)

        if accepted_premiums:
            metrics.avg_premium_accepted = statistics.mean(accepted_premiums)
            metrics.total_premium_accepted = sum(accepted_premiums)

        return metrics

    def _compute_lead_metrics(self, state: "EnvironmentState") -> LeadMetrics:
        """Compute lead-related metrics."""
        metrics = LeadMetrics()
        metrics.total_leads = len(state.leads)

        # Count leads by status
        contacted_leads = set()
        converted_leads = set()

        for call in state.call_history:
            contacted_leads.add(call.lead_id)
            from salesbench.core.types import BuyerDecision
            if call.outcome == BuyerDecision.ACCEPT_PLAN:
                converted_leads.add(call.lead_id)

        metrics.leads_contacted = len(contacted_leads)
        metrics.leads_converted = len(converted_leads)
        metrics.leads_on_dnc = sum(1 for lead in state.leads.values() if lead.on_dnc_list)

        # Calculate rates
        if metrics.total_leads > 0:
            metrics.contact_rate = metrics.leads_contacted / metrics.total_leads
            metrics.dnc_rate = metrics.leads_on_dnc / metrics.total_leads

        if metrics.leads_contacted > 0:
            metrics.conversion_rate = metrics.leads_converted / metrics.leads_contacted

        return metrics

    def _compute_efficiency_metrics(
        self,
        state: "EnvironmentState",
        other_metrics: EpisodeMetrics,
    ) -> EfficiencyMetrics:
        """Compute efficiency-related metrics."""
        metrics = EfficiencyMetrics()
        metrics.total_tool_calls = state.total_tool_calls
        metrics.days_used = state.time.current_day

        accepts = other_metrics.offers.accepted_offers

        if accepts > 0:
            metrics.tool_calls_per_accept = metrics.total_tool_calls / accepts
            metrics.minutes_per_accept = state.stats.total_call_minutes / accepts

        # Time efficiency: how much of the time budget was used effectively
        if self.total_days > 0:
            # Higher score for finishing faster with more accepts
            days_ratio = metrics.days_used / self.total_days
            if accepts > 0:
                # Efficiency = accepts / days_ratio (more accepts in less time = better)
                metrics.time_efficiency = min(1.0, accepts / (days_ratio * 10))
            else:
                metrics.time_efficiency = 0.0

        return metrics

    def _compute_temperature_breakdown(self, state: "EnvironmentState") -> TemperatureBreakdown:
        """Compute breakdown by lead temperature."""
        from salesbench.core.types import LeadTemperature, BuyerDecision

        breakdown = TemperatureBreakdown()

        # Map temperatures to breakdown attributes
        temp_map = {
            LeadTemperature.HOT: breakdown.hot,
            LeadTemperature.WARM: breakdown.warm,
            LeadTemperature.LUKEWARM: breakdown.lukewarm,
            LeadTemperature.COLD: breakdown.cold,
            LeadTemperature.HOSTILE: breakdown.hostile,
        }

        # Track which leads were contacted/converted
        contacted_leads: dict[str, bool] = {}  # lead_id -> converted
        for call in state.call_history:
            lead_id = call.lead_id
            if lead_id not in contacted_leads:
                contacted_leads[lead_id] = False
            if call.outcome == BuyerDecision.ACCEPT_PLAN:
                contacted_leads[lead_id] = True

        # Aggregate by temperature
        for lead_id, converted in contacted_leads.items():
            lead = state.leads.get(lead_id)
            if lead:
                temp_dict = temp_map.get(lead.temperature)
                if temp_dict:
                    temp_dict["contacted"] += 1
                    if converted:
                        temp_dict["converted"] += 1

        return breakdown


def compute_episode_metrics(
    state: "EnvironmentState",
    total_days: int = 10,
) -> EpisodeMetrics:
    """Compute episode metrics from state.

    Args:
        state: Environment state.
        total_days: Total available days.

    Returns:
        Complete episode metrics.
    """
    engine = MetricsEngine(total_days=total_days)
    return engine.compute(state)
