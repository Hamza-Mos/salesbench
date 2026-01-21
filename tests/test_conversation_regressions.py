def test_observation_mapping_in_call_sets_lead_id():
    """Regression: EpisodeExecutor observation mapping must reflect active_call."""
    from salesbench.runner.executor import EpisodeExecutor

    # Create an uninitialized instance (method does not depend on __init__ state)
    executor = object.__new__(EpisodeExecutor)

    obs_dict = {
        "time": {"current_day": 1, "current_hour": 10, "current_minute": 30},
        "stats": {
            "total_calls": 1,
            "accepted_offers": 0,
            "rejected_offers": 0,
            "dnc_violations": 0,
        },
        "has_active_call": True,
        "active_call": {
            "call_id": "call_test",
            "lead_id": "lead_123",
            "duration_minutes": 7,
            "offers_presented": 2,
        },
        "last_tool_results": [],
    }

    obs = executor._dict_to_observation(obs_dict)
    assert obs.in_call is True
    assert obs.current_lead_id == "lead_123"
    assert obs.call_duration == 7
    assert obs.offers_this_call == 2


def test_conversational_buyer_message_does_not_end_call():
    """Regression: conversational buyer dialogue must not update decision state/end call."""
    from salesbench.context.episode import EpisodeContext

    ctx = EpisodeContext()
    ctx.record_call_start("lead_abc", lead_name="Test Lead")

    assert ctx.current_lead_id == "lead_abc"
    assert ctx._anchored_state.active_call_lead_id == "lead_abc"  # internal invariant

    ctx.record_buyer_message("lead_abc", "Can you explain that a bit more?")

    # Call should remain active; no decision side-effects.
    assert ctx.current_lead_id == "lead_abc"
    assert ctx._anchored_state.active_call_lead_id == "lead_abc"


def test_accept_marks_converted_and_filters_from_search(sales_env, accepting_buyer_simulator):
    """Regression: accepted leads should not be searchable or callable again."""
    from salesbench.core.types import ToolCall

    sales_env.set_buyer_simulator(accepting_buyer_simulator)

    # Find any lead
    lead_id = next(iter(sales_env.state.leads.keys()))
    lead = sales_env.state.leads[lead_id]
    assert lead.converted is False

    # Start call + propose plan (auto-accept ends call)
    start = sales_env.execute_tool(
        ToolCall(tool_name="calling.start_call", arguments={"lead_id": str(lead_id)})
    )
    assert start.success is True

    propose = sales_env.execute_tool(
        ToolCall(
            tool_name="calling.propose_plan",
            arguments={
                "plan_id": "TERM",
                "monthly_premium": 10.0,
                "coverage_amount": 100000.0,
                "next_step": "close_now",
                "term_years": 20,
            },
        )
    )
    assert propose.success is True
    assert propose.data and propose.data.get("decision") == "accept_plan"

    # Lead is converted and call ended
    assert lead.converted is True
    assert sales_env.state.active_call is None

    # Search should not return converted lead
    search = sales_env.execute_tool(ToolCall(tool_name="crm.search_leads", arguments={"limit": 50}))
    assert search.success is True
    ids = {l["lead_id"] for l in (search.data or {}).get("leads", [])}
    assert str(lead_id) not in ids

    # Calling again should fail
    start_again = sales_env.execute_tool(
        ToolCall(tool_name="calling.start_call", arguments={"lead_id": str(lead_id)})
    )
    assert start_again.success is False
    assert start_again.data and start_again.data.get("converted") is True


def test_orchestrator_rejects_propose_plan_without_seller_message(
    default_config, mock_buyer_simulator
):
    """Benchmark hygiene: don't allow propose_plan tool-only turns."""
    from salesbench.core.types import ToolCall
    from salesbench.orchestrator.orchestrator import Orchestrator

    orch = Orchestrator(default_config)
    orch.set_buyer_simulator(mock_buyer_simulator)
    orch.reset()

    # Search and start a call
    r1 = orch.step(
        [ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})],
        seller_message="Searching.",
    )
    lead_id = r1.tool_results[0].data["leads"][0]["lead_id"]
    orch.step(
        [ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})],
        seller_message="Calling now.",
    )

    # Propose with no seller_message should be rejected (and not end the call)
    r3 = orch.step(
        [
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "monthly_premium": 10.0,
                    "coverage_amount": 100000.0,
                    "next_step": "schedule_followup",
                    "term_years": 20,
                },
            )
        ],
        seller_message=None,
    )
    assert r3.tool_results[0].success is False
    assert "requires a seller message" in (r3.tool_results[0].error or "")
    assert orch.env.state.active_call is not None


def test_orchestrator_stops_after_call_ended(default_config, accepting_buyer_simulator):
    """Benchmark hygiene: ignore extra tool calls after accept ends call."""
    from salesbench.core.types import ToolCall
    from salesbench.orchestrator.orchestrator import Orchestrator

    orch = Orchestrator(default_config)
    orch.set_buyer_simulator(accepting_buyer_simulator)
    orch.reset()

    # Search and start a call
    r1 = orch.step(
        [ToolCall(tool_name="crm.search_leads", arguments={"limit": 1})],
        seller_message="Searching.",
    )
    lead_id = r1.tool_results[0].data["leads"][0]["lead_id"]
    orch.step(
        [ToolCall(tool_name="calling.start_call", arguments={"lead_id": lead_id})],
        seller_message="Calling now.",
    )

    # Two propose_plan calls in same turn; first accept ends call, second should be ignored (break)
    r3 = orch.step(
        [
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "monthly_premium": 10.0,
                    "coverage_amount": 100000.0,
                    "next_step": "close_now",
                    "term_years": 20,
                },
            ),
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "monthly_premium": 12.0,
                    "coverage_amount": 120000.0,
                    "next_step": "close_now",
                    "term_years": 20,
                },
            ),
        ],
        seller_message="Here's the plan that fits what you said.",
    )
    assert len(r3.tool_results) == 1
    assert r3.tool_results[0].success is True
    assert orch.env.state.active_call is None


def test_orchestrator_blocks_duplicate_search_consecutive_turns(default_config):
    """Benchmark hygiene: prevent immediate search thrashing."""
    from salesbench.core.types import ToolCall
    from salesbench.orchestrator.orchestrator import Orchestrator

    orch = Orchestrator(default_config)
    orch.reset()

    r1 = orch.step(
        [ToolCall(tool_name="crm.search_leads", arguments={"temperature": "cold", "limit": 10})],
        seller_message="Searching cold.",
    )
    assert r1.tool_results[0].success is True

    r2 = orch.step(
        [ToolCall(tool_name="crm.search_leads", arguments={"temperature": "cold", "limit": 10})],
        seller_message="Searching cold again.",
    )
    assert r2.tool_results[0].success is False
    assert "Duplicate crm.search_leads" in (r2.tool_results[0].error or "")


def test_has_propose_plan_helper():
    """Verify _has_propose_plan correctly detects calling.propose_plan tool calls."""
    from salesbench.agents.seller_llm import _has_propose_plan
    from salesbench.core.types import ToolCall

    # Empty list
    assert _has_propose_plan([]) is False

    # No propose_plan
    tool_calls_without = [
        ToolCall(tool_name="crm.search_leads", arguments={"limit": 10}),
        ToolCall(tool_name="calling.start_call", arguments={"lead_id": "abc"}),
    ]
    assert _has_propose_plan(tool_calls_without) is False

    # Has propose_plan
    tool_calls_with = [
        ToolCall(tool_name="crm.search_leads", arguments={"limit": 10}),
        ToolCall(
            tool_name="calling.propose_plan",
            arguments={
                "plan_id": "TERM",
                "monthly_premium": 50,
                "coverage_amount": 100000,
                "next_step": "close_now",
                "term_years": 20,
            },
        ),
    ]
    assert _has_propose_plan(tool_calls_with) is True

    # Only propose_plan
    tool_calls_only = [
        ToolCall(
            tool_name="calling.propose_plan",
            arguments={
                "plan_id": "WHOLE",
                "monthly_premium": 100,
                "coverage_amount": 250000,
                "next_step": "schedule_followup",
            },
        ),
    ]
    assert _has_propose_plan(tool_calls_only) is True


def test_llm_seller_postprocess_dedupes_and_caps_propose_plan():
    """Regression: prevent LLM tool-call spam within a single action."""
    from salesbench.agents.seller_llm import _postprocess_tool_calls
    from salesbench.core.types import ToolCall

    tool_calls = [
        ToolCall(
            tool_name="calling.propose_plan",
            arguments={
                "plan_id": "TERM",
                "monthly_premium": 46,
                "coverage_amount": 500000,
                "next_step": "schedule_followup",
                "term_years": 20,
            },
            call_id="a",
        ),
        # Exact duplicate (different call_id) should be removed by dedupe
        ToolCall(
            tool_name="calling.propose_plan",
            arguments={
                "plan_id": "TERM",
                "monthly_premium": 46,
                "coverage_amount": 500000,
                "next_step": "schedule_followup",
                "term_years": 20,
            },
            call_id="b",
        ),
        # Different propose_plan should also be removed by the cap (only 1 per action)
        ToolCall(
            tool_name="calling.propose_plan",
            arguments={
                "plan_id": "TERM",
                "monthly_premium": 60,
                "coverage_amount": 750000,
                "next_step": "close_now",
                "term_years": 20,
            },
            call_id="c",
        ),
        # Other duplicated tools should be de-duped but still preserved if unique
        ToolCall(tool_name="products.quote_premium", arguments={"plan_id": "TERM", "age": 26}),
        ToolCall(tool_name="products.quote_premium", arguments={"plan_id": "TERM", "age": 26}),
    ]

    filtered = _postprocess_tool_calls(tool_calls)

    # One propose_plan total (cap) and one quote_premium total (dedupe)
    assert [tc.tool_name for tc in filtered] == [
        "calling.propose_plan",
        "products.quote_premium",
    ]
