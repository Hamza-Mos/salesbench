"""Gradio leaderboard for SalesBench.

A simple, HuggingFace Spaces-compatible leaderboard that displays
benchmark results from JSON files.

Usage:
    python -m salesbench.ui.app

Or deploy to HuggingFace Spaces by pushing this file along with
requirements.txt containing 'gradio>=4.0.0'.
"""

import json
import os
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio is required for the UI. Install with: pip install 'salesbench[ui]'")

from salesbench.storage.json_writer import JSONResultsWriter

# ============================================================================
# Trace Formatting Helpers
# ============================================================================


def format_tool_call(tool_call: dict) -> str:
    """Format a tool call for display."""
    name = tool_call.get("tool_name", tool_call.get("name", "unknown"))
    args = tool_call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
    return f"**{name}**\n```json\n{args_str}\n```"


def format_tool_result(result: dict) -> str:
    """Format a tool result for display."""
    success = result.get("success", True)
    icon = "✅" if success else "❌"
    output = result.get("data", result.get("output", result.get("result", "")))
    if isinstance(output, dict):
        output = json.dumps(output, indent=2)
    # Truncate long outputs
    if len(str(output)) > 500:
        output = str(output)[:500] + "..."
    return f"{icon} **Result:**\n```\n{output}\n```"


def format_turn_markdown(turn: dict, turn_number: int) -> str:
    """Format a complete turn for display."""
    parts = [f"## Turn {turn_number}"]

    # Seller message
    seller_msg = turn.get("seller_message", turn.get("agent_message", ""))
    if seller_msg:
        parts.append("### Seller Message")
        parts.append(f"> {seller_msg.replace(chr(10), chr(10) + '> ')}")

    # Tool calls
    tool_calls = turn.get("tool_calls", [])
    if tool_calls:
        parts.append("### Tool Calls")
        for tc in tool_calls:
            parts.append(format_tool_call(tc))

    # Tool results
    tool_results = turn.get("tool_results", [])
    if tool_results:
        parts.append("### Tool Results")
        for tr in tool_results:
            parts.append(format_tool_result(tr))

    # Buyer response
    buyer_msg = turn.get("buyer_response", turn.get("environment_response", ""))
    if buyer_msg:
        parts.append("### Buyer Response")
        parts.append(f"> {buyer_msg.replace(chr(10), chr(10) + '> ')}")

    # Score
    score = turn.get("score", 0)
    if score != 0:
        parts.append(f"### Score: **{score}**")

    return "\n\n".join(parts)


TURNS_PER_PAGE = 10  # Limit turns per page to prevent browser crashes


def format_all_turns_markdown(trajectory: list[dict], page: int = 1) -> str:
    """Format turns in a trajectory for display with pagination.

    Args:
        trajectory: List of turn dictionaries.
        page: Page number (1-indexed).

    Returns:
        Formatted markdown string for the current page.
    """
    if not trajectory:
        return "No turns to display."

    total_turns = len(trajectory)
    total_pages = (total_turns + TURNS_PER_PAGE - 1) // TURNS_PER_PAGE
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * TURNS_PER_PAGE
    end_idx = min(start_idx + TURNS_PER_PAGE, total_turns)

    parts = [f"# Conversation Turns {start_idx + 1}-{end_idx} of {total_turns}\n"]
    parts.append(f"**Page {page} of {total_pages}**\n")
    parts.append("---\n")

    for i in range(start_idx, end_idx):
        parts.append(format_turn_markdown(trajectory[i], i + 1))
        parts.append("\n---\n")

    return "\n".join(parts)


def create_leaderboard(
    results_dir: Optional[str] = None,
    title: str = "SalesBench Leaderboard",
    refresh_interval: int = 60,
) -> gr.Blocks:
    """Create Gradio leaderboard interface.

    Args:
        results_dir: Directory containing JSON result files.
                    Defaults to 'results' in current directory.
        title: Title displayed on the leaderboard.
        refresh_interval: Seconds between data refreshes.

    Returns:
        Gradio Blocks app.
    """
    if results_dir is None:
        results_dir = os.getenv("SALESBENCH_RESULTS_DIR", "results")

    writer = JSONResultsWriter(results_dir)

    def load_leaderboard_data():
        """Load and format data for the leaderboard table."""
        data = writer.get_leaderboard_data()

        # Sort by mean score (descending)
        data = sorted(data, key=lambda x: x.get("mean_score", 0), reverse=True)

        # Format for display
        rows = []
        for i, r in enumerate(data, 1):
            # Get token usage
            total_tokens = r.get("total_tokens", 0)
            total_cost = r.get("total_cost", 0)

            rows.append(
                [
                    i,  # Rank
                    r.get("model", "unknown"),
                    f"{r.get('mean_score', 0):.1f}",
                    f"{r.get('acceptance_rate', 0):.1%}",
                    r.get("total_dnc_violations", 0),
                    r.get("episodes", 0),
                    f"{total_tokens:,}" if total_tokens else "N/A",
                    f"${total_cost:.2f}" if total_cost else "N/A",
                    f"{r.get('duration_seconds', 0):.0f}s",
                    r.get("timestamp", "")[:10] if r.get("timestamp") else "",
                ]
            )

        return rows

    def get_run_choices():
        """Get list of available runs for dropdown."""
        data = writer.get_leaderboard_data()
        choices = []
        for r in data:
            # Format timestamp: "2024-01-20T14:30:45" -> "2024-01-20 14:30"
            timestamp = r.get("timestamp", "")[:16].replace("T", " ")
            model = r.get("model", "unknown")
            episodes = r.get("episodes", 0)
            label = f"[{timestamp}] {model} ({episodes} eps)"
            choices.append((label, r.get("benchmark_id", "")))
        return choices

    def load_result_details(benchmark_id: str) -> str:
        """Load detailed results for a specific benchmark."""
        if not benchmark_id:
            return "Select a benchmark run from the dropdown above to view details."

        result = writer.get_result_by_id(benchmark_id)
        if not result:
            return f"Benchmark {benchmark_id} not found."

        config = result.get("config", {})
        aggregate = result.get("aggregate_metrics", {})
        token_usage = aggregate.get("total_token_usage", {})
        cost_breakdown = aggregate.get("total_cost_breakdown", {})

        # Token counts
        seller_input = token_usage.get("seller_input_tokens", 0)
        seller_output = token_usage.get("seller_output_tokens", 0)
        buyer_input = token_usage.get("buyer_input_tokens", 0)
        buyer_output = token_usage.get("buyer_output_tokens", 0)
        total_tokens = seller_input + seller_output + buyer_input + buyer_output

        # Costs
        total_cost = cost_breakdown.get("total_cost", 0)
        seller_cost = cost_breakdown.get("seller_input_cost", 0) + cost_breakdown.get(
            "seller_output_cost", 0
        )
        buyer_cost = cost_breakdown.get("buyer_input_cost", 0) + cost_breakdown.get(
            "buyer_output_cost", 0
        )

        # Time metrics
        action_minutes = aggregate.get("mean_action_based_minutes", 0)
        token_minutes = aggregate.get("mean_token_based_minutes", 0)
        conversation_turns = aggregate.get("mean_conversation_turns", 0)

        # Lead outcome metrics
        total_leads = aggregate.get("total_leads", 0)
        leads_contacted = aggregate.get("total_leads_contacted", 0)
        leads_converted = aggregate.get("total_leads_converted", 0)
        leads_dnc = aggregate.get("total_leads_dnc", 0)
        leads_uncontacted = aggregate.get("total_leads_uncontacted", 0)
        lead_contact_rate = aggregate.get("lead_contact_rate", 0)

        details = f"""## Benchmark: {result.get('benchmark_id', 'unknown')}

### Configuration
| Setting | Value |
|---------|-------|
| Seller Model | {config.get('seller_model', 'unknown')} |
| Buyer Model | {config.get('buyer_model', 'unknown')} |
| Domain | {config.get('domain', 'insurance')} |
| Episodes | {result.get('completed_episodes', 0)} |
| Mode | {config.get('mode', 'unknown')} |
| Parallelism | {config.get('parallelism', 1)} |

### Performance Results
| Metric | Value |
|--------|-------|
| Mean Score | {aggregate.get('mean_score', 0):.2f} (±{aggregate.get('std_score', 0):.2f}) |
| Score Range | {aggregate.get('min_score', 0):.1f} - {aggregate.get('max_score', 0):.1f} |
| Median Score | {aggregate.get('median_score', 0):.2f} |
| Acceptance Rate | {aggregate.get('mean_acceptance_rate', 0):.1%} |
| Conversion Rate | {aggregate.get('conversion_rate', 0):.1%} |
| Mean Calls | {aggregate.get('mean_calls', 0):.1f} |
| Mean Offers | {aggregate.get('mean_offers', 0):.1f} |
| Total Accepts | {aggregate.get('total_accepts', 0)} |
| Episodes with Accepts | {aggregate.get('episodes_with_accepts', 0)}/{aggregate.get('n_episodes', 0)} |
| Buyer Ended Calls | {aggregate.get('total_end_calls', 0)} |
| Episode Success Rate | {aggregate.get('episode_success_rate', 0):.1%} |
| DNC Violations | {aggregate.get('total_dnc_violations', 0)} |

### Lead Outcomes (All Episodes)
| Metric | Value |
|--------|-------|
| Total Leads | {total_leads} |
| Leads Contacted | {leads_contacted} ({lead_contact_rate:.1%}) |
| Leads Converted | {leads_converted} |
| Leads DNC | {leads_dnc} |
| Leads Uncontacted | {leads_uncontacted} |

### Time Metrics
| Metric | Value |
|--------|-------|
| Action-Based Minutes | {action_minutes:.1f} |
| Token-Based Minutes | {token_minutes:.1f} |
| Conversation Turns | {conversation_turns:.1f} |
| Mean Episode Duration | {aggregate.get('mean_episode_duration', 0):.1f}s |
| Wall-Clock Duration | {result.get('duration_seconds', 0):.1f}s |

### Token Usage (All Episodes)
| Component | Input | Output | Total |
|-----------|-------|--------|-------|
| Seller | {seller_input:,} | {seller_output:,} | {seller_input + seller_output:,} |
| Buyer | {buyer_input:,} | {buyer_output:,} | {buyer_input + buyer_output:,} |
| **Total** | **{seller_input + buyer_input:,}** | **{seller_output + buyer_output:,}** | **{total_tokens:,}** |

### Cost Breakdown
| Component | Cost |
|-----------|------|
| Seller | ${seller_cost:.4f} |
| Buyer | ${buyer_cost:.4f} |
| **Total** | **${total_cost:.4f}** |

### Timing
- **Started**: {result.get('started_at', 'unknown')}
- **Completed**: {result.get('ended_at', 'unknown')}
"""
        return details

    def load_episode_table(benchmark_id: str):
        """Load episode-level results for a specific benchmark."""
        if not benchmark_id:
            return []

        result = writer.get_result_by_id(benchmark_id)
        if not result:
            return []

        episodes = result.get("episode_results", [])
        rows = []
        for ep in episodes:
            metrics = ep.get("metrics", {})
            token_usage = ep.get("token_usage", {})
            cost_breakdown = ep.get("cost_breakdown", {})

            # Calculate totals
            total_tokens = (
                token_usage.get("seller_input_tokens", 0)
                + token_usage.get("seller_output_tokens", 0)
                + token_usage.get("buyer_input_tokens", 0)
                + token_usage.get("buyer_output_tokens", 0)
            )
            total_cost = cost_breakdown.get("total_cost", 0)

            # Lead outcome stats (compact format: contacted/total)
            total_leads = metrics.get("total_leads", 0)
            leads_contacted = metrics.get("leads_contacted", 0)
            leads_converted = metrics.get("leads_converted", 0)
            leads_str = f"{leads_contacted}/{total_leads}" if total_leads > 0 else "N/A"

            rows.append(
                [
                    ep.get("episode_index", 0) + 1,
                    ep.get("seed", ""),
                    f"{ep.get('final_score', ep.get('score', 0)):.1f}",
                    metrics.get("accepted_offers", 0),
                    metrics.get("rejected_offers", 0),
                    ep.get("dnc_violations", 0),
                    metrics.get("total_calls", 0),
                    leads_str,
                    leads_converted,
                    ep.get("total_turns", 0),
                    f"{metrics.get('action_based_minutes', 0):.0f}",
                    f"{total_tokens:,}" if total_tokens else "0",
                    f"${total_cost:.3f}" if total_cost else "$0",
                    ep.get("termination_reason", "")[:30],
                ]
            )
        return rows

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(),
        css="""
        .leaderboard-table { font-size: 14px; }
        .rank-1 { background-color: #ffd700 !important; }
        .rank-2 { background-color: #c0c0c0 !important; }
        .rank-3 { background-color: #cd7f32 !important; }
        """,
    ) as demo:
        gr.Markdown(f"""
# {title}

Benchmarking AI agents on sales conversations.

[GitHub](https://github.com/Hamza-Mos/salesbench) | [Documentation](#docs)
        """)

        with gr.Tabs():
            with gr.Tab("Leaderboard"):
                leaderboard_table = gr.Dataframe(
                    value=load_leaderboard_data,
                    headers=[
                        "Rank",
                        "Model",
                        "Score",
                        "Accept%",
                        "DNC",
                        "Episodes",
                        "Tokens",
                        "Cost",
                        "Duration",
                        "Date",
                    ],
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "str",
                        "number",
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    every=refresh_interval,
                    interactive=False,
                    elem_classes=["leaderboard-table"],
                )

                gr.Markdown("""
### Metrics Explained
- **Score**: Composite score based on sales outcomes (higher is better)
- **Accept%**: Percentage of offers accepted by buyers
- **Tokens**: Total tokens used across all episodes (seller + buyer)
- **Cost**: Exact API cost based on token usage and model pricing
- **Duration**: Total wall-clock time for benchmark run
                """)

            with gr.Tab("Run Details"):
                gr.Markdown("### Select a benchmark run to view detailed results")

                run_dropdown = gr.Dropdown(
                    choices=get_run_choices(),
                    label="Select Benchmark Run",
                    value=None,
                    interactive=True,
                )

                refresh_btn = gr.Button("🔄 Refresh Run List", size="sm")

                details_output = gr.Markdown(
                    value="Select a benchmark run from the dropdown above to view details."
                )

                gr.Markdown("### Episode Results")
                episode_table = gr.Dataframe(
                    value=[],
                    headers=[
                        "Ep",
                        "Seed",
                        "Score",
                        "Accepts",
                        "Rejects",
                        "DNC",
                        "Calls",
                        "Leads",
                        "Converted",
                        "Turns",
                        "Minutes",
                        "Tokens",
                        "Cost",
                        "Termination",
                    ],
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "number",
                        "number",
                        "number",
                        "number",
                        "str",
                        "number",
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=False,
                )

                # Wire up the dropdown to update details and episode table
                run_dropdown.change(
                    fn=load_result_details,
                    inputs=[run_dropdown],
                    outputs=[details_output],
                )
                run_dropdown.change(
                    fn=load_episode_table,
                    inputs=[run_dropdown],
                    outputs=[episode_table],
                )

                # Refresh button updates dropdown choices
                def refresh_choices():
                    return gr.Dropdown(choices=get_run_choices())

                refresh_btn.click(
                    fn=refresh_choices,
                    inputs=[],
                    outputs=[run_dropdown],
                )

            with gr.Tab("Trace Viewer"):
                gr.Markdown("### View Conversation Traces")
                gr.Markdown(
                    "*Traces are only available for benchmark runs executed with the `-v` (verbose) flag.*"
                )

                trace_run_dropdown = gr.Dropdown(
                    choices=get_run_choices(),
                    label="Select Benchmark Run",
                    value=None,
                    interactive=True,
                )

                trace_refresh_btn = gr.Button("Refresh Run List", size="sm")

                traces_status = gr.Markdown("Select a benchmark run to check for traces.")

                episode_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Episode",
                    value=None,
                    interactive=True,
                    visible=False,
                )

                # Toggle for viewing multiple turns at once (paginated)
                show_all_turns = gr.Checkbox(
                    label="Show multiple turns per page (10 turns/page)",
                    value=False,
                    visible=False,
                )

                turn_slider = gr.Slider(
                    minimum=1,
                    maximum=1,
                    step=1,
                    value=1,
                    label="Turn",
                    visible=False,
                )

                # Page slider for paginated view
                page_slider = gr.Slider(
                    minimum=1,
                    maximum=1,
                    step=1,
                    value=1,
                    label="Page",
                    visible=False,
                )

                with gr.Row(visible=False) as nav_row:
                    prev_btn = gr.Button("Previous Turn", size="sm")
                    next_btn = gr.Button("Next Turn", size="sm")

                with gr.Row(visible=False) as page_nav_row:
                    prev_page_btn = gr.Button("Previous Page", size="sm")
                    next_page_btn = gr.Button("Next Page", size="sm")

                turn_display = gr.Markdown("")

                # State to store current trajectory, benchmark ID, and page
                current_trajectory = gr.State([])
                current_benchmark_id = gr.State(None)
                current_page = gr.State(1)

                def check_traces_available(benchmark_id):
                    """Check if traces exist and load episode list (metadata only)."""
                    if not benchmark_id:
                        return (
                            "Select a benchmark run to check for traces.",
                            gr.Dropdown(choices=[], visible=False),
                            gr.Checkbox(visible=False),
                            gr.Slider(visible=False),
                            gr.Slider(visible=False),
                            gr.Row(visible=False),
                            gr.Row(visible=False),
                            "",
                            [],
                            None,
                            1,
                        )

                    has_traces = writer.has_traces(benchmark_id)
                    if not has_traces:
                        return (
                            "**No traces available** for this run. Run benchmark with `-v` flag to generate traces.",
                            gr.Dropdown(choices=[], visible=False),
                            gr.Checkbox(visible=False),
                            gr.Slider(visible=False),
                            gr.Slider(visible=False),
                            gr.Row(visible=False),
                            gr.Row(visible=False),
                            "",
                            [],
                            None,
                            1,
                        )

                    # Load only episode metadata, not full trajectories (memory efficient)
                    episode_metadata = writer.load_episode_metadata(benchmark_id)
                    if not episode_metadata:
                        return (
                            "**Error loading traces.**",
                            gr.Dropdown(choices=[], visible=False),
                            gr.Checkbox(visible=False),
                            gr.Slider(visible=False),
                            gr.Slider(visible=False),
                            gr.Row(visible=False),
                            gr.Row(visible=False),
                            "",
                            [],
                            None,
                            1,
                        )

                    episode_choices = []
                    for ep in episode_metadata:
                        idx = ep.get("episode_index", 0)
                        seed = ep.get("seed", "")
                        turn_count = ep.get("turn_count", 0)
                        label = f"Episode {idx + 1} (seed: {seed}, {turn_count} turns)"
                        episode_choices.append((label, idx))

                    return (
                        f"**Traces available** - {len(episode_metadata)} episodes",
                        gr.Dropdown(choices=episode_choices, visible=True, value=None),
                        gr.Checkbox(visible=False, value=False),
                        gr.Slider(visible=False),
                        gr.Slider(visible=False),
                        gr.Row(visible=False),
                        gr.Row(visible=False),
                        "",
                        [],
                        benchmark_id,
                        1,
                    )

                def load_episode_trajectory(episode_index, benchmark_id, show_all):
                    """Load trajectory for selected episode on-demand."""
                    if benchmark_id is None or episode_index is None:
                        return (
                            gr.Checkbox(visible=False),
                            gr.Slider(visible=False),
                            gr.Slider(visible=False),
                            gr.Row(visible=False),
                            gr.Row(visible=False),
                            "",
                            [],
                            1,
                        )

                    # Load only this episode's trajectory from disk
                    trajectory = writer.load_episode_trajectory(benchmark_id, episode_index)

                    if not trajectory:
                        return (
                            gr.Checkbox(visible=False),
                            gr.Slider(visible=False),
                            gr.Slider(visible=False),
                            gr.Row(visible=False),
                            gr.Row(visible=False),
                            "No turns in this episode.",
                            [],
                            1,
                        )

                    max_turns = len(trajectory)
                    total_pages = (max_turns + TURNS_PER_PAGE - 1) // TURNS_PER_PAGE

                    if show_all:
                        # Show paginated view (10 turns per page)
                        display = format_all_turns_markdown(trajectory, page=1)
                        return (
                            gr.Checkbox(visible=True, value=True),
                            gr.Slider(visible=False),
                            gr.Slider(minimum=1, maximum=total_pages, value=1, visible=True),
                            gr.Row(visible=False),
                            gr.Row(visible=True),
                            display,
                            trajectory,
                            1,
                        )
                    else:
                        # Show single turn with navigation
                        first_turn_display = format_turn_markdown(trajectory[0], 1)
                        return (
                            gr.Checkbox(visible=True, value=False),
                            gr.Slider(minimum=1, maximum=max_turns, value=1, visible=True),
                            gr.Slider(visible=False),
                            gr.Row(visible=True),
                            gr.Row(visible=False),
                            first_turn_display,
                            trajectory,
                            1,
                        )

                def toggle_view_mode(show_all, trajectory):
                    """Toggle between single-turn and paginated view."""
                    if not trajectory:
                        return (
                            gr.Slider(visible=False),
                            gr.Slider(visible=False),
                            gr.Row(visible=False),
                            gr.Row(visible=False),
                            "",
                            1,
                        )

                    max_turns = len(trajectory)
                    total_pages = (max_turns + TURNS_PER_PAGE - 1) // TURNS_PER_PAGE

                    if show_all:
                        # Show paginated view (10 turns per page)
                        display = format_all_turns_markdown(trajectory, page=1)
                        return (
                            gr.Slider(visible=False),
                            gr.Slider(minimum=1, maximum=total_pages, value=1, visible=True),
                            gr.Row(visible=False),
                            gr.Row(visible=True),
                            display,
                            1,
                        )
                    else:
                        # Show single turn with navigation
                        first_turn_display = format_turn_markdown(trajectory[0], 1)
                        return (
                            gr.Slider(minimum=1, maximum=max_turns, value=1, visible=True),
                            gr.Slider(visible=False),
                            gr.Row(visible=True),
                            gr.Row(visible=False),
                            first_turn_display,
                            1,
                        )

                def display_turn(turn_number, trajectory):
                    """Display a specific turn."""
                    if not trajectory or turn_number < 1:
                        return ""
                    idx = int(turn_number) - 1
                    if idx >= len(trajectory):
                        idx = len(trajectory) - 1
                    return format_turn_markdown(trajectory[idx], idx + 1)

                def prev_turn(current, trajectory):
                    """Navigate to previous turn."""
                    if not trajectory:
                        return 1, ""
                    new_turn = max(1, int(current) - 1)
                    return new_turn, display_turn(new_turn, trajectory)

                def next_turn(current, trajectory):
                    """Navigate to next turn."""
                    if not trajectory:
                        return 1, ""
                    new_turn = min(len(trajectory), int(current) + 1)
                    return new_turn, display_turn(new_turn, trajectory)

                def display_page(page_number, trajectory):
                    """Display a specific page of turns."""
                    if not trajectory:
                        return ""
                    return format_all_turns_markdown(trajectory, page=int(page_number))

                def prev_page(current, trajectory):
                    """Navigate to previous page."""
                    if not trajectory:
                        return 1, "", 1
                    new_page = max(1, int(current) - 1)
                    return new_page, display_page(new_page, trajectory), new_page

                def next_page(current, trajectory):
                    """Navigate to next page."""
                    if not trajectory:
                        return 1, "", 1
                    total_pages = (len(trajectory) + TURNS_PER_PAGE - 1) // TURNS_PER_PAGE
                    new_page = min(total_pages, int(current) + 1)
                    return new_page, display_page(new_page, trajectory), new_page

                def refresh_trace_choices():
                    return gr.Dropdown(choices=get_run_choices())

                # Event wiring
                trace_run_dropdown.change(
                    fn=check_traces_available,
                    inputs=[trace_run_dropdown],
                    outputs=[
                        traces_status,
                        episode_dropdown,
                        show_all_turns,
                        turn_slider,
                        page_slider,
                        nav_row,
                        page_nav_row,
                        turn_display,
                        current_trajectory,
                        current_benchmark_id,
                        current_page,
                    ],
                )

                episode_dropdown.change(
                    fn=load_episode_trajectory,
                    inputs=[episode_dropdown, current_benchmark_id, show_all_turns],
                    outputs=[
                        show_all_turns,
                        turn_slider,
                        page_slider,
                        nav_row,
                        page_nav_row,
                        turn_display,
                        current_trajectory,
                        current_page,
                    ],
                )

                show_all_turns.change(
                    fn=toggle_view_mode,
                    inputs=[show_all_turns, current_trajectory],
                    outputs=[
                        turn_slider,
                        page_slider,
                        nav_row,
                        page_nav_row,
                        turn_display,
                        current_page,
                    ],
                )

                turn_slider.change(
                    fn=display_turn,
                    inputs=[turn_slider, current_trajectory],
                    outputs=[turn_display],
                )

                def display_page_and_update_state(page_number, trajectory):
                    """Display page and return page number for state sync."""
                    return display_page(page_number, trajectory), int(page_number)

                page_slider.change(
                    fn=display_page_and_update_state,
                    inputs=[page_slider, current_trajectory],
                    outputs=[turn_display, current_page],
                )

                prev_btn.click(
                    fn=prev_turn,
                    inputs=[turn_slider, current_trajectory],
                    outputs=[turn_slider, turn_display],
                )

                next_btn.click(
                    fn=next_turn,
                    inputs=[turn_slider, current_trajectory],
                    outputs=[turn_slider, turn_display],
                )

                prev_page_btn.click(
                    fn=prev_page,
                    inputs=[current_page, current_trajectory],
                    outputs=[page_slider, turn_display, current_page],
                )

                next_page_btn.click(
                    fn=next_page,
                    inputs=[current_page, current_trajectory],
                    outputs=[page_slider, turn_display, current_page],
                )

                trace_refresh_btn.click(
                    fn=refresh_trace_choices,
                    inputs=[],
                    outputs=[trace_run_dropdown],
                )

            with gr.Tab("How to Submit"):
                gr.Markdown("""
## Submitting Results

### 1. Run the Benchmark

```bash
# Install salesbench
pip install salesbench

# Run benchmark with your model
salesbench run-benchmark \\
    --seller-model YOUR_MODEL \\
    --domain insurance \\
    --mode production \\
    -n 100
```

### 2. Find Your Results

Results are saved to the `results/` directory as JSON files:
```
results/
  bench_abc123_20240115_143022.json
```

### 3. Submit

**Option A: GitHub PR**
1. Fork the repository
2. Add your results file to `results/`
3. Submit a pull request

**Option B: Upload**
Contact us to add your results directly.

### Requirements
- Minimum 100 episodes for official rankings
- Use `--mode production` for comparable results
- Include model name and configuration
                """)

            with gr.Tab("About"):
                gr.Markdown("""
## About SalesBench

SalesBench is a benchmark for evaluating AI agents on sales conversations.
It provides a standardized environment where agents must:

1. **Search and prioritize leads** using CRM tools
2. **Engage in sales calls** with simulated buyers
3. **Understand customer needs** and propose appropriate products
4. **Handle objections** and close deals

### Domains

Currently supported domains:
- **Insurance**: Life insurance cold-calling

### Evaluation

Agents are scored on:
- Successful sales (accepted offers)
- Sales efficiency (time and calls used)
- Customer satisfaction (objection handling)
- Compliance (DNC list violations)

### Technical Details

- Built on the [Verifiers](https://github.com/primeintellect-ai/verifiers) framework
- Deterministic persona generation for reproducibility
- Multi-provider LLM support (OpenAI, Anthropic, etc.)
                """)

        gr.Markdown(
            """
---
*Last updated every {} seconds*
        """.format(refresh_interval)
        )

    return demo


def main():
    """Launch the leaderboard app."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch SalesBench Leaderboard")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing result files (default: results/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )

    args = parser.parse_args()

    demo = create_leaderboard(results_dir=args.results_dir)
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
