"""
Vireo Audio – Agent Tools
All tools have detailed docstrings so the LLM understands how to use them.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from io import BytesIO, StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Annotated
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
import base64
import os

CLEAN_DIR = Path("cleaned_data")
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(exist_ok=True)

# Load data once at module level
refunds = pd.read_csv(CLEAN_DIR / "clean_refunds.csv", parse_dates=["created_at"])
agents_df = pd.read_csv(CLEAN_DIR / "agents.csv")


# ============================================================
# 1. STRUCTURED QUERY TOOLS
# ============================================================

@tool
def get_monthly_refund_by_reason(month: Optional[str] = None) -> str:
    """Get monthly refund summary grouped by reason code.

    Args:
        month: Month in format '2025-01'. Leave as None to get all months.

    Returns:
        str: Table of year_month, refund_reason_code, count, total_inr
    """
    df = refunds.copy()
    if month:
        df = df[df["year_month"] == month]

    result = (
        df.groupby(["year_month", "refund_reason_code"])
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .reset_index()
        .sort_values(["year_month", "total_inr"], ascending=[True, False])
    )
    return result.to_string(index=False)


@tool
def get_monthly_refund_by_agent(month: Optional[str] = None) -> str:
    """Get monthly refund summary grouped by agent.

    Args:
        month: Month in format '2025-01'. Leave as None to get all months.

    Returns:
        str: Table of year_month, agent_id, name, count, total_inr
    """
    df = refunds.copy()
    if month:
        df = df[df["year_month"] == month]

    result = (
        df.groupby(["year_month", "agent_id", "name"])
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .reset_index()
        .sort_values(["year_month", "total_inr"], ascending=[True, False])
    )
    return result.to_string(index=False)


@tool
def get_quarter_total(quarter: str) -> str:
    """Get total refunds and detailed breakdown for a specific quarter.

    Args:
        quarter: Quarter in format '2025Q1', '2025Q2', '2026Q1', etc.

    Returns:
        str: Total amount, ticket count, breakdown by reason and top agents.
    """
    df = refunds[refunds["quarter"] == quarter]
    if df.empty:
        return f"No data found for quarter: {quarter}"

    total = df["refund_amount_inr"].sum()
    count = len(df)

    by_reason = (
        df.groupby("refund_reason_code")
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .sort_values("total_inr", ascending=False)
        .reset_index()
    )

    by_agent = (
        df.groupby(["agent_id", "name"])
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .sort_values("total_inr", ascending=False)
        .head(10)
        .reset_index()
    )

    return (
        f"Quarter: {quarter}\n"
        f"Total Refund Amount: ₹{total:,.0f}\n"
        f"Number of Tickets: {count}\n\n"
        f"=== By Reason ===\n{by_reason.to_string(index=False)}\n\n"
        f"=== Top 10 Agents ===\n{by_agent.to_string(index=False)}"
    )


@tool
def get_agent_detail(agent_id: str) -> str:
    """Get detailed refund history for one specific agent.

    Args:
        agent_id: The agent_id (e.g. 'A001')

    Returns:
        str: Agent name, total refunded, monthly breakdown and reason breakdown.
    """
    df = refunds[refunds["agent_id"] == agent_id]
    if df.empty:
        return f"No refunds found for agent_id: {agent_id}"

    name = df["name"].iloc[0]
    total = df["refund_amount_inr"].sum()
    count = len(df)

    by_month = (
        df.groupby("year_month")
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .reset_index()
    )

    by_reason = (
        df.groupby("refund_reason_code")
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .sort_values("total_inr", ascending=False)
        .reset_index()
    )

    return (
        f"Agent: {name} ({agent_id})\n"
        f"Total Refunded: ₹{total:,.0f} across {count} tickets\n\n"
        f"=== By Month ===\n{by_month.to_string(index=False)}\n\n"
        f"=== By Reason ===\n{by_reason.to_string(index=False)}"
    )


@tool
def get_reason_breakdown(quarter: Optional[str] = None) -> str:
    """Get breakdown of refunds by reason code.

    Args:
        quarter: Optional quarter filter (e.g. '2025Q1'). None = all time.

    Returns:
        str: Table with reason, count, total_inr and percentage.
    """
    df = refunds.copy()
    if quarter:
        df = df[df["quarter"] == quarter]

    result = (
        df.groupby("refund_reason_code")
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .sort_values("total_inr", ascending=False)
        .reset_index()
    )
    result["pct"] = (result["total_inr"] / result["total_inr"].sum() * 100).round(1)
    return result.to_string(index=False)


@tool
def get_top_agents(n: int = 5, quarter: Optional[str] = None) -> str:
    """Get top N agents by total refund amount.

    Args:
        n: Number of top agents to return (default 5)
        quarter: Optional quarter filter (e.g. '2025Q1')

    Returns:
        str: Table of top agents with count and total_inr.
    """
    df = refunds.copy()
    if quarter:
        df = df[df["quarter"] == quarter]

    result = (
        df.groupby(["agent_id", "name", "site", "team"])
        .agg(count=("ticket_id", "count"), total_inr=("refund_amount_inr", "sum"))
        .sort_values("total_inr", ascending=False)
        .head(n)
        .reset_index()
    )
    return result.to_string(index=False)


@tool
def compare_quarters(quarter1: str, quarter2: str) -> str:
    """Compare two quarters side-by-side.

    Args:
        quarter1: First quarter (e.g. '2025Q1')
        quarter2: Second quarter (e.g. '2025Q2')

    Returns:
        str: Side-by-side comparison of total, count and average refund.
    """
    def summary(q):
        df = refunds[refunds["quarter"] == q]
        return {
            "total": df["refund_amount_inr"].sum(),
            "count": len(df),
            "avg": df["refund_amount_inr"].mean() if len(df) > 0 else 0
        }

    s1 = summary(quarter1)
    s2 = summary(quarter2)

    return (
        f"Comparison: {quarter1} vs {quarter2}\n\n"
        f"{'Metric':<20} {quarter1:>12} {quarter2:>12} {'Change':>12}\n"
        f"{'-'*60}\n"
        f"{'Total Refund (₹)':<20} {s1['total']:>12,.0f} {s2['total']:>12,.0f} {s2['total']-s1['total']:>+12,.0f}\n"
        f"{'Ticket Count':<20} {s1['count']:>12} {s2['count']:>12} {s2['count']-s1['count']:>+12}\n"
        f"{'Avg Refund (₹)':<20} {s1['avg']:>12,.0f} {s2['avg']:>12,.0f} {s2['avg']-s1['avg']:>+12,.0f}"
    )


# ============================================================
# 2. DYNAMIC VISUALIZATION TOOL (like your example)
# ============================================================

@tool
def generate_visualization(
    name: str,
    pandas_code: str,
    plotly_code: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Generate a dynamic visualization using pandas + Plotly.
    The visualization is automatically rendered on the frontend if successful.

    Args:
        name: Short name for the visualization (use underscores, no spaces). Example: top_agents_2025q1
        pandas_code: Python code that creates a DataFrame named 'df' from the global 'refunds' DataFrame.
                     You can filter, group, aggregate etc.
        plotly_code: Python code that creates a Plotly figure named 'fig' from the 'df' DataFrame.

    Returns:
        str: Success message or error message.

    ## Assumptions
    The following are already available:
    - A global DataFrame named `refunds` (the cleaned refund data)
    - import pandas as pd
    - import plotly.express as px
    - import plotly.graph_objects as go
    - import plotly.io as pio

    ## Example
    User asks: "Show me top 5 agents by refund amount in 2025Q1"

    pandas_code = '''
    df = (refunds[refunds["quarter"] == "2025Q1"]
          .groupby("name")["refund_amount_inr"]
          .sum()
          .sort_values(ascending=False)
          .head(5)
          .reset_index())
    '''

    plotly_code = '''
    fig = px.bar(df, x="name", y="refund_amount_inr",
                 title="Top 5 Agents by Refund Amount – 2025Q1",
                 labels={"name": "Agent", "refund_amount_inr": "Refund Amount (₹)"})
    fig.update_layout(xaxis_tickangle=-45)
    '''
    """
    file_path = VIS_DIR / f"{name}.json"

    # Build the full executable code
    full_code = f"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# === User pandas code ===
{pandas_code}

# === User plotly code ===
{plotly_code}

# Save figure
if 'fig' in locals() or 'fig' in globals():
    fig_json = pio.to_json(fig)
    with open(r'{file_path}', 'w') as f:
        f.write(fig_json)
"""

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    exec_globals = {
        "refunds": refunds,
        "pd": pd,
        "px": px,
        "go": go,
        "pio": pio,
    }

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(full_code, exec_globals)

        if file_path.exists():
            with open(file_path, "r") as f:
                fig_json = f.read()

            # Return a special message that the frontend can detect
            return f"VISUALIZATION_SUCCESS::{name}::{fig_json}"
        else:
            return f"Error: Figure was not created.\nSTDERR:\n{stderr_capture.getvalue()}"

    except Exception as e:
        return f"Error executing visualization code: {str(e)}\n\nSTDERR:\n{stderr_capture.getvalue()}"


# ============================================================
# Export all tools
# ============================================================
ALL_TOOLS = [
    get_monthly_refund_by_reason,
    get_monthly_refund_by_agent,
    get_quarter_total,
    get_agent_detail,
    get_reason_breakdown,
    get_top_agents,
    compare_quarters,
    generate_visualization,
]