"""
VeriLayer — Phase 8: Metrics Panel Component.
Displays aggregated evaluation and pipeline health metrics.
"""
from __future__ import annotations

import gradio as gr


def format_metrics(data: dict) -> str:
    """Format aggregate pipeline evaluation metrics into a Markdown table."""
    if not data or data.get("total_queries", 0) == 0:
        return "_No data yet — run a query first._"

    total = data.get("total_queries", 0)
    hall = data.get("hallucination_rate", 0.0)
    conf = data.get("avg_confidence", 0.0)
    retry = data.get("retry_rate", 0.0)
    latency = data.get("avg_latency_ms", 0.0)
    verified = data.get("verified_rate", 0.0)
    partial = data.get("partial_rate", 0.0)
    unsafe = data.get("unsafe_rate", 0.0)

    return (
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total Queries | **{total}** |\n"
        f"| 🟢 Verified Rate | **{verified:.1%}** |\n"
        f"| 🟡 Partial Rate | **{partial:.1%}** |\n"
        f"| 🔴 Unsafe Rate | **{unsafe:.1%}** |\n"
        f"| Hallucination Rate | **{hall:.1%}** |\n"
        f"| Avg Confidence | **{conf:.1%}** |\n"
        f"| Retry Rate | **{retry:.1%}** |\n"
        f"| Avg Latency | **{latency:.0f}ms** |\n"
    )


def render_metrics_panel() -> tuple[gr.Markdown, gr.Button]:
    """Renders the metrics markdown display and refresh button."""
    metrics_output = gr.Markdown(value=format_metrics({}))
    refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
    return metrics_output, refresh_btn
