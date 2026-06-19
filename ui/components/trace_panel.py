"""
VeriLayer — Phase 8: Trace Panel Component.
Displays execution traces of agent stages (planner, grader, claims, verification, confidence, etc.).
"""
from __future__ import annotations

import gradio as gr


def format_trace(data: dict) -> str:
    """Format agent trace steps into readable markdown blocks."""
    trace = data.get("trace", [])
    if not trace:
        return "_No trace available._"

    result = []
    for t in trace:
        step = t.get("step", "?")
        details = t.get("details", "")
        latency = t.get("latency_ms", 0)
        label = f"🔹 {step.upper()} — {latency}ms"
        result.append(f"**{label}**\n{details}")
    return "\n\n".join(result)


def render_trace_panel() -> gr.Markdown:
    """Renders the markdown area for the trace list."""
    return gr.Markdown(value="_Trace will appear here after a query._")
