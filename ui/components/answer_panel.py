"""
VeriLayer — Phase 8: Answer Panel Component.
Renders the final generated answer markdown block along with the status badge and confidence scores.
"""
from __future__ import annotations

import gradio as gr

STATUS_EMOJI = {
    "verified": "🟢 Verified",
    "partial":  "🟡 Partial",
    "unsafe":   "🔴 Unsafe",
}


def format_answer(data: dict) -> str:
    """Format the raw API verification result into the markdown structure for the answer panel."""
    status = data.get("status", "unsafe")
    confidence = data.get("confidence", 0.0)
    answer = data.get("final_answer", "No answer generated.")
    badge = STATUS_EMOJI.get(status, status)
    conf_pct = f"{confidence * 100:.1f}%"
    return (
        f"### {badge} &nbsp;&nbsp; Confidence: **{conf_pct}**\n\n"
        f"---\n\n"
        f"{answer}"
    )


def render_answer_panel() -> gr.Markdown:
    """Renders the markdown area where the formatted answer is outputted."""
    return gr.Markdown(
        value="_Submit a query to see the verified answer._",
        label="Answer",
    )
