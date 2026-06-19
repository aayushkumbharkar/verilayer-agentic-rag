"""
VeriLayer — Phase 8: Claims Table Component.
Displays extracted claims, their verdicts (supported, unsupported, partial), and confidence levels.
"""
from __future__ import annotations

import gradio as gr

VERDICT_EMOJI = {
    "supported":   "🟢",
    "partial":     "🟡",
    "unsupported": "🔴",
}


def format_claims(data: dict) -> str:
    """Format the claims data into a Markdown table for display."""
    claims = data.get("claims", [])
    if not claims:
        return "_No claims extracted._"

    lines = ["| # | Claim | Verdict | Confidence |",
             "|---|-------|---------|------------|"]
    for i, c in enumerate(claims, 1):
        verdict = c.get("verdict", "unsupported")
        conf = float(c.get("confidence", 0.0))
        emoji = VERDICT_EMOJI.get(verdict, "❓")
        bar_filled = int(conf * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        text = c.get("text", "")[:120].replace("|", "\\|")
        lines.append(
            f"| {i} | {text} | {emoji} {verdict} | `{bar}` {conf:.0%} |"
        )
    return "\n".join(lines)


def render_claims_table() -> gr.Markdown:
    """Renders the markdown area for the claims table."""
    return gr.Markdown(value="_Claims will appear here._")
