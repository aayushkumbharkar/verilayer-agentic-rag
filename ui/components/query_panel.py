"""
VeriLayer — Phase 8: Query Panel Component.
Defines the user query textbox, the top_k slider, the verify button, and pipeline metadata output.
"""
from __future__ import annotations

import gradio as gr


def render_query_panel() -> tuple[gr.Textbox, gr.Slider, gr.Button, gr.Markdown]:
    """
    Renders the query input controls in the current Gradio layout block.
    Returns:
        tuple containing (query_input, top_k_slider, submit_btn, meta_output)
    """
    query_input = gr.Textbox(
        label="Your Question",
        placeholder="e.g. What are the payment obligations under clause 5.2?",
        lines=4,
        elem_id="query-box",
    )
    top_k_slider = gr.Slider(
        minimum=1, maximum=15, step=1, value=5,
        label="Documents to retrieve (top_k)",
    )
    submit_btn = gr.Button("🔍 Verify", variant="primary", size="lg")
    meta_output = gr.Markdown(value="", label="Pipeline Metadata")
    
    return query_input, top_k_slider, submit_btn, meta_output
