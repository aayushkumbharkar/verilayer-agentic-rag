"""
VeriLayer — Phase 8: Gradio UI.

A rich Gradio Blocks interface that imports modular components:
  - Query Input panel with top_k slider
  - Answer Panel: final answer + confidence badge + status chip
  - Claims Table: claim text, verdict (🟢/🔴/🟡), confidence bar
  - Sources Panel: document ID, chunk text, relevance score
  - Trace Panel: collapsible accordion per agent step with latency
  - Metrics Panel: live hallucination rate, avg confidence, retry rate
"""
from __future__ import annotations

import asyncio
import gradio as gr

from ui.utils.api_client import call_verify, call_metrics
from ui.components.query_panel import render_query_panel
from ui.components.answer_panel import render_answer_panel, format_answer
from ui.components.claims_table import render_claims_table, format_claims
from ui.components.sources_panel import render_sources_panel, format_sources
from ui.components.trace_panel import render_trace_panel, format_trace
from ui.components.metrics_panel import render_metrics_panel, format_metrics


# ── Synchronous API call (Gradio runs sync functions) ─────────────────────────

def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Main UI handlers ───────────────────────────────────────────────────────────

def on_submit(query: str, top_k: int):
    """Called when the user clicks Submit."""
    if not query.strip():
        empty = {
            "status": "unsafe",
            "confidence": 0.0,
            "final_answer": "Please enter a query.",
            "claims": [],
            "trace": [],
            "metadata": {},
        }
        return (
            format_answer(empty),
            format_claims(empty),
            format_sources(empty),
            "",
            "_",
        )

    try:
        data = _run_async(call_verify(query, top_k))
    except Exception as exc:
        err = {
            "status": "unsafe",
            "confidence": 0.0,
            "final_answer": f"❌ Error: {exc}",
            "claims": [],
            "trace": [],
            "metadata": {},
        }
        return (
            format_answer(err),
            format_claims(err),
            format_sources(err),
            f"Error: {exc}",
            "_",
        )

    trace_md = format_trace(data)
    meta = data.get("metadata", {})
    meta_str = (
        f"📄 Retrieved: **{meta.get('retrieval_docs', 0)}** docs | "
        f"🔄 Retries: **{meta.get('retries', 0)}** | "
        f"⏱ Latency: **{meta.get('latency_total_ms', 0)}ms**"
    )

    return (
        format_answer(data),
        format_claims(data),
        format_sources(data),
        trace_md,
        meta_str,
    )


def on_metrics_refresh():
    try:
        data = _run_async(call_metrics())
        return format_metrics(data)
    except Exception as exc:
        return f"❌ Could not fetch metrics: {exc}"


# ── Gradio Blocks UI ───────────────────────────────────────────────────────────

CSS = """
#verilayer-header { text-align: center; padding: 1.5rem 0 0.5rem; }
#query-box textarea { font-size: 1rem; }
.status-chip { border-radius: 999px; padding: 4px 14px; font-weight: 700; }
"""

with gr.Blocks(
    title="VeriLayer — Trust Layer for LLMs",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css=CSS,
) as demo:

    # Header
    gr.HTML("""
    <div id='verilayer-header'>
        <h1 style='font-size:2rem;font-weight:800;color:#6366f1;'>
            🛡 VeriLayer
        </h1>
        <p style='color:#94a3b8;font-size:1rem;margin-top:4px;'>
            Agentic RAG Trust Layer · Claim Verification · Full Audit Trail
        </p>
    </div>
    """)

    with gr.Row():
        # ── Left column: query inputs ─────────────────────────────────────────
        with gr.Column(scale=1):
            query_input, top_k_slider, submit_btn, meta_output = render_query_panel()

        # ── Right column: answer display ──────────────────────────────────────
        with gr.Column(scale=2):
            answer_output = render_answer_panel()

    gr.Markdown("---")

    # ── Detailed tabs (Claims, Sources, Trace, Metrics) ───────────────────────
    with gr.Tabs():
        with gr.TabItem("📋 Claims"):
            claims_output = render_claims_table()

        with gr.TabItem("📚 Sources"):
            sources_output = render_sources_panel()

        with gr.TabItem("🔎 Pipeline Trace"):
            trace_output = render_trace_panel()

        with gr.TabItem("📊 Metrics"):
            metrics_output, refresh_btn = render_metrics_panel()

    # ── Event handlers ─────────────────────────────────────────────────────────
    submit_btn.click(
        fn=on_submit,
        inputs=[query_input, top_k_slider],
        outputs=[answer_output, claims_output, sources_output, trace_output, meta_output],
    )
    query_input.submit(
        fn=on_submit,
        inputs=[query_input, top_k_slider],
        outputs=[answer_output, claims_output, sources_output, trace_output, meta_output],
    )
    refresh_btn.click(fn=on_metrics_refresh, inputs=[], outputs=[metrics_output])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
