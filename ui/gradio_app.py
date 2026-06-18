"""
VeriLayer — Phase 8: Gradio UI.

A rich Gradio Blocks interface with:
  - Query Input panel with top_k slider
  - Answer Panel: final answer + confidence badge + status chip
  - Claims Table: claim text, verdict (🟢/🔴/🟡), confidence bar
  - Sources Panel: document ID, chunk text, relevance score
  - Trace Panel: collapsible accordion per agent step with latency
  - Metrics Panel: live hallucination rate, avg confidence, retry rate
"""
from __future__ import annotations

import asyncio
import json
import httpx
import gradio as gr

BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0


# ── Status helpers ─────────────────────────────────────────────────────────────

STATUS_EMOJI = {
    "verified": "🟢 Verified",
    "partial":  "🟡 Partial",
    "unsafe":   "🔴 Unsafe",
}

VERDICT_EMOJI = {
    "supported":   "🟢",
    "partial":     "🟡",
    "unsupported": "🔴",
}

STATUS_CSS = {
    "verified": "color: #22c55e; font-weight: bold;",
    "partial":  "color: #eab308; font-weight: bold;",
    "unsafe":   "color: #ef4444; font-weight: bold;",
}


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


async def _post_verify(query: str, top_k: int) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{BASE_URL}/verify",
            json={"query": query, "top_k": int(top_k)},
        )
        resp.raise_for_status()
        return resp.json()


async def _get_metrics() -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{BASE_URL}/metrics")
        resp.raise_for_status()
        return resp.json()


# ── Result formatters ──────────────────────────────────────────────────────────

def _format_answer(data: dict) -> str:
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


def _format_claims(data: dict) -> str:
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


def _format_sources(data: dict) -> str:
    claims = data.get("claims", [])
    seen: dict[str, dict] = {}
    for c in claims:
        for s in c.get("sources", []):
            cid = s.get("chunk_id", "")
            if cid and cid not in seen:
                seen[cid] = s

    if not seen:
        return "_No source chunks cited._"

    parts = []
    for i, (_, s) in enumerate(seen.items(), 1):
        doc_id = s.get("document_id", "?")[:20]
        chunk_text = s.get("text", "")[:300].replace("\n", " ")
        score = s.get("score", 0.0)
        parts.append(
            f"**Source {i}** — `{doc_id}` (score: {score:.4f})\n\n"
            f"> {chunk_text}…\n"
        )
    return "\n---\n".join(parts)


def _format_trace(data: dict) -> list[tuple[str, str]]:
    """Return list of (label, content) for gr.Accordion steps."""
    trace = data.get("trace", [])
    result = []
    for t in trace:
        step = t.get("step", "?")
        details = t.get("details", "")
        latency = t.get("latency_ms", 0)
        label = f"🔹 {step.upper()} — {latency}ms"
        result.append((label, details))
    return result


def _format_metrics(data: dict) -> str:
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


# ── Main handler ───────────────────────────────────────────────────────────────

def on_submit(query: str, top_k: int):
    """Called when the user clicks Submit."""
    if not query.strip():
        empty = {"status": "unsafe", "confidence": 0.0, "final_answer": "Please enter a query.", "claims": [], "trace": [], "metadata": {}}
        return (
            _format_answer(empty),
            _format_claims(empty),
            _format_sources(empty),
            gr.update(value=""),  # trace placeholder
            "_",
        )

    try:
        data = _run_async(_post_verify(query, top_k))
    except Exception as exc:
        err = {"status": "unsafe", "confidence": 0.0, "final_answer": f"❌ Error: {exc}", "claims": [], "trace": [], "metadata": {}}
        return (
            _format_answer(err),
            _format_claims(err),
            _format_sources(err),
            f"Error: {exc}",
            "_",
        )

    trace_steps = _format_trace(data)
    trace_md = "\n\n".join(
        f"**{label}**\n{content}" for label, content in trace_steps
    ) if trace_steps else "_No trace available._"

    meta = data.get("metadata", {})
    meta_str = (
        f"📄 Retrieved: **{meta.get('retrieval_docs', 0)}** docs | "
        f"🔄 Retries: **{meta.get('retries', 0)}** | "
        f"⏱ Latency: **{meta.get('latency_total_ms', 0)}ms**"
    )

    return (
        _format_answer(data),
        _format_claims(data),
        _format_sources(data),
        trace_md,
        meta_str,
    )


def on_metrics_refresh():
    try:
        data = _run_async(_get_metrics())
        return _format_metrics(data)
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
        # ── Left column: input ────────────────────────────────────────────────
        with gr.Column(scale=1):
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

        # ── Right column: answer ──────────────────────────────────────────────
        with gr.Column(scale=2):
            answer_output = gr.Markdown(
                value="*Submit a query to see the verified answer.*",
                label="Answer",
            )

    gr.Markdown("---")

    # ── Claims, Sources, Trace ─────────────────────────────────────────────────
    with gr.Tabs():
        with gr.TabItem("📋 Claims"):
            claims_output = gr.Markdown(value="_Claims will appear here._")

        with gr.TabItem("📚 Sources"):
            sources_output = gr.Markdown(value="_Sources will appear here._")

        with gr.TabItem("🔎 Pipeline Trace"):
            trace_output = gr.Markdown(value="_Trace will appear here after a query._")

        with gr.TabItem("📊 Metrics"):
            metrics_output = gr.Markdown(value=_format_metrics({}))
            refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")

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
