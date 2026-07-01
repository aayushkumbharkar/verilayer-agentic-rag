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

from ui.utils.api_client import call_verify, call_metrics, call_ingest_text, call_ingest_pdf
from ui.components.query_panel import render_query_panel
from ui.components.answer_panel import render_answer_panel, format_answer
from ui.components.claims_table import render_claims_table, format_claims
from ui.components.sources_panel import render_sources_panel, format_sources
from ui.components.trace_panel import render_trace_panel, format_trace
from ui.components.metrics_panel import render_metrics_panel, format_metrics
from ui.components.ingest_panel import render_ingest_panel


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


def on_ingest_pdf(pdf_file, section: str, clause: str) -> str:
    """Called when the user clicks Ingest PDF."""
    if pdf_file is None:
        return "⚠️ Please select a PDF file first."
    try:
        data = _run_async(call_ingest_pdf(
            pdf_path=pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file),
            section=section.strip() or None,
            clause=clause.strip() or None,
        ))
        chunks = data.get("chunks_created", data.get("chunk_count", "?"))
        doc_id = data.get("document_id", "")
        return (
            f"✅ **PDF ingested successfully!**\n\n"
            f"- 📄 Document ID: `{doc_id}`\n"
            f"- 🧮 Chunks created: **{chunks}**\n"
            f"- 📊 Ready to query!"
        )
    except Exception as exc:
        return f"❌ Ingestion failed: {exc}"


def on_ingest_text(source_name: str, content: str, section: str, clause: str) -> str:
    """Called when the user clicks Ingest Text."""
    if not source_name.strip():
        return "⚠️ Please enter a document name."
    if not content.strip():
        return "⚠️ Please paste some content to ingest."
    try:
        data = _run_async(call_ingest_text(
            source_name=source_name.strip(),
            content=content.strip(),
            section=section.strip() or None,
            clause=clause.strip() or None,
        ))
        chunks = data.get("chunks_created", data.get("chunk_count", "?"))
        doc_id = data.get("document_id", "")
        return (
            f"✅ **Text ingested successfully!**\n\n"
            f"- 📄 Document ID: `{doc_id}`\n"
            f"- 🧮 Chunks created: **{chunks}**\n"
            f"- 📊 Ready to query!"
        )
    except Exception as exc:
        return f"❌ Ingestion failed: {exc}"


# ── Gradio Blocks UI ───────────────────────────────────────────────────────────

CSS = """
#verilayer-header { text-align: center; padding: 1.5rem 0 0.5rem; }
#query-box textarea { font-size: 1rem; }
.status-chip { border-radius: 999px; padding: 4px 14px; font-weight: 700; }
#recording-info { background: #1e293b; border-radius: 10px; padding: 12px 18px; color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px; }
"""

with gr.Blocks(
    title="VeriLayer — Trust Layer for LLMs",
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

    # ── Tabs: Ingest first, then query results ─────────────────────────────────
    with gr.Tabs():
        # ── Ingest Tab ────────────────────────────────────────────────────────
        with gr.TabItem("📥 Ingest Documents"):
            (
                pdf_file, pdf_section, pdf_clause, ingest_pdf_btn,
                text_source_name, text_content, text_section, text_clause, ingest_text_btn,
                ingest_status,
            ) = render_ingest_panel()

        with gr.TabItem("📋 Claims"):
            claims_output = render_claims_table()

        with gr.TabItem("📚 Sources"):
            sources_output = render_sources_panel()

        with gr.TabItem("🔎 Pipeline Trace"):
            trace_output = render_trace_panel()

        with gr.TabItem("📊 Metrics"):
            metrics_output, refresh_btn = render_metrics_panel()

        # ── Screen Recording Tab ──────────────────────────────────────────────
        with gr.TabItem("🎥 Screen Recording"):
            gr.HTML("""
            <div id='recording-info'>
              <h3 style='color:#6366f1;margin-top:0'>🎬 How to record your screen</h3>
              <p>Gradio 6 uses your <strong>browser's built-in screen recorder</strong>. Follow these steps:</p>
              <ol style='line-height:2'>
                <li>Press <kbd style='background:#334155;padding:2px 8px;border-radius:4px;color:#fff'>Windows + G</kbd> to open <strong>Xbox Game Bar</strong> (Windows built-in recorder)</li>
                <li><strong>Or</strong> use <a href='https://chrome.google.com/webstore/detail/screen-recorder' target='_blank' style='color:#6366f1'>Chrome Screen Recorder extension</a></li>
                <li><strong>Or</strong> press <kbd style='background:#334155;padding:2px 8px;border-radius:4px;color:#fff'>Win + Alt + R</kbd> to start/stop recording immediately</li>
                <li>Once done, <strong>upload your recording</strong> below to view &amp; save it in VeriLayer</li>
              </ol>
              <p style='color:#f59e0b'>⚠️ Gradio 6.x removed <code>sources=["screen"]</code> — use the browser/OS tools above to capture screen, then upload here.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=3):
                    uploaded_recording = gr.Video(
                        sources=["upload", "webcam"],
                        label="📹 Upload or Webcam Recording — drag &amp; drop your screen recording here",
                        include_audio=True,
                        height=420,
                        elem_id="screen-recorder",
                    )
                with gr.Column(scale=2):
                    recording_status = gr.Markdown(
                        value=(
                            "**📹 Quick recording options:**\n\n"
                            "**Windows (built-in):**\n"
                            "- `Win + G` → Xbox Game Bar → ● Record\n"
                            "- `Win + Alt + R` → instant record/stop\n"
                            "- Saves to: `Videos\\Captures\\`\n\n"
                            "**Browser extension:**\n"
                            "- [Loom](https://www.loom.com) — free, instant share\n"
                            "- [Screenity](https://chrome.google.com/webstore/detail/screenity) — Chrome\n\n"
                            "**After recording:** drag the `.mp4`/`.webm` file into the video player ⬅"
                        ),
                        label="Recording Options",
                    )

            # When a video is uploaded/recorded, save it server-side
            def on_recording_upload(video_path):
                import os, shutil, datetime
                if video_path is None:
                    return "⚠️ No video provided."
                save_dir = "/app/uploads/recordings"
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ext = os.path.splitext(video_path)[-1] or ".mp4"
                dest = os.path.join(save_dir, f"recording_{ts}{ext}")
                shutil.copy2(video_path, dest)
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                return (
                    f"✅ **Recording saved server-side!**\n\n"
                    f"- 📁 Path: `uploads/recordings/recording_{ts}{ext}`\n"
                    f"- 💾 Size: **{size_mb:.2f} MB**\n"
                    f"- 🕒 Saved at: `{ts}`"
                )

            uploaded_recording.change(
                fn=on_recording_upload,
                inputs=[uploaded_recording],
                outputs=[recording_status],
            )

    # ── Event handlers ───────────────────────────────────────────────────────────────────
    # Query handlers
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

    # Ingest handlers
    ingest_pdf_btn.click(
        fn=on_ingest_pdf,
        inputs=[pdf_file, pdf_section, pdf_clause],
        outputs=[ingest_status],
    )
    ingest_text_btn.click(
        fn=on_ingest_text,
        inputs=[text_source_name, text_content, text_section, text_clause],
        outputs=[ingest_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        pwa=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=CSS,
    )
