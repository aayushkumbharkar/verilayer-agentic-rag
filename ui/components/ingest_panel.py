"""
VeriLayer — Ingest Panel Component.

Two-mode document ingestion UI:
  - PDF Upload:   drag-and-drop or browse for a .pdf file (max 50MB)
  - Plain Text:   paste raw text with a source name

Optional metadata fields: section, clause (mirroring the /ingest API).
"""
from __future__ import annotations

import gradio as gr


def render_ingest_panel() -> tuple:
    """
    Renders the full ingestion panel inside the current Gradio layout block.

    Returns:
        (pdf_file, text_source_name, text_content, section_input,
         clause_input, ingest_pdf_btn, ingest_text_btn, ingest_status)
    """
    gr.Markdown(
        """
        ### 📥 Ingest a Document
        Add documents to the VeriLayer knowledge base before running queries.
        Uploaded documents are chunked, embedded, and indexed automatically.
        """
    )

    with gr.Tabs():
        # ── Tab 1: PDF Upload ─────────────────────────────────────────────────
        with gr.TabItem("📄 Upload PDF"):
            gr.Markdown("*Supported: text-based PDFs up to 50 MB.*")
            pdf_file = gr.File(
                label="Select PDF file",
                file_types=[".pdf"],
                file_count="single",
                elem_id="pdf-upload",
            )
            with gr.Row():
                pdf_section = gr.Textbox(
                    label="Section (optional)",
                    placeholder="e.g. Liability",
                    scale=1,
                )
                pdf_clause = gr.Textbox(
                    label="Clause (optional)",
                    placeholder="e.g. 5.2",
                    scale=1,
                )
            ingest_pdf_btn = gr.Button("⬆️ Ingest PDF", variant="primary", size="lg")

        # ── Tab 2: Plain Text ─────────────────────────────────────────────────
        with gr.TabItem("📝 Paste Text"):
            gr.Markdown("*Paste raw text — articles, notes, scraped content, etc.*")
            text_source_name = gr.Textbox(
                label="Document name",
                placeholder="e.g. contract-2024, meeting-notes",
                max_lines=1,
            )
            text_content = gr.Textbox(
                label="Content",
                placeholder="Paste your text here…",
                lines=10,
                elem_id="text-ingest-box",
            )
            with gr.Row():
                text_section = gr.Textbox(
                    label="Section (optional)",
                    placeholder="e.g. Overview",
                    scale=1,
                )
                text_clause = gr.Textbox(
                    label="Clause (optional)",
                    placeholder="e.g. 1",
                    scale=1,
                )
            ingest_text_btn = gr.Button("⬆️ Ingest Text", variant="primary", size="lg")

    # ── Shared status output ──────────────────────────────────────────────────
    ingest_status = gr.Markdown(value="", label="Ingestion Status")

    return (
        pdf_file, pdf_section, pdf_clause, ingest_pdf_btn,
        text_source_name, text_content, text_section, text_clause, ingest_text_btn,
        ingest_status,
    )
