"""
VeriLayer — Phase 8: Sources Panel Component.
Displays referenced source document chunks, their relevance scores, and metadata.
"""
from __future__ import annotations

import gradio as gr


def format_sources(data: dict) -> str:
    """Format cited source chunks into readable markdown blocks."""
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


def render_sources_panel() -> gr.Markdown:
    """Renders the markdown area for the sources list."""
    return gr.Markdown(value="_Sources will appear here._")
