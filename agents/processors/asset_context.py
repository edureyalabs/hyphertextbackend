# agents/processors/asset_context.py
"""
Builds the asset context string that gets injected into the orchestrator system prompt.
Called once per agent run, after all assets for the page have been processed.
"""

from database import get_page_assets_ready


def build_asset_context(page_id: str) -> str:
    """
    Returns a formatted string describing all ready assets for the page.
    Returns empty string if no assets exist.
    """
    assets = get_page_assets_ready(page_id)
    if not assets:
        return ""

    sections: list[str] = ["=" * 60, "UPLOADED FILES CONTEXT", "=" * 60]

    images   = [a for a in assets if a["asset_type"] == "image"]
    docs     = [a for a in assets if a["asset_type"] == "document"]
    extracted = [a for a in assets if a["asset_type"] == "extracted_image"]

    # ── images ───────────────────────────────────────────────────────────────
    if images:
        sections.append("\nIMAGES AVAILABLE FOR USE IN HTML")
        sections.append("You can embed these directly using their URL in <img> tags or CSS background-image.\n")
        for img in images:
            block = [f"  File: {img['original_file_name']}"]
            if img.get("public_url"):
                block.append(f"  URL:  {img['public_url']}")
            if img.get("vision_description"):
                block.append(f"  What it shows: {img['vision_description']}")
            if img.get("vision_suggested_use"):
                block.append(f"  Suggested use: {img['vision_suggested_use']}")
            if img.get("vision_alt_text"):
                block.append(f"  Alt text: {img['vision_alt_text']}")
            if img.get("vision_tags"):
                tags = img["vision_tags"]
                if isinstance(tags, list) and tags:
                    block.append(f"  Tags: {', '.join(str(t) for t in tags)}")
            if img.get("dominant_colors"):
                colors = img["dominant_colors"]
                if isinstance(colors, list) and colors:
                    block.append(f"  Dominant colors: {', '.join(str(c) for c in colors)}")
            if img.get("width") and img.get("height"):
                block.append(f"  Dimensions: {img['width']}x{img['height']}px")
            if img.get("vision_contains_text") and img.get("vision_extracted_text"):
                block.append(f"  Text in image: {img['vision_extracted_text']}")
            sections.append("\n".join(block))
            sections.append("")  # blank line between entries

    # ── documents ────────────────────────────────────────────────────────────
    if docs:
        sections.append("\nDOCUMENT CONTENT")
        sections.append("Text extracted from uploaded documents. Use this content when building the page.\n")
        for doc in docs:
            block = [f"  File: {doc['original_file_name']}"]
            if doc.get("extracted_summary"):
                block.append(f"  Content:\n{_indent(doc['extracted_summary'], 4)}")
            elif doc.get("extracted_text"):
                # show first 600 chars if no summary
                preview = doc["extracted_text"][:600]
                if len(doc["extracted_text"]) > 600:
                    preview += "\n  [... truncated]"
                block.append(f"  Content:\n{_indent(preview, 4)}")
            sections.append("\n".join(block))
            sections.append("")

    # ── extracted images (from PDFs/DOCX) ────────────────────────────────────
    if extracted:
        sections.append("\nIMAGES EXTRACTED FROM DOCUMENTS")
        sections.append("These images were found inside uploaded documents.\n")
        for img in extracted:
            block = [f"  File: {img['original_file_name']} (from document)"]
            if img.get("public_url"):
                block.append(f"  URL:  {img['public_url']}")
            if img.get("vision_description"):
                block.append(f"  What it shows: {img['vision_description']}")
            if img.get("vision_suggested_use"):
                block.append(f"  Suggested use: {img['vision_suggested_use']}")
            sections.append("\n".join(block))
            sections.append("")

    sections.append("=" * 60)
    return "\n".join(sections)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())