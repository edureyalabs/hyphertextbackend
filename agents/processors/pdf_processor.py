# agents/processors/pdf_processor.py
"""
Extracts text and embedded images from PDF files using PyMuPDF (fitz).
Returns extracted text (markdown-formatted) and a list of image blobs.
"""

import io
from dataclasses import dataclass, field
from typing import Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# Max characters of text to extract (to control token usage)
MAX_TEXT_CHARS = 12_000

# Min image size to bother extracting (skip tiny icons/bullets)
MIN_IMAGE_WIDTH  = 80
MIN_IMAGE_HEIGHT = 80


@dataclass
class ExtractedImage:
    bytes: bytes
    mime_type: str
    page_number: int
    width: int
    height: int
    index_on_page: int


@dataclass
class PDFExtractionResult:
    text: str                             # full extracted text (possibly truncated)
    summary: str                          # first ~800 chars — used as quick preview
    page_count: int
    was_truncated: bool
    images: list[ExtractedImage] = field(default_factory=list)
    error: Optional[str] = None


def extract_pdf(file_bytes: bytes) -> PDFExtractionResult:
    """
    Extract text and images from PDF bytes.
    Returns PDFExtractionResult.
    """
    if not PYMUPDF_AVAILABLE:
        return PDFExtractionResult(
            text="",
            summary="",
            page_count=0,
            was_truncated=False,
            error="PyMuPDF not installed. Run: pip install pymupdf"
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return PDFExtractionResult(
            text="",
            summary="",
            page_count=0,
            was_truncated=False,
            error=f"Could not open PDF: {e}"
        )

    page_count = doc.page_count
    text_parts = []
    images: list[ExtractedImage] = []
    total_chars = 0
    was_truncated = False

    for page_num in range(page_count):
        if was_truncated:
            break

        page = doc[page_num]

        # ── text extraction ──────────────────────────────────────────────────
        page_text = page.get_text("text").strip()
        if page_text:
            header = f"\n\n--- Page {page_num + 1} ---\n"
            chunk = header + page_text
            if total_chars + len(chunk) > MAX_TEXT_CHARS:
                remaining = MAX_TEXT_CHARS - total_chars
                if remaining > 100:
                    text_parts.append(chunk[:remaining])
                text_parts.append(f"\n\n[... content truncated at {MAX_TEXT_CHARS} characters. Document has {page_count} pages total.]")
                was_truncated = True
                total_chars = MAX_TEXT_CHARS
            else:
                text_parts.append(chunk)
                total_chars += len(chunk)

        # ── image extraction ─────────────────────────────────────────────────
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_ext    = base_image["ext"]          # e.g. "jpeg", "png"
                img_w      = base_image.get("width", 0)
                img_h      = base_image.get("height", 0)

                if img_w < MIN_IMAGE_WIDTH or img_h < MIN_IMAGE_HEIGHT:
                    continue  # skip tiny decorative elements

                mime = _ext_to_mime(img_ext)
                if mime is None:
                    continue  # skip unsupported formats

                images.append(ExtractedImage(
                    bytes=img_bytes,
                    mime_type=mime,
                    page_number=page_num + 1,
                    width=img_w,
                    height=img_h,
                    index_on_page=img_index,
                ))
            except Exception:
                continue  # skip problematic images silently

    doc.close()

    full_text = "".join(text_parts).strip()
    summary   = full_text[:800] if full_text else ""

    return PDFExtractionResult(
        text=full_text,
        summary=summary,
        page_count=page_count,
        was_truncated=was_truncated,
        images=images,
    )


def _ext_to_mime(ext: str) -> Optional[str]:
    mapping = {
        "jpeg": "image/jpeg",
        "jpg":  "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp",
    }
    return mapping.get(ext.lower())