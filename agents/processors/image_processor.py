# agents/processors/image_processor.py
"""
Vision analysis using AsyncAnthropic (claude-haiku).
A process-level semaphore bounds concurrent Anthropic calls to avoid
rate-limit errors when many users upload images simultaneously.
"""

import asyncio
import base64
import json
import logging
from anthropic import AsyncAnthropic
from config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

VISION_MODEL = "claude-haiku-4-5-20251001"

# Max concurrent Anthropic vision calls per worker process.
# 4 workers × 50 = 200 simultaneous vision calls across the fleet.
# Anthropic's default RPM for Haiku is high, but this prevents burst spikes.
_VISION_SEMAPHORE = asyncio.Semaphore(50)

VISION_PROMPT = """Analyze this image carefully and return a JSON object with the following fields.
Be concise but precise — this description will be given to an AI coding agent that will use the image in an HTML page.

{
  "description": "2-3 sentence description of what this image shows. Be specific and useful for an AI agent building a web page.",
  "detected_objects": ["list", "of", "main", "objects", "or", "subjects"],
  "contains_people": true or false,
  "contains_text": true or false,
  "extracted_text": "any text visible in the image, verbatim. empty string if none.",
  "dominant_colors": ["#hexcolor1", "#hexcolor2", "#hexcolor3"],
  "suggested_use": one of: "profile_photo" | "product_image" | "logo" | "background" | "diagram" | "illustration" | "document_scan" | "other",
  "alt_text": "concise accessible alt text for the image"
}

Return only the JSON object. No markdown fences. No explanation."""


async def analyze_image(image_bytes: bytes, mime_type: str) -> dict:
    """
    Run vision analysis on image bytes.
    Bounded by _VISION_SEMAPHORE — safe to call from hundreds of concurrent tasks.
    """
    supported = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
    if mime_type not in supported:
        return _svg_placeholder()

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    async with _VISION_SEMAPHORE:
        response = await _client.messages.create(
            model=VISION_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                }
            ],
        )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "description": "Image uploaded by user.",
            "detected_objects": [],
            "contains_people": False,
            "contains_text": False,
            "extracted_text": "",
            "dominant_colors": [],
            "suggested_use": "other",
            "alt_text": "Uploaded image",
        }

    return {
        "description":      str(data.get("description", "")),
        "detected_objects": list(data.get("detected_objects", [])),
        "contains_people":  bool(data.get("contains_people", False)),
        "contains_text":    bool(data.get("contains_text", False)),
        "extracted_text":   str(data.get("extracted_text", "")),
        "dominant_colors":  list(data.get("dominant_colors", [])),
        "suggested_use":    str(data.get("suggested_use", "other")),
        "alt_text":         str(data.get("alt_text", "Uploaded image")),
    }


def _svg_placeholder() -> dict:
    return {
        "description": "SVG vector image uploaded by user.",
        "detected_objects": [],
        "contains_people": False,
        "contains_text": False,
        "extracted_text": "",
        "dominant_colors": [],
        "suggested_use": "illustration",
        "alt_text": "SVG image",
    }