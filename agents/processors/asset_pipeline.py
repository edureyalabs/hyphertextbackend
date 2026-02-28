# agents/processors/asset_pipeline.py
"""
The asset pipeline is called at the start of every agent run.
It finds all 'pending' assets for the page, processes them
(vision for images, text extraction for documents), then updates the DB.

Design: fire-and-forget per asset. If one fails, it's marked 'failed'
and the rest continue. The agent run is never blocked.
"""

import asyncio
import uuid
from typing import Optional

from database import (
    get_pending_assets_for_page,
    update_asset_processing_started,
    update_asset_image_result,
    update_asset_document_result,
    insert_extracted_image_asset,
    mark_asset_failed,
    get_supabase_client,
)
from agents.processors.image_processor import analyze_image
from agents.processors.pdf_processor   import extract_pdf
from agents.processors.docx_processor  import extract_docx

SUPABASE_STORAGE_BUCKET = "page-assets"


async def process_pending_assets(page_id: str, owner_id: str) -> int:
    """
    Process all pending assets for a page.
    Returns the count of successfully processed assets.
    """
    pending = get_pending_assets_for_page(page_id)
    if not pending:
        return 0

    tasks = [_process_one(asset, owner_id) for asset in pending]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if r is True)
    return success_count


async def _process_one(asset: dict, owner_id: str) -> bool:
    """Process a single asset. Returns True on success."""
    asset_id   = asset["id"]
    asset_type = asset["asset_type"]
    file_type  = asset["file_type"]
    storage_path = asset.get("storage_path")

    try:
        update_asset_processing_started(asset_id)

        # download the file from Supabase storage
        file_bytes = await _download_from_storage(storage_path)
        if file_bytes is None:
            mark_asset_failed(asset_id, "Could not download file from storage")
            return False

        if asset_type == "image":
            return await _process_image(asset, file_bytes)
        elif asset_type == "document":
            return await _process_document(asset, file_bytes, owner_id)
        else:
            mark_asset_failed(asset_id, f"Unknown asset_type: {asset_type}")
            return False

    except Exception as e:
        mark_asset_failed(asset_id, str(e))
        return False


async def _process_image(asset: dict, file_bytes: bytes) -> bool:
    asset_id  = asset["id"]
    file_type = asset["file_type"]

    result = await analyze_image(file_bytes, file_type)

    update_asset_image_result(
        asset_id=asset_id,
        vision_description=result["description"],
        vision_tags=result["detected_objects"],
        vision_suggested_use=result["suggested_use"],
        vision_alt_text=result["alt_text"],
        vision_contains_text=result["contains_text"],
        vision_extracted_text=result.get("extracted_text", ""),
        dominant_colors=result["dominant_colors"],
    )
    return True


async def _process_document(asset: dict, file_bytes: bytes, owner_id: str) -> bool:
    asset_id  = asset["id"]
    page_id   = asset["page_id"]
    file_type = asset["file_type"]

    if file_type == "application/pdf":
        from agents.processors.pdf_processor import extract_pdf
        extraction = extract_pdf(file_bytes)
    else:
        from agents.processors.docx_processor import extract_docx
        extraction = extract_docx(file_bytes, file_type)

    if extraction.error:
        # if the extractor returned a handled error (like legacy .doc), store it
        # but still mark as completed with the error note in extracted_text
        update_asset_document_result(
            asset_id=asset_id,
            extracted_text=extraction.error,
            extracted_summary=extraction.error,
        )
        return True  # considered "processed" — error is surfaced in the text

    update_asset_document_result(
        asset_id=asset_id,
        extracted_text=extraction.text,
        extracted_summary=extraction.summary,
    )

    # process any embedded images
    if extraction.images:
        image_tasks = [
            _process_embedded_image(img, asset, owner_id)
            for img in extraction.images
        ]
        await asyncio.gather(*image_tasks, return_exceptions=True)

    return True


async def _process_embedded_image(
    img,   # ExtractedImage dataclass
    parent_asset: dict,
    owner_id: str,
) -> None:
    """Upload an extracted image to storage, run vision, insert as child asset."""
    page_id   = parent_asset["page_id"]
    parent_id = parent_asset["id"]

    # generate a storage path for this extracted image
    ext = img.mime_type.split("/")[-1].replace("jpeg", "jpg")
    filename = f"extracted_{uuid.uuid4().hex[:8]}.{ext}"
    storage_path = f"{owner_id}/{page_id}/{filename}"

    # upload to Supabase storage
    public_url = await _upload_to_storage(
        path=storage_path,
        data=img.bytes,
        content_type=img.mime_type,
    )
    if public_url is None:
        return  # skip if upload failed

    # insert child asset record (processing_status=pending so pipeline will pick it up next time,
    # but we call vision directly here for efficiency)
    child_id = insert_extracted_image_asset(
        page_id=page_id,
        owner_id=owner_id,
        parent_asset_id=parent_id,
        file_name=filename,
        original_file_name=filename,
        file_type=img.mime_type,
        storage_path=storage_path,
        public_url=public_url,
        width=img.width,
        height=img.height,
        file_size_bytes=len(img.bytes),
    )

    if child_id is None:
        return

    # run vision on the extracted image directly
    try:
        vision_result = await analyze_image(img.bytes, img.mime_type)
        update_asset_image_result(
            asset_id=child_id,
            vision_description=vision_result["description"],
            vision_tags=vision_result["detected_objects"],
            vision_suggested_use=vision_result["suggested_use"],
            vision_alt_text=vision_result["alt_text"],
            vision_contains_text=vision_result["contains_text"],
            vision_extracted_text=vision_result.get("extracted_text", ""),
            dominant_colors=vision_result["dominant_colors"],
        )
    except Exception:
        mark_asset_failed(child_id, "Vision analysis failed for extracted image")


async def _download_from_storage(storage_path: Optional[str]) -> Optional[bytes]:
    """Download file bytes from Supabase storage."""
    if not storage_path:
        return None
    try:
        supabase = get_supabase_client()
        response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(storage_path)
        return response
    except Exception as e:
        print(f"[ASSET PIPELINE] Storage download error for {storage_path}: {e}")
        return None


async def _upload_to_storage(path: str, data: bytes, content_type: str) -> Optional[str]:
    """Upload bytes to Supabase storage, return public URL."""
    try:
        supabase = get_supabase_client()
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type}
        )
        public_url = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(path)
        return public_url
    except Exception as e:
        print(f"[ASSET PIPELINE] Storage upload error for {path}: {e}")
        return None