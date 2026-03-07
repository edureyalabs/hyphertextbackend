# agents/models/coding_router.py
"""
Coding model router.

Economy mode (default)
──────────────────────────────────────────────────────────────────────
Uses Together AI. Routing table:
  is_new_page = True                         → CODING_MODEL_COMPLEX (GLM-5)
  page_source = import                       → CODING_MODEL_COMPLEX (GLM-5)
  plan.decision = full_rewrite               → CODING_MODEL_COMPLEX (GLM-5)
  plan.complexity = complex / moderate       → CODING_MODEL_COMPLEX (GLM-5)
  plan.complexity = simple + surgical_edit   → CODING_MODEL_SIMPLE  (GLM-4.7-Flash)

Speed mode
──────────────────────────────────────────────────────────────────────
Uses Cerebras for ALL coding tasks regardless of complexity or decision.
  ALL tasks → CODING_MODEL_SPEED (cerebras/glm-4.7)

The inference_mode is chosen once on the first user message (persisted to
the page record as `inference_mode`). All subsequent edits on the same page
use the same mode — ensuring consistency between the model that built the
page and the model that edits it.

The override_model_id path honours a previously persisted coding_model_id
so the model choice stays stable across sessions.
"""

from config import (
    CODING_MODEL_COMPLEX,
    CODING_MODEL_SIMPLE,
    CODING_MODEL_SPEED,
)


def select_coding_model(
    plan: dict,
    is_new_page: bool,
    is_imported: bool,
    inference_mode: str = "economy",
    override_model_id: str | None = None,
) -> str:
    """
    Return the model alias to use for this coding task.

    Args:
        plan:              Parsed plan dict from the planning step.
        is_new_page:       True if the page still contains the boilerplate placeholder.
        is_imported:       True if page_source == "import" and no summary exists yet.
        inference_mode:    "economy" (Together AI) or "speed" (Cerebras).
        override_model_id: If the page already has a persisted model from a prior run,
                           pass it here to skip routing and stay consistent.

    Returns:
        One of CODING_MODEL_COMPLEX, CODING_MODEL_SIMPLE, or CODING_MODEL_SPEED.
    """
    # ── honour a previously persisted decision ────────────────────────────────
    valid_models = {CODING_MODEL_COMPLEX, CODING_MODEL_SIMPLE, CODING_MODEL_SPEED}
    if override_model_id and override_model_id in valid_models:
        return override_model_id

    # ── Speed mode: always use Cerebras regardless of complexity ─────────────
    if inference_mode == "speed":
        return CODING_MODEL_SPEED

    # ── Economy mode routing (Together AI) ───────────────────────────────────

    # Always use the heavy model for first-time / imported pages
    if is_new_page or is_imported:
        return CODING_MODEL_COMPLEX

    decision   = plan.get("decision", "surgical_edit")
    complexity = plan.get("complexity", "simple")

    if decision == "full_rewrite":
        return CODING_MODEL_COMPLEX

    if complexity in ("complex", "moderate"):
        return CODING_MODEL_COMPLEX

    # simple surgical edit → fast model
    return CODING_MODEL_SIMPLE