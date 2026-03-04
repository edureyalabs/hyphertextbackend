# agents/models/coding_router.py
"""
Coding model router.

Decides between GLM-5 (complex) and GLM-4.7-Flash (simple) for every
HTML coding task based on signals from the planning step.

Decision is made ONCE per page (on the first user message that triggers
a code change) and then persisted to the page record so that all future
edits on that page use the same model — ensuring the model that built
the page is also the model that edits it.

Routing table
─────────────────────────────────────────────────────────────────────────
Signal                                          → Model
─────────────────────────────────────────────────────────────────────────
is_new_page = True   (vibe-coding a new page)  → GLM-5
page_source = import (imported existing code)  → GLM-5
plan.decision = full_rewrite                   → GLM-5
plan.complexity = complex                       → GLM-5
plan.complexity = moderate                      → GLM-5
plan.complexity = simple + surgical_edit        → GLM-4.7-Flash
─────────────────────────────────────────────────────────────────────────

Once a model is persisted on the page, it is always returned for
subsequent edits (the override_model_id path).
"""

from config import CODING_MODEL_COMPLEX, CODING_MODEL_SIMPLE


def select_coding_model(
    plan: dict,
    is_new_page: bool,
    is_imported: bool,
    override_model_id: str | None = None,
) -> str:
    """
    Return the DeepInfra model alias to use for this coding task.

    Args:
        plan:              Parsed plan dict from the planning step.
        is_new_page:       True if the page still contains the boilerplate placeholder.
        is_imported:       True if page_source == "import" and no summary exists yet.
        override_model_id: If the page already has a persisted model from a prior run,
                           pass it here to skip routing and stay consistent.

    Returns:
        One of CODING_MODEL_COMPLEX or CODING_MODEL_SIMPLE.
    """
    # ── honour a previously persisted decision ────────────────────────────────
    if override_model_id and override_model_id in (CODING_MODEL_COMPLEX, CODING_MODEL_SIMPLE):
        return override_model_id

    # ── always use the heavy model for first-time / imported pages ────────────
    if is_new_page or is_imported:
        return CODING_MODEL_COMPLEX

    # ── plan-based routing ────────────────────────────────────────────────────
    decision   = plan.get("decision", "surgical_edit")
    complexity = plan.get("complexity", "simple")

    if decision == "full_rewrite":
        return CODING_MODEL_COMPLEX

    if complexity in ("complex", "moderate"):
        return CODING_MODEL_COMPLEX

    # simple surgical edit → fast model
    return CODING_MODEL_SIMPLE