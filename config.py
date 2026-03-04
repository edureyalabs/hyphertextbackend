import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # kept for image vision (claude-haiku still used in image_processor)
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ── Groq: used for intent classification, planning, and simple conversation ──
GROQ_MODELS = {
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-3.1-8b":  "llama-3.1-8b-instant",
}

# ── DeepInfra: used for all HTML coding tasks ────────────────────────────────
# GLM-5        → complex / full rewrites / new page builds / imported pages
# GLM-4.7-Flash → surgical edits, simple fixes, single-component changes
DEEPINFRA_MODELS = {
    "deepinfra/glm-5":         "zai-org/GLM-5",
    "deepinfra/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
}

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# ── Internal role aliases (never exposed to the frontend) ────────────────────
# These are the logical names the orchestrator uses when selecting a model.
CODING_MODEL_COMPLEX = "deepinfra/glm-5"          # full rewrites, new pages, complex edits
CODING_MODEL_SIMPLE  = "deepinfra/glm-4.7-flash"  # surgical edits, simple changes
PLANNING_MODEL       = "groq/llama-3.3-70b"        # planning + intent classification
CONVERSATION_MODEL   = "groq/llama-3.1-8b"         # lightweight chat replies

ALL_MODELS = list(GROQ_MODELS.keys()) + list(DEEPINFRA_MODELS.keys())

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"