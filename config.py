import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")   # kept for image vision (claude-haiku)
TOGETHER_API_KEY     = os.getenv("TOGETHER_API_KEY")
CEREBRAS_API_KEY     = os.getenv("CEREBRAS_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

SUPABASE_URL              = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ── Groq: intent classification, planning, conversation, coding fallback ──────
GROQ_MODELS = {
    "groq/llama-3.3-70b":    "llama-3.3-70b-versatile",
    "groq/llama-3.1-8b":     "llama-3.1-8b-instant",
    "groq/gpt-oss-120b":     "openai/gpt-oss-120b",
}

# ── Together AI: kept for future use, not used for coding currently ───────────
TOGETHER_MODELS = {
    "together/glm-5":         "zai-org/GLM-5",
    "together/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
}

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# ── Cerebras: primary coding model (~1000-1700 tokens/sec) ───────────────────
CEREBRAS_MODELS = {
    "cerebras/glm-4.7": "zai-glm-4.7",
}

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# ── Coding model chain ────────────────────────────────────────────────────────
# Primary:  Cerebras GLM-4.7  — fast, used for ALL coding tasks
# Fallback: Groq GPT-OSS-120B — triggered after 2 consecutive Cerebras failures
CODING_MODEL_PRIMARY  = "cerebras/glm-4.7"    # tried up to 2 times
CODING_MODEL_FALLBACK = "groq/gpt-oss-120b"   # tried up to 2 times if primary fails

# ── Internal role aliases ─────────────────────────────────────────────────────
PLANNING_MODEL      = "groq/llama-3.3-70b"    # planning + intent classification
CONVERSATION_MODEL  = "groq/llama-3.1-8b"     # lightweight chat replies

ALL_MODELS = (
    list(GROQ_MODELS.keys())
    + list(TOGETHER_MODELS.keys())
    + list(CEREBRAS_MODELS.keys())
)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"