# config.py — Central configuration for BC Bank Chatbot

# ── ChromaDB Retrieval ─────────────────────────────────────────────────────
DISTANCE_THRESHOLD = 0.8          # Max distance for a KB result to be "relevant"
CONFIDENCE_HIGH_THRESHOLD = 0.4   # dist < 0.4 → high confidence
CONFIDENCE_MED_THRESHOLD = 0.6    # dist 0.4–0.6 → medium; > 0.6 → low → ask clarification

# ── Conversation ───────────────────────────────────────────────────────────
HISTORY_WINDOW = 8                # Recent messages included in Gemini prompt
SUMMARY_TRIGGER = 16              # Total message count that triggers summarization

# ── Query Cache ────────────────────────────────────────────────────────────
CACHE_MAX_SIZE = 100              # Max KB query cache entries before pruning

# ── Exchange Rate Cache ────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 300           # 5-minute TTL for exchange rate data

# ── Follow-up Detection ────────────────────────────────────────────────────
FOLLOWUP_SIMILARITY_THRESHOLD = 0.65  # Cosine similarity above this → follow-up

# ── Tenacity Retry ─────────────────────────────────────────────────────────
RETRY_MAX_ATTEMPTS = 3
RETRY_WAIT_MIN = 1                # Seconds
RETRY_WAIT_MAX = 4                # Seconds

# ── Intent → ChromaDB Category Mapping ────────────────────────────────────
INTENT_CATEGORY_MAP = {
    "account_info":    "accounts",
    "card_services":   "cards",
    "funds_transfer":  "fundstransfer",
    "loans":           "loans",
    "insurance":       "insurance",
    "investments":     "investments",
    "security":        "security",
    "exchange_rate":   None,
    "complaint":       None,
    "general":         None,
}

# ── Support Contact ────────────────────────────────────────────────────────
SUPPORT_PHONE = "+91-9876543210"
SUPPORT_TOLLFREE = "1800-XXX-XXXX"
