# query_processor.py — Core query processing pipeline

import random

import numpy as np
import streamlit as st

import db_manager
from config import (
    CACHE_MAX_SIZE,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MED_THRESHOLD,
    DISTANCE_THRESHOLD,
    FOLLOWUP_SIMILARITY_THRESHOLD,
    HISTORY_WINDOW,
    INTENT_CATEGORY_MAP,
    SUMMARY_TRIGGER,
    SUPPORT_PHONE,
)
from exchange_rate_utils import get_exchange_rate_data, is_exchange_rate_query
from gemini_api import build_prompt, call_llm, summarize_conversation

# ── Loading messages ──────────────────────────────────────────────────────────

_LOADING_MESSAGES = [
    "Please hold on while I prepare the most accurate information for you.",
    "Just a moment, I'm gathering the details for your request.",
    "I'm checking the latest information to give you the best answer.",
    "Working on your request, this will only take a few seconds…",
    "Let me review your query and provide the right details shortly.",
    "One moment, I'm putting together the information you need.",
    "Retrieving the details for you, please hold on.",
]

# ── Intent keyword map ────────────────────────────────────────────────────────

_INTENT_KEYWORDS: dict = {
    "account_info":   ["account", "balance", "statement", "savings", "current account",
                       "open account", "close account", "account number"],
    "card_services":  ["card", "atm", "credit card", "debit card", "card limit",
                       "card activation", "block card", "replace card", "lost card"],
    "funds_transfer": ["transfer", "neft", "rtgs", "imps", "upi", "send money",
                       "payment", "fund transfer", "money transfer"],
    "loans":          ["loan", "emi", "personal loan", "home loan", "car loan",
                       "education loan", "top-up loan", "loan application"],
    "insurance":      ["insurance", "policy", "premium", "life insurance",
                       "health insurance", "claim", "coverage"],
    "investments":    ["investment", "mutual fund", "fixed deposit", "fd", "rd",
                       "recurring deposit", "bonds", "stocks", "returns"],
    "security":       ["security", "password", "pin", "otp", "fraud", "scam",
                       "phishing", "authentication", "cybersecurity"],
    "exchange_rate":  ["exchange rate", "convert", "forex", "fx rate", "currency"],
    "complaint":      ["complaint", "issue", "problem", "unhappy", "dissatisfied",
                       "wrong", "error", "mistake"],
}


# ── Helper functions ──────────────────────────────────────────────────────────

def classify_intent(query_lower: str) -> str:
    """Classify query into a known intent using keyword matching."""
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return intent
    return "general"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_followup(current_query: str, prev_query: str) -> bool:
    """Use embedding cosine similarity to detect follow-up queries."""
    try:
        model = st.session_state.get('embed_model')
        if model is None:
            return False
        emb_cur = model.encode(current_query)
        emb_prev = model.encode(prev_query)
        return _cosine_similarity(emb_cur, emb_prev) > FOLLOWUP_SIMILARITY_THRESHOLD
    except Exception as e:
        print(f"[WARN] Follow-up detection failed: {e}")
        return False


def score_confidence(distances: list) -> str:
    """Map best ChromaDB distance to high / medium / low confidence band."""
    if not distances:
        return "low"
    best = min(distances)
    if best < CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    elif best < CONFIDENCE_MED_THRESHOLD:
        return "medium"
    return "low"


def detect_language(query: str) -> str:
    """Detect query language; returns English name for Gemini's use."""
    try:
        from langdetect import detect
        code = detect(query)
        lang_map = {
            "en": "English", "hi": "Hindi", "fr": "French",
            "de": "German", "es": "Spanish", "ar": "Arabic",
            "zh-cn": "Chinese", "ja": "Japanese", "pt": "Portuguese",
            "ru": "Russian", "it": "Italian",
        }
        return lang_map.get(code, "English")
    except Exception:
        return "English"


def maybe_summarize(messages: list) -> str:
    """Trigger conversation summarization when history exceeds SUMMARY_TRIGGER."""
    if len(messages) <= SUMMARY_TRIGGER:
        return st.session_state.get('conversation_summary', '')
    if 'conversation_summary' not in st.session_state:
        old_turns = messages[:-HISTORY_WINDOW]
        print(f"[INFO] Summarizing {len(old_turns)} older turns.")
        st.session_state.conversation_summary = summarize_conversation(old_turns)
    return st.session_state.get('conversation_summary', '')


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(search_query: str, category: str | None) -> tuple:
    """
    Query ChromaDB for relevant documents, with in-memory caching.
    Returns (documents, distances, metadatas).
    """
    cache_key = f"{search_query}_{category}"
    if cache_key in st.session_state.query_cache:
        cached = st.session_state.query_cache[cache_key]
        return cached['documents'], cached['distances'], cached['metadatas']

    query_kwargs = dict(query_texts=[search_query], n_results=5)
    if category:
        query_kwargs['where'] = {"Class": category}

    results = st.session_state.collection.query(**query_kwargs)
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    st.session_state.query_cache[cache_key] = {
        'documents': documents, 'distances': distances, 'metadatas': metadatas,
    }
    if len(st.session_state.query_cache) > CACHE_MAX_SIZE:
        for k in list(st.session_state.query_cache.keys())[:50]:
            del st.session_state.query_cache[k]

    return documents, distances, metadatas


# ── Core response generation ──────────────────────────────────────────────────

def generate_response(user_query: str) -> tuple:
    """
    Full pipeline: detect language, classify intent, retrieve context,
    score confidence, summarize history if needed, call LLM.

    Returns:
        (response_text: str, metadata: dict)
    """
    query_lower = user_query.lower().strip()
    messages = st.session_state.messages

    # 1. Detect language
    language = detect_language(user_query)

    # 2. Follow-up detection via embeddings
    search_query = user_query
    followup = False
    if len(messages) >= 3:
        prev_user_msg = next(
            (m['content'] for m in reversed(messages[:-1]) if m['role'] == 'user'),
            None,
        )
        if prev_user_msg and is_followup(user_query, prev_user_msg):
            followup = True
            search_query = f"{prev_user_msg} {user_query}"

    # 3. Intent classification → ChromaDB category
    intent = classify_intent(query_lower)
    category = INTENT_CATEGORY_MAP.get(intent)

    # 4. KB retrieval (direct call — ThreadPoolExecutor cannot access st.session_state)
    documents, distances, metadatas = [], [], []
    try:
        documents, distances, metadatas = retrieve_context(search_query, category)
    except Exception as e:
        print(f"[ERROR] KB retrieval failed: {e}")

    # 5. Filter by distance threshold and compute confidence
    relevant_pairs = [(doc, dist, meta) for doc, dist, meta in
                      zip(documents, distances, metadatas) if dist <= DISTANCE_THRESHOLD]
    relevant_docs = [p[0] for p in relevant_pairs]
    relevant_dists = [p[1] for p in relevant_pairs]
    relevant_metas = [p[2] for p in relevant_pairs]
    # Confidence is informational only — it does NOT block the response
    confidence = score_confidence(relevant_dists) if relevant_docs else "low"

    # 6. Build source list for UI
    sources = []
    for meta, dist in zip(relevant_metas, relevant_dists):
        if isinstance(meta, dict):
            sources.append({
                "question": meta.get("Question", ""),
                "category": meta.get("Class", ""),
                "distance": round(dist, 3),
            })

    # 7. Conversation summary
    summary = maybe_summarize(messages)

    # 8. Route to the appropriate response path
    # Any document within DISTANCE_THRESHOLD is relevant enough to answer from.
    # Confidence only adds an optional caveat to the prompt — it never blocks a response.
    if relevant_docs:
        kb_context = "\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)]
        )
        if confidence == "low":
            # Docs exist but are borderline — ask Gemini to answer with a light caveat
            kb_context += (
                "\n\nNote to assistant: the retrieved context has lower relevance. "
                "Answer as helpfully as possible from the context above, "
                "and suggest the customer verify with the bank if needed."
            )
        elif confidence == "medium":
            kb_context += (
                "\n\nNote to assistant: confidence is moderate. "
                "Answer helpfully but suggest the customer verify details with the bank."
            )
        prompt = build_prompt(user_query, kb_context, messages, language, summary)
        response_text = call_llm(prompt)

    elif intent == "exchange_rate" or (is_exchange_rate_query(user_query) and not followup):
        response_text = get_exchange_rate_data(user_query)
        confidence = "high"

    else:
        response_text = (
            "I don't have enough information to answer that accurately. "
            "Could you provide more details or rephrase your question? "
            f"You can also call our Customer Support at **{SUPPORT_PHONE}**."
        )
        confidence = "low"

    metadata = {
        "sources": sources,
        "confidence_level": confidence,
        "intent": intent,
        "language": language,
    }
    return response_text, metadata


# ── Public entry point ────────────────────────────────────────────────────────

def process_query(user_query: str):
    """Add user message to history, generate response, persist both to SQLite."""
    session_id = st.session_state.session_id

    # Persist and display user message
    db_manager.save_message(session_id, "user", user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Show typing indicator
        response_placeholder.markdown(
            '<div class="typing-dots">'
            '<span></span><span></span><span></span>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.spinner(random.choice(_LOADING_MESSAGES)):
            response_text, metadata = generate_response(user_query)

        response_placeholder.markdown(response_text)

        # Persist assistant message
        msg_id = db_manager.save_message(
            session_id, "assistant", response_text, metadata
        )
        metadata["db_message_id"] = msg_id

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "metadata": metadata,
        })
