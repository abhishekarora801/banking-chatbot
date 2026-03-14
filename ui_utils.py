# ui_utils.py — UI configuration, rendering, and interactive components

from datetime import datetime, timezone

import streamlit as st

import db_manager
from config import SUPPORT_TOLLFREE

# ── Static content ────────────────────────────────────────────────────────────

_CATEGORY_QUERIES = {
    "🏦 Accounts":      "What types of accounts does BC Bank offer?",
    "💳 Cards":         "Tell me about credit and debit card options at BC Bank.",
    "💸 Fund Transfer": "How do I transfer money using NEFT, RTGS, or UPI?",
    "🏠 Loans":         "What loan products does BC Bank offer and how do I apply?",
    "🛡️ Insurance":     "What insurance products are available at BC Bank?",
    "📈 Investments":   "Tell me about fixed deposits, mutual funds, and other investment options.",
    "🔒 Security":      "How can I keep my BC Bank account safe from fraud?",
}

_TOP_FAQS = [
    "How do I open a savings account?",
    "How do I block my lost or stolen card?",
    "What are the current home loan interest rates?",
    "How do I reset my net banking password?",
    "What documents are required for a personal loan?",
]

# ── CSS helpers ───────────────────────────────────────────────────────────────

@st.cache_data
def _load_base_css() -> str:
    try:
        with open("resources/styles.css", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to load CSS: {e}")
        return ""


def _theme_css(theme: str) -> str:
    """Return CSS variable overrides for light theme (dark is the default in styles.css)."""
    if theme == "Light":
        return """
        <style>
        :root {
          --bg-primary:     #f7f8fa;
          --bg-secondary:   #ffffff;
          --bg-surface:     #edf2f7;
          --text-primary:   #1a202c;
          --text-secondary: #4a5568;
          --text-muted:     #718096;
          --accent:         #2b6cb0;
          --accent-hover:   #2c5282;
          --border:         #e2e8f0;
          --chip-bg:        #e2e8f0;
          --chip-hover:     #cbd5e0;
          --chip-text:      #2d3748;
          --shadow:         rgba(0,0,0,0.12);
          --input-bg:       #ffffff;
        }
        </style>
        """
    return ""


# ── Page setup ────────────────────────────────────────────────────────────────

def setup_ui():
    """Configure page, inject CSS, render header and disclaimer."""
    st.set_page_config(
        page_title="BC Bank Assistant",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Base CSS
    st.markdown(f"<style>{_load_base_css()}</style>", unsafe_allow_html=True)

    # Theme override (applied after base so it wins on specificity)
    theme = st.session_state.get("theme", "Dark")
    st.markdown(_theme_css(theme), unsafe_allow_html=True)

    # Header
    st.markdown(
        "<h1 class='header-title'>🏦 BC Bank Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='header-subtitle'>"
        "Ask questions about accounts, cards, loans, insurance, investments, or exchange rates."
        "</p>",
        unsafe_allow_html=True,
    )

    # Disclaimer banner
    st.info(
        f"This chatbot provides **general information only**. "
        f"For account-specific queries, please call **{SUPPORT_TOLLFREE}** "
        f"or visit your nearest BC Bank branch.",
        icon="ℹ️",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render all sidebar elements: theme, categories, FAQs, clear chat."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        _render_theme_toggle()

        st.divider()
        st.markdown("### 🗂️ Browse Topics")
        _render_category_nav()

        st.divider()
        st.markdown("### ❓ Popular Questions")
        _render_faq_panel()

        st.divider()
        _render_clear_chat()


def _render_theme_toggle():
    theme = st.radio(
        "Theme",
        options=["Dark", "Light"],
        index=0 if st.session_state.get("theme", "Dark") == "Dark" else 1,
        key="theme",
        horizontal=True,
    )
    return theme


def _render_category_nav():
    for label, query in _CATEGORY_QUERIES.items():
        if st.button(label, key=f"cat_{label}", use_container_width=True):
            st.session_state.prefill_query = query
            st.rerun()


def _render_faq_panel():
    with st.expander("Frequently Asked Questions", expanded=False):
        for i, faq in enumerate(_TOP_FAQS):
            if st.button(faq, key=f"faq_{i}", use_container_width=True):
                st.session_state.prefill_query = faq
                st.rerun()


def _render_clear_chat():
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        session_id = st.session_state.get("session_id")
        if session_id:
            db_manager.clear_session(session_id)
        # Reset in-memory state
        for key in ("messages", "query_cache", "conversation_summary",
                    "exchange_rate_cache", "prefill_query"):
            st.session_state.pop(key, None)
        st.rerun()


# ── Chat history display ──────────────────────────────────────────────────────

def display_chat_history():
    """Render all messages with timestamps only."""
    messages = st.session_state.messages

    for message in messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            # Timestamp
            ts_raw = message.get("timestamp")
            if ts_raw:
                try:
                    dt = datetime.fromisoformat(ts_raw)
                    ts_display = dt.astimezone().strftime("%I:%M %p")
                except Exception:
                    ts_display = ts_raw
                st.markdown(
                    f"<div class='msg-timestamp'>{ts_display}</div>",
                    unsafe_allow_html=True,
                )

    # Spacer for fixed input bar
    st.markdown("<br><br><br>", unsafe_allow_html=True)
