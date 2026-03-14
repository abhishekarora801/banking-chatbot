# chatbot_app.py — Main application entry point

import os

import streamlit as st

from gemini_api import initialize_gemini_api
from query_processor import process_query
from session_manager import initialize_session_state
from ui_utils import display_chat_history, render_sidebar, setup_ui


def main():
    """Main application entry point."""

    # 1. Initialise session (DB, embed model, ChromaDB, history load)
    initialize_session_state()

    # 2. Page config, CSS, header, disclaimer
    setup_ui()

    # 3. Sidebar (theme toggle, category nav, FAQs, clear chat)
    render_sidebar()

    # 4. Gemini API key validation
    try:
        initialize_gemini_api(os.environ.get("GEMINI_API_KEY"))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # 5. Display persisted chat history
    display_chat_history()

    # 6. Handle prefill_query from sidebar/quick-reply buttons
    prefill = st.session_state.pop("prefill_query", None)

    # 7. Chat input — prefilled value wins if set
    user_query = st.chat_input("Ask a question about banking…")
    active_query = prefill or user_query

    # 8. Process the active query
    if active_query:
        process_query(active_query)


if __name__ == "__main__":
    main()
