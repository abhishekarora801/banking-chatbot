# session_manager.py — Session state initialisation, ChromaDB connection, and model loading

import uuid

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

import db_manager

# Welcome message shown to new sessions
_WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "👋 Welcome to BC Bank Chatbot! How can I assist you today?",
}


def initialize_session_state():
    """
    Initialise all session-level resources.
    Safe to call on every Streamlit rerun — guards with 'not in session_state'.
    """
    # ── Session ID ────────────────────────────────────────────────────────────
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # ── SQLite DB ─────────────────────────────────────────────────────────────
    db_manager.init_db()

    # ── Chat history (load from SQLite or start fresh) ────────────────────────
    if 'messages' not in st.session_state:
        persisted = db_manager.load_messages(st.session_state.session_id)
        if persisted:
            st.session_state.messages = persisted
        else:
            # First visit — save and display welcome message
            msg_id = db_manager.save_message(
                st.session_state.session_id,
                _WELCOME_MESSAGE['role'],
                _WELCOME_MESSAGE['content'],
            )
            welcome = dict(_WELCOME_MESSAGE, id=msg_id)
            st.session_state.messages = [welcome]

    # ── Sentence-Transformer embedding model ──────────────────────────────────
    if 'embed_model' not in st.session_state:
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Embedding model loaded.")

    # ── ChromaDB collection ───────────────────────────────────────────────────
    if 'collection' not in st.session_state:
        _connect_to_chromadb()

    # ── Query cache ───────────────────────────────────────────────────────────
    if 'query_cache' not in st.session_state:
        st.session_state.query_cache = {}


def _connect_to_chromadb():
    """Connect to the ChromaDB cloud collection."""
    try:
        client = chromadb.CloudClient(
            api_key='ck-7bD95NuNebzB4NuD8goHvnDrt1GCqfKEbghob6bkRRMS',
            tenant='f780702a-cea3-4602-b65f-e7f41f6546ec',
            database='my-project',
        )
        collection = client.get_collection(name="knowledge_base")
        st.session_state.collection = collection
        print(f"[INFO] Connected to ChromaDB collection ({collection.count()} entries).")
    except Exception as e:
        print(f"[ERROR] ChromaDB connection failed: {e}")
        st.error(
            f"Could not connect to the knowledge base: {e}. "
            "Please run populate_db.py first to create the collection."
        )
