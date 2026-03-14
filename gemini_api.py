# gemini_api.py — Google Gemini API wrapper with retry and summarization

import os

import google.generativeai as genai
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    HISTORY_WINDOW,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)

GEMINI_MODEL = "gemini-2.5-flash"

GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.4,
    top_p=0.92,
    top_k=40,
    max_output_tokens=2500,
)

# Module-level prompt template cache
_PROMPT_TEMPLATE: str | None = None


def get_prompt_template() -> str:
    """Load and cache the detailed prompt template."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_path = os.path.join('resources', 'detailed_prompt_template.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            _PROMPT_TEMPLATE = f.read()
    return _PROMPT_TEMPLATE


def _get_model():
    """Return a cached Gemini model instance from session state."""
    if 'gemini_model' not in st.session_state:
        model = genai.GenerativeModel(GEMINI_MODEL)
        st.session_state.gemini_model = model
        st.session_state.generation_config = GENERATION_CONFIG
    return st.session_state.gemini_model, st.session_state.generation_config


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    reraise=True,
)
def _call_gemini(prompt: str) -> str:
    """Call Gemini API with tenacity retry on transient failures."""
    model, config = _get_model()
    response = model.generate_content(prompt, generation_config=config)
    return response.text


def build_prompt(query: str, context: str, history: list,
                 language: str = "English", summary: str = "") -> str:
    """
    Assemble the full prompt from the template, injecting all dynamic values.

    Args:
        query:    Current user question.
        context:  Knowledge base excerpts.
        history:  Full message list (dicts with 'role'/'content').
        language: Detected language for the response.
        summary:  Optional summary of older conversation turns.
    """
    recent = history[-HISTORY_WINDOW:] if len(history) > HISTORY_WINDOW else history
    history_lines = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ]
    history_text = "\n".join(history_lines)

    template = get_prompt_template()
    return template.format(
        history_text=history_text,
        query=query,
        context=context,
        language=language,
        summary=summary,
    )


def call_llm(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the text response.
    Wraps _call_gemini so callers don't need to know about retries.
    """
    try:
        return _call_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Gemini API failed after retries: {e}")
        return f"I'm having trouble connecting right now. Please try again in a moment."


def get_gemini_response(query: str, context: str, history: list,
                        language: str = "English", summary: str = "") -> str:
    """
    High-level helper: build prompt then call Gemini.
    Kept for backwards compatibility with any callers that import it directly.
    """
    prompt = build_prompt(query, context, history, language, summary)
    return call_llm(prompt)


def summarize_conversation(old_turns: list) -> str:
    """
    Ask Gemini to produce a short summary of older conversation turns.
    Used when history exceeds SUMMARY_TRIGGER messages.

    Args:
        old_turns: List of message dicts (role/content) to summarise.

    Returns:
        A 2-3 sentence plain-text summary string.
    """
    lines = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in old_turns
    ]
    history_text = "\n".join(lines)
    prompt = (
        "Summarise the following banking chatbot conversation in 2-3 concise sentences. "
        "Focus on the topics discussed and any decisions or information provided. "
        "Do not include greetings or pleasantries.\n\n"
        f"Conversation:\n{history_text}\n\nSummary:"
    )
    try:
        return _call_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        return ""


def initialize_gemini_api(api_key: str):
    """Configure the Gemini SDK with the provided API key."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
