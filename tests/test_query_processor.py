# tests/test_query_processor.py
# Run with: pytest tests/

import sys
import os

import numpy as np
import pytest

# Add project root to path so imports work without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_processor import classify_intent, score_confidence, _cosine_similarity
from config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MED_THRESHOLD,
    INTENT_CATEGORY_MAP,
)


# ── classify_intent ───────────────────────────────────────────────────────────

class TestClassifyIntent:
    def test_account_info(self):
        assert classify_intent("what is my account balance") == "account_info"

    def test_card_services(self):
        assert classify_intent("i lost my debit card") == "card_services"

    def test_loans(self):
        assert classify_intent("i want to apply for a home loan") == "loans"

    def test_insurance(self):
        assert classify_intent("what is the premium for life insurance") == "insurance"

    def test_investments(self):
        assert classify_intent("tell me about fixed deposit returns") == "investments"

    def test_funds_transfer(self):
        assert classify_intent("how do i do a neft transfer") == "funds_transfer"

    def test_security(self):
        assert classify_intent("i received a suspicious otp") == "security"

    def test_exchange_rate(self):
        assert classify_intent("convert usd to eur") == "exchange_rate"

    def test_complaint(self):
        assert classify_intent("i have a complaint about my transaction") == "complaint"

    def test_general_fallback(self):
        assert classify_intent("hello how are you") == "general"


# ── INTENT_CATEGORY_MAP round-trip ────────────────────────────────────────────

class TestIntentCategoryMap:
    def test_account_info_maps_to_accounts(self):
        assert INTENT_CATEGORY_MAP["account_info"] == "accounts"

    def test_card_services_maps_to_cards(self):
        assert INTENT_CATEGORY_MAP["card_services"] == "cards"

    def test_general_maps_to_none(self):
        assert INTENT_CATEGORY_MAP["general"] is None

    def test_exchange_rate_maps_to_none(self):
        assert INTENT_CATEGORY_MAP["exchange_rate"] is None


# ── score_confidence ──────────────────────────────────────────────────────────

class TestScoreConfidence:
    def test_high_confidence(self):
        dist = CONFIDENCE_HIGH_THRESHOLD - 0.05
        assert score_confidence([dist]) == "high"

    def test_medium_confidence(self):
        dist = (CONFIDENCE_HIGH_THRESHOLD + CONFIDENCE_MED_THRESHOLD) / 2
        assert score_confidence([dist]) == "medium"

    def test_low_confidence(self):
        dist = CONFIDENCE_MED_THRESHOLD + 0.1
        assert score_confidence([dist]) == "low"

    def test_empty_distances_returns_low(self):
        assert score_confidence([]) == "low"

    def test_uses_minimum_distance(self):
        # Even if one distance is low, best (min) should determine confidence
        assert score_confidence([0.9, 0.1]) == "high"


# ── _cosine_similarity ────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0])
        assert abs(_cosine_similarity(v, -v) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        v = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        assert _cosine_similarity(v, u) == 0.0


# ── build_prompt (integration, no API call) ───────────────────────────────────

class TestBuildPrompt:
    def test_contains_query(self):
        from gemini_api import build_prompt
        query = "What are the loan types?"
        prompt = build_prompt(
            query=query,
            context="Document 1: BC Bank offers personal and home loans.",
            history=[],
            language="English",
            summary="",
        )
        assert query in prompt

    def test_contains_context(self):
        from gemini_api import build_prompt
        context = "BC Bank offers FD with 7% interest."
        prompt = build_prompt(
            query="Tell me about FD",
            context=context,
            history=[],
        )
        assert context in prompt

    def test_language_in_prompt(self):
        from gemini_api import build_prompt
        prompt = build_prompt(
            query="What is UPI?",
            context="UPI is a payment method.",
            history=[],
            language="Hindi",
        )
        assert "Hindi" in prompt

    def test_summary_in_prompt(self):
        from gemini_api import build_prompt
        summary = "Customer asked about savings accounts earlier."
        prompt = build_prompt(
            query="What about FD?",
            context="FD info here.",
            history=[],
            summary=summary,
        )
        assert summary in prompt

    def test_history_is_limited_to_window(self):
        from gemini_api import build_prompt
        from config import HISTORY_WINDOW
        # Create more messages than the window
        history = [{"role": "user", "content": f"msg {i}"} for i in range(HISTORY_WINDOW + 5)]
        prompt = build_prompt("latest query", "context", history)
        # First message (outside window) should not appear
        assert "msg 0" not in prompt
        # Last message (inside window) should appear
        assert f"msg {HISTORY_WINDOW + 4}" in prompt
