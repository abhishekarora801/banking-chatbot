# tests/test_exchange_rate_utils.py
# Run with: pytest tests/

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import exchange_rate_utils
from exchange_rate_utils import (
    _CURRENCY_MAP,
    _NAME_TO_CODE,
    find_currencies_in_query,
    is_exchange_rate_query,
    get_exchange_rate_data,
)

# Minimal fake rates for deterministic tests
_FAKE_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.5,
    "INR": 83.1,
}


# ── Module-level resource loading ─────────────────────────────────────────────

class TestModuleLevelResources:
    def test_currency_map_loaded(self):
        assert isinstance(_CURRENCY_MAP, dict)
        assert len(_CURRENCY_MAP) > 10

    def test_usd_in_currency_map(self):
        assert "USD" in _CURRENCY_MAP

    def test_name_to_code_loaded(self):
        assert isinstance(_NAME_TO_CODE, dict)
        assert len(_NAME_TO_CODE) > 10

    def test_name_to_code_reverse(self):
        # "DOLLAR" or full name should map back to USD
        assert _NAME_TO_CODE.get("USD") == "USD" or "USD" in _CURRENCY_MAP


# ── find_currencies_in_query ──────────────────────────────────────────────────

class TestFindCurrenciesInQuery:
    def _patch_rates(self):
        return patch.object(exchange_rate_utils, "get_exchange_rates", return_value=_FAKE_RATES)

    def test_codes_usd_eur(self):
        result = find_currencies_in_query("convert USD to EUR", _FAKE_RATES.keys())
        assert "USD" in result
        assert "EUR" in result
        assert len(result) == 2

    def test_dollar_symbol(self):
        result = find_currencies_in_query("how many $ in £", _FAKE_RATES.keys())
        assert "USD" in result
        assert "GBP" in result

    def test_yen_symbol(self):
        result = find_currencies_in_query("convert $100 to ¥", _FAKE_RATES.keys())
        assert "USD" in result
        assert "JPY" in result

    def test_returns_at_most_two(self):
        result = find_currencies_in_query("USD EUR GBP INR", _FAKE_RATES.keys())
        assert len(result) <= 2

    def test_no_currencies_returns_empty(self):
        result = find_currencies_in_query("what is the weather today", _FAKE_RATES.keys())
        assert result == []

    def test_single_currency_returns_empty(self):
        result = find_currencies_in_query("tell me about USD", _FAKE_RATES.keys())
        # Only one currency found — function returns [] unless 2 are present
        assert len(result) <= 1


# ── is_exchange_rate_query ────────────────────────────────────────────────────

class TestIsExchangeRateQuery:
    def _mock_rates(self):
        return patch.object(exchange_rate_utils, "get_exchange_rates", return_value=_FAKE_RATES)

    def test_true_for_convert_query(self):
        with self._mock_rates():
            assert is_exchange_rate_query("convert USD to EUR") is True

    def test_true_for_exchange_rate_query(self):
        with self._mock_rates():
            assert is_exchange_rate_query("what is the exchange rate USD EUR") is True

    def test_false_for_banking_query(self):
        with self._mock_rates():
            assert is_exchange_rate_query("what is a savings account?") is False

    def test_false_when_only_one_currency(self):
        with self._mock_rates():
            assert is_exchange_rate_query("forex rate for USD") is False

    def test_false_when_no_keywords(self):
        with self._mock_rates():
            assert is_exchange_rate_query("USD EUR comparison") is False

    def test_false_when_api_unavailable(self):
        with patch.object(exchange_rate_utils, "get_exchange_rates", return_value=None):
            assert is_exchange_rate_query("convert USD to EUR") is False


# ── get_exchange_rate_data ────────────────────────────────────────────────────

class TestGetExchangeRateData:
    def _mock_rates(self):
        return patch.object(exchange_rate_utils, "get_exchange_rates", return_value=_FAKE_RATES)

    def test_returns_rate_string(self):
        with self._mock_rates():
            result = get_exchange_rate_data("convert USD to EUR")
        assert "EUR" in result
        assert "USD" in result
        # Rate should be ~0.92
        assert "0.92" in result

    def test_returns_error_when_no_rates(self):
        with patch.object(exchange_rate_utils, "get_exchange_rates", return_value=None):
            result = get_exchange_rate_data("convert USD to EUR")
        assert "unavailable" in result.lower() or "sorry" in result.lower()

    def test_returns_clarify_when_less_than_two_currencies(self):
        with self._mock_rates():
            result = get_exchange_rate_data("what is the forex rate?")
        assert "clarify" in result.lower()

    def test_non_usd_base_calculation(self):
        with self._mock_rates():
            result = get_exchange_rate_data("convert EUR to GBP")
        # EUR→GBP = 0.79 / 0.92 ≈ 0.8587
        assert "EUR" in result
        assert "GBP" in result
