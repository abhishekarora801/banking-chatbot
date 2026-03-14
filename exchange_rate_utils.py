# exchange_rate_utils.py — Currency detection and exchange rate retrieval

import re
import json
import time

import requests
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CACHE_TTL_SECONDS, RETRY_MAX_ATTEMPTS, RETRY_WAIT_MIN, RETRY_WAIT_MAX

# ── Module-level constants (loaded once) ─────────────────────────────────────

def _load_currency_map() -> dict:
    try:
        with open('resources/exchange_rate_mapping.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load currency mappings: {e}")
        return {}


_CURRENCY_MAP: dict = _load_currency_map()

# Build reverse mapping: "UNITED STATES DOLLAR" → "USD", "DOLLAR" → "USD", etc.
_NAME_TO_CODE: dict = {}
for _code, _name in _CURRENCY_MAP.items():
    _NAME_TO_CODE[_name.upper()] = _code
    for _word in _name.upper().split():
        if _word not in ('AND', 'OF', 'NEW'):
            _NAME_TO_CODE.setdefault(_word, _code)


# ── Exchange rate fetching with retry + TTL cache ────────────────────────────

@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    reraise=True,
)
def _fetch_rates() -> dict | None:
    """Fetch live rates from ExchangeRate-API (with tenacity retry)."""
    api_url = "https://v6.exchangerate-api.com/v6/1132ea8d26455cd85dfe935e/latest/USD"
    response = requests.get(api_url, timeout=10)
    data = response.json()
    if data.get("result") == "success":
        return data["conversion_rates"]
    return None


def get_exchange_rates() -> dict | None:
    """
    Return exchange rates, using a 5-minute session cache to avoid
    hitting the API on every query.
    """
    now = time.time()
    cached = st.session_state.get('exchange_rate_cache')
    if cached and (now - cached['timestamp']) < CACHE_TTL_SECONDS:
        return cached['rates']

    try:
        rates = _fetch_rates()
    except Exception as e:
        print(f"[ERROR] Exchange rate API failed after retries: {e}")
        rates = None

    if rates:
        st.session_state['exchange_rate_cache'] = {'rates': rates, 'timestamp': now}
    return rates


# ── Currency detection ────────────────────────────────────────────────────────

def find_currencies_in_query(query: str, valid_currencies) -> list:
    """Extract up to two currency codes from the query string."""
    query_upper = query.upper()
    currencies = []

    # 1. Explicit 3-letter codes (USD, EUR, …)
    codes = re.findall(r'\b[A-Z]{3}\b', query_upper)
    currencies.extend([c for c in codes if c in valid_currencies])
    if len(currencies) >= 2:
        return currencies[:2]

    # 2. Currency symbols
    symbols = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
    for symbol, code in symbols.items():
        if symbol in query and code in valid_currencies and code not in currencies:
            currencies.append(code)
    if len(currencies) >= 2:
        return currencies[:2]

    # 3. Multi-word and single-word currency names (uses module-level _NAME_TO_CODE)
    query_words = query_upper.split()
    i = 0
    while i < len(query_words):
        for j in range(min(5, len(query_words) - i), 0, -1):
            phrase = " ".join(query_words[i:i + j])
            if phrase in _NAME_TO_CODE:
                code = _NAME_TO_CODE[phrase]
                if code in valid_currencies and code not in currencies:
                    currencies.append(code)
                    i += j - 1
                    break
        i += 1
        if len(currencies) >= 2:
            return currencies[:2]

    # 4. Individual word fallback
    for word in query_words:
        if word in _NAME_TO_CODE:
            code = _NAME_TO_CODE[word]
            if code in valid_currencies and code not in currencies:
                currencies.append(code)
        if len(currencies) >= 2:
            return currencies[:2]

    return currencies[:2] if len(currencies) >= 2 else []


def is_exchange_rate_query(query: str) -> bool:
    """Return True if the query asks about an exchange rate and contains two currencies."""
    query_lower = query.lower()
    keywords = ['exchange rate', 'currency conversion', 'convert currency',
                'forex rate', 'fx rate', 'convert']
    if not any(kw in query_lower for kw in keywords):
        return False
    rates = get_exchange_rates()
    if not rates:
        return False
    return len(find_currencies_in_query(query, rates.keys())) >= 2


def get_exchange_rate_data(query: str) -> str:
    """Compute and return a formatted exchange rate string for the query."""
    try:
        rates = get_exchange_rates()
        if not rates:
            return "Sorry, exchange rate data is temporarily unavailable. Please try again shortly."

        currencies = find_currencies_in_query(query, rates.keys())
        if len(currencies) < 2:
            return "Could you please clarify which currencies you'd like to convert between?"

        from_curr, to_curr = currencies[:2]
        rate = rates[to_curr] if from_curr == "USD" else rates[to_curr] / rates[from_curr]
        from_name = _CURRENCY_MAP.get(from_curr, from_curr)
        to_name = _CURRENCY_MAP.get(to_curr, to_curr)
        return (
            f"The current exchange rate from **{from_curr}** ({from_name}) "
            f"to **{to_curr}** ({to_name}) is **{rate:.4f}**.\n\n"
            "_Rates are sourced live and may vary slightly from market rates._"
        )
    except Exception as e:
        print(f"[ERROR] Exchange rate calculation failed: {e}")
        return "Sorry, I couldn't retrieve the exchange rate at this moment. Please try again."
