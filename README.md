# BC Bank Chatbot

An AI-powered banking assistant built with Streamlit, ChromaDB, Sentence Transformers, and Google Gemini 2.5 Flash. It answers customer queries about accounts, cards, loans, insurance, investments, fund transfers, security, and live currency exchange rates.

---

## Prerequisites

- **Python 3.12** or higher
- A **Google Gemini API key** — get one free at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Internet connection (for Gemini API, ChromaDB Cloud, and live exchange rates)

---

## Setup Instructions

### Step 1 — Clone or download the project

Place all project files in a folder, e.g.:
```
C:/your-path/final/
```

### Step 2 — Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `chromadb` | Vector database for knowledge base |
| `sentence-transformers` | Embedding model for semantic search |
| `google-generativeai` | Gemini API client |
| `pandas` | CSV data processing (for populate_db.py) |
| `requests` | HTTP calls (exchange rate API) |
| `tenacity` | Retry logic with exponential backoff |
| `langdetect` | Detect query language |
| `numpy` | Vector math for follow-up detection |
| `pytest` | Unit testing |

### Step 4 — Set the Gemini API key

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**macOS / Linux:**
```bash
export GEMINI_API_KEY=your_api_key_here
```

> **Tip:** To avoid setting this every session, add it to your system environment variables permanently.

### Step 5 — Populate the knowledge base (first time only)

The ChromaDB cloud collection must be populated before the chatbot can answer questions. Run this **once**:

```bash
python populate_db.py
```

> **Note:** `populate_db.py` expects the CSV files (`BankFAQs_Part1.csv` through `BankFAQs_Part7.csv`) to be present. Update the file paths inside `populate_db.py` if they are stored in a different location on your machine.

### Step 6 — Run the chatbot

```bash
streamlit run chatbot_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Project Structure

```
final/
├── chatbot_app.py          # Main entry point
├── config.py               # All tunable constants (thresholds, limits, etc.)
├── db_manager.py           # SQLite persistence for chat history
├── session_manager.py      # Session init: ChromaDB, embedding model, DB
├── query_processor.py      # Core pipeline: intent → retrieval → LLM
├── gemini_api.py           # Gemini API wrapper with retry and summarization
├── exchange_rate_utils.py  # Live currency exchange rate queries
├── ui_utils.py             # Streamlit UI: sidebar, header, chat display
├── populate_db.py          # One-time script to load FAQs into ChromaDB
├── requirements.txt        # Python dependencies
├── resources/
│   ├── styles.css                    # Custom CSS (dark/light themes, mobile)
│   ├── categories.json               # Banking category keyword mappings
│   ├── exchange_rate_mapping.json    # Currency code to name mappings
│   ├── detailed_prompt_template.txt  # Gemini prompt template
│   └── brief_prompt_template.txt     # Compact prompt template
├── tests/
│   ├── test_query_processor.py       # Unit tests for query logic
│   └── test_exchange_rate_utils.py   # Unit tests for currency detection
└── chat_history.db         # SQLite DB (auto-created on first run)
```

---

## Running Unit Tests

```bash
pytest tests/
```

---

## Features

- **AI responses** powered by Google Gemini 2.5 Flash
- **Semantic search** over bank FAQ knowledge base using ChromaDB + Sentence Transformers
- **Intent classification** routes queries to the right knowledge category
- **Follow-up detection** via embedding cosine similarity
- **Conversation summarization** for long sessions
- **Multi-language support** — responds in the language the user writes in
- **Live exchange rates** with 5-minute caching
- **Retry logic** on all external API calls (Gemini, ChromaDB, exchange rate)
- **Persistent chat history** stored in local SQLite database
- **Dark / Light theme** toggle in sidebar
- **Category navigation** sidebar with quick-access topic buttons
- **FAQ panel** with top 5 popular questions
- **Message timestamps** on every message
- **Mobile-friendly** responsive layout

---

## Configuration

All tunable constants are in `config.py`. Key settings:

| Constant | Default | Description |
|---|---|---|
| `DISTANCE_THRESHOLD` | `0.8` | Max ChromaDB distance for a result to be used |
| `HISTORY_WINDOW` | `8` | Number of recent messages included in the prompt |
| `SUMMARY_TRIGGER` | `16` | Message count that triggers conversation summarization |
| `CACHE_TTL_SECONDS` | `300` | Exchange rate cache duration (5 minutes) |
| `FOLLOWUP_SIMILARITY_THRESHOLD` | `0.65` | Cosine similarity threshold for follow-up detection |
| `RETRY_MAX_ATTEMPTS` | `3` | Max retries on failed API calls |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY environment variable not set` | Set the env variable (Step 4 above) |
| `Error connecting to ChromaDB` | Run `populate_db.py` first (Step 5 above) |
| App gives fallback answer for every query | ChromaDB collection may be empty — re-run `populate_db.py` |
| Exchange rate not loading | Check internet connection; the API has a free-tier rate limit |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your active virtual environment |
