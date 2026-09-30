
# DocuSphere AI

**Professional intelligent AI workspace for conversation, documents, code, structured data, charts, and web-grounded questions.**

## What it does

DocuSphere is intentionally **not** only a RAG chatbot.

### 1. Normal AI conversation
You can open the app without uploading anything and ask general questions, explanations, writing requests, coding questions, brainstorming, project-management questions, etc.

### 2. File intelligence
Upload multiple:
- PDF / scanned PDFs (text extraction with OCR fallback)
- DOCX
- PPTX
- XLSX
- CSV
- TXT / Markdown
- Python / Java / C / C++ / JavaScript / TypeScript
- HTML / CSS / SQL / JSON / XML
- JPG / PNG / WEBP (native Gemini multimodal vision + OCR fallback)

The application extracts content, chunks it, embeds it, performs hybrid lexical + semantic retrieval, and sends relevant context to Gemini.

### 3. Structured-data intelligence
CSV/XLSX files are also loaded as real pandas tables.

The application can answer questions such as:
- total sales/revenue/debit/credit
- category/type/status/customer/project breakdowns
- weekly/monthly/yearly analysis when a real calendar date column exists
- daily trends
- comparisons
- record counts
- professional interactive Plotly charts
- period-over-period growth calculations when a real date field exists
- native Gemini image generation for explicit image/picture requests
- downloadable CSV result tables

**Important:** date-based questions never silently treat a time-only field as a calendar date. If a reliable date column is missing, the application says so instead of inventing a period.

### 4. Project-management analytics
The structured-data layer uses semantic column detection, so common fields such as:
`Project`, `Status`, `Owner`, `Budget`, `Cost`, `Revenue`, `Progress`, `Start Date`, `Due Date`, `Department`
can be used without hardcoding one company's schema.

### 5. Web-grounded AI
Gemini can use Google Search grounding for requests that clearly ask for current/latest/search information. The app enables this automatically; there is no manual mode switch.

### 6. Multimodal vision and image generation

Image uploads are passed to Gemini as actual image input for visual question answering, while OCR remains available as a text fallback. Explicit requests such as “generate an image”, “create a picture”, or “image banao” are routed to Gemini’s native image model and displayed as an actual image with a download option.

### 7. Code understanding
For code uploads, relevant full-file context can be supplied to Gemini so it can explain and modify code without reducing the task to one tiny retrieved chunk.

## Architecture

```text
                    ┌──────────────────────┐
                    │      User Chat       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │      Auto Router      │
                    └───┬──────┬──────┬────┘
                        │      │      │
             ┌──────────▼┐ ┌──▼────┐ ┌▼──────────┐
             │ Documents │ │ Data  │ │ Web / AI  │
             │ + RAG     │ │Pandas │ │ Gemini    │
             └─────┬─────┘ └──┬────┘ └────┬──────┘
                   │           │           │
                   └───────────┬┴───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Gemini Answer Engine │
                    └──────────┬───────────┘
                               ▼
                 Answer + Sources + Chart + Export
```

## Project files

```text
DocuSphere-AI/
├── app.py
├── requirements.txt
├── packages.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── .gitkeep
├── assets/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── extraction.py
    ├── ingestion.py
    ├── chunking.py
    ├── embeddings.py
    ├── retrieval.py
    ├── reranking.py
    ├── citations.py
    ├── security.py
    ├── llm.py
    ├── csv_analytics.py
    ├── tabular.py
    └── utils.py
```

## Verification

The repository includes a small standard-library test suite for the deterministic CSV/date layer:

```bash
python tests/test_core.py
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Cloud, add:

```toml
GEMINI_API_KEY = "YOUR_KEY"
```

Do not commit the real API key to GitHub.

## Design principles

- No document is required for normal AI chat.
- File facts and general AI knowledge are kept conceptually separate.
- Exact numerical data work is performed by Python/pandas.
- Graphs are generated from real structured data.
- Period queries require a reliable calendar date.
- Web search is used for current/external questions when selected or appropriate.
- Uploaded documents are treated as untrusted data, not instructions.
- The main product surface does not include experimental RAG evaluation UI.
