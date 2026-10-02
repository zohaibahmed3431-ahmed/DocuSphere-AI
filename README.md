# DocuSphere AI — Professional Thorough Final v13

DocuSphere AI is a single-chat AI workspace for general questions, documents, CSV/XLSX analysis, graphs, image generation, code understanding, current/web questions, exports, and bank reconciliation.

## Bank reconciliation — current first priority

- Bank and Software original serial numbers **do not need to match**.
- Original source serials remain unchanged and are preserved in the output.
- A new shared reconciliation ID is generated only for a safely matched pair, e.g. `REC-000001`.
- Amount alone is **never** sufficient evidence.
- Matching is one-to-one.
- Reference/description, amount, date and debit/credit direction are checked conservatively.
- Different debit/credit conventions can be handled when there is strong identity evidence.
- Duplicate or ambiguous candidates remain **UNMATCHED** rather than being guessed.
- A single XLSX workbook containing multiple sheets can be inspected for a Bank/Statement sheet and a Software/Ledger sheet.
- Matched and unmatched outputs are provided separately as CSV and, when within Excel's worksheet limit, Excel workbooks.
- Natural-language reconciliation requests are recognized without requiring a mode selector.
- The matching engine uses indexed candidate lookup rather than a Bank×Software Cartesian comparison.

## Conversation and AI behavior

The user asks naturally in one chat. DocuSphere decides which processing path is appropriate. Simple greetings and frustration/feedback messages are handled as conversation, with intent-aware replies rather than parroting the user's wording or consuming an unnecessary AI request. General AI questions, file-grounded explanations, web-grounded questions, coding, and image generation can use Gemini when configured.

For exact financial matching, the application combines schema understanding, conservative text/identity matching, indexed candidate selection, and explicit validation rules so an AI/provider response cannot silently invent a financial match.

## Graphs and images

- Data graphs are generated from actual uploaded structured data.
- If a graph is requested as a picture, the app can provide the real data graph plus a PNG download.
- Creative image requests are routed to the configured image model and provide a downloadable image when generation succeeds.

## Deployment

Set these Streamlit Secrets:

```toml
GEMINI_API_KEY = "your_key"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
```

Do not commit a real API key.

## Verification

The source tree passes the automated test suite and Python compilation checks. The latest local run was **64 passed** from the project root. A fresh 100,000-row Bank + 100,000-row Software synthetic reconciliation matched **100,000/100,000** in about **3.4 seconds** in the available environment. This is a benchmark, not a deployment capacity guarantee.

A Streamlit server was not started in this build environment because the Streamlit executable is not installed there; deployment/runtime behavior should therefore also be checked in the target Streamlit environment. The package layout places `app.py`, `src/`, `tests/`, and deployment files at the ZIP root so it can be run directly after extraction.

## Important capacity note

No honest software build can guarantee a fixed maximum such as 1 crore+ rows on every deployment regardless of RAM/CPU. The reconciliation algorithm avoids a full Bank×Software Cartesian comparison and uses indexed candidate lookup, but real maximum capacity still depends on deployment resources and file format.
