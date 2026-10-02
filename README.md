# DocuSphere AI — Professional Audited Build

DocuSphere AI is a single-chat AI workspace for general questions, documents, CSV/XLSX analysis, graphs, image generation, code understanding, current/web questions, exports, and bank reconciliation.

## Bank reconciliation — core behavior

- Bank and Software original serial numbers **do not need to match**.
- Original source serials remain unchanged.
- A new shared reconciliation ID is generated only for a safely matched pair, e.g. `REC-000001`.
- Amount alone is never sufficient evidence.
- Matching is one-to-one.
- Reference/description, amount, date and debit/credit direction are checked conservatively.
- Duplicate or ambiguous candidates remain **UNMATCHED** rather than being guessed.
- Matched and unmatched outputs can be downloaded separately as CSV/Excel when the Excel row limit permits.
- Reconciliation is deterministic and does not depend on Gemini.

## AI behavior

The chat uses natural-language routing; the user does not select a mode. Deterministic features such as reconciliation, exact tabular calculations, file profiles, exports and data-driven charts are handled locally. Gemini is used for general AI, file-grounded explanation, web-grounded answers and image generation.

Simple greetings and basic conversational inputs are handled locally, so they do not consume an API request. If a Gemini provider error occurs, the app attempts the configured stable fallback model when appropriate and uses bounded retries and fallback handling for provider failures without exposing a generic empty-response message.

## Graphs and images

- A data graph is generated from the actual uploaded data.
- If the user asks for a graph as a picture, the app provides the interactive graph **and a PNG download**.
- A creative image request is routed to the image model and provides a downloadable image when generation succeeds.

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

The audited source tree passes the automated test suite and Python compilation checks. The test count is intentionally reported from the actual run; no synthetic pass count is claimed.

## Important capacity note

No honest software build can guarantee a fixed maximum row count such as 1 crore+ on every deployment regardless of RAM/CPU. The reconciliation algorithm avoids a full Bank×Software Cartesian comparison and uses indexed candidate lookup, but real maximum capacity still depends on deployment resources and file format.
