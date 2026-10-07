# DocuSphere AI — Integrated Professional Build

This build keeps the general AI/document/data workspace while adding conservative bank reconciliation.

## Core behavior
- One natural-language chat; no manual mode selection.
- General AI, document Q&A/RAG, exact CSV/XLSX analysis, charts, image generation, PDF/CSV/XLSX exports, code understanding, and current/web questions remain available.
- Uploaded CSV/XLSX files are read with tolerant header detection for report titles, blank preambles, and generic sheet names.
- Reconciliation accepts separate bank/software files or multiple workbook sheets.

## Reconciliation safety
- Bank and software original serial/voucher numbers do not need to be equal.
- Original serials are preserved unchanged.
- A new shared `REC-000001` style ID is created only for a safe one-to-one match.
- Amount alone never matches.
- If both sides contain dates, a date mismatch beyond the configured tolerance blocks the match even when description/reference/amount agree.
- Debit/credit direction conflicts block the match.
- Duplicate/ambiguous candidates remain unmatched.
- Non-transaction rows without a usable amount/debit/credit value are not exported as transactions.
- Matched and unmatched results are available as CSV; Excel is also offered when the Excel row limit permits it.
- Download artifacts are persisted in chat history so a download click does not remove the other download buttons after Streamlit reruns.

## Capacity
No fixed row-count guarantee is claimed. CSV/XLSX processing uses pandas and therefore depends on available RAM/CPU and deployment limits. Excel itself has a 1,048,576-row worksheet limit; CSV remains the complete export for larger results.
