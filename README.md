# DocuSphere AI — Intelligent Document Assistant

DocuSphere AI is a Streamlit AI workspace for general conversation, uploaded documents, structured CSV/XLSX analysis, professional charts, code understanding, citations, and web-grounded current information.

## What it does

- Normal AI conversation even when no file is uploaded.
- Multi-file document ingestion: PDF, DOCX, PPTX, TXT/Markdown, CSV/XLSX, images, and common code files.
- Text extraction, OCR for images, normalization, chunking, embeddings, hybrid retrieval, and lightweight reranking.
- Conversational follow-ups with document context and source citations.
- Deterministic CSV/XLSX analytics with pandas for totals, records, date periods, growth/change, breakdowns, tables, exports, and Plotly charts.
- Complete-file dashboard requests such as “give me full details with graphs” without relying on Gemini for the numeric calculations.
- Code-file context so questions can use the actual uploaded code instead of generic snippets.
- Prompt-injection-aware document handling.
- Natural-language image requests routed to Gemini image generation instead of pretending that text output is an image.

## Important image-generation note

Gemini image generation is separate from normal text/document Gemini usage. The current Gemini 3.1 Flash Image model uses the Interactions API and supports JPEG image output configuration. Image-generation access is not available on the Gemini API Free Tier for this model, so an API key/project without image access can still use the rest of DocuSphere normally but will receive a clear image-availability message instead of a fake image answer.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` in Streamlit Secrets or the environment. Never commit a real API key to GitHub.

## Streamlit Cloud

1. Upload the extracted project contents to the repository root.
2. Main file: `app.py`.
3. Keep `requirements.txt` and `packages.txt` in the repository root.
4. Add `GEMINI_API_KEY` under Streamlit Secrets.
5. Reboot/redeploy after a code update.

Do **not** upload the ZIP itself as the application source.

## Project structure

```text
DocuSphere-AI/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── extraction.py
│   ├── chunking.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── citations.py
│   ├── security.py
│   ├── llm.py
│   ├── csv_analytics.py
│   ├── tabular.py
│   └── utils.py
├── data/.gitkeep
├── assets/.gitkeep
└── tests/
```

## Verification

The bundled regression suite covers deterministic CSV analytics, date detection, full-file reporting, and the image-output-format regression that previously caused the Gemini `image/png` error.


### Supported common file formats
PDF (including scanned/image-only pages with OCR), DOCX, PPTX, XLSX, CSV, common text/markup/config/code files, and common image formats including JPG/JPEG/PNG/WEBP/GIF/BMP/TIF/TIFF/AVIF. Legacy binary DOC/PPT and proprietary formats are not treated as universally readable.
