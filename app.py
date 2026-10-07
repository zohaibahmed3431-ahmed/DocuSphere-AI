from __future__ import annotations

import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant
from src.security import sanitize_question
from src.citations import format_sources
from src.csv_analytics import analyze_csv, dataframe_to_csv_bytes, dataframe_to_excel_bytes, is_full_details_request, full_file_report
from src.tabular import read_tabular, dataframe_context
from src.utils import is_code_file
from src.pdf_export import build_pdf_report
from src.reconciliation import reconcile, classify_frames, csv_bytes, matched_excel_bytes, unmatched_excel_bytes

load_dotenv()

st.set_page_config(page_title="DocuSphere AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.5rem;padding-bottom:5rem}
[data-testid="stSidebar"]{min-width:300px;max-width:340px}
.hero{padding:1.35rem 1.5rem;border:1px solid rgba(128,128,128,.20);border-radius:20px;background:linear-gradient(135deg,rgba(90,80,180,.12),rgba(30,120,180,.08));margin-bottom:1rem}
.hero h1{margin:0 0 .35rem 0;font-size:2.1rem}.hero p{margin:0;opacity:.72}
.answer-card{padding:.4rem 0}
.small-muted{opacity:.68;font-size:.88rem}
[data-testid="stChatMessage"]{border-radius:16px}
</style>
""", unsafe_allow_html=True)

SUPPORTED=["pdf","docx","pptx","xlsx","xls","csv","txt","md","log","py","java","cpp","c","h","hpp","js","jsx","ts","tsx","html","css","scss","sql","json","xml","yaml","yml","toml","ini","cfg","conf","sh","bat","ps1","php","go","rs","rb","kt","kts","swift","r","m","vue","svelte","tex","jpg","jpeg","png","webp","gif","bmp","tif","tiff","avif"]

if "records" not in st.session_state: st.session_state.records=[]
if "chat" not in st.session_state: st.session_state.chat=[]
if "retriever" not in st.session_state: st.session_state.retriever=HybridRetriever()
if "assistant" not in st.session_state: st.session_state.assistant=GeminiAssistant()
if "tabular" not in st.session_state: st.session_state.tabular={}

st.markdown('<div class="hero"><h1>🧠 DocuSphere AI</h1><p>Your AI workspace for conversation, documents, data, charts, code and current information.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Workspace")
    st.caption("Ask naturally. DocuSphere automatically chooses document RAG, exact data analysis, reconciliation, charts, image generation, web information, or general AI.")
    uploads=st.file_uploader("Add files", type=SUPPORTED, accept_multiple_files=True)
    processed_names = {r["source"] for r in st.session_state.records}
    pending_uploads = [u for u in uploads if u.name not in processed_names] if uploads else []
    if pending_uploads:
        progress=st.progress(0); status=st.empty(); added=[]; tab={}; failures=[]
        for i,u in enumerate(pending_uploads):
            status.write(f"Reading `{u.name}`…")
            try:
                added.extend(ingest_uploaded_file(u))
                tab.update(read_tabular(u))
            except Exception as e:
                failures.append(f"{u.name}: {e}")
            progress.progress((i+1)/len(pending_uploads))
        if added:
            names={r["source"] for r in added}
            st.session_state.records=[r for r in st.session_state.records if r["source"] not in names]+added
            st.session_state.tabular.update(tab)
            st.session_state.retriever.build(st.session_state.records)
        status.empty(); progress.empty()
        for failure in failures: st.error(failure)
        if added: st.rerun()

    if st.session_state.records:
        st.divider(); st.subheader("📚 Connected files")
        counts={}
        for r in st.session_state.records: counts[r["source"]]=counts.get(r["source"],0)+1
        for name,count in counts.items(): st.write(f"📄 **{name}** · {count} chunks")
        for label,df in st.session_state.tabular.items():
            st.caption(f"📊 `{label}` · {len(df):,} rows × {len(df.columns)} columns")
        if st.button("🗑️ Remove all files", use_container_width=True):
            st.session_state.records=[]; st.session_state.tabular={}; st.session_state.retriever.clear(); st.rerun()

    st.divider()
    st.subheader("⚙️ AI behavior")
    st.caption("No mode selection or special command is required. Ask in normal language.")
    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.chat=[]; st.rerun()

if not st.session_state.records:
    st.info("No file is required. Start a normal AI conversation below, or add files when you want file-specific answers.")
else:
    st.success(f"{len({r['source'] for r in st.session_state.records})} file(s) connected. You can still ask completely general questions.")

for i,item in enumerate(st.session_state.chat):
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("chart") is not None: st.plotly_chart(item["chart"], use_container_width=True, key=f"history-chart-{i}")
        if item.get("table") is not None and not item["table"].empty: st.dataframe(item["table"], use_container_width=True, hide_index=True)
        if item.get("downloads"):
            st.markdown("**Downloads**")
            for di, dl in enumerate(item["downloads"]):
                st.download_button(dl["label"], dl["data"], dl["name"], dl.get("mime", "application/octet-stream"), key=f"history-download-{i}-{di}")
        elif item.get("download_data") is not None:
            st.download_button("⬇️ Download file", item["download_data"], item.get("download_name","docusphere_export.csv"), item.get("download_mime", "text/csv"), key=f"history-download-{i}")
        if item.get("sources"): st.markdown(format_sources(item["sources"]))


def build_code_context(results):
    if not any(is_code_file(r.get("source","")) for r in st.session_state.records): return ""
    by={}
    for r in st.session_state.records:
        if is_code_file(r.get("source","")): by.setdefault(r["source"],[]).append(r["text"])
    return "\n\n".join(f"[FULL CODE FILE: {src}]\n"+"\n\n".join(parts)[:60000] for src,parts in by.items())



def is_pdf_export_request(q: str) -> bool:
    ql = re.sub(r"\s+", " ", q.lower().strip())
    pdf_words = ("pdf", "pdf file", "pdf report", "pdf bana", "pdf banao", "pdf bana do", "pdf mein", "pdf me")
    export_words = ("make", "create", "generate", "give", "send", "download", "export", "save", "bana", "banao", "chahiye", "do", "de do", "nikal")
    return any(x in ql for x in pdf_words) and (any(x in ql for x in export_words) or "as pdf" in ql or "in pdf" in ql or "pdf report" in ql)

def is_download_request(q: str) -> bool:
    ql = re.sub(r"\s+", " ", q.lower().strip())
    return any(x in ql for x in ("download", "export", "save as", "file bana", "file banao", "csv chahiye", "excel chahiye", "xlsx chahiye"))


def should_use_web(q: str) -> bool:
    ql = q.lower()
    triggers = (
        "latest", "today", "current", "recent", "news", "search web", "search online",
        "internet", "price now", "right now", "as of today", "what happened",
        "who is the current", "live update", "currently", "this morning", "this week",
    )
    return any(x in ql for x in triggers)


def should_generate_image(q: str) -> bool:
    """Detect natural-language requests for an actual generated image."""
    ql = re.sub(r"\s+", " ", q.lower().strip())
    # Explicit image/picture/visual creation requests.
    if re.search(r"\b(generate|create|make|draw|render|design|produce)\b.{0,60}\b(image|picture|photo|illustration|graphic|poster|artwork|visual|tasveer|pic)\b", ql):
        return True
    if re.search(r"\b(image|picture|photo|illustration|graphic|poster|artwork|visual|tasveer|pic)\b.{0,50}\b(banao|bana do|generate|create|make|draw|dikhao|do)\b", ql):
        return True
    return any(x in ql for x in (
        "tasveer banao", "tasveer bana do", "tasveer dikhao", "tasveer chahiye",
        "image banao", "image bana do", "image chahiye",
        "picture banao", "picture bana do", "picture chahiye",
        "photo banao", "photo bana do", "photo chahiye",
        "pic banao", "pic bana do", "pic chahiye",
        "image generate karo", "picture generate karo", "photo generate karo",
        "ek image banao", "ek picture banao", "mujhe ek picture chahiye",
        "mujhe ek pic chahiye", "mujhe picture chahiye", "mujhe image chahiye",
        "show me a picture", "show me an image", "show me a photo",
    ))

def is_graph_picture_request(q: str) -> bool:
    """Detect requests for a graph/chart as an image/visual."""
    ql = re.sub(r"\s+", " ", q.lower().strip())
    has_graph = any(x in ql for x in ("graph", "chart", "plot", "bar chart", "line chart", "pie chart"))
    has_picture = any(x in ql for x in ("picture", "image", "photo", "pic", "tasveer", "visual"))
    has_create = any(x in ql for x in ("banao", "bana do", "generate", "create", "make", "draw", "render", "dikhao"))
    return has_graph and (has_picture or has_create)


def _tabular_target(query: str, frames: dict) -> list[tuple[str, pd.DataFrame]]:
    """Rank tables by explicit filename/column mentions, then by relevance."""
    if not frames:
        return []
    q = query.lower()
    scored = []
    for label, df in frames.items():
        score = 0
        ll = label.lower()
        if ll in q:
            score += 1000
        stem = re.sub(r"\.[^.]+$", "", ll)
        if stem and stem in q:
            score += 800
        for col in df.columns:
            c = str(col).lower()
            if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", q):
                score += 80
        scored.append((score, label, df))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    # If the user explicitly names a file/column, use that target first. Otherwise
    # let each table try deterministic analysis; this avoids silently choosing a random file.
    return [(label, df) for _, label, df in scored]


def is_reconciliation_request(q: str) -> bool:
    ql = re.sub(r"\s+", " ", q.lower().strip())
    terms = ("reconciliation", "reconcile", "reconciliate", "bank reconcile", "bank reconciliation", "match bank", "match ledger", "matched transactions", "unmatched transactions")
    return any(t in ql for t in terms)


def _reconciliation_frames() -> dict:
    return dict(st.session_state.tabular)


def _render_reconciliation_result(result: dict, bank_name: str, software_name: str):
    matched = result["matched"]
    bank_un = result["bank_unmatched"]
    soft_un = result["software_unmatched"]
    unmatched = result["unmatched"]
    st.markdown("### 🏦 Bank Reconciliation")
    st.markdown(
        f"**Bank:** `{bank_name}`  •  **Software:** `{software_name}`\n\n"
        f"Bank rows: **{result['bank_rows']:,}**  •  Software rows: **{result['software_rows']:,}**  •  "
        f"Matched: **{result['matched_rows']:,}**  •  Unmatched: **{result['unmatched_rows']:,}**"
    )
    st.info("Original serial numbers are preserved. Every safe match receives a new common Reconciliation ID. Amount alone is never used as a match; ambiguous duplicates remain unmatched.")
    st.markdown("#### Matched transactions")
    if matched.empty:
        st.info("No transactions met the conservative matching rules.")
    else:
        st.dataframe(matched.head(500), use_container_width=True, hide_index=True)
    st.markdown("#### Unmatched transactions")
    if unmatched.empty:
        st.success("All rows were safely matched.")
    else:
        st.dataframe(unmatched.head(500), use_container_width=True, hide_index=True)

    downloads=[]
    downloads.append({"label":"⬇️ Matched — CSV", "data":csv_bytes(matched), "name":"docusphere_matched.csv", "mime":"text/csv"})
    try:
        downloads.append({"label":"⬇️ Matched — Excel", "data":matched_excel_bytes(matched), "name":"docusphere_matched.xlsx", "mime":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
    except ValueError:
        st.caption("Matched Excel is unavailable because the result exceeds Excel's worksheet row limit; the complete CSV remains available.")
    downloads.append({"label":"⬇️ Unmatched — CSV", "data":csv_bytes(unmatched), "name":"docusphere_unmatched.csv", "mime":"text/csv"})
    try:
        downloads.append({"label":"⬇️ Unmatched — Excel", "data":unmatched_excel_bytes(bank_un, soft_un), "name":"docusphere_unmatched.xlsx", "mime":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
    except ValueError:
        st.caption("Unmatched Excel is unavailable because the result exceeds Excel's worksheet row limit; the complete CSV remains available.")
    st.markdown("#### Downloads")
    for di,dl in enumerate(downloads):
        st.download_button(dl["label"], dl["data"], dl["name"], dl["mime"], key=f"recon-download-{len(st.session_state.chat)}-{di}")
    return downloads


question=st.chat_input("Ask anything — files, CSV, graphs, coding, projects, current topics, or general questions…")

if question:
    clean=sanitize_question(question)
    if not clean: st.stop()
    st.session_state.chat.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        # Handle a pure image request before retrieval/OCR/RAG. This prevents an
        # unrelated uploaded document from making a simple image request feel slow.
        if should_generate_image(clean) and not is_graph_picture_request(clean):
            with st.spinner("🎨 Generating your image…"):
                try:
                    image_bytes = st.session_state.assistant.generate_image(clean)
                    st.image(image_bytes, caption="Generated by Gemini", use_container_width=True)
                    st.download_button(
                        "⬇️ Download image", image_bytes, "docusphere_generated.jpg",
                        "image/jpeg", key=f"generated-image-{len(st.session_state.chat)}"
                    )
                    answer = "I generated the requested image above."
                    st.session_state.chat.append({"role":"assistant","content":answer,"sources":[],"chart":None,"table":None,"download_data":None,"download_name":"docusphere_generated.jpg"})
                    st.stop()
                except Exception as exc:
                    st.error(f"Image generation failed: {exc}")
                    st.info("The request was stopped safely; DocuSphere will not keep loading indefinitely.")
                    st.stop()

        with st.spinner("Analyzing your request…"):
            results=st.session_state.retriever.search(clean, top_k=12) if st.session_state.records else []
            analysis=None; source_name=None; full_report=None; full_reports=[]

            # Reconciliation is a first-class data operation. It runs before generic
            # LLM/file analysis so exact matching is not delegated to prose generation.
            if is_reconciliation_request(clean):
                frames = _reconciliation_frames()
                bank_target, software_target, role_warnings = classify_frames(frames)
                if bank_target and software_target:
                    try:
                        recon = reconcile(bank_target[1], software_target[1], bank_target[0], software_target[0])
                        downloads = _render_reconciliation_result(recon, bank_target[0], software_target[0])
                        answer = (
                            f"Reconciliation completed using `{bank_target[0]}` as the bank ledger and `{software_target[0]}` as the software ledger. "
                            f"I found **{recon['matched_rows']:,} matched** and **{recon['unmatched_rows']:,} unmatched** rows. "
                            "Original serials were preserved and matched pairs received shared Reconciliation IDs. "
                            "The matched and unmatched CSV/Excel files are available above."
                        )
                        st.markdown(answer)
                        st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":None,"table":recon["matched"].head(500),"downloads":downloads})
                        st.stop()
                    except Exception as exc:
                        st.error(f"Reconciliation could not be completed safely: {exc}")
                        st.info("No match was guessed. Check that the uploaded ledgers contain usable transaction amount fields and enough identifying transaction data.")
                        st.stop()
                else:
                    st.warning("I need both a bank statement/ledger and a software ledger before I can reconcile safely.")
                    for warning in role_warnings:
                        st.write(f"• {warning}")
                    st.info("I will not guess which table is the bank or software ledger, and I will not match transactions from amount alone.")
                    st.stop()
            targets = _tabular_target(clean, st.session_state.tabular)
            if is_full_details_request(clean):
                for label, df in targets:
                    full_reports.append(full_file_report(df, label))
                if len(full_reports) == 1:
                    full_report = full_reports[0]; source_name = full_report["filename"]
            else:
                for label, df in targets:
                    candidate = analyze_csv(df, clean)
                    if candidate is not None:
                        analysis = candidate; source_name = label; break

            chart=None; table=None; download_data=None; download_name="docusphere_export.csv"

            # A natural-language download/export request gets a real file artifact.
            # Keep both formats when Excel can represent the table; CSV remains available for large data.
            if is_download_request(clean) and st.session_state.tabular and not is_reconciliation_request(clean):
                export_df = None
                export_name = source_name
                if analysis is not None and analysis.records is not None and not analysis.records.empty:
                    export_df = analysis.records.copy()
                    export_name = source_name or "analysis"
                elif full_reports and len(full_reports) == 1:
                    export_name = full_reports[0]["filename"]
                    export_df = st.session_state.tabular.get(export_name)
                elif targets:
                    export_name, export_df = targets[0]
                if export_df is not None and not export_df.empty:
                    safe = re.sub(r"[^a-zA-Z0-9]+", "_", (export_name or "data").rsplit(".", 1)[0]).strip("_") or "data"
                    dls = [{"label":"⬇️ CSV", "data":dataframe_to_csv_bytes(export_df), "name":f"{safe}.csv", "mime":"text/csv"}]
                    try:
                        dls.append({"label":"⬇️ Excel", "data":dataframe_to_excel_bytes(export_df, "Data"), "name":f"{safe}.xlsx", "mime":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
                    except ValueError:
                        st.caption("Excel is unavailable for this result because it exceeds Excel's worksheet limit; the complete CSV is available.")
                    st.markdown("### 📥 Download")
                    for di, dl in enumerate(dls):
                        st.download_button(dl["label"], dl["data"], dl["name"], dl["mime"], key=f"generic-export-{len(st.session_state.chat)}-{di}")
                    answer = f"I prepared the requested data from `{export_name}`. The available download formats are shown above."
                    st.markdown(answer)
                    st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":export_df.head(500),"downloads":dls})
                    st.stop()

            # PDF export is a first-class output for ANY uploaded file, including CSV/XLSX.
            # It uses deterministic extracted/file data instead of asking Gemini to fabricate a report.
            if is_pdf_export_request(clean) and st.session_state.records:
                target_name = source_name or (targets[0][0] if targets else st.session_state.records[0]["source"])
                target_df = st.session_state.tabular.get(target_name)
                target_records = [r for r in st.session_state.records if r.get("source") == target_name]
                report = full_file_report(target_df, target_name) if target_df is not None else None
                pdf_bytes = build_pdf_report(
                    target_name, dataframe=target_df, records=target_records,
                    title=f"DocuSphere AI — {target_name}",
                    summary=(report or {}).get("summary") if report else None,
                    metrics=(report or {}).get("metrics") if report else None,
                    profile=(report or {}).get("profile") if report else None,
                    result_table=(analysis.table if analysis is not None else None),
                )
                safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", target_name.rsplit(".",1)[0]).strip("_") or "docusphere_report"
                st.success(f"PDF report created for `{target_name}`.")
                st.download_button("⬇️ Download PDF report", pdf_bytes, f"{safe_name}_report.pdf", "application/pdf", key=f"pdf-export-{len(st.session_state.chat)}")
                answer = f"I created a PDF report for `{target_name}` using the uploaded file data. The PDF contains the available details, profile, and requested results without inventing missing values."
                st.markdown(answer)
                st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":pdf_bytes,"download_name":f"{safe_name}_report.pdf","download_mime":"application/pdf"})
                st.stop()

            # A request for a graph picture should use the real uploaded data when a
            # suitable structured table exists. This prevents an AI-drawn decorative
            # graph from replacing an exact business/data chart.
            if is_graph_picture_request(clean) and analysis and analysis.chart is not None:
                st.markdown("#### 📈 Graph")
                st.plotly_chart(analysis.chart, use_container_width=True)
                chart = analysis.chart
                st.success("This graph was generated directly from the uploaded data.")
                answer = "I generated the graph from the uploaded data and displayed it above."
                st.markdown(answer)
                st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":download_data,"download_name":download_name,"download_mime":"text/csv"})
                if results: st.markdown(format_sources(results))
                st.stop()

            # If the user explicitly asks for a graph as a picture but there is no
            # structured-data chart available, route it to image generation rather than
            # pretending a text answer is a graph image.
            if is_graph_picture_request(clean) and (analysis is None or analysis.chart is None) and not full_reports:
                with st.spinner("🎨 Generating your graph image…"):
                    try:
                        image_bytes = st.session_state.assistant.generate_image(clean)
                        st.image(image_bytes, caption="Generated by Gemini", use_container_width=True)
                        st.download_button(
                            "⬇️ Download image", image_bytes, "docusphere_graph.jpg",
                            "image/jpeg", key=f"generated-graph-{len(st.session_state.chat)}"
                        )
                        answer = "I generated the requested graph image above."
                        st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":None,"table":None,"download_data":None,"download_name":"docusphere_graph.jpg"})
                        st.stop()
                    except Exception as exc:
                        st.error(f"Graph image generation failed: {exc}")
                        st.info("The request was stopped safely; DocuSphere will not keep loading indefinitely.")
                        st.stop()

            # Full-file dashboards are deterministic and never depend on Gemini.
            if full_reports:
                for report_idx, report in enumerate(full_reports):
                    st.markdown(f"### 📊 Complete file analysis — `{report['filename']}`")
                    st.info(report.get("summary", "Complete file analysis calculated directly from the uploaded data."))
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("Rows", f"{report['rows']:,}")
                    m2.metric("Columns", f"{report['columns']:,}")
                    m3.metric("Numeric fields", f"{len(report['numeric_columns']):,}")
                    m4.metric("Date field", report['date_column'] or "Not detected")
                    if not report['metrics'].empty:
                        st.markdown("#### Financial / numeric overview")
                        st.dataframe(report['metrics'], use_container_width=True, hide_index=True)
                        table=report['metrics']
                    for idx,fig in enumerate(report['charts']):
                        st.plotly_chart(fig, use_container_width=True, key=f"full-report-chart-{report_idx}-{idx}-{len(st.session_state.chat)}")
                        if chart is None: chart=fig
                    st.markdown("#### Column profile")
                    st.dataframe(report['profile'], use_container_width=True, hide_index=True)
                    table=report['profile']
                    export_bytes=dataframe_to_csv_bytes(st.session_state.tabular[report['filename']])
                    safe_name=re.sub(r'[^a-zA-Z0-9]+','_',report['filename'].rsplit('.',1)[0]).strip('_')
                    st.download_button("⬇️ Download complete CSV", export_bytes, f"{safe_name}_full_data.csv", "text/csv", key=f"full-export-{report_idx}-{len(st.session_state.chat)}")
                source_name = ", ".join(r['filename'] for r in full_reports)
                st.success("Complete file details were calculated directly from the uploaded tables. No LLM-generated numbers were used.")

            data_context=""
            if st.session_state.tabular:
                data_context += "\nSTRUCTURED FILE SCHEMAS:\n" + dataframe_context(st.session_state.tabular, max_rows=12, max_cols=35)

            if analysis:
                st.markdown(f"#### {analysis.title}")
                st.markdown(analysis.summary)
                if analysis.chart is not None:
                    st.plotly_chart(analysis.chart, use_container_width=True)
                    chart=analysis.chart
                if analysis.table is not None and not analysis.table.empty:
                    st.dataframe(analysis.table, use_container_width=True, hide_index=True)
                    table=analysis.table
                export_df=analysis.records if analysis.records is not None and not analysis.records.empty else None
                if export_df is not None and (analysis.kind=="export" or "csv" in clean.lower() or "download" in clean.lower()):
                    download_data=dataframe_to_csv_bytes(export_df)
                    safe=re.sub(r"[^a-zA-Z0-9]+","_",analysis.title.lower()).strip("_")[:60]
                    download_name=f"docusphere_{safe or 'export'}.csv"
                    st.download_button(f"⬇️ Download {len(export_df):,} record(s) as CSV", download_data, download_name, "text/csv", key=f"export-{len(st.session_state.chat)}")
                data_context += f"\nEXACT ANALYTICS RESULT FROM {source_name}:\nTitle: {analysis.title}\nSummary: {analysis.summary}\nPeriod: {analysis.period}\nValue column: {analysis.value_column}\nDate column: {analysis.date_column}\nRESULT:\n{analysis.table.head(150).to_csv(index=False) if analysis.table is not None and not analysis.table.empty else '(no aggregate table)'}"

            history=st.session_state.chat[-16:]
            if full_reports:
                answer = (
                    f"I analyzed {len(full_reports)} uploaded table file(s) directly. "
                    "The numeric summary, column profile, professional charts, and CSV downloads are shown above."
                )
            else:
                answer=st.session_state.assistant.answer(
                    question=clean,
                    retrieved=results,
                    conversation=history,
                    web_search=should_use_web(clean) and analysis is None and not full_reports,
                    data_context=data_context,
                    code_context=build_code_context(results),
                    vision_images=(
                        [(r["mime_type"], r["image_bytes"], r["source"]) for r in st.session_state.records if r.get("image_bytes") and r.get("mime_type")]
                        if any(token in clean.lower() for token in ("image", "picture", "photo", "pic", "tasveer", "what is shown", "what do you see", "describe the image", "look at", "ocr", "read this image"))
                        else []
                    ),
                )

        # If exact analytics already gave the requested result, Gemini adds concise interpretation rather than replacing it.
        st.markdown(answer)
        if results: st.markdown(format_sources(results))

    st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":download_data,"download_name":download_name,"download_mime":"text/csv"})
