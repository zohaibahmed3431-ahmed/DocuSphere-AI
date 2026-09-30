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
from src.csv_analytics import analyze_csv, dataframe_to_csv_bytes, is_full_details_request, full_file_report
from src.tabular import read_tabular, dataframe_context
from src.utils import is_code_file

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

SUPPORTED=["pdf","docx","txt","md","pptx","xlsx","csv","py","java","cpp","c","h","hpp","js","ts","html","css","sql","json","xml","jpg","jpeg","png","webp"]

if "records" not in st.session_state: st.session_state.records=[]
if "chat" not in st.session_state: st.session_state.chat=[]
if "retriever" not in st.session_state: st.session_state.retriever=HybridRetriever()
if "assistant" not in st.session_state: st.session_state.assistant=GeminiAssistant()
if "tabular" not in st.session_state: st.session_state.tabular={}

st.markdown('<div class="hero"><h1>🧠 DocuSphere AI</h1><p>Your AI workspace for conversation, documents, data, charts, code and current information.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Workspace")
    st.caption("AI automatically decides whether your question needs file knowledge, exact data analysis, general AI, or web information.")
    uploads=st.file_uploader("Add files", type=SUPPORTED, accept_multiple_files=True)
    if st.button("➕ Process files", use_container_width=True, disabled=not uploads):
        progress=st.progress(0); status=st.empty(); added=[]; tab={}
        for i,u in enumerate(uploads):
            status.write(f"Processing `{u.name}`…")
            try:
                added.extend(ingest_uploaded_file(u))
                tab.update(read_tabular(u))
            except Exception as e:
                st.error(f"{u.name}: {e}")
            progress.progress((i+1)/len(uploads))
        if added:
            names={r["source"] for r in added}
            st.session_state.records=[r for r in st.session_state.records if r["source"] not in names]+added
            st.session_state.tabular.update(tab)
            st.session_state.retriever.build(st.session_state.records)
            st.rerun()

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
    st.caption("No mode selection is required. Ask naturally; DocuSphere chooses the appropriate tool automatically.")
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
        if item.get("download_data") is not None:
            st.download_button("⬇️ Download CSV", item["download_data"], item.get("download_name","docusphere_export.csv"), "text/csv", key=f"history-download-{i}")
        if item.get("sources"): st.markdown(format_sources(item["sources"]))


def build_code_context(results):
    if not any(is_code_file(r.get("source","")) for r in st.session_state.records): return ""
    by={}
    for r in st.session_state.records:
        if is_code_file(r.get("source","")): by.setdefault(r["source"],[]).append(r["text"])
    return "\n\n".join(f"[FULL CODE FILE: {src}]\n"+"\n\n".join(parts)[:60000] for src,parts in by.items())


def should_use_web(q: str) -> bool:
    ql=q.lower()
    return any(x in ql for x in ["latest","today","current","recent","news","search web","search online","internet","price now","right now","as of today"])

question=st.chat_input("Ask anything — files, CSV, graphs, coding, projects, current topics, or general questions…")

if question:
    clean=sanitize_question(question)
    if not clean: st.stop()
    st.session_state.chat.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your request…"):
            results=st.session_state.retriever.search(clean, top_k=12) if st.session_state.records else []
            analysis=None; source_name=None; full_report=None
            for label,df in st.session_state.tabular.items():
                if is_full_details_request(clean):
                    full_report=full_file_report(df, label)
                    source_name=label
                    break
                candidate=analyze_csv(df, clean)
                if candidate is not None:
                    analysis=candidate; source_name=label; break

            chart=None; table=None; download_data=None; download_name="docusphere_export.csv"

            # Full-file dashboard is deterministic and never depends on Gemini.
            if full_report:
                st.markdown(f"### 📊 Complete file analysis — `{full_report['filename']}`")
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Rows", f"{full_report['rows']:,}")
                m2.metric("Columns", f"{full_report['columns']:,}")
                m3.metric("Numeric fields", f"{len(full_report['numeric_columns']):,}")
                m4.metric("Date field", full_report['date_column'] or "Not detected")
                if not full_report['metrics'].empty:
                    st.markdown("#### Financial / numeric overview")
                    st.dataframe(full_report['metrics'], use_container_width=True, hide_index=True)
                    table=full_report['metrics']
                for idx,fig in enumerate(full_report['charts']):
                    st.plotly_chart(fig, use_container_width=True, key=f"full-report-chart-{idx}-{len(st.session_state.chat)}")
                    if chart is None: chart=fig
                st.markdown("#### Column profile")
                st.dataframe(full_report['profile'], use_container_width=True, hide_index=True)
                table=full_report['profile']
                export_bytes=dataframe_to_csv_bytes(st.session_state.tabular[source_name])
                download_data=export_bytes
                download_name=f"{re.sub(r'[^a-zA-Z0-9]+','_',source_name.rsplit('.',1)[0]).strip('_')}_full_data.csv"
                st.download_button("⬇️ Download complete CSV", download_data, download_name, "text/csv", key=f"full-export-{len(st.session_state.chat)}")
                st.success("Complete file details and professional charts were generated directly from the CSV. No AI guesswork was used.")

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
            if full_report:
                answer = (
                    f"I analyzed the complete file `{source_name}` directly. "
                    f"It contains {full_report['rows']:,} rows and {full_report['columns']:,} columns. "
                    "The financial/numeric summary, column profile, charts, and full CSV download are shown above."
                )
            else:
                answer=st.session_state.assistant.answer(
                    question=clean,
                    retrieved=results,
                    conversation=history,
                    web_search=should_use_web(clean),
                    data_context=data_context,
                    code_context=build_code_context(results),
                )

        # If exact analytics already gave the requested result, Gemini adds concise interpretation rather than replacing it.
        st.markdown(answer)
        if results: st.markdown(format_sources(results))

    st.session_state.chat.append({"role":"assistant","content":answer,"sources":results,"chart":chart,"table":table,"download_data":download_data,"download_name":download_name})
