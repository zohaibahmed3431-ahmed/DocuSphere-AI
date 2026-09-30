
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant
from src.security import sanitize_question
from src.citations import format_sources
from src.csv_analytics import analyze_csv, dataframe_to_csv_bytes
from src.tabular import read_tabular, dataframe_context
from src.utils import is_code_file

load_dotenv()

st.set_page_config(page_title="DocuSphere AI", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.1rem;}
[data-testid="stSidebar"] {min-width: 300px; max-width: 340px;}
.hero {padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,.2); border-radius: 16px; margin-bottom: 1rem;}
.muted {opacity:.72;}
</style>
""", unsafe_allow_html=True)

SUPPORTED=[
    "pdf","docx","txt","md","pptx","xlsx","csv","py","java","cpp","c","h","hpp",
    "js","ts","html","css","sql","json","xml","jpg","jpeg","png","webp"
]

defaults={
    "records": [], "chat": [], "retriever": HybridRetriever(), "assistant": GeminiAssistant(),
    "tabular": {}, "last_analysis": None, "last_export": None
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

st.title("📚 DocuSphere AI")
st.caption("Intelligent AI workspace • Documents • Data analytics • Charts • Web search • Conversation")

with st.sidebar:
    st.header("🧠 AI Mode")
    mode=st.selectbox(
        "How should I answer?",
        ["Auto", "General AI", "Document & File", "Web Search"],
        help="Auto chooses the useful source. General AI works without uploads."
    )

    st.divider()
    st.header("📁 Files")
    uploads=st.file_uploader(
        "Upload files (optional)",
        type=SUPPORTED,
        accept_multiple_files=True,
        help="You can chat without uploading anything. Upload documents/data when you want file-specific answers."
    )

    if st.button("➕ Add / Process files", use_container_width=True, disabled=not uploads):
        progress=st.progress(0)
        status=st.empty()
        added=[]
        new_tabular={}
        for i,u in enumerate(uploads):
            status.write(f"Processing `{u.name}`…")
            try:
                recs=ingest_uploaded_file(u)
                added.extend(recs)
                new_tabular.update(read_tabular(u))
            except Exception as e:
                st.error(f"{u.name}: {e}")
            progress.progress((i+1)/len(uploads))
        if added:
            names={r["source"] for r in added}
            st.session_state.records=[r for r in st.session_state.records if r["source"] not in names]+added
            st.session_state.tabular.update(new_tabular)
            st.session_state.retriever.build(st.session_state.records)
            st.success(f"{len(names)} file(s) ready.")
            st.rerun()

    if st.session_state.records:
        st.divider()
        st.subheader("Ready")
        counts={}
        for r in st.session_state.records: counts[r["source"]]=counts.get(r["source"],0)+1
        for n,c in counts.items(): st.write(f"📄 **{n}** · `{c} chunks`")

        if st.session_state.tabular:
            st.caption("📊 Structured data:")
            for label,df in st.session_state.tabular.items():
                st.write(f"`{label}` · {len(df):,} rows × {len(df.columns)} columns")

        if st.button("🗑️ Clear files", use_container_width=True):
            st.session_state.records=[]
            st.session_state.tabular={}
            st.session_state.retriever.clear()
            st.session_state.last_analysis=None
            st.session_state.last_export=None
            st.rerun()

    st.divider()
    st.subheader("Pipeline")
    st.caption("Upload → Extract/OCR → Normalize → Chunk → Embeddings → Hybrid retrieval → Rerank → Gemini")
    st.caption("Data → Exact pandas analysis → Professional Plotly chart → Gemini explanation")
    st.caption("General/current questions → Gemini → optional Google Search grounding")

    st.divider()
    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.chat=[]
        st.session_state.last_analysis=None
        st.session_state.last_export=None
        st.rerun()

# Main landing
if not st.session_state.records:
    st.info("You can start chatting immediately — no document upload is required.")
else:
    st.success(f"{len({r['source'] for r in st.session_state.records})} file(s) connected to this conversation.")

# Render history
for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("chart") is not None:
            st.plotly_chart(item["chart"], use_container_width=True)
        if item.get("table") is not None and not item["table"].empty:
            st.dataframe(item["table"], use_container_width=True, hide_index=True)
        if item.get("download_data") is not None:
            st.download_button(
                "⬇️ Download CSV",
                data=item["download_data"],
                file_name=item.get("download_name","docusphere_export.csv"),
                mime="text/csv",
                key=item.get("download_key","history-export")
            )
        if item.get("sources"):
            st.markdown(format_sources(item["sources"]))

question=st.chat_input("Ask anything — about your files, data, coding, projects, or general topics…")

def build_code_context(results):
    if not any(is_code_file(r.get("source","")) for r in results) and not any(is_code_file(r.get("source","")) for r in st.session_state.records):
        return ""
    by={}
    for r in st.session_state.records:
        if is_code_file(r.get("source","")):
            by.setdefault(r["source"],[]).append(r["text"])
    return "\n\n".join(f"[FULL CODE FILE: {src}]\n"+"\n\n".join(parts)[:60000] for src,parts in by.items())

def wants_web(q, selected_mode):
    if selected_mode=="Web Search":
        return True
    if selected_mode!="Auto":
        return False
    # With no files, Auto behaves like a general AI assistant with optional
    # real-time grounding; Gemini itself decides whether search is useful.
    if not st.session_state.records:
        return True
    ql=q.lower()
    return any(x in ql for x in [
        "latest","today","current","recent","news","search","web","internet",
        "online","price now","what happened","who is","what is the current"
    ])

if question:
    clean=sanitize_question(question)
    if not clean: st.stop()

    st.session_state.chat.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            results=st.session_state.retriever.search(clean, top_k=10) if st.session_state.records else []
            analysis=None
            analysis_source=None

            # Exact structured-data layer.
            for label,df in st.session_state.tabular.items():
                candidate=analyze_csv(df, clean)
                if candidate is not None:
                    analysis=candidate
                    analysis_source=label
                    break

            data_context=""
            chart=None
            table=None
            download_data=None
            download_name="docusphere_export.csv"

            if analysis:
                st.markdown(f"### 📊 {analysis.title}")
                st.write(analysis.summary)
                if analysis.chart is not None:
                    st.plotly_chart(analysis.chart, use_container_width=True)
                    chart=analysis.chart
                if analysis.table is not None and not analysis.table.empty:
                    st.dataframe(analysis.table, use_container_width=True, hide_index=True)
                    table=analysis.table
                export_df = analysis.records if analysis.records is not None and not analysis.records.empty else analysis.table
                if export_df is not None and not export_df.empty:
                    download_data=dataframe_to_csv_bytes(export_df)
                    download_name=f"docusphere_{re.sub(r'[^a-zA-Z0-9]+','_',analysis.title.lower()).strip('_')[:60]}.csv"
                    st.download_button(
                        f"⬇️ Download {len(export_df):,} matching record(s) as CSV",
                        download_data, download_name, "text/csv",
                        key=f"export-{len(st.session_state.chat)}"
                    )
                data_context=f"""
Exact structured-data result from `{analysis_source}`:
Title: {analysis.title}
Summary: {analysis.summary}
Period: {analysis.period}
Value column: {analysis.value_column}
Date column: {analysis.date_column}
Result table:
{analysis.table.head(100).to_csv(index=False) if not analysis.table.empty else "(no result rows)"}
"""

            # If user explicitly requests export and an exact table exists, do not ask Gemini to recreate it.
            history=st.session_state.chat[-14:]
            answer=st.session_state.assistant.answer(
                question=clean,
                retrieved=results,
                conversation=history,
                web_search=wants_web(clean,mode),
                data_context=data_context,
                code_context=build_code_context(results),
            )

        # Don't duplicate a clean data-analysis result with a generic answer unless useful.
        if analysis and ("unavailable" in analysis.title.lower() or analysis.chart is None):
            st.markdown(answer)
        else:
            st.markdown(answer)

        if results:
            st.markdown(format_sources(results))

    st.session_state.chat.append({
        "role":"assistant","content":answer,"sources":results,
        "chart":chart,"table":table,"download_data":download_data,
        "download_name":download_name,"download_key":f"history-export-{len(st.session_state.chat)}"
    })
