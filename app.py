import io
import hashlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant
from src.security import sanitize_question
from src.citations import format_sources
from src.data_analysis import analyze_csv, format_total

load_dotenv()

st.set_page_config(
    page_title="DocuSphere AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
[data-testid="stSidebar"] {min-width: 290px; max-width: 330px;}
</style>
""", unsafe_allow_html=True)

st.title("📚 DocuSphere AI")
st.caption("Intelligent Document Assistant • Multimodal ingestion • Hybrid RAG • Gemini • CSV Analytics")

SUPPORTED = [
    "pdf", "docx", "txt", "md", "pptx", "xlsx", "csv",
    "py", "java", "cpp", "c", "h", "hpp", "js", "ts",
    "html", "css", "sql", "json", "xml", "jpg", "jpeg", "png", "webp",
]

for key, default in {
    "records": [],
    "chat": [],
    "retriever": None,
    "assistant": None,
    "csv_frames": {},
    "processed_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def file_signature(files):
    parts = []
    for f in sorted(files or [], key=lambda x: x.name.lower()):
        data = f.getvalue()
        parts.append(f"{f.name}:{len(data)}:{hashlib.sha256(data).hexdigest()}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest() if parts else None


def render_csv_charts(charts):
    if not charts:
        return
    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly is missing. Add plotly>=6.0.0 to requirements.txt and redeploy.")
        return

    st.subheader("📊 Professional visualization")
    for chart in charts:
        data = chart.get("data")
        x_col, y_col = chart.get("x"), chart.get("y")
        if data is None or data.empty or x_col not in data.columns or y_col not in data.columns:
            continue

        if chart.get("kind") == "line":
            fig = px.line(data, x=x_col, y=y_col, markers=True, title=chart.get("title", "Trend"))
            fig.update_traces(line={"width": 3}, marker={"size": 7})
        elif chart.get("kind") == "bar":
            fig = px.bar(data, x=x_col, y=y_col, title=chart.get("title", "Distribution"), text_auto=True)
        else:
            continue

        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=70, b=45),
            hovermode="x unified" if chart.get("kind") == "line" else "closest",
            template="plotly_white",
            title=dict(x=0.02),
            xaxis_title=x_col,
            yaxis_title=y_col,
        )
        fig.update_xaxes(showgrid=False, automargin=True)
        fig.update_yaxes(showgrid=True, automargin=True)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})
        if chart.get("description"):
            st.caption(chart["description"])


with st.sidebar:
    st.header("📁 Documents")
    uploads = st.file_uploader(
        "Upload one or more files",
        type=SUPPORTED,
        accept_multiple_files=True,
        help="Upload documents, spreadsheets, images, or source-code files.",
    )

    if st.button("➕ Process uploads", use_container_width=True, disabled=not uploads):
        new_records = []
        new_csv_frames = {}
        progress = st.progress(0)
        status = st.empty()

        for i, uploaded in enumerate(uploads):
            status.write(f"Processing `{uploaded.name}`…")
            try:
                new_records.extend(ingest_uploaded_file(uploaded))
            except Exception as exc:
                st.error(f"{uploaded.name}: {exc}")

            if uploaded.name.lower().endswith(".csv"):
                try:
                    new_csv_frames[uploaded.name] = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                except Exception as exc:
                    st.error(f"{uploaded.name} CSV analysis failed: {exc}")

            progress.progress((i + 1) / len(uploads))

        names = {r["source"] for r in new_records}
        names.update(new_csv_frames.keys())
        st.session_state.records = [r for r in st.session_state.records if r.get("source") not in names] + new_records
        st.session_state.csv_frames = {
            name: frame for name, frame in st.session_state.csv_frames.items() if name not in names
        }
        st.session_state.csv_frames.update(new_csv_frames)

        retriever = HybridRetriever()
        retriever.build(st.session_state.records)
        st.session_state.retriever = retriever
        st.session_state.processed_signature = file_signature(uploads)
        st.success("All uploads processed successfully.")
        st.rerun()

    if st.session_state.records or st.session_state.csv_frames:
        st.divider()
        st.subheader("Ready documents")
        names = sorted(set(r.get("source") for r in st.session_state.records if r.get("source")))
        names = sorted(set(names) | set(st.session_state.csv_frames.keys()))
        for name in names:
            chunk_count = sum(1 for r in st.session_state.records if r.get("source") == name)
            extra = f" • {chunk_count} chunks" if chunk_count else " • CSV analytics ready"
            st.write(f"📄 **{name}**{extra}")

        if st.button("🗑️ Clear documents", use_container_width=True):
            st.session_state.records = []
            st.session_state.csv_frames = {}
            st.session_state.retriever = None
            st.session_state.chat = []
            st.rerun()

    st.divider()
    st.subheader("Pipeline")
    st.write("📤 Upload → Detect → Extract/OCR → Normalize → Chunk")
    st.write("🧩 Embed → Vector index → Hybrid retrieval → Rerank → Gemini")
    st.write("📊 CSV → Pandas → Exact analysis → Interactive Plotly")
    st.write("📌 Answer → Citation")

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

if not st.session_state.records and not st.session_state.csv_frames:
    st.info("Upload documents from the left to start. You can also ask general AI questions.")
else:
    ready_count = len(set(r.get("source") for r in st.session_state.records if r.get("source")) | set(st.session_state.csv_frames.keys()))
    st.success(f"{ready_count} document(s) ready.")

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            st.markdown(format_sources(item["sources"]))

question = st.chat_input("Ask about your files, analyze CSV data, explain code, or modify uploaded code…")

if question:
    clean_question = sanitize_question(question)
    if not clean_question:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ------------------------------------------------------------
    # Exact CSV path: never send deterministic numeric questions to Gemini.
    # ------------------------------------------------------------
    csv_result = None
    csv_name = None
    for name, frame in st.session_state.csv_frames.items():
        try:
            result = analyze_csv(frame, clean_question)
        except Exception as exc:
            result = None
            st.warning(f"CSV analysis error in {name}: {exc}")
        if result is not None:
            csv_result, csv_name = result, name
            break

    if csv_result is not None:
        with st.chat_message("assistant"):
            if csv_result["mode"] == "visualization":
                st.markdown(f"### 📊 Analysis of `{csv_name}`")
                st.caption(f"Using the complete dataset ({csv_result['file_rows']:,} rows).")
                render_csv_charts(csv_result.get("charts", []))
                chat_answer = f"Created a professional visualization from the complete `{csv_name}` dataset."
            else:
                st.markdown("### 📊 CSV Analysis")
                st.markdown(
                    f"**{csv_result['value_column']} — {csv_result['period']}:** "
                    f"{format_total(csv_result['total'])}"
                )
                st.markdown(f"**Period:** {csv_result['start'].date()} → {csv_result['end'].date()}")
                st.markdown(f"**Matching records:** {csv_result['record_count']}")
                st.markdown(f"**Date column:** `{csv_result['date_column']}`")
                st.markdown(f"**File:** `{csv_name}`")
                st.subheader("Complete matching records")
                st.dataframe(csv_result["records"], use_container_width=True, hide_index=True)
                render_csv_charts(csv_result.get("charts", []))
                st.caption("Numerical results are calculated directly from the complete CSV using pandas.")
                chat_answer = (
                    f"CSV analysis from `{csv_name}`: {format_total(csv_result['total'])} "
                    f"{csv_result['value_column']} for {csv_result['period']}."
                )

        st.session_state.chat.append({"role": "assistant", "content": chat_answer})

    else:
        with st.chat_message("assistant"):
            if st.session_state.retriever is None:
                answer = "Please upload and process a document first."
                results = []
            else:
                results = st.session_state.retriever.search(clean_question, top_k=8)
                if st.session_state.assistant is None:
                    st.session_state.assistant = GeminiAssistant()
                history = st.session_state.chat[-12:]
                try:
                    answer = st.session_state.assistant.answer(
                        question=clean_question,
                        retrieved=results,
                        all_records=st.session_state.records,
                        conversation=history,
                    )
                except Exception as exc:
                    answer = f"I could not generate the AI answer: {exc}"

            st.markdown(answer)
            if results:
                st.markdown(format_sources(results))

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer,
            "sources": results,
        })
