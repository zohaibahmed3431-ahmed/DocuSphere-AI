import io
import hashlib
import pandas as pd
import streamlit as st

from src.data_analysis import (
    analyze_csv,
    format_total,
)
from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuSphere AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuSphere AI")
st.caption(
    "Intelligent document assistant with RAG, citations, "
    "code understanding, and professional CSV analytics."
)


SUPPORTED = [
    "pdf", "docx", "txt", "md", "pptx", "xlsx", "csv",
    "py", "java", "cpp", "c", "h", "hpp", "js", "ts",
    "html", "css", "sql", "json", "xml",
    "jpg", "jpeg", "png", "webp",
]


# ============================================================
# SESSION STATE
# ============================================================

if "files_signature" not in st.session_state:
    st.session_state.files_signature = None

if "records" not in st.session_state:
    st.session_state.records = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "csv_frames" not in st.session_state:
    st.session_state.csv_frames = {}

if "chat" not in st.session_state:
    st.session_state.chat = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Documents")

    uploads = st.file_uploader(
        "Upload files",
        type=SUPPORTED,
        accept_multiple_files=True,
    )

    st.divider()

    st.caption(
        "CSV calculations use pandas on the complete uploaded "
        "dataset. Gemini is not used to invent numerical results."
    )


# ============================================================
# FILE SIGNATURE
# ============================================================

def signature(files):

    parts = []

    for f in sorted(
        files or [],
        key=lambda x: x.name.lower(),
    ):

        data = f.getvalue()

        parts.append(
            f"{f.name}:{len(data)}:"
            f"{hashlib.sha256(data).hexdigest()}"
        )

    if not parts:
        return None

    return hashlib.sha256(
        "\n".join(parts).encode()
    ).hexdigest()


# ============================================================
# PROFESSIONAL CSV CHART RENDERER
# ============================================================

def render_csv_charts(charts):

    if not charts:
        return

    try:
        import plotly.express as px
    except ImportError:
        st.warning(
            "Professional interactive charts require Plotly. "
            "Add `plotly>=6.0.0` to requirements.txt."
        )
        return

    st.subheader("📊 Data Visualization")

    for chart in charts:

        data = chart.get("data")

        if data is None or data.empty:
            continue

        chart_kind = chart.get("kind")
        x_col = chart.get("x")
        y_col = chart.get("y")

        if (
            x_col not in data.columns
            or y_col not in data.columns
        ):
            continue

        if chart_kind == "line":

            fig = px.line(
                data,
                x=x_col,
                y=y_col,
                markers=True,
                title=chart.get(
                    "title",
                    "Trend",
                ),
            )

        elif chart_kind == "bar":

            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                title=chart.get(
                    "title",
                    "Distribution",
                ),
            )

        else:
            continue

        fig.update_layout(
            height=430,
            margin={
                "l": 20,
                "r": 20,
                "t": 70,
                "b": 30,
            },
            hovermode="x unified",
            title={
                "text": chart.get(
                    "title",
                    "Data Visualization",
                ),
                "x": 0.02,
            },
            xaxis_title=x_col,
            yaxis_title=y_col,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            showgrid=False,
            automargin=True,
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.18)",
            automargin=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

        if chart.get("description"):
            st.caption(
                chart["description"]
            )


# ============================================================
# PROCESS UPLOADED FILES
# ============================================================

if uploads:

    sig = signature(uploads)

    if sig != st.session_state.files_signature:

        with st.spinner(
            "Processing documents and building search index..."
        ):

            records = []
            csv_frames = {}

            for file in uploads:

                # ------------------------------------------------
                # Normal document ingestion
                # ------------------------------------------------

                try:

                    extracted_records = (
                        ingest_uploaded_file(file)
                    )

                    records.extend(
                        extracted_records
                    )

                except Exception as exc:

                    st.warning(
                        f"Could not process "
                        f"{file.name}: {exc}"
                    )

                # ------------------------------------------------
                # Full CSV dataframe
                # ------------------------------------------------

                if file.name.lower().endswith(".csv"):

                    try:

                        csv_frames[file.name] = (
                            pd.read_csv(
                                io.BytesIO(
                                    file.getvalue()
                                )
                            )
                        )

                    except Exception as exc:

                        st.warning(
                            f"Could not analyze CSV "
                            f"{file.name}: {exc}"
                        )

            # ----------------------------------------------------
            # Save state
            # ----------------------------------------------------

            st.session_state.records = records
            st.session_state.chunks = records
            st.session_state.csv_frames = csv_frames
            st.session_state.files_signature = sig

            # ----------------------------------------------------
            # Build hybrid retrieval
            # ----------------------------------------------------

            try:

                retriever = HybridRetriever()

                retriever.build(
                    records
                )

                st.session_state.retriever = retriever

            except Exception as exc:

                st.session_state.retriever = None

                st.error(
                    "Could not build search index: "
                    f"{exc}"
                )

    st.success(
        f"Ready: {len(uploads)} file(s)"
    )

    if st.session_state.csv_frames:

        st.caption(
            "CSV files available for exact analysis: "
            + ", ".join(
                st.session_state.csv_frames.keys()
            )
        )

else:

    st.info(
        "Upload one or more documents to begin."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for item in st.session_state.chat:

    with st.chat_message("user"):
        st.markdown(
            item["q"]
        )

    with st.chat_message("assistant"):
        st.markdown(
            item["a"]
        )


# ============================================================
# QUESTION
# ============================================================

question = st.chat_input(
    "Ask about your documents or CSV data..."
)


# ============================================================
# HANDLE QUESTION
# ============================================================

if question:

    with st.chat_message("user"):
        st.markdown(
            question
        )


    # ========================================================
    # CSV ANALYTICS / VISUALIZATION
    # ========================================================

    csv_result = None
    csv_name = None

    for name, frame in (
        st.session_state.csv_frames.items()
    ):

        try:

            result = analyze_csv(
                frame,
                question,
            )

        except Exception:

            result = None

        if result is not None:

            csv_result = result
            csv_name = name
            break


    # ========================================================
    # CSV RESPONSE
    # ========================================================

    if csv_result is not None:

        with st.chat_message("assistant"):

            # ------------------------------------------------
            # VISUALIZATION REQUEST
            # ------------------------------------------------

            if csv_result.get("mode") == "visualization":

                st.markdown(
                    f"### 📊 Analysis of `{csv_name}`"
                )

                st.caption(
                    f"Using the complete dataset "
                    f"({csv_result['file_rows']:,} rows)."
                )

                render_csv_charts(
                    csv_result.get("charts", [])
                )


            # ------------------------------------------------
            # EXACT CALCULATION
            # ------------------------------------------------

            else:

                st.markdown(
                    f"""
### 📊 CSV Analysis

**{csv_result['value_column']} — "
{csv_result['period']}:**
{format_total(csv_result['total'])}

**Period:** {csv_result['start'].date()}
→ {csv_result['end'].date()}

**Records:** {csv_result['record_count']}

**Date column:** `{csv_result['date_column']}`

**File:** `{csv_name}`
"""
                )

                st.subheader(
                    "Complete matching records"
                )

                st.dataframe(
                    csv_result["records"],
                    use_container_width=True,
                    hide_index=True,
                )

                render_csv_charts(
                    csv_result.get("charts", [])
                )

                st.caption(
                    "All numerical results are calculated "
                    "directly from the complete CSV using "
                    "pandas."
                )


        # ----------------------------------------------------
        # Save CSV conversation
        # ----------------------------------------------------

        if csv_result.get("mode") == "visualization":

            chat_answer = (
                f"Created professional visualizations "
                f"from the complete `{csv_name}` dataset."
            )

        else:

            chat_answer = (
                f"CSV analysis from `{csv_name}`: "
                f"{format_total(csv_result['total'])} "
                f"{csv_result['value_column']} "
                f"for {csv_result['period']}."
            )

        st.session_state.chat.append(
            {
                "q": question,
                "a": chat_answer,
            }
        )


    # ========================================================
    # DOCUMENT / GEMINI RESPONSE
    # ========================================================

    else:

        with st.chat_message("assistant"):

            if st.session_state.retriever is None:

                answer = (
                    "Please upload and process "
                    "a document first."
                )

                results = []

                st.markdown(
                    answer
                )

            else:

                # --------------------------------------------
                # RETRIEVAL
                # --------------------------------------------

                try:

                    results = (
                        st.session_state
                        .retriever
                        .search(
                            question,
                            top_k=6,
                        )
                    )

                except Exception as exc:

                    results = []

                    st.warning(
                        f"Search error: {exc}"
                    )


                # --------------------------------------------
                # CONVERSATION
                # --------------------------------------------

                conversation = []

                for item in (
                    st.session_state.chat
                ):

                    conversation.append(
                        {
                            "role": "user",
                            "content": item["q"],
                        }
                    )

                    conversation.append(
                        {
                            "role": "assistant",
                            "content": item["a"],
                        }
                    )


                # --------------------------------------------
                # GEMINI
                # --------------------------------------------

                try:

                    assistant = GeminiAssistant()

                    answer = assistant.answer(
                        question,
                        results,
                        st.session_state.records,
                        conversation,
                    )

                except Exception as exc:

                    answer = (
                        "I could not generate "
                        "the AI answer: "
                        f"{exc}"
                    )


                st.markdown(
                    answer
                )


                # --------------------------------------------
                # SOURCES
                # --------------------------------------------

                if results:

                    st.caption(
                        "Sources"
                    )

                    shown_sources = set()

                    for result in results[:6]:

                        location = (
                            result.get("location")
                            or result.get("source")
                            or "Unknown source"
                        )

                        if location in shown_sources:
                            continue

                        shown_sources.add(
                            location
                        )

                        st.write(
                            f"- {location}"
                        )


        # ----------------------------------------------------
        # Save AI conversation
        # ----------------------------------------------------

        st.session_state.chat.append(
            {
                "q": question,
                "a": answer,
            }
        )
