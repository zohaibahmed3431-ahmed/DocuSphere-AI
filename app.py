import io
import hashlib
import pandas as pd
import streamlit as st

from src.data_analysis import analyze_csv, format_total
from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuSphere AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuSphere AI")
st.caption(
    "Intelligent document assistant with RAG, citations, "
    "code understanding, and CSV analytics."
)


SUPPORTED = [
    "pdf", "docx", "txt", "md", "pptx", "xlsx", "csv",
    "py", "java", "cpp", "c", "h", "hpp", "js", "ts",
    "html", "css", "sql", "json", "xml",
    "jpg", "jpeg", "png", "webp"
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
        accept_multiple_files=True
    )

    st.divider()

    st.caption(
        "CSV questions use deterministic pandas calculations. "
        "Gemini is not used to invent totals."
    )


# ============================================================
# FILE SIGNATURE
# ============================================================

def signature(files):
    parts = []

    for f in sorted(
        files or [],
        key=lambda x: x.name.lower()
    ):
        b = f.getvalue()

        parts.append(
            f"{f.name}:{len(b)}:{hashlib.sha256(b).hexdigest()}"
        )

    if not parts:
        return None

    return hashlib.sha256(
        "\n".join(parts).encode()
    ).hexdigest()


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

            for f in uploads:

                # --------------------------------------------
                # Normal document ingestion
                # --------------------------------------------

                try:
                    extracted_records = ingest_uploaded_file(f)
                    records.extend(extracted_records)

                except Exception as exc:
                    st.warning(
                        f"Could not process {f.name}: {exc}"
                    )

                # --------------------------------------------
                # CSV dataframe for exact analytics
                # --------------------------------------------

                if f.name.lower().endswith(".csv"):

                    try:
                        csv_frames[f.name] = pd.read_csv(
                            io.BytesIO(f.getvalue())
                        )

                    except Exception as exc:
                        st.warning(
                            f"Could not analyze CSV {f.name}: {exc}"
                        )

            # --------------------------------------------
            # Save records
            # --------------------------------------------

            st.session_state.records = records
            st.session_state.chunks = records
            st.session_state.csv_frames = csv_frames
            st.session_state.files_signature = sig

            # --------------------------------------------
            # Build hybrid retrieval index
            # --------------------------------------------

            try:
                retriever = HybridRetriever()
                retriever.build(records)

                st.session_state.retriever = retriever

            except Exception as exc:
                st.session_state.retriever = None

                st.error(
                    f"Could not build search index: {exc}"
                )

    st.success(
        f"Ready: {len(uploads)} file(s)"
    )

    if st.session_state.csv_frames:

        st.caption(
            "CSV files available for exact data analysis: "
            + ", ".join(st.session_state.csv_frames.keys())
        )

else:

    st.info(
        "Upload one or more documents to begin."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for item in st.session_state.chat:

    with st.chat_message("user"):
        st.markdown(item["q"])

    with st.chat_message("assistant"):
        st.markdown(item["a"])


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask about your documents or CSV data..."
)


# ============================================================
# HANDLE QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(question)


    # ========================================================
    # CSV ANALYTICS PATH
    # Exact calculation using pandas
    # ========================================================

    csv_result = None
    csv_name = None

    for name, frame in st.session_state.csv_frames.items():

        result = analyze_csv(
            frame,
            question
        )

        if result is not None:
            csv_result = result
            csv_name = name
            break


    # ========================================================
    # CSV RESULT
    # ========================================================

    if csv_result is not None:

        with st.chat_message("assistant"):

            st.markdown(
                f"### 📊 CSV Analysis\n\n"
                f"**{csv_result['value_column']} — "
                f"{csv_result['period']}:** "
                f"{format_total(csv_result['total'])}\n\n"
                f"**Period:** "
                f"{csv_result['start'].date()} → "
                f"{csv_result['end'].date()}  \n"
                f"**Records:** "
                f"{csv_result['record_count']}  \n"
                f"**Date column:** "
                f"`{csv_result['date_column']}`  \n"
                f"**File:** "
                f"`{csv_name}`"
            )

            st.subheader(
                "Complete matching records"
            )

            st.dataframe(
                csv_result["records"],
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Daily trend"
            )

            chart_df = (
                csv_result["daily"]
                .set_index("Date")
            )

            st.line_chart(chart_df)

            st.caption(
                "Numbers and chart are calculated directly "
                "from the CSV with pandas; Gemini is not "
                "used for arithmetic."
            )


        # Save chat history

        st.session_state.chat.append(
            {
                "q": question,
                "a": (
                    f"CSV analysis from `{csv_name}`: "
                    f"{format_total(csv_result['total'])} "
                    f"{csv_result['value_column']} "
                    f"for {csv_result['period']}."
                )
            }
        )


    # ========================================================
    # DOCUMENT / AI PATH
    # ========================================================

    else:

        with st.chat_message("assistant"):

            # --------------------------------------------
            # No documents
            # --------------------------------------------

            if st.session_state.retriever is None:

                answer = (
                    "Please upload and process a document first."
                )

                st.markdown(answer)


            else:

                # ----------------------------------------
                # Hybrid retrieval
                # ----------------------------------------

                try:

                    results = (
                        st.session_state.retriever.search(
                            question,
                            top_k=6
                        )
                    )

                except Exception as exc:

                    results = []

                    st.warning(
                        f"Search error: {exc}"
                    )


                # ----------------------------------------
                # Gemini answer
                # ----------------------------------------

                try:

                    assistant = GeminiAssistant()

                    conversation = []

for item in st.session_state.chat:
    conversation.append({
        "role": "user",
        "content": item["q"]
    })

    conversation.append({
        "role": "assistant",
        "content": item["a"]
    })

answer = assistant.answer(
    question,
    results,
    st.session_state.records,
    conversation
)

                except Exception as exc:

                    answer = (
                        f"I could not generate the AI answer: "
                        f"{exc}"
                    )


                # ----------------------------------------
                # Display answer
                # ----------------------------------------

                st.markdown(answer)


                # ----------------------------------------
                # Display sources ONCE
                # ----------------------------------------

                if results:

                    st.caption("Sources")

                    shown_sources = set()

                    for r in results[:6]:

                        location = (
                            r.get("location")
                            or r.get("source")
                            or "Unknown source"
                        )

                        if location in shown_sources:
                            continue

                        shown_sources.add(location)

                        st.write(
                            f"- {location}"
                        )


        # ----------------------------------------------------
        # Save conversation
        # ----------------------------------------------------

        st.session_state.chat.append(
            {
                "q": question,
                "a": answer
            }
        )
