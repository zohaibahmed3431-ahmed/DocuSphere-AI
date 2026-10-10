import os

import streamlit as st
import pandas as pd

from config.settings import (
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    SUPPORTED_TYPES,
    MAX_FILE_SIZE_MB,
    DEFAULT_DATE_TOLERANCE_DAYS,
    OPENAI_DEFAULT_MODEL,
    GEMINI_DEFAULT_MODEL,
)

from core.file_loader import load_file
from core.document_detector import detect_document_type
from core.data_cleaner import clean_sheet, standardize
from core.reconciler import run_reconciliation
from core.exporter import create_excel_report
from core.utils import dataframe_to_context

from services.reconciliation_service import process_reconciliation

from ai.assistant import ask_ai, build_context
from ai.prompts import SYSTEM_PROMPT

st.set_page_config(
    page_title="Bank Reconciliation AI",
    page_icon="🏦",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "uploaded_files_raw" not in st.session_state:
    st.session_state.uploaded_files_raw = []

if "reconciliation_result" not in st.session_state:
    st.session_state.reconciliation_result = None

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_uploaded_file(uploaded_file):
    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name,
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())

    return file_path


with st.sidebar:
    st.title("🏦 Bank Reconciliation AI")
    st.caption("Professional AI-powered reconciliation workspace")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_files_raw = []
        st.session_state.reconciliation_result = None
        st.rerun()

    st.divider()

    st.subheader("AI Settings")

    provider = st.selectbox(
        "AI Provider",
        ["Gemini", "OpenAI"],
    )

    if provider == "Gemini":
        default_model = GEMINI_DEFAULT_MODEL
    else:
        default_model = OPENAI_DEFAULT_MODEL

    api_key = st.text_input(
        "API Key",
        type="password",
    )

    model = st.text_input(
        "Model",
        value=default_model,
    )

    st.divider()

    st.subheader("Reconciliation Settings")

    date_tolerance = st.number_input(
        "Date Tolerance (days)",
        min_value=0,
        max_value=30,
        value=DEFAULT_DATE_TOLERANCE_DAYS,
    )

    st.divider()

    st.subheader("Uploaded Files")

    if st.session_state.uploaded_files:
        for index, file_info in enumerate(st.session_state.uploaded_files):
            st.markdown(f"📄 **{file_info['name']}**")

            file_info["type"] = st.selectbox(
                f"Type for {file_info['name']}",
                [
                    "Bank Statement",
                    "Company Ledger",
                    "Supporting Document",
                ],
                key=f"file_type_{index}",
            )
    else:
        st.caption("Koi file upload nahi hui.")

st.title("💬 Bank Reconciliation AI")
st.caption(
    "Upload your bank statement, ledger, PDF, Word or CSV file. "
    "Then ask anything or run reconciliation."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_files = st.file_uploader(
    "Attach files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    new_files_added = False

    for uploaded_file in uploaded_files:
        already_uploaded = any(
            file_info["name"] == uploaded_file.name
            for file_info in st.session_state.uploaded_files
        )

        if already_uploaded:
            continue

        try:
            saved_path = save_uploaded_file(uploaded_file)

            sheets = load_file(uploaded_file)
            detected_type = detect_document_type(sheets)

            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "path": saved_path,
                "type": detected_type,
                "sheets": sheets,
            })

            st.session_state.uploaded_files_raw.append(uploaded_file)

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"📄 **{uploaded_file.name}** upload ho gayi.\n\n"
                    f"Detected type: **{detected_type}**\n\n"
                    "Ab aap chat mein likh sakte hain:\n"
                    "- `Run reconciliation`\n"
                    "- `Show matched transactions`\n"
                    "- `Show unmatched transactions`\n"
                    "- Ya koi bhi file-related sawal poochein."
                ),
            })

            new_files_added = True

        except Exception as error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"⚠️ **{uploaded_file.name}** process nahi ho saki.\n\n"
                    f"Error: {error}"
                ),
            })

    if new_files_added:
        st.rerun()

user_input = st.chat_input(
    "Ask anything about your files, or type: Run reconciliation"
)

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    command = user_input.lower()

    if "run reconciliation" in command:
        try:
            bank_file = next(
                (
                    file_info
                    for file_info in st.session_state.uploaded_files
                    if file_info["type"] == "Bank Statement"
                ),
                None,
            )

            ledger_file = next(
                (
                    file_info
                    for file_info in st.session_state.uploaded_files
                    if file_info["type"] == "Company Ledger"
                ),
                None,
            )

            if not bank_file or not ledger_file:
                reply = (
                    "⚠️ Reconciliation ke liye Bank Statement aur "
                    "Company Ledger dono files select karein."
                )

            else:
                bank_sheet_name = list(bank_file["sheets"].keys())[0]
                ledger_sheet_name = list(ledger_file["sheets"].keys())[0]

                bank_clean = clean_sheet(
                    bank_file["sheets"][bank_sheet_name]
                )

                ledger_clean = clean_sheet(
                    ledger_file["sheets"][ledger_sheet_name]
                )

                bank_df = standardize(
                    bank_clean,
                    "Bank Statement",
                    bank_sheet_name,
                    bank_file["name"],
                )

                ledger_df = standardize(
                    ledger_clean,
                    "Company Ledger",
                    ledger_sheet_name,
                    ledger_file["name"],
                )

                bank_df, ledger_df, summary = run_reconciliation(
                    bank_df,
                    ledger_df,
                    int(date_tolerance),
                )

                st.session_state.reconciliation_result = {
                    "bank": bank_df,
                    "ledger": ledger_df,
                    "summary": summary,
                }

                reply = f"""
✅ **Reconciliation completed successfully**

| Metric | Value |
|---|---:|
| Bank Transactions | {summary['Bank Transactions']} |
| Ledger Transactions | {summary['Ledger Transactions']} |
| Matched Pairs | {summary['Matched Pairs']} |
| Bank Unmatched | {summary['Bank Unmatched']} |
| Ledger Unmatched | {summary['Ledger Unmatched']} |
| Status | {summary['Reconciliation Status']} |
"""

        except Exception as error:
            reply = f"⚠️ Reconciliation error:\n\n`{error}`"

    elif command in [
        "show matched transactions",
        "show matched",
        "matched transactions",
    ]:
        if st.session_state.reconciliation_result:
            bank_df = st.session_state.reconciliation_result["bank"]
            matched = bank_df[
                bank_df["Status"].isin(["Matched", "Suggested Match"])
            ]

            reply = f"✅ Total matched transactions: **{len(matched)}**"

        else:
            reply = "Pehle `Run reconciliation` karein."

    elif command in [
        "show unmatched transactions",
        "show unmatched",
        "unmatched transactions",
    ]:
        if st.session_state.reconciliation_result:
            bank_df = st.session_state.reconciliation_result["bank"]
            ledger_df = st.session_state.reconciliation_result["ledger"]

            bank_unmatched = bank_df[bank_df["Status"] == "Unmatched"]
            ledger_unmatched = ledger_df[ledger_df["Status"] == "Unmatched"]

            reply = f"""
🔴 **Bank Unmatched:** {len(bank_unmatched)}  
🔴 **Ledger Unmatched:** {len(ledger_unmatched)}
"""

        else:
            reply = "Pehle `Run reconciliation` karein."

    else:
        bank_df = None
        ledger_df = None
        summary = None

        if st.session_state.reconciliation_result:
            bank_df = st.session_state.reconciliation_result["bank"]
            ledger_df = st.session_state.reconciliation_result["ledger"]
            summary = st.session_state.reconciliation_result["summary"]

        context = build_context(
            bank_df,
            ledger_df,
            summary,
            st.session_state.uploaded_files,
        )

        with st.chat_message("assistant"):
            with st.spinner("AI is analyzing your files..."):
                reply = ask_ai(
                    provider,
                    api_key,
                    model,
                    user_input,
                    context,
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
    })

    st.rerun()

if st.session_state.reconciliation_result:
    st.divider()
    st.header("📊 Reconciliation Dashboard")

    result = st.session_state.reconciliation_result
    summary = result["summary"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Bank Transactions",
        summary["Bank Transactions"],
    )

    col2.metric(
        "Ledger Transactions",
        summary["Ledger Transactions"],
    )

    col3.metric(
        "Matched Pairs",
        summary["Matched Pairs"],
    )

    col4.metric(
        "Status",
        summary["Reconciliation Status"],
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bank Transactions",
        "Ledger Transactions",
        "Bank Unmatched",
        "Ledger Unmatched",
        "Matched Pairs",
    ])

    with tab1:
        st.dataframe(
            result["bank"],
            use_container_width=True,
        )

    with tab2:
        st.dataframe(
            result["ledger"],
            use_container_width=True,
        )

    with tab3:
        st.dataframe(
            result["bank"][result["bank"]["Status"] == "Unmatched"],
            use_container_width=True,
        )

    with tab4:
        st.dataframe(
            result["ledger"][result["ledger"]["Status"] == "Unmatched"],
            use_container_width=True,
        )

    with tab5:
        matched_bank = result["bank"][
            result["bank"]["Status"].isin(["Matched", "Suggested Match"])
        ]

        matched_ledger = result["ledger"][
            result["ledger"]["Status"].isin(["Matched", "Suggested Match"])
        ]

        matched_pairs = pd.merge(
            matched_bank,
            matched_ledger,
            on="Match ID",
            suffixes=("_Bank", "_Ledger"),
            how="inner",
        )

        st.dataframe(
            matched_pairs,
            use_container_width=True,
        )

    st.divider()
    st.subheader("⬇️ Download Reports")

    excel_report = create_excel_report(
        result["bank"],
        result["ledger"],
        result["summary"],
    )

    st.download_button(
        label="Download Full Reconciliation Report",
        data=excel_report,
        file_name="Bank_Reconciliation_Report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

    bank_unmatched = result["bank"][
        result["bank"]["Status"] == "Unmatched"
    ]

    ledger_unmatched = result["ledger"][
        result["ledger"]["Status"] == "Unmatched"
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Bank Unmatched",
            data=bank_unmatched.to_csv(index=False).encode("utf-8"),
            file_name="Bank_Unmatched.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="Download Ledger Unmatched",
            data=ledger_unmatched.to_csv(index=False).encode("utf-8"),
            file_name="Ledger_Unmatched.csv",
            mime="text/csv",
            use_container_width=True,
        )