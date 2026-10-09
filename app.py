import os

import streamlit as st

from services.reconciliation_service import process_uploaded_files
from core.exporter import create_excel_report
from config.settings import UPLOAD_FOLDER

st.set_page_config(
    page_title="Bank Reconciliation AI",
    page_icon="🏦",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_raw" not in st.session_state:
    st.session_state.uploaded_files_raw = []

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []

if "result" not in st.session_state:
    st.session_state.result = None

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_uploaded_file(uploaded_file):
    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name,
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())

    return file_path


with st.sidebar:
    st.title("🏦 Reconciliation AI")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_files_raw = []
        st.session_state.uploaded_files_info = []
        st.session_state.result = None
        st.rerun()

    st.divider()
    st.subheader("API Key")
    st.text_input(
        "AI API Key",
        type="password",
        key="api_key",
    )
    st.caption("API key sirf is session mein use hogi.")

    st.divider()
    st.subheader("Uploaded Files")

    if st.session_state.uploaded_files_info:
        for file_info in st.session_state.uploaded_files_info:
            st.markdown(
                f"📄 **{file_info['name']}**  \n"
                f"`{file_info['type']}`"
            )
    else:
        st.caption("Koi file upload nahi hui.")


st.title("Bank Reconciliation AI")
st.caption(
    "Bank statement aur company ledger upload karein, "
    "phir chat mein reconciliation run karein."
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


uploaded_files = st.file_uploader(
    "Attach files",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


if uploaded_files:
    new_files_added = False

    for uploaded_file in uploaded_files:
        already_uploaded = any(
            file_info["name"] == uploaded_file.name
            for file_info in st.session_state.uploaded_files_info
        )

        if already_uploaded:
            continue

        try:
            saved_path = save_uploaded_file(uploaded_file)

            st.session_state.uploaded_files_raw.append(uploaded_file)

            st.session_state.uploaded_files_info.append({
                "name": uploaded_file.name,
                "type": "Uploaded",
                "path": saved_path,
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"📄 **{uploaded_file.name}** upload ho gayi.\n\n"
                    "Ab chat mein likhein: `Run reconciliation`"
                ),
            })

            new_files_added = True

        except Exception as error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"⚠️ **{uploaded_file.name}** upload nahi ho saki.\n\n"
                    f"Error: {error}"
                ),
            })

    if new_files_added:
        st.rerun()


user_input = st.chat_input(
    "Ask anything or type: Run reconciliation"
)


if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    command = user_input.lower()

    if "run reconciliation" in command:
        try:
            if not st.session_state.uploaded_files_raw:
                reply = (
                    "⚠️ Pehle bank statement aur company ledger "
                    "upload karein."
                )

            else:
                result = process_uploaded_files(
                    st.session_state.uploaded_files_raw
                )

                st.session_state.result = result

                summary = result["summary"]

                reply = f"""
✅ **Reconciliation completed**

| Metric | Value |
|---|---:|
| Bank Transactions | {summary['Bank Transactions']} |
| Ledger Transactions | {summary['Ledger Transactions']} |
| Matched | {summary['Matched']} |
| Bank Unmatched | {summary['Bank Unmatched']} |
| Ledger Unmatched | {summary['Ledger Unmatched']} |
| Status | {summary['Status']} |
"""

        except Exception as error:
            reply = f"⚠️ Reconciliation error:\n\n`{error}`"

    elif "matched" in command:
        if st.session_state.result:
            matched = st.session_state.result["bank"]
            matched = matched[matched["Status"] == "Matched"]

            reply = f"✅ Total matched transactions: **{len(matched)}**"

        else:
            reply = "Pehle `Run reconciliation` karein."

    elif "unmatched" in command:
        if st.session_state.result:
            bank_unmatched = st.session_state.result["bank"]
            bank_unmatched = bank_unmatched[
                bank_unmatched["Status"] == "Unmatched"
            ]

            ledger_unmatched = st.session_state.result["ledger"]
            ledger_unmatched = ledger_unmatched[
                ledger_unmatched["Status"] == "Unmatched"
            ]

            reply = f"""
🔴 **Bank Unmatched:** {len(bank_unmatched)}  
🔴 **Ledger Unmatched:** {len(ledger_unmatched)}
"""

        else:
            reply = "Pehle `Run reconciliation` karein."

    else:
        reply = (
            "Main abhi ye commands support karta hoon:\n\n"
            "- `Run reconciliation`\n"
            "- `Show matched transactions`\n"
            "- `Show unmatched transactions`"
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
    })

    st.rerun()


if st.session_state.result:
    st.divider()
    st.header("📊 Reconciliation Results")

    result = st.session_state.result

    tab1, tab2, tab3, tab4 = st.tabs([
        "Bank",
        "Ledger",
        "Bank Unmatched",
        "Ledger Unmatched",
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

    st.download_button(
        label="⬇️ Download Reconciliation Report",
        data=create_excel_report(
            result["bank"],
            result["ledger"],
            result["summary"],
        ),
        file_name="Bank_Reconciliation_Report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        use_container_width=True,
    )