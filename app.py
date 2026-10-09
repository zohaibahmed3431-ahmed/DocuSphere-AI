import streamlit as st

import pandas as pd

from core.file_loader import load_file, get_sheet_names
from core.data_cleaner import clean_raw_sheet, standardize_transactions
from core.reconciler import run_reconciliation
from core.exporter import create_reconciliation_excel

st.set_page_config(
    page_title="Bank Reconciliation AI",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Reconciliation AI")
st.caption(
    "Upload bank statement and company ledger, run reconciliation, "
    "review matched and unmatched transactions, and download reports."
)

with st.sidebar:
    st.header("📁 Upload Files")

    bank_file = st.file_uploader(
        "Bank Statement",
        type=["xlsx", "xls", "csv"],
    )

    ledger_file = st.file_uploader(
        "Company Ledger",
        type=["xlsx", "xls", "csv"],
    )

    run_button = st.button(
        "Run Reconciliation",
        use_container_width=True,
    )

if "bank_df" not in st.session_state:
    st.session_state.bank_df = None

if "ledger_df" not in st.session_state:
    st.session_state.ledger_df = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if run_button:
    if bank_file is None or ledger_file is None:
        st.error("Please upload both Bank Statement and Company Ledger.")
        st.stop()

    try:
        with st.spinner("Reading files..."):
            bank_sheets = load_file(bank_file)
            ledger_sheets = load_file(ledger_file)

        bank_sheet_name = st.selectbox(
            "Select Bank Statement Sheet",
            get_sheet_names(bank_sheets),
        )

        ledger_sheet_name = st.selectbox(
            "Select Company Ledger Sheet",
            get_sheet_names(ledger_sheets),
        )

        with st.spinner("Cleaning and standardizing data..."):
            bank_raw = clean_raw_sheet(bank_sheets[bank_sheet_name])
            ledger_raw = clean_raw_sheet(ledger_sheets[ledger_sheet_name])

            bank_df = standardize_transactions(
                bank_raw,
                "Bank",
                bank_sheet_name,
            )

            ledger_df = standardize_transactions(
                ledger_raw,
                "Ledger",
                ledger_sheet_name,
            )

        with st.spinner("Running reconciliation..."):
            bank_df, ledger_df, summary = run_reconciliation(
                bank_df,
                ledger_df,
            )

        st.session_state.bank_df = bank_df
        st.session_state.ledger_df = ledger_df
        st.session_state.summary = summary

        st.success("Reconciliation completed successfully.")

    except Exception as error:
        st.error(f"Error: {str(error)}")

if st.session_state.summary is not None:
    summary = st.session_state.summary

    st.header("📊 Reconciliation Summary")

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
        "Reconciliation Status",
        summary["Reconciliation Status"],
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Bank Unmatched",
        summary["Bank Unmatched"],
    )

    col2.metric(
        "Ledger Unmatched",
        summary["Ledger Unmatched"],
    )

    st.header("📑 Reconciliation Results")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Bank Transactions",
            "Ledger Transactions",
            "Bank Unmatched",
            "Ledger Unmatched",
        ]
    )

    with tab1:
        st.dataframe(
            st.session_state.bank_df,
            use_container_width=True,
        )

    with tab2:
        st.dataframe(
            st.session_state.ledger_df,
            use_container_width=True,
        )

    with tab3:
        bank_unmatched = st.session_state.bank_df[
            st.session_state.bank_df["Match Status"] == "Unmatched"
        ]

        st.dataframe(
            bank_unmatched,
            use_container_width=True,
        )

    with tab4:
        ledger_unmatched = st.session_state.ledger_df[
            st.session_state.ledger_df["Match Status"] == "Unmatched"
        ]

        st.dataframe(
            ledger_unmatched,
            use_container_width=True,
        )

    st.header("⬇️ Download Reconciliation Report")

    excel_file = create_reconciliation_excel(
        st.session_state.bank_df,
        st.session_state.ledger_df,
        st.session_state.summary,
    )

    st.download_button(
        label="Download Full Reconciliation Excel",
        data=excel_file,
        file_name="bank_reconciliation_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
        use_container_width=True,
    )