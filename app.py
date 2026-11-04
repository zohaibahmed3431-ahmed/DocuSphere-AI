import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from services.reconciliation_service import process_reconciliation_files
from core.powerbi_engine import generate_powerbi_dashboard
from core.exporter import export_reconciliation_report
from ai.assistant import query_openrouter_ai

st.set_page_config(page_title="DocuSphere AI", page_icon="🏦", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "reconciliation_result" not in st.session_state:
    st.session_state.reconciliation_result = None

with st.sidebar:
    st.title("🏦 DocuSphere AI")
    if st.button("＋ New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.reconciliation_result = None
        st.rerun()
        
    st.divider()
    openrouter_key = st.text_input("OpenRouter API Key", type="password", key="user_api_key")
    if not openrouter_key:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        
    st.divider()
    st.subheader("📁 File Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Financial Data / Ledgers",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

st.title("DocuSphere AI")
st.caption("Enterprise Bank Reconciliation & Power BI Level Analytics Platform")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask a question, request Power BI analysis, or type 'reconcile'...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    cmd = user_input.lower().strip()
    
    if "reconcile" in cmd or "reconciliation" in cmd:
        if not uploaded_files:
            reply = "⚠️ Reconciliation run karne ke liye sidebar mein Bank Statement aur Ledger upload karein."
        else:
            try:
                with st.spinner("Reconciliation Engine executing..."):
                    b_df, l_df, m_df, summary = process_reconciliation_files(uploaded_files)
                    st.session_state.reconciliation_result = {
                        "bank": b_df,
                        "ledger": l_df,
                        "matched": m_df,
                        "summary": summary
                    }
                    reply = f"""✅ **Reconciliation Completed Successfully**

| Metric | Value |
|---|---:|
| Bank Transactions | {summary['Bank Transactions']} |
| Ledger Transactions | {summary['Ledger Transactions']} |
| Matched Pairs | {summary['Matched Pairs']} |
| Bank Unmatched | {summary['Bank Unmatched']} |
| Ledger Unmatched | {summary['Ledger Unmatched']} |
| Status | **{summary['Status']}** |
"""
            except Exception as e:
                reply = f"❌ Reconciliation Error: {str(e)}"
    elif "power bi" in cmd or "chart" in cmd or "analytics" in cmd or "dashboard" in cmd:
        if uploaded_files:
            try:
                df_sample = pd.read_csv(uploaded_files[0]) if uploaded_files[0].name.endswith('.csv') else pd.read_excel(uploaded_files[0])
                kpis, charts = generate_powerbi_dashboard(df_sample)
                reply = "### 📊 Power BI Analytics Summary\n"
                for k, v in kpis.items():
                    reply += f"- **{k}:** `{v}`\n"
                st.session_state["active_charts"] = charts
            except Exception as e:
                reply = f"❌ Analytics Error: {str(e)}"
        else:
            reply = "⚠️ Analytics ke liye kam se kam ek CSV ya Excel file upload karein."
    else:
        context_text = ""
        if st.session_state.reconciliation_result:
            context_text = str(st.session_state.reconciliation_result["summary"])
        reply = query_openrouter_ai(user_input, data_context=context_text, api_key=openrouter_key)
        
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

if "active_charts" in st.session_state and st.session_state["active_charts"]:
    st.divider()
    st.header("📈 Power BI Visual Analytics")
    cols = st.columns(len(st.session_state["active_charts"]))
    for i, (name, fig) in enumerate(st.session_state["active_charts"].items()):
        with cols[i % len(cols)]:
            st.plotly_chart(fig, use_container_width=True)

if st.session_state.reconciliation_result:
    st.divider()
    st.header("📊 Reconciliation Results & Workspace")
    res = st.session_state.reconciliation_result
    
    t1, t2, t3, t4 = st.tabs(["Matched Pairs", "Bank Unmatched", "Ledger Unmatched", "Full Summary"])
    with t1:
        st.dataframe(res["matched"], use_container_width=True)
    with t2:
        st.dataframe(res["bank"][res["bank"]["Status"] == "Unmatched"], use_container_width=True)
    with t3:
        st.dataframe(res["ledger"][res["ledger"]["Status"] == "Unmatched"], use_container_width=True)
    with t4:
        st.json(res["summary"])
        
    excel_report = export_reconciliation_report(res["bank"], res["ledger"], res["matched"], res["summary"])
    st.download_button(
        label="⬇️ Download Reconciliation Excel Report",
        data=excel_report,
        file_name="DocuSphere_Reconciliation_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
