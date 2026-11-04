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

st.set_page_config(
    page_title="DocuSphere AI — Financial Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Sidebar & Apply Modern CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; font-weight: 600; border: none; }
    .stButton>button:hover { background-color: #2ea043; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "reconciliation_result" not in st.session_state:
    st.session_state.reconciliation_result = None

# Top Configuration Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚡ DocuSphere AI — Financial Intelligence")
    st.caption("Automated Multi-Entity Bank Reconciliation, Audit Analytics & Power BI Dashboarding")
with col2:
    openrouter_key = st.text_input("OpenRouter API Key", type="password", key="user_api_key", placeholder="sk-or-v1-...")
    if not openrouter_key:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

st.divider()

# Direct Main Area File Upload
uploaded_files = st.file_uploader(
    "📥 Drag and drop your Financial Statements / Company Ledgers here (CSV, XLSX, XLS)",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

st.divider()

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Type your command (e.g. 'reconcile', 'show charts', or ask any accounting query)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    cmd = user_input.lower().strip()
    
    if "reconcile" in cmd or "reconciliation" in cmd:
        if not uploaded_files:
            reply = "⚠️ **Execution Error**: Please upload both Bank Statement and Company Ledger files above to run reconciliation."
        else:
            try:
                with st.spinner("Executing Deterministic Matching Engine..."):
                    b_df, l_df, m_df, summary = process_reconciliation_files(uploaded_files)
                    st.session_state.reconciliation_result = {
                        "bank": b_df,
                        "ledger": l_df,
                        "matched": m_df,
                        "summary": summary
                    }
                    reply = f"""### ✅ Reconciliation Execution Completed

| Audit Metric | Recorded Value |
|---|---:|
| **Bank Transactions Processed** | `{summary.get('Bank Transactions', 0):,}` |
| **Ledger Transactions Processed** | `{summary.get('Ledger Transactions', 0):,}` |
| **Matched Pairs Identified** | `{summary.get('Matched Pairs', 0):,}` |
| **Unmatched Bank Items** | `{summary.get('Bank Unmatched', 0):,}` |
| **Unmatched Ledger Items** | `{summary.get('Ledger Unmatched', 0):,}` |
| **Audit Status** | **{summary.get('Reconciliation Status', 'Needs Review')}** |
"""
            except Exception as e:
                reply = f"❌ **Reconciliation Engine Error**: {str(e)}"
    elif "power bi" in cmd or "chart" in cmd or "analytics" in cmd or "dashboard" in cmd:
        if uploaded_files:
            try:
                df_sample = pd.read_csv(uploaded_files[0]) if uploaded_files[0].name.endswith('.csv') else pd.read_excel(uploaded_files[0])
                kpis, charts = generate_powerbi_dashboard(df_sample)
                reply = "### 📊 Visual Analytics Dashboard Generated\n\n"
                for k, v in kpis.items():
                    reply += f"- **{k}:** `{v}`\n"
                st.session_state["active_charts"] = charts
            except Exception as e:
                reply = f"❌ **Analytics Engine Error**: {str(e)}"
        else:
            reply = "⚠️ **Input Required**: Please upload at least one financial dataset to generate visual analytics."
    else:
        context_text = ""
        if st.session_state.reconciliation_result:
            context_text = str(st.session_state.reconciliation_result["summary"])
        reply = query_openrouter_ai(user_input, data_context=context_text, api_key=openrouter_key)
        
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

if "active_charts" in st.session_state and st.session_state["active_charts"]:
    st.divider()
    st.subheader("📈 Power BI Analytics Dashboard")
    cols = st.columns(len(st.session_state["active_charts"]))
    for i, (name, fig) in enumerate(st.session_state["active_charts"].items()):
        with cols[i % len(cols)]:
            st.plotly_chart(fig, use_container_width=True)

if st.session_state.reconciliation_result:
    st.divider()
    st.subheader("📊 Complete Audit Workspace & Results")
    res = st.session_state.reconciliation_result
    
    t1, t2, t3, t4 = st.tabs(["Matched Pairs (REC-000001)", "Unmatched Bank Items", "Unmatched Ledger Items", "Summary Metrics"])
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
        label="⬇️ Download Complete Multi-Sheet Financial Audit Report (XLSX)",
        data=excel_report,
        file_name="DocuSphere_Full_Reconciliation_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
