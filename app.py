import os
import sys

# Root path add karna taake Streamlit Cloud core aur modules ko dhoond sake
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from core.document_parser import extract_file_context
from core.llm_engine import generate_ai_response
from modules.reconciliation import reconcile_ledgers

# 1. Page Configuration
st.set_page_config(page_title="DocuSphere", page_icon="📄", layout="wide")

# 2. Session States Initializations
if "matched_csv" not in st.session_state:
    st.session_state.matched_csv = None
if "unmatched_csv" not in st.session_state:
    st.session_state.unmatched_csv = None
if "reconciled" not in st.session_state:
    st.session_state.reconciled = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar UI
st.sidebar.markdown("## 📁 Documents")
st.sidebar.markdown("**Upload one or more files**")

uploaded_files = st.sidebar.file_uploader(
    "Upload area",
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.sidebar.caption("200MB per file • PDF, DOCX, TXT, MD, PNG, JPG, CSV, XLSX")
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("Gemini API Key (Optional)", type="password")

if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.matched_csv = None
    st.session_state.unmatched_csv = None
    st.session_state.reconciled = False
    st.rerun()

# 4. Main Title
st.markdown("# DocuSphere")
st.markdown("")

# 5. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about your documents or any general query...")

# 6. Automatic Request Routing
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AUTO PROCESS FILES IN REAL-TIME (No manual button required)
    file_context = extract_file_context(uploaded_files)
    low_input = user_input.lower().strip()

    # Route A: Bank Reconciliation
    if ("reconcile" in low_input or "reconciliation" in low_input) and uploaded_files and len(uploaded_files) >= 2:
        with st.spinner("Executing Bank vs Ledger Reconciliation..."):
            try:
                f1, f2 = uploaded_files[0], uploaded_files[1]
                f1.seek(0)
                f2.seek(0)
                df1 = pd.read_csv(f1) if f1.name.lower().endswith('.csv') else pd.read_excel(f1)
                df2 = pd.read_csv(f2) if f2.name.lower().endswith('.csv') else pd.read_excel(f2)

                matched_df, unmatched_df = reconcile_ledgers(df1, df2)

                st.session_state.matched_csv = matched_df.to_csv(index=False).encode('utf-8')
                st.session_state.unmatched_csv = unmatched_df.to_csv(index=False).encode('utf-8')
                st.session_state.reconciled = True

                reply = f"""
### 📊 Reconciliation Results Summary

* **Matched Entries (REC-XXXXXX):** `{len(matched_df)}`
* **Unmatched Entries Total:** `{len(unmatched_df)}`

---

#### 🟢 Matched Preview
{matched_df.head(5).to_markdown(index=False) if not matched_df.empty else "No matched records found."}

---

#### 🔴 Unmatched Preview
{unmatched_df.head(5).to_markdown(index=False) if not unmatched_df.empty else "No unmatched records found."}
"""
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)

            except Exception as e:
                err = f"❌ Reconciliation Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.markdown(err)

    # Route B: Multimodal Gemini AI Generation
    else:
        with st.spinner("DocuSphere AI processing..."):
            reply = generate_ai_response(user_input, file_context, api_key_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

# 7. Persistent Downloads (Won't disappear on click)
if st.session_state.reconciled:
    st.markdown("---")
    st.subheader("📥 Download Reconciliation Data")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("🟢 Download Matched CSV", st.session_state.matched_csv, "docusphere_matched.csv", "text/csv")
    with col2:
        st.download_button("🔴 Download Unmatched CSV", st.session_state.unmatched_csv, "docusphere_unmatched.csv", "text/csv")
