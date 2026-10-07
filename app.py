import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="DocuSphere AI — Enterprise Multi-File Engine",
    page_icon="📁",
    layout="wide"
)

# Persistent States
if "matched_csv" not in st.session_state:
    st.session_state.matched_csv = None
if "unmatched_csv" not in st.session_state:
    st.session_state.unmatched_csv = None
if "reconciled" not in st.session_state:
    st.session_state.reconciled = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. HELPER FUNCTIONS & DATA PARSER
# ==========================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all').dropna(how='all', axis=1)
    df.columns = [re.sub(r'\s+', '_', str(c).strip().lower()) for c in df.columns]
    return df

def find_best_column(df: pd.DataFrame, keywords: list) -> str:
    for kw in keywords:
        for col in df.columns:
            if kw in col:
                return col
    return None

def normalize_text(val) -> str:
    if pd.isna(val):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(val).lower().strip())

def extract_identifiers(val) -> set:
    if pd.isna(val):
        return set()
    matches = re.findall(r'[a-zA-Z0-9]{4,}', str(val))
    return set(matches)

def extract_file_context(files) -> str:
    """Dynamically converts uploaded CSV/Excel files into AI-readable context."""
    if not files:
        return "No files currently uploaded."
    
    context_str = "UPLOADED FILES DATA SUMMARY:\n\n"
    for idx, f in enumerate(files, 1):
        try:
            f.seek(0)
            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            clean_df = clean_column_names(df)
            
            context_str += f"=== FILE {idx}: {f.name} ===\n"
            context_str += f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}\n"
            context_str += f"Columns List: {list(df.columns)}\n"
            context_str += f"Numeric Columns Summary:\n{df.describe().to_string()}\n\n"
            context_str += f"First 5 Sample Rows:\n{clean_df.head(5).to_string()}\n"
            context_str += "--------------------------------------------------\n\n"
        except Exception as e:
            context_str += f"=== FILE {idx}: {f.name} (Error parsing file: {str(e)}) ===\n\n"
            
    return context_str

# ==========================================
# 3. DETERMINISTIC RECONCILIATION ENGINE
# ==========================================
def reconcile_ledgers(bank_df: pd.DataFrame, ledger_df: pd.DataFrame):
    b_df = clean_column_names(bank_df)
    l_df = clean_column_names(ledger_df)

    b_serial_col = find_best_column(b_df, ['serial', 'ref', 'doc', 'no', 'trans', 'id', 'cheque']) or b_df.columns[0]
    l_serial_col = find_best_column(l_df, ['voucher', 'serial', 'ref', 'doc', 'no', 'trans', 'id', 'cheque']) or l_df.columns[0]

    b_amt_col = find_best_column(b_df, ['amount', 'amt', 'credit', 'debit', 'bal']) or b_df.columns[1]
    l_amt_col = find_best_column(l_df, ['amount', 'amt', 'debit', 'credit', 'bal']) or l_df.columns[1]

    b_date_col = find_best_column(b_df, ['date', 'time', 'day'])
    l_date_col = find_best_column(l_df, ['date', 'time', 'day'])

    matched_records = []
    unmatched_bank = []
    ledger_matched_indices = set()
    rec_counter = 1

    for b_idx, b_row in b_df.iterrows():
        try:
            b_amt = float(re.sub(r'[^0-9.-]', '', str(b_row[b_amt_col]))) if pd.notna(b_row[b_amt_col]) else 0.0
        except Exception:
            b_amt = 0.0

        b_str = " ".join([str(v) for v in b_row.values if pd.notna(v)])
        b_ids = extract_identifiers(b_str)
        found_match = False

        for l_idx, l_row in l_df.iterrows():
            if l_idx in ledger_matched_indices:
                continue

            try:
                l_amt = float(re.sub(r'[^0-9.-]', '', str(l_row[l_amt_col]))) if pd.notna(l_row[l_amt_col]) else 0.0
            except Exception:
                l_amt = 0.0

            if abs(abs(b_amt) - abs(l_amt)) > 0.01:
                continue

            l_str = " ".join([str(v) for v in l_row.values if pd.notna(v)])
            l_ids = extract_identifiers(l_str)

            common_ids = b_ids.intersection(l_ids)
            b_norm = normalize_text(b_str)
            l_norm = normalize_text(l_str)

            if len(common_ids) > 0 or (len(b_norm) > 4 and b_norm in l_norm) or (len(l_norm) > 4 and l_norm in b_norm) or b_amt != 0:
                rec_id = f"REC-{rec_counter:03d}"
                rec_counter += 1
                ledger_matched_indices.add(l_idx)

                b_date = b_row[b_date_col] if b_date_col and pd.notna(b_row[b_date_col]) else "N/A"
                
                matched_records.append({
                    "Reconciliation_ID": rec_id,
                    "Date": b_date,
                    "Bank_Original_Serial": b_row[b_serial_col],
                    "Ledger_Original_Serial": l_row[l_serial_col],
                    "Bank_Amount": b_amt,
                    "Ledger_Amount": l_amt,
                    "Match_Status": "MATCHED"
                })
                found_match = True
                break

        if not found_match:
            unmatched_bank.append({
                "Source": "Bank Only (Unmatched)",
                "Original_Serial": b_row[b_serial_col],
                "Date": b_row[b_date_col] if b_date_col and pd.notna(b_row[b_date_col]) else "N/A",
                "Amount": b_amt,
                "Type_Recommendation": "Bank Charges / Direct Inflows",
                "Details": b_str
            })

    unmatched_ledger = []
    for l_idx, l_row in l_df.iterrows():
        if l_idx not in ledger_matched_indices:
            try:
                l_amt = float(re.sub(r'[^0-9.-]', '', str(l_row[l_amt_col]))) if pd.notna(l_row[l_amt_col]) else 0.0
            except Exception:
                l_amt = 0.0

            unmatched_ledger.append({
                "Source": "Ledger Only (Unmatched)",
                "Original_Serial": l_row[l_serial_col],
                "Date": l_row[l_date_col] if l_date_col and pd.notna(l_row[l_date_col]) else "N/A",
                "Amount": l_amt,
                "Type_Recommendation": "Unpresented Cheques / Pending Clearance",
                "Details": " ".join([str(v) for v in l_row.values if pd.notna(v)])
            })

    matched_df = pd.DataFrame(matched_records)
    unmatched_df = pd.concat([pd.DataFrame(unmatched_bank), pd.DataFrame(unmatched_ledger)], ignore_index=True)

    summary = {
        "total_bank_txns": len(b_df),
        "total_ledger_txns": len(l_df),
        "matched_pairs": len(matched_records),
        "unmatched_bank_count": len(unmatched_bank),
        "unmatched_ledger_count": len(unmatched_ledger),
        "net_variance": round(abs(sum([r.get('Bank_Amount', 0) for r in matched_records]) - sum([r.get('Ledger_Amount', 0) for r in matched_records])), 2)
    }

    return matched_df, unmatched_df, summary

# ==========================================
# 4. STREAMLIT INTERFACE & GEMINI AI INTEGRATION
# ==========================================
st.title("📁 DocuSphere AI — Enterprise Multi-File Data Engine")
st.caption("Autonomous Lead AI Accountant & Multi-Format Document Intelligence")

# Sidebar Configuration
st.sidebar.header("🔑 Gemini AI API Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password", help="Required to enable full AI understanding.")

if api_key:
    genai.configure(api_key=api_key)

st.sidebar.markdown("---")
st.sidebar.header("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Bank Statements, General Ledgers, Shopify/Amazon CSVs, or Excel",
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} File(s) Active in Context")
    for idx, f in enumerate(uploaded_files, 1):
        st.sidebar.caption(f"📄 **File {idx}:** {f.name} ({round(f.size/1024, 1)} KB)")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.matched_csv = None
    st.session_state.unmatched_csv = None
    st.session_state.reconciled = False
    st.rerun()

# Chat History Rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask any question about your uploaded files or requested analysis...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    low_input = user_input.lower().strip()

    # Special Trigger: Reconciliation Engine
    if "reconcile" in low_input and uploaded_files and len(uploaded_files) >= 2:
        with st.spinner("Executing Deterministic Bank Reconciliation Engine..."):
            f1, f2 = uploaded_files[0], uploaded_files[1]
            f1.seek(0)
            f2.seek(0)
            df1 = pd.read_csv(f1) if f1.name.endswith('.csv') else pd.read_excel(f1)
            df2 = pd.read_csv(f2) if f2.name.endswith('.csv') else pd.read_excel(f2)
            
            matched_df, unmatched_df, summary = reconcile_ledgers(df1, df2)
            
            st.session_state.matched_csv = matched_df.to_csv(index=False).encode('utf-8')
            st.session_state.unmatched_csv = unmatched_df.to_csv(index=False).encode('utf-8')
            st.session_state.reconciled = True
            
            response_text = f"""
### 📊 Executive Summary — Bank vs General Ledger Reconciliation

* **Bank File:** `{f1.name}` ({summary['total_bank_txns']} records)
* **Ledger File:** `{f2.name}` ({summary['total_ledger_txns']} records)
* **Matched Transactions (Sequential REC-XXX Assigned):** `{summary['matched_pairs']}`
* **Unmatched Bank Entries:** `{summary['unmatched_bank_count']}`
* **Unmatched Ledger Entries:** `{summary['unmatched_ledger_count']}`
* **Net Financial Variance:** `${summary['net_variance']}`

---

#### 🟢 Matched Transactions Preview
{matched_df.head(5).to_markdown(index=False) if not matched_df.empty else "No matched records found."}

---

#### 🔴 Unmatched Transactions Preview
{unmatched_df.head(5).to_markdown(index=False) if not unmatched_df.empty else "No unmatched records found."}
"""
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)

    # All Other Queries Handled Dynamically by Gemini AI Engine
    else:
        if not api_key:
            reply = "⚠️ Please enter your **Gemini API Key** in the sidebar to enable full AI document intelligence."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            with st.spinner("DocuSphere AI is processing query & file context..."):
                try:
                    file_context = extract_file_context(uploaded_files)
                    
                    system_prompt = f"""
You are "DocuSphere AI" — an enterprise-grade Autonomous Financial Data Analyst and Multi-File AI Engine.
You specialize in Accounting, Bank Reconciliation, E-Commerce Analytics, and Big Data Processing with zero hallucinations.

Acknowledge the user's specific query directly and provide a professional, highly detailed answer.
When files are uploaded, use the provided file context below to give exact figures, detailed column breakdowns, or relevant summaries.

CONTEXT OF CURRENTLY UPLOADED FILES:
{file_context}

USER QUESTION:
{user_input}
"""
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    ai_response = model.generate_content(system_prompt)
                    
                    st.session_state.messages.append({"role": "assistant", "content": ai_response.text})
                    with st.chat_message("assistant"):
                        st.markdown(ai_response.text)

                except Exception as e:
                    error_msg = f"❌ Error executing Gemini AI Engine: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.markdown(error_msg)

# Downloads Rendering
if st.session_state.reconciled:
    st.markdown("---")
    st.subheader("📥 Download Reconciled Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("🟢 Download Matched CSV", st.session_state.matched_csv, "docusphere_matched.csv", "text/csv", key="btn_m")
    with col2:
        st.download_button("🔴 Download Unmatched CSV", st.session_state.unmatched_csv, "docusphere_unmatched.csv", "text/csv", key="btn_u")
