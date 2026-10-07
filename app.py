import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

# Safe Gemini Library Import
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="DocuSphere",
    page_icon="📄",
    layout="wide"
)

if "matched_csv" not in st.session_state:
    st.session_state.matched_csv = None
if "unmatched_csv" not in st.session_state:
    st.session_state.unmatched_csv = None
if "reconciled" not in st.session_state:
    st.session_state.reconciled = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. HELPER FUNCTIONS & DATA EXTRACTION
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
    return set(re.findall(r'[a-zA-Z0-9]{4,}', str(val)))

def build_files_context(files) -> str:
    if not files:
        return "No files uploaded."
    
    context = "=== UPLOADED DOCUMENTS & FILES CONTEXT ===\n\n"
    for idx, f in enumerate(files, 1):
        context += f"FILE #{idx}: {f.name}\n"
        try:
            f.seek(0)
            if f.name.lower().endswith(('.csv', '.xlsx', '.xls')):
                df = pd.read_csv(f) if f.name.lower().endswith('.csv') else pd.read_excel(f)
                clean_df = clean_column_names(df)
                context += f"Type: Tabular Data ({len(df)} rows, {len(df.columns)} columns)\n"
                context += f"Columns: {list(df.columns)}\n"
                context += f"Data Sample:\n{clean_df.head(5).to_string()}\n"
            else:
                content = f.read()
                try:
                    text = content.decode('utf-8', errors='ignore')
                    context += f"Content Preview:\n{text[:2500]}\n"
                except Exception:
                    context += "Binary / Image / Document file loaded.\n"
        except Exception as e:
            context += f"File Read Note: {str(e)}\n"
        context += "-----------------------------------\n"
    return context

# ==========================================
# 3. BANK RECONCILIATION ENGINE
# ==========================================
def reconcile_ledgers(bank_df: pd.DataFrame, ledger_df: pd.DataFrame):
    b_df = clean_column_names(bank_df)
    l_df = clean_column_names(ledger_df)

    b_serial_col = find_best_column(b_df, ['serial', 'ref', 'doc', 'no', 'trans', 'id', 'cheque', 'voucher']) or b_df.columns[0]
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
                rec_id = f"REC-{rec_counter:06d}"
                rec_counter += 1
                ledger_matched_indices.add(l_idx)

                b_date = b_row[b_date_col] if b_date_col and pd.notna(b_row[b_date_col]) else "N/A"
                
                matched_records.append({
                    "Reconciliation_ID": rec_id,
                    "Date": b_date,
                    "Bank_Original_Serial": b_row[b_serial_col],
                    "Software_Original_Serial": l_row[l_serial_col],
                    "Bank_Amount": b_amt,
                    "Software_Amount": l_amt,
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
                "Type_Recommendation": "Bank Charges / Direct Credit",
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
                "Source": "Software Ledger Only (Unmatched)",
                "Original_Serial": l_row[l_serial_col],
                "Date": l_row[l_date_col] if l_date_col and pd.notna(l_row[l_date_col]) else "N/A",
                "Amount": l_amt,
                "Type_Recommendation": "Unpresented Cheques / Pending Clearance",
                "Details": " ".join([str(v) for v in l_row.values if pd.notna(v)])
            })

    matched_df = pd.DataFrame(matched_records)
    unmatched_df = pd.concat([pd.DataFrame(unmatched_bank), pd.DataFrame(unmatched_ledger)], ignore_index=True)

    summary = {
        "total_bank": len(b_df),
        "total_ledger": len(l_df),
        "matched_pairs": len(matched_records),
        "unmatched_bank": len(unmatched_bank),
        "unmatched_ledger": len(unmatched_ledger)
    }

    return matched_df, unmatched_df, summary

# ==========================================
# 4. SIDEBAR LAYOUT
# ==========================================
st.sidebar.markdown("## 📁 Documents")
st.sidebar.markdown("**Upload one or more files**")

uploaded_files = st.sidebar.file_uploader(
    "Upload area",
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.sidebar.caption("200MB per file • PDF, DOCX, TXT, MD, PNG, JPG, CSV, XLSX")

process_btn = st.sidebar.button("➕ Process uploads", use_container_width=True)

st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("Gemini API Key (Optional)", type="password")

if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.matched_csv = None
    st.session_state.unmatched_csv = None
    st.session_state.reconciled = False
    st.rerun()

# ==========================================
# 5. MAIN INTERFACE (SIMPLE "DocuSphere")
# ==========================================
st.markdown("# DocuSphere")
st.markdown("")

if not uploaded_files and len(st.session_state.messages) == 0:
    st.info("Upload documents from the left to start. You can also ask general AI questions in any language.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about your documents, request analysis, or type anything...")

if uploaded_files and process_btn:
    st.sidebar.success(f"✅ {len(uploaded_files)} File(s) Processed Automatically!")

# Handling Chat Input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    low_input = user_input.lower().strip()

    # Route 1: Bank Reconciliation Execution
    if ("reconcile" in low_input or "match" in low_input) and uploaded_files and len(uploaded_files) >= 2:
        with st.spinner("Reconciling Bank Statement & Software Ledger..."):
            f1, f2 = uploaded_files[0], uploaded_files[1]
            f1.seek(0)
            f2.seek(0)
            df1 = pd.read_csv(f1) if f1.name.lower().endswith('.csv') else pd.read_excel(f1)
            df2 = pd.read_csv(f2) if f2.name.lower().endswith('.csv') else pd.read_excel(f2)

            matched_df, unmatched_df, summary = reconcile_ledgers(df1, df2)

            st.session_state.matched_csv = matched_df.to_csv(index=False).encode('utf-8')
            st.session_state.unmatched_csv = unmatched_df.to_csv(index=False).encode('utf-8')
            st.session_state.reconciled = True

            reply = f"""
### 📊 Reconciliation Results

* **Bank Records:** `{summary['total_bank']}` | **Software Records:** `{summary['total_ledger']}`
* **Matched Entries (REC-000001 series):** `{summary['matched_pairs']}`
* **Unmatched Bank Entries:** `{summary['unmatched_bank']}`
* **Unmatched Software Entries:** `{summary['unmatched_ledger']}`

---

#### 🟢 Matched Records Preview
{matched_df.head(5).to_markdown(index=False) if not matched_df.empty else "No matched records."}

---

#### 🔴 Unmatched Records Preview
{unmatched_df.head(5).to_markdown(index=False) if not unmatched_df.empty else "No unmatched records."}
"""
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    # Route 2: Multi-Language Conversational & Document AI
    else:
        key = api_key_input or os.environ.get("GEMINI_API_KEY")
        if not key and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]

        if HAS_GEMINI and key:
            with st.spinner("DocuSphere AI is processing..."):
                try:
                    genai.configure(api_key=key)
                    file_context = build_files_context(uploaded_files)

                    prompt = f"""
You are DocuSphere — an enterprise-grade multimodal AI assistant.

IMPORTANT INSTRUCTIONS:
1. Detect the user's language (English, Urdu, Roman Urdu, Hindi, Arabic, Spanish, etc.) and respond in that exact language naturally.
2. If the query asks about uploaded documents, cite exact details and page/file references.
3. If no documents are referenced or needed, act as a helpful general AI.

DOCUMENT CONTEXT:
{file_context}

USER QUERY:
{user_input}
"""
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(prompt)

                    st.session_state.messages.append({"role": "assistant", "content": res.text})
                    with st.chat_message("assistant"):
                        st.markdown(res.text)
                except Exception as e:
                    err_msg = f"❌ AI Engine Notice: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    with st.chat_message("assistant"):
                        st.markdown(err_msg)
        else:
            file_context = build_files_context(uploaded_files)
            reply = f"Hello! I am **DocuSphere**.\n\nQuery received: *'{user_input}'*\n\n**Processed Documents:**\n{file_context[:400]}...\n\n*(Add your Gemini API Key in the sidebar for real-time generative responses)*"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

# Persistent Download Buttons
if st.session_state.reconciled:
    st.markdown("---")
    st.subheader("📥 Download Reconciliation Data")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("🟢 Download Matched Records CSV", st.session_state.matched_csv, "docusphere_matched.csv", "text/csv")
    with col2:
        st.download_button("🔴 Download Unmatched Records CSV", st.session_state.unmatched_csv, "docusphere_unmatched.csv", "text/csv")
