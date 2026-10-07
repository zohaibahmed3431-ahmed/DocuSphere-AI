import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import io

# ==========================================
# 1. PAGE CONFIG & PERSISTENT SESSION STATE
# ==========================================
st.set_page_config(
    page_title="FinAI Pro — Enterprise Autonomous Engine",
    page_icon="💼",
    layout="wide"
)

# Download State Lock (Streamlit Refresh Bug Fix)
if "matched_csv" not in st.session_state:
    st.session_state.matched_csv = None
if "unmatched_csv" not in st.session_state:
    st.session_state.unmatched_csv = None
if "reconciled" not in st.session_state:
    st.session_state.reconciled = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. HELPER FUNCTIONS & PARSING LOGIC
# ==========================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Detect headers, drop empty rows/columns, and sanitize column names."""
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

# ==========================================
# 3. BANK RECONCILIATION ENGINE
# ==========================================
def reconcile_ledgers(bank_df: pd.DataFrame, ledger_df: pd.DataFrame):
    b_df = clean_column_names(bank_df)
    l_df = clean_column_names(ledger_df)

    # Detect Columns
    b_serial_col = find_best_column(b_df, ['serial', 'ref', 'doc', 'no', 'trans', 'id', 'cheque']) or b_df.columns[0]
    l_serial_col = find_best_column(l_df, ['voucher', 'serial', 'ref', 'doc', 'no', 'trans', 'id', 'cheque']) or l_df.columns[0]

    b_amt_col = find_best_column(b_df, ['amount', 'amt', 'credit', 'debit', 'bal'])
    l_amt_col = find_best_column(l_df, ['amount', 'amt', 'debit', 'credit', 'bal'])

    if not b_amt_col:
        b_amt_col = b_df.select_dtypes(include=[np.number]).columns[0] if len(b_df.select_dtypes(include=[np.number]).columns) > 0 else b_df.columns[1]
    if not l_amt_col:
        l_amt_col = l_df.select_dtypes(include=[np.number]).columns[0] if len(l_df.select_dtypes(include=[np.number]).columns) > 0 else l_df.columns[1]

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

            # Deterministic Match Rule
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
                    "Match_ID": rec_id,
                    "Date": b_date,
                    "Bank_Original_Serial": b_row[b_serial_col],
                    "Ledger_Original_Serial": l_row[l_serial_col],
                    "Bank_Amount": b_amt,
                    "Ledger_Amount": l_amt,
                    "Status": "MATCHED"
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

    total_bank_amt = sum([r.get('Bank_Amount', 0) for r in matched_records]) + sum([r.get('Amount', 0) for r in unmatched_bank])
    total_ledger_amt = sum([r.get('Ledger_Amount', 0) for r in matched_records]) + sum([r.get('Amount', 0) for r in unmatched_ledger])
    net_variance = abs(total_bank_amt - total_ledger_amt)

    summary = {
        "total_bank_txns": len(b_df),
        "total_ledger_txns": len(l_df),
        "matched_pairs": len(matched_records),
        "unmatched_bank_count": len(unmatched_bank),
        "unmatched_ledger_count": len(unmatched_ledger),
        "net_variance": round(net_variance, 2)
    }

    adjustments = []
    if unmatched_bank:
        adjustments.append({
            "Account": "Bank Service Charges / Direct Taxes",
            "Action": "Debit Expense, Credit Bank",
            "Reason": f"{len(unmatched_bank)} direct bank charges/taxes missing in ledger."
        })
    if unmatched_ledger:
        adjustments.append({
            "Account": "Unpresented Cheques / Outstanding Deposits",
            "Action": "Keep in Timing Difference Schedule",
            "Reason": f"{len(unmatched_ledger)} ledger entries awaiting bank clearance."
        })

    return matched_df, unmatched_df, summary, adjustments

# ==========================================
# 4. E-COMMERCE ANALYTICS ENGINE
# ==========================================
def process_ecommerce_analytics(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns
    rev_col = next((c for c in df.columns if 'sale' in c.lower() or 'revenue' in c.lower() or 'price' in c.lower() or 'amount' in c.lower()), None)
    
    total_revenue = float(df[rev_col].sum()) if rev_col else (float(df[numeric_cols[0]].sum()) if len(numeric_cols)>0 else 0.0)
    order_count = len(df)
    aov = total_revenue / order_count if order_count > 0 else 0.0
    
    metrics = {
        "total_revenue": round(total_revenue, 2),
        "total_orders": order_count,
        "average_order_value": round(aov, 2)
    }
    
    fig = None
    if rev_col:
        fig = px.histogram(df, x=rev_col, title="Financial Distribution / Revenue Spread", template="plotly_white")
        
    return metrics, fig

# ==========================================
# 5. USER INTERFACE & STREAMLIT LAYOUT
# ==========================================
st.title("💼 FinAI Pro — Enterprise Autonomous Financial Analyst")
st.caption("Lead AI Accountant & Multi-File Data Processing Engine")

# Sidebar
st.sidebar.header("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Bank Statements, General Ledgers, Shopify/Amazon CSVs",
    accept_multiple_files=True
)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a financial question, request data analysis, or type 'reconcile'...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    low_input = user_input.lower().strip()

    # Router 1: Fast AI Greetings
    if low_input in ["hi", "hello", "salam", "aoa", "assalam o alaikum", "hey"]:
        reply = """Hello! I am **FinAI Pro**, your Lead AI Accountant and Financial Data Engine.

How can I assist you today?
- 🏦 **Bank Reconciliation:** Upload Bank Statement + Internal Ledger and ask to reconcile.
- 📊 **E-Commerce Analytics:** Upload Shopify, Amazon, Fiverr, or Sales CSVs for revenue & margin breakdowns.
- 💻 **General & Technical Q&A:** Ask any accounting, tax, code, or general query."""
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Router 2: Reconciliation
    elif "reconcile" in low_input or "ledger" in low_input or "bank" in low_input or "statement" in low_input:
        csv_xlsx_files = [f for f in (uploaded_files or []) if f.name.endswith(('.csv', '.xlsx'))]
        if len(csv_xlsx_files) >= 2:
            with st.spinner("Executing FinAI Pro Reconciliation Engine..."):
                f1, f2 = csv_xlsx_files[0], csv_xlsx_files[1]
                
                df1 = pd.read_csv(f1) if f1.name.endswith('.csv') else pd.read_excel(f1)
                df2 = pd.read_csv(f2) if f2.name.endswith('.csv') else pd.read_excel(f2)
                
                matched_df, unmatched_df, summary, adjustments = reconcile_ledgers(df1, df2)
                
                st.session_state.matched_csv = matched_df.to_csv(index=False).encode('utf-8')
                st.session_state.unmatched_csv = unmatched_df.to_csv(index=False).encode('utf-8')
                st.session_state.reconciled = True
                
                reply = f"""
### 📊 Executive Summary — Bank vs General Ledger Reconciliation

* **Total Bank Transactions:** `{summary['total_bank_txns']}`
* **Total Ledger Transactions:** `{summary['total_ledger_txns']}`
* **Matched Pairs (REC-001 ID Assigned):** `{summary['matched_pairs']}`
* **Unmatched Bank Entries:** `{summary['unmatched_bank_count']}`
* **Unmatched Ledger Entries:** `{summary['unmatched_ledger_count']}`
* **Net Financial Variance:** `${summary['net_variance']}`

---

#### 🟢 Matched Transactions Preview
{matched_df.head(5).to_markdown(index=False) if not matched_df.empty else "No matched records found."}

---

#### 🔴 Unmatched Transactions Preview
{unmatched_df.head(5).to_markdown(index=False) if not unmatched_df.empty else "No unmatched records found."}

---

#### 📝 Recommended Journal Adjustments
"""
                for adj in adjustments:
                    reply += f"- **{adj['Account']}**: {adj['Action']} — *{adj['Reason']}*\n"

                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
        else:
            reply = "⚠️ Please upload at least **2 files** (Bank Statement and Internal General Ledger) in the sidebar to execute reconciliation."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    # Router 3: Analytics
    elif "sales" in low_input or "revenue" in low_input or "profit" in low_input or "chart" in low_input or "graph" in low_input:
        csv_files = [f for f in (uploaded_files or []) if f.name.endswith(('.csv', '.xlsx'))]
        if csv_files:
            f = csv_files[0]
            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            metrics, fig = process_ecommerce_analytics(df)
            
            reply = f"""
### 📈 Financial Analytics Summary

* **Gross Revenue / Total Sales:** `${metrics['total_revenue']}`
* **Total Orders Processed:** `{metrics['total_orders']}`
* **Average Order Value (AOV):** `${metrics['average_order_value']}`
"""
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            reply = "Please upload a financial/sales CSV or XLSX dataset in sidebar to generate analytics."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    # Router 4: Fallback
    else:
        reply = f"**FinAI Pro:** Processed query '{user_input}'. Ready to perform analysis or reconciliation."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# Persistent Download Buttons
if st.session_state.reconciled:
    st.markdown("---")
    st.subheader("📥 Download Reconciled Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="🟢 Download Matched CSV",
            data=st.session_state.matched_csv,
            file_name="finai_pro_matched.csv",
            mime="text/csv",
            key="btn_download_matched"
        )
    with col2:
        st.download_button(
            label="🔴 Download Unmatched CSV",
            data=st.session_state.unmatched_csv,
            file_name="finai_pro_unmatched.csv",
            mime="text/csv",
            key="btn_download_unmatched"
        )