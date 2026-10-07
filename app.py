import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

# Try importing Google Generative AI safely
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==========================================
# 1. PAGE CONFIG & PERSISTENT STATE
# ==========================================
st.set_page_config(
    page_title="DocuSphere AI",
    page_icon="📚",
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
# 2. HELPER & DATA EXTRACTION FUNCTIONS
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
        return "No files currently uploaded."
    
    context = "=== UPLOADED DOCUMENTS CONTEXT ===\n\n"
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
                    context += f"Content Preview:\n{text[:2000]}\n"
                except Exception:
                    context += "Binary file loaded.\n"
        except Exception as e:
            context += f"Could not read content: {str(e)}\n"
        context += "-----------------------------------\n"
    return context

# ==========================================
# 3. RECONCILIATION ENGINE
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
