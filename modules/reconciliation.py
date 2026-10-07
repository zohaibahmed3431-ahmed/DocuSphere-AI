import pandas as pd
import re
from core.document_parser import clean_column_names

def find_best_column(df: pd.DataFrame, keywords: list) -> str:
    for kw in keywords:
        for col in df.columns:
            if kw in col:
                return col
    return None

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

        found_match = False
        for l_idx, l_row in l_df.iterrows():
            if l_idx in ledger_matched_indices:
                continue

            try:
                l_amt = float(re.sub(r'[^0-9.-]', '', str(l_row[l_amt_col]))) if pd.notna(l_row[l_amt_col]) else 0.0
            except Exception:
                l_amt = 0.0

            if abs(abs(b_amt) - abs(l_amt)) <= 0.01 and b_amt != 0:
                rec_id = f"REC-{rec_counter:06d}"
                rec_counter += 1
                ledger_matched_indices.add(l_idx)

                matched_records.append({
                    "Reconciliation_ID": rec_id,
                    "Date": b_row[b_date_col] if b_date_col and pd.notna(b_row[b_date_col]) else "N/A",
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
                "Source": "Bank Statement Only",
                "Original_Serial": b_row[b_serial_col],
                "Date": b_row[b_date_col] if b_date_col and pd.notna(b_row[b_date_col]) else "N/A",
                "Amount": b_amt,
                "Details": "Unmatched in Software Ledger"
            })

    unmatched_ledger = []
    for l_idx, l_row in l_df.iterrows():
        if l_idx not in ledger_matched_indices:
            try:
                l_amt = float(re.sub(r'[^0-9.-]', '', str(l_row[l_amt_col]))) if pd.notna(l_row[l_amt_col]) else 0.0
            except Exception:
                l_amt = 0.0

            unmatched_ledger.append({
                "Source": "Software Ledger Only",
                "Original_Serial": l_row[l_serial_col],
                "Date": l_row[l_date_col] if l_date_col and pd.notna(l_row[l_date_col]) else "N/A",
                "Amount": l_amt,
                "Details": "Unmatched in Bank Statement"
            })

    matched_df = pd.DataFrame(matched_records)
    unmatched_df = pd.concat([pd.DataFrame(unmatched_bank), pd.DataFrame(unmatched_ledger)], ignore_index=True)

    return matched_df, unmatched_df