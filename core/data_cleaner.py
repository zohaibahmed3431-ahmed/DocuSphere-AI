import pandas as pd
import re
from core.utils import clean_amount, clean_date, clean_text, extract_identifiers

def find_best_column(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw in str(col).lower():
                return col
    return None

def clean_sheet(df):
    df = df.dropna(how="all").reset_index(drop=True)
    header_row = 0
    for i in range(min(20, len(df))):
        row_text = " ".join(str(x).lower() for x in df.iloc[i].values)
        if "date" in row_text and ("debit" in row_text or "credit" in row_text or "amount" in row_text):
            header_row = i
            break
    df = df.iloc[header_row:]
    df.columns = [str(c).strip() for c in df.iloc[0]]
    df = df.iloc[1:].reset_index(drop=True)
    return df

def standardize(df, source_type, sheet_name="Sheet1"):
    date_col = find_best_column(df, ["date", "time", "day"])
    desc_col = find_best_column(df, ["description", "narration", "particulars", "details"])
    debit_col = find_best_column(df, ["debit", "withdrawal", "money out"])
    credit_col = find_best_column(df, ["credit", "deposit", "money in"])
    serial_col = find_best_column(df, ["serial", "voucher", "ref", "doc", "stan", "id", "cheque"]) or df.columns[0]

    if not date_col or not desc_col:
        raise ValueError(f"{source_type} mein Date ya Description column detect nahi hua.")

    rows = []
    for i, row in df.iterrows():
        dt = clean_date(row[date_col])
        desc = clean_text(row[desc_col])
        debit = clean_amount(row[debit_col]) if debit_col else 0.0
        credit = clean_amount(row[credit_col]) if credit_col else 0.0
        serial_val = clean_text(row[serial_col]) if serial_col else f"ROW-{i+2}"

        if not dt or (debit == 0 and credit == 0):
            continue
        if "opening balance" in desc.lower():
            continue

        if source_type == "Bank Statement":
            direction = "Outflow" if debit > 0 else "Inflow"
            amount = debit if debit > 0 else credit
        else:
            direction = "Inflow" if debit > 0 else "Outflow"
            amount = debit if debit > 0 else credit

        rows.append({
            "Source": source_type,
            "Sheet": sheet_name,
            "Row": i + 2,
            "Original_Serial": serial_val,
            "Date": dt,
            "Description": desc,
            "Direction": direction,
            "Amount": amount,
            "Identifiers": extract_identifiers(f"{serial_val} {desc}"),
            "Status": "Unmatched",
            "Match ID": "",
            "Match Rule": ""
        })
    return pd.DataFrame(rows)
