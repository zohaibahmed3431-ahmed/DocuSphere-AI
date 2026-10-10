import re
import pandas as pd


def clean_amount(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value)
    for token in ["PKR", "Rs.", "Rs", "USD", "$", ","]:
        text = text.replace(token, "")
    text = text.strip()
    try:
        return round(float(text), 2)
    except ValueError:
        return 0.0


def clean_date(value):
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def clean_text(value):
    return "" if pd.isna(value) else str(value).strip()


def extract_reference(text):
    text = str(text).upper()
    patterns = [
        r"REF#\s*([A-Z0-9]+)",
        r"STAN\s*\(?([A-Z0-9]+)\)?",
        r"SLIP#:\s*\(?([A-Z0-9]+)\)?",
        r"DOC\s*#\s*([A-Z0-9]+)",
        r"CHEQUE\s*(?:NO\.?|NUMBER)?\s*[:#]?\s*([A-Z0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def get_file_extension(filename):
    return filename.split(".")[-1].lower()


def dataframe_to_context(df, max_rows=40):
    if df is None or df.empty:
        return "No data available."
    sample = df.head(max_rows).copy()
    return sample.to_csv(index=False)