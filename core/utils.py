import re
from datetime import datetime
import pandas as pd

def clean_amount(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value)
    text = re.sub(r'[^\d.-]', '', text)
    try:
        return round(float(text), 2)
    except ValueError:
        return 0.0

def clean_date(value):
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.date()

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def extract_identifiers(text):
    if pd.isna(text):
        return set()
    matches = re.findall(r'[a-zA-Z0-9]{4,}', str(text))
    return set(matches)

def get_file_extension(file_name):
    if not file_name:
        return ""
    return file_name.split(".")[-1].lower() if "." in file_name else ""
