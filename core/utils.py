import re
from datetime import datetime

import pandas as pd


def clean_amount(value):
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return round(float(value), 2)

    text = str(value)
    text = text.replace("PKR", "")
    text = text.replace("Rs.", "")
    text = text.replace("Rs", "")
    text = text.replace(",", "")
    text = text.strip()

    try:
        return round(float(text), 2)
    except ValueError:
        return 0.0


def clean_date(value):
    if pd.isna(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)

    if pd.isna(parsed):
        return None

    return parsed.date()


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def extract_reference(text):
    text = str(text).upper()

    patterns = [
        r"REF#\s*([A-Z0-9]+)",
        r"STAN\s*\(?([A-Z0-9]+)\)?",
        r"SLIP#:\s*([A-Z0-9]+)",
        r"DOC\s*#\s*([A-Z0-9]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return ""


def get_file_extension(filename):
    return filename.split(".")[-1].lower()