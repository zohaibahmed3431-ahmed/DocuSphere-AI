import re
from datetime import datetime

import pandas as pd


def clean_amount(value):
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return round(float(value), 2)

    text = str(value).strip()
    text = text.replace("PKR", "")
    text = text.replace("Rs.", "")
    text = text.replace("Rs", "")
    text = text.replace(",", "")
    text = text.replace(" ", "")

    try:
        return round(float(text), 2)
    except ValueError:
        return 0.0


def clean_date(value):
    if pd.isna(value):
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, pd.Timestamp):
        return value.date()

    text = str(value).strip()

    parsed_date = pd.to_datetime(text, errors="coerce", dayfirst=True)

    if pd.isna(parsed_date):
        return None

    return parsed_date.date()


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def extract_reference(text):
    if pd.isna(text):
        return ""

    text = str(text).upper()

    patterns = [
        r"REF#\s*([A-Z0-9]+)",
        r"REF\s*([A-Z0-9]+)",
        r"STAN\s*\(?([A-Z0-9]+)\)?",
        r"SLIP#:\s*([A-Z0-9]+)",
        r"SLIP#\s*([A-Z0-9]+)",
        r"DOC\s*#\s*([A-Z0-9]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return ""


def get_file_extension(filename):
    return filename.split(".")[-1].lower()


def is_supported_file(filename):
    from config.settings import SUPPORTED_FILE_TYPES

    return get_file_extension(filename) in SUPPORTED_FILE_TYPES