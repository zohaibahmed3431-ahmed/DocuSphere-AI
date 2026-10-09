import io

import pandas as pd

from core.utils import get_file_extension, is_supported_file


def load_file(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    filename = uploaded_file.name

    if not is_supported_file(filename):
        raise ValueError(
            f"Unsupported file type: {filename}. "
            "Please upload Excel or CSV file."
        )

    extension = get_file_extension(filename)

    if extension in ["xlsx", "xls"]:
        return load_excel_file(uploaded_file)

    if extension == "csv":
        return load_csv_file(uploaded_file)

    raise ValueError("Unsupported file format.")


def load_excel_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))

    sheets = {}

    for sheet_name in excel_file.sheet_names:
        dataframe = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            header=None,
        )

        sheets[sheet_name] = dataframe

    return sheets


def load_csv_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    dataframe = pd.read_csv(
        io.BytesIO(file_bytes),
        header=None,
        dtype=str,
    )

    return {"CSV": dataframe}


def get_sheet_names(sheets):
    return list(sheets.keys())