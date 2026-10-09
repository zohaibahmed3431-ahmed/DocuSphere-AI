import io

import pandas as pd

from core.utils import get_file_extension


def load_file(uploaded_file):
    extension = get_file_extension(uploaded_file.name)

    if extension in ["xlsx", "xls"]:
        return load_excel(uploaded_file)

    if extension == "csv":
        return load_csv(uploaded_file)

    raise ValueError("Sirf Excel ya CSV file upload karein.")


def load_excel(uploaded_file):
    sheets = pd.read_excel(
        io.BytesIO(uploaded_file.getvalue()),
        sheet_name=None,
        header=None,
    )

    return sheets


def load_csv(uploaded_file):
    dataframe = pd.read_csv(
        io.BytesIO(uploaded_file.getvalue()),
        header=None,
        dtype=str,
    )

    return {"CSV": dataframe}