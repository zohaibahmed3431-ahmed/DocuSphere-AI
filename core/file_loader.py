import io
import pandas as pd
from core.utils import get_file_extension

def load_excel_or_csv(uploaded_file):
    name = uploaded_file.name.lower()
    ext = get_file_extension(name)
    
    if ext in ["xlsx", "xls"]:
        sheets = pd.read_excel(
            io.BytesIO(uploaded_file.getvalue()),
            sheet_name=None,
            header=None,
        )
        return sheets
    elif ext == "csv":
        df = pd.read_csv(
            io.BytesIO(uploaded_file.getvalue()),
            header=None,
            dtype=str,
        )
        return {"CSV": df}
    else:
        raise ValueError("Sirf Excel (.xlsx, .xls) ya CSV file upload karein.")
