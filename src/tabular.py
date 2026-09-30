
from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

def read_tabular(uploaded_file):
    """Return {label: dataframe} for CSV/XLSX, or {} for other formats."""
    name=uploaded_file.name
    suffix=Path(name).suffix.lower()
    uploaded_file.seek(0)
    data=uploaded_file.read()
    if suffix==".csv":
        bio=io.BytesIO(data)
        try: df=pd.read_csv(bio)
        except Exception:
            bio.seek(0); df=pd.read_csv(bio,encoding="latin1")
        df.columns=[str(c).strip() for c in df.columns]
        return {name: df}
    if suffix==".xlsx":
        xls=pd.ExcelFile(io.BytesIO(data))
        return {f"{name}::{sheet}": pd.read_excel(io.BytesIO(data), sheet_name=sheet) for sheet in xls.sheet_names}
    return {}

def dataframe_context(frames: dict, max_rows=25, max_cols=30) -> str:
    parts=[]
    for label, df in frames.items():
        sample=df.head(max_rows).iloc[:, :max_cols]
        parts.append(
            f"TABLE {label}\nCOLUMNS: {list(df.columns)}\nROWS: {len(df)}\n"
            f"DATA SAMPLE:\n{sample.to_csv(index=False)}"
        )
    return "\n\n".join(parts)
