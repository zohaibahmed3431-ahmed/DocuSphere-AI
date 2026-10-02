
from __future__ import annotations
import io
from pathlib import Path
import re
import pandas as pd
from openpyxl import load_workbook

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() if str(c).strip() else f"Unnamed_{i}" for i, c in enumerate(df.columns)]
    return df.reset_index(drop=True)


def _header_score(values) -> int:
    keys = {
        "date", "transaction", "txn", "trans", "posting", "entry", "reference", "ref", "utr", "description", "particular", "particulars",
        "narration", "remarks", "particulars", "debit", "credit", "dr", "cr", "amount", "value", "balance",
        "voucher", "account", "ledger", "payee", "beneficiary", "withdrawal", "deposit", "serial", "sr",
    }
    score = 0
    for v in values:
        n = str(v).strip().lower()
        compact = re.sub(r"[^a-z0-9]+", " ", n)
        tokens = set(compact.split())
        if tokens & keys:
            score += 2
        if any(k in compact for k in ("debit", "credit", "transaction", "narration", "description", "balance", "voucher", "reference")):
            score += 1
    return score


def _read_excel_sheet(data: bytes, sheet: str, engine: str) -> pd.DataFrame:
    # First read normally. If headers are clearly metadata/unnamed, inspect the
    # first few rows and promote the strongest transaction-like row to the header.
    normal = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine=engine)
    normal = _clean_dataframe(normal)
    current_score = _header_score(normal.columns)
    if current_score >= 4 and not all(str(c).lower().startswith("unnamed") for c in normal.columns):
        return normal
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine=engine, header=None, nrows=20)
    if raw.empty:
        return normal
    scores = [(i, _header_score(raw.iloc[i].tolist())) for i in range(min(len(raw), 12))]
    best_i, best_score = max(scores, key=lambda x: (x[1], -x[0]))
    if best_score < 4 or best_i == 0:
        return normal
    full = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine=engine, header=best_i)
    return _clean_dataframe(full)


def read_tabular(uploaded_file):
    """Return {label: dataframe} for CSV/XLSX/XLS.

    Real ledgers often contain a title/reporting-period row above the actual
    header. The reader detects and promotes a strong transaction-header row while
    preserving ordinary tables unchanged.
    """
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    uploaded_file.seek(0)
    data = uploaded_file.read()
    if suffix == ".csv":
        bio = io.BytesIO(data)
        try:
            df = pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, encoding="latin1")
        df = _clean_dataframe(df)
        if _header_score(df.columns) < 4:
            try:
                raw = pd.read_csv(io.BytesIO(data), header=None, nrows=20)
                scores = [(i, _header_score(raw.iloc[i].tolist())) for i in range(min(len(raw), 12))]
                best_i, best_score = max(scores, key=lambda x: (x[1], -x[0]))
                if best_score >= 4 and best_i > 0:
                    df = _clean_dataframe(pd.read_csv(io.BytesIO(data), header=best_i))
            except Exception:
                pass
        return {name: df}
    if suffix in (".xlsx", ".xls"):
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        xls = pd.ExcelFile(io.BytesIO(data), engine=engine)
        return {f"{name}::{sheet}": _read_excel_sheet(data, sheet, engine) for sheet in xls.sheet_names}
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
