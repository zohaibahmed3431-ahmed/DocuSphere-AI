from __future__ import annotations

import io
from pathlib import Path
import pandas as pd


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    seen = {}
    for i, c in enumerate(df.columns):
        base = str(c).strip() if c is not None else f"Column_{i+1}"
        if not base or base.lower().startswith("unnamed:"):
            base = f"Column_{i+1}"
        n = seen.get(base.lower(), 0)
        seen[base.lower()] = n + 1
        cols.append(base if n == 0 else f"{base}_{n+1}")
    df.columns = cols
    return df


def _header_score(row) -> float:
    vals = [str(v).strip() for v in row.tolist() if pd.notna(v) and str(v).strip()]
    if not vals: return -1
    alpha = sum(any(ch.isalpha() for ch in v) for v in vals)
    unique = len({v.lower() for v in vals})
    return alpha * 2 + unique * .25 - sum(len(v) > 80 for v in vals) * 2


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last = None
    for enc in (None, "utf-8-sig", "latin1"):
        try:
            bio = io.BytesIO(data)
            kwargs = {} if enc is None else {"encoding": enc}
            probe = pd.read_csv(bio, header=None, nrows=25, **kwargs)
            best_row = max(range(len(probe)), key=lambda i: _header_score(probe.iloc[i])) if len(probe) else 0
            bio.seek(0)
            df = pd.read_csv(bio, header=best_row, **kwargs)
            return _normalise_columns(df.dropna(how="all"))
        except Exception as exc:
            last = exc
    raise ValueError(f"Could not read CSV: {last}")


def _read_excel_sheet(data: bytes, sheet, engine: str) -> pd.DataFrame:
    probe = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, nrows=35, engine=engine)
    if probe.empty:
        return pd.DataFrame()
    scores = [_header_score(probe.iloc[i]) for i in range(len(probe))]
    best_row = max(range(len(scores)), key=lambda i: scores[i])
    # Prefer rows containing domain-like header words when available.
    domain_words = ("date", "amount", "debit", "credit", "description", "narration", "reference", "voucher", "serial", "account")
    for i in range(min(len(probe), 20)):
        text = " ".join(str(v).lower() for v in probe.iloc[i].tolist() if pd.notna(v))
        if sum(w in text for w in domain_words) >= 2:
            best_row = i
            break
    df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=best_row, engine=engine)
    return _normalise_columns(df.dropna(how="all"))


def read_tabular(uploaded_file):
    """Return {label: dataframe} for common tabular uploads.

    Header detection tolerates report titles, blank rows, merged-style preambles,
    and generic sheet names. Only genuinely tabular sheets are returned.
    """
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    uploaded_file.seek(0)
    data = uploaded_file.read()
    if suffix == ".csv":
        df = _read_csv_bytes(data)
        return {name: df} if len(df.columns) >= 1 and not df.empty else {}
    if suffix in (".xlsx", ".xls"):
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        xls = pd.ExcelFile(io.BytesIO(data), engine=engine)
        out = {}
        for sheet in xls.sheet_names:
            try:
                df = _read_excel_sheet(data, sheet, engine)
                if not df.empty and len(df.columns) >= 1:
                    out[f"{name}::{sheet}"] = df
            except Exception:
                continue
        return out
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
