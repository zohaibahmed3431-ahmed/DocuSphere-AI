from __future__ import annotations
import re
from datetime import datetime
from typing import Optional
import pandas as pd

DATE_HINTS = ("date", "day", "time", "timestamp", "created", "order", "sale", "sold")
VALUE_HINTS = ("sales", "sale", "revenue", "amount", "total", "value", "price", "income", "profit", "net")
COUNT_HINTS = ("qty", "quantity", "units", "count", "orders", "transactions")

def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()

def _score_column(column: object, hints: tuple[str, ...]) -> int:
    name = _norm(column)
    return sum(5 if name == h else 2 if h in name else 0 for h in hints)

def detect_columns(df: pd.DataFrame) -> dict:
    date_col = max(df.columns, key=lambda c: _score_column(c, DATE_HINTS), default=None)
    value_col = max(df.columns, key=lambda c: _score_column(c, VALUE_HINTS), default=None)
    count_col = max(df.columns, key=lambda c: _score_column(c, COUNT_HINTS), default=None)
    if date_col is not None and _score_column(date_col, DATE_HINTS) == 0:
        candidates = []
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().mean() >= 0.60:
                candidates.append((parsed.notna().mean(), col))
        if candidates:
            date_col = max(candidates)[1]
    if value_col is not None and _score_column(value_col, VALUE_HINTS) == 0:
        numeric = []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= 0.60:
                numeric.append((converted.notna().mean(), col))
        if numeric:
            value_col = max(numeric, key=lambda x: (_score_column(x[1], VALUE_HINTS), x[0]))[1]
    return {"date": date_col, "value": value_col, "count": count_col}

def _previous_calendar_week(reference: pd.Timestamp):
    reference = pd.Timestamp(reference).normalize()
    monday = reference - pd.Timedelta(days=reference.weekday())
    return monday - pd.Timedelta(days=7), monday - pd.Timedelta(days=1)

def _period(question: str, dates: pd.Series):
    q = question.lower()
    latest = dates.max()
    if pd.isna(latest):
        return None, None, ""
    if re.search(r"\blast\s+(calendar\s+)?week\b|previous\s+week", q):
        start, end = _previous_calendar_week(latest)
        return start, end, "previous calendar week"
    if re.search(r"\bthis\s+week\b", q):
        start = latest.normalize() - pd.Timedelta(days=latest.weekday())
        return start, latest.normalize(), "this dataset week"
    if re.search(r"\blast\s+7\s+days?\b|past\s+7\s+days?\b", q):
        end = latest.normalize()
        return end - pd.Timedelta(days=6), end, "last 7 days"
    if re.search(r"\blast\s+30\s+days?\b|past\s+30\s+days?\b", q):
        end = latest.normalize()
        return end - pd.Timedelta(days=29), end, "last 30 days"
    if re.search(r"\bthis\s+month\b", q):
        return latest.replace(day=1).normalize(), latest.normalize(), "this month"
    if re.search(r"\blast\s+month\b|previous\s+month\b", q):
        first = latest.replace(day=1).normalize()
        end = first - pd.Timedelta(days=1)
        return end.replace(day=1).normalize(), end, "previous month"
    return None, None, ""

def analyze_csv(df: pd.DataFrame, question: str) -> Optional[dict]:
    if df is None or df.empty:
        return None
    cols = detect_columns(df)
    date_col = cols["date"]
    if date_col is None:
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if dates.notna().sum() == 0:
        return None
    start, end, period = _period(question, dates)
    if start is None:
        return None
    work = df.copy()
    work["__date"] = dates
    work = work[work["__date"].notna()]
    mask = (work["__date"].dt.normalize() >= start) & (work["__date"].dt.normalize() <= end)
    filtered = work.loc[mask].copy()
    value_col = cols["value"]
    count_col = cols["count"]
    if value_col is not None:
        metric = pd.to_numeric(filtered[value_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        metric_name = str(value_col)
    elif count_col is not None:
        metric = pd.to_numeric(filtered[count_col], errors="coerce")
        metric_name = str(count_col)
    else:
        metric = pd.Series(1.0, index=filtered.index)
        metric_name = "records"
    filtered["__metric"] = metric
    daily = (filtered.assign(Date=filtered["__date"].dt.normalize())
             .groupby("Date", as_index=False)["__metric"].sum()
             .rename(columns={"__metric": metric_name})
             .sort_values("Date"))
    records = filtered.drop(columns=["__date", "__metric"], errors="ignore").copy()
    records.insert(0, "Date", filtered["__date"].dt.strftime("%Y-%m-%d").values)
    return {"period": period, "start": start, "end": end, "date_column": str(date_col),
            "value_column": metric_name, "total": float(metric.sum(skipna=True)),
            "record_count": len(filtered), "records": records, "daily": daily}

def format_total(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"
