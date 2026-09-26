from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CSV ANALYTICS
# ============================================================

DATE_HINTS = (
    "date", "day", "time", "timestamp", "created", "updated",
    "joined", "join", "dob", "birth", "month", "year"
)

VALUE_HINTS = (
    "sales", "sale", "revenue", "amount", "total", "price",
    "cost", "profit", "income", "value", "salary", "payment",
    "expense", "quantity", "qty", "units"
)

ID_HINTS = (
    "id", "code", "number", "no", "zip", "postal", "phone",
    "employee", "department"
)

GRAPH_WORDS = (
    "graph", "chart", "plot", "visualize", "visualise",
    "trend", "distribution", "visualization", "visualisation",
    "show me", "display"
)

TIME_WORDS = (
    "today", "yesterday", "week", "month", "year",
    "daily", "weekly", "monthly", "last", "this"
)


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def _candidate_date_columns(df: pd.DataFrame) -> list[tuple[str, float]]:
    candidates = []

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append((col, 1.0))
            continue

        name = _clean_name(col)
        name_score = 0.8 if any(h in name.split() for h in DATE_HINTS) else 0.0

        if not (s.dtype == "object" or pd.api.types.is_string_dtype(s)):
            if name_score:
                parsed = pd.to_datetime(s, errors="coerce")
            else:
                continue
        else:
            parsed = pd.to_datetime(s, errors="coerce")

        valid_ratio = float(parsed.notna().mean()) if len(s) else 0.0

        # Only accept a text column as a date when most values parse.
        if valid_ratio >= 0.70:
            score = max(valid_ratio, name_score)
            candidates.append((col, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _candidate_value_columns(df: pd.DataFrame) -> list[str]:
    numeric = _numeric_columns(df)
    scored = []

    for col in numeric:
        name = _clean_name(col)
        score = 0

        if any(h in name.split() for h in VALUE_HINTS):
            score += 5

        if any(h in name.split() for h in ID_HINTS):
            score -= 4

        # Avoid obvious identifier columns when a real metric exists.
        if df[col].nunique(dropna=True) <= 1:
            score -= 5

        scored.append((score, col))

    scored.sort(reverse=True)
    return [c for _, c in scored]


def _candidate_category_columns(df: pd.DataFrame) -> list[str]:
    candidates = []

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            # Numeric low-cardinality fields such as department IDs
            # are useful for categorical charts.
            unique = s.nunique(dropna=True)
            if 2 <= unique <= min(20, max(2, len(df) // 2)):
                candidates.append(col)
            continue

        unique = s.nunique(dropna=True)

        if 2 <= unique <= min(15, max(2, len(df) // 2)):
            candidates.append(col)

    # Prefer human-readable categorical names.
    candidates.sort(
        key=lambda c: (
            0 if any(x in _clean_name(c).split()
                     for x in ("name", "gender", "status", "category", "type", "department"))
            else 1,
            df[c].nunique(dropna=True),
        )
    )

    return candidates


def _parse_date_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _period_from_question(
    question: str,
    date_series: pd.Series,
) -> Optional[tuple[pd.Timestamp, pd.Timestamp, str]]:
    q = question.lower()
    clean = date_series.dropna()

    if clean.empty:
        return None

    latest = clean.max().normalize()

    # Previous calendar week: Monday-Sunday.
    if "last week" in q or "previous week" in q:
        start_this = latest - pd.Timedelta(days=latest.weekday())
        end_this = start_this + pd.Timedelta(days=6)
        start = start_this - pd.Timedelta(days=7)
        end = end_this - pd.Timedelta(days=7)
        return start, end, "last week"

    if "this week" in q:
        start = latest - pd.Timedelta(days=latest.weekday())
        end = start + pd.Timedelta(days=6)
        return start, end, "this week"

    if "last 7 days" in q or "past 7 days" in q:
        return latest - pd.Timedelta(days=6), latest, "last 7 days"

    if "last month" in q or "previous month" in q:
        first_this = latest.replace(day=1)
        end = first_this - pd.Timedelta(days=1)
        start = end.replace(day=1)
        return start, end, "last month"

    if "this month" in q:
        start = latest.replace(day=1)
        return start, latest, "this month"

    if "last year" in q or "previous year" in q:
        start = pd.Timestamp(year=latest.year - 1, month=1, day=1)
        end = pd.Timestamp(year=latest.year - 1, month=12, day=31)
        return start, end, "last year"

    if "this year" in q:
        start = pd.Timestamp(year=latest.year, month=1, day=1)
        return start, latest, "this year"

    return None


def _find_columns_for_question(
    df: pd.DataFrame,
    question: str,
) -> tuple[Optional[str], Optional[str]]:
    q = _clean_name(question)

    date_candidates = _candidate_date_columns(df)
    value_candidates = _candidate_value_columns(df)

    date_col = None
    value_col = None

    for col, _ in date_candidates:
        if _clean_name(col) in q:
            date_col = col
            break

    if date_col is None and date_candidates:
        date_col = date_candidates[0][0]

    for col in value_candidates:
        tokens = _clean_name(col).split()
        if any(token in q for token in tokens):
            value_col = col
            break

    if value_col is None and value_candidates:
        value_col = value_candidates[0]

    return date_col, value_col


def _format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.date()} → {end.date()}"


def _build_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col])

    if temp.empty:
        return pd.DataFrame(columns=["Date", value_col])

    daily = (
        temp.assign(Date=temp[date_col].dt.normalize())
        .groupby("Date", as_index=False)[value_col]
        .sum()
        .sort_values("Date")
    )

    return daily


def _is_graph_request(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in GRAPH_WORDS)


def _is_time_question(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in TIME_WORDS)


def build_professional_charts(
    df: pd.DataFrame,
    question: str = "",
) -> list[dict]:
    """
    Build chart specifications from the complete CSV.

    The function does not invent values. Every chart value comes directly
    from the dataframe supplied by the app.
    """
    if df is None or df.empty:
        return []

    charts: list[dict] = []
    q = question.lower()

    date_candidates = _candidate_date_columns(df)
    value_candidates = _candidate_value_columns(df)
    category_candidates = _candidate_category_columns(df)

    # --------------------------------------------------------
    # 1. Time-series request / dataset with a clear date metric
    # --------------------------------------------------------
    if date_candidates and value_candidates and (
        _is_time_question(question)
        or any(x in q for x in ("sales", "revenue", "profit", "income", "amount"))
    ):
        date_col = date_candidates[0][0]
        value_col = value_candidates[0]

        daily = _build_daily(df, date_col, value_col)

        if len(daily) >= 2:
            charts.append({
                "kind": "line",
                "title": f"{value_col} over time",
                "description": f"Daily {value_col.lower()} from the uploaded CSV.",
                "x": "Date",
                "y": value_col,
                "data": daily,
            })

    # --------------------------------------------------------
    # 2. Explicit graph request: best categorical distribution
    # --------------------------------------------------------
    if _is_graph_request(question) and category_candidates:
        for col in category_candidates[:2]:
            counts = (
                df[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(12)
                .rename_axis(col)
                .reset_index(name="Count")
            )

            charts.append({
                "kind": "bar",
                "title": f"{col} distribution",
                "description": f"Records grouped by {col}.",
                "x": col,
                "y": "Count",
                "data": counts,
            })

    # --------------------------------------------------------
    # 3. If explicit graph request but no category, use numeric
    # --------------------------------------------------------
    if _is_graph_request(question) and not charts and value_candidates:
        for col in value_candidates[:2]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()

            if series.empty:
                continue

            # Histogram-like distribution using 10 stable bins.
            bins = min(10, max(2, series.nunique()))
            counts, edges = np.histogram(series, bins=bins)

            rows = []
            for i in range(len(counts)):
                rows.append({
                    "Range": f"{edges[i]:g} – {edges[i + 1]:g}",
                    "Count": int(counts[i]),
                })

            charts.append({
                "kind": "bar",
                "title": f"{col} distribution",
                "description": f"Distribution of values in {col}.",
                "x": "Range",
                "y": "Count",
                "data": pd.DataFrame(rows),
            })

    # Remove duplicate chart specifications.
    unique = []
    seen = set()

    for chart in charts:
        key = (
            chart["kind"],
            chart["title"],
            chart["x"],
            chart["y"],
        )

        if key not in seen:
            seen.add(key)
            unique.append(chart)

    return unique[:2]


def analyze_csv(
    frame: pd.DataFrame,
    question: str,
) -> Optional[dict]:
    """
    Exact CSV analytics for natural-language questions.

    Returns None when the question is not a deterministic CSV calculation
    or an explicit visualization request.
    """
    if frame is None or frame.empty:
        return None

    df = frame.copy()

    # Explicit visualization request.
    if _is_graph_request(question):
        charts = build_professional_charts(df, question)

        if charts:
            return {
                "mode": "visualization",
                "file_rows": len(df),
                "charts": charts,
                "period": "full dataset",
            }

    date_candidates = _candidate_date_columns(df)

    if not date_candidates:
        return None

    date_col, _ = _find_columns_for_question(df, question)

    if not date_col:
        return None

    date_series = _parse_date_series(df, date_col)

    period = _period_from_question(
        question,
        date_series,
    )

    # Without a recognized period, do not guess an aggregation window.
    if period is None:
        return None

    start, end, period_name = period

    value_col_candidates = _candidate_value_columns(df)

    if not value_col_candidates:
        return None

    _, value_col = _find_columns_for_question(
        df,
        question,
    )

    if not value_col:
        return None

    work = df.copy()
    work["_parsed_date"] = date_series
    work["_numeric_value"] = pd.to_numeric(
        work[value_col],
        errors="coerce",
    )

    matching = work[
        (work["_parsed_date"] >= start)
        & (work["_parsed_date"] <= end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    ].copy()

    matching = matching.dropna(
        subset=["_parsed_date", "_numeric_value"]
    )

    total = float(
        matching["_numeric_value"].sum()
    )

    daily = _build_daily(
        matching,
        "_parsed_date",
        "_numeric_value",
    )

    if not daily.empty:
        daily = daily.rename(
            columns={"_numeric_value": value_col}
        )

    # Keep original CSV columns in matching records.
    records = matching[
        [c for c in df.columns if c in matching.columns]
    ].copy()

    return {
        "mode": "calculation",
        "date_column": date_col,
        "value_column": value_col,
        "period": period_name,
        "start": start,
        "end": end,
        "total": total,
        "record_count": len(records),
        "records": records,
        "daily": daily,
        "charts": [{
            "kind": "line",
            "title": f"{value_col} — {period_name}",
            "description": _format_period_label(start, end),
            "x": "Date",
            "y": value_col,
            "data": daily,
        }] if not daily.empty else [],
    }


def format_total(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return f"{int(number):,}"

    return f"{number:,.2f}"
