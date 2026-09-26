from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

DATE_HINTS = {
    "date", "day", "time", "timestamp", "created", "updated", "joined",
    "join", "dob", "birth", "month", "year", "entry", "order", "sale",
}
VALUE_HINTS = {
    "sales", "sale", "revenue", "amount", "total", "price", "cost", "profit",
    "income", "value", "salary", "payment", "expense", "quantity", "qty",
    "units", "net", "commission", "avance", "advance", "balance", "budget",
}
ID_HINTS = {
    "id", "code", "number", "no", "zip", "postal", "phone", "employee",
    "department", "branch", "user", "cnic", "cid", "emp", "desig",
}
GRAPH_WORDS = {
    "graph", "chart", "plot", "visualize", "visualise", "visualization",
    "visualisation", "distribution", "trend", "dashboard", "bar chart",
    "line chart", "pie chart", "show graph", "show chart", "graph this",
}
TIME_WORDS = {
    "today", "yesterday", "week", "month", "year", "daily", "weekly",
    "monthly", "last", "this", "previous", "past",
}


def _clean_name(value: str) -> str:
    text = str(value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(value: str) -> set[str]:
    return set(_clean_name(value).split())


def _numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[,₹$€£%]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        parsed = _numeric_series(df[col])
        if parsed.notna().mean() >= 0.80:
            cols.append(col)
    return cols


def _candidate_date_columns(df: pd.DataFrame) -> list[tuple[str, float]]:
    candidates = []
    for col in df.columns:
        s = df[col]
        name_tokens = _tokens(col)
        hint_score = 1.0 if name_tokens & DATE_HINTS else 0.0
        if "entry" in name_tokens or "updated" in name_tokens or "created" in name_tokens:
            hint_score = 1.25
        elif "join" in name_tokens or "joined" in name_tokens or "order" in name_tokens:
            hint_score = 1.15
        elif "dob" in name_tokens or "birth" in name_tokens:
            hint_score = 0.65

        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append((col, 1.0))
            continue

        parsed = pd.to_datetime(s, format="mixed", errors="coerce")
        ratio = float(parsed.notna().mean()) if len(s) else 0.0

        # Numeric columns are only dates when the column name strongly says so.
        if pd.api.types.is_numeric_dtype(s) and not hint_score:
            continue

        if ratio >= 0.70 and (hint_score or not pd.api.types.is_numeric_dtype(s)):
            candidates.append((col, max(ratio, hint_score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _candidate_value_columns(df: pd.DataFrame) -> list[str]:
    scored = []
    for col in _numeric_columns(df):
        tokens = _tokens(col)
        score = 0
        score += 6 * len(tokens & VALUE_HINTS)
        score -= 4 * len(tokens & ID_HINTS)
        unique = df[col].nunique(dropna=True)
        if unique <= 1:
            score -= 8
        # Prefer real metrics over identifier-like columns.
        scored.append((score, unique, col))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [col for _, _, col in scored]


def _candidate_category_columns(df: pd.DataFrame) -> list[str]:
    result = []
    preferred = {"name", "gender", "gander", "status", "category", "type", "department", "branch", "religion", "workingstatus"}

    for col in df.columns:
        s = df[col]
        unique = s.nunique(dropna=True)
        if unique < 2:
            continue
        if unique > min(20, max(20, len(df))):
            continue
        # Exclude obvious free-text fields.
        avg_len = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
        if avg_len > 60 and unique > 10:
            continue
        result.append(col)

    result.sort(key=lambda c: (
        0 if _tokens(c) & preferred else 1,
        df[c].nunique(dropna=True),
    ))
    return result


def _find_named_column(df: pd.DataFrame, question: str, candidates: list[str]) -> Optional[str]:
    q_tokens = _tokens(question)
    q_clean = _clean_name(question)

    aliases = {
        "gender": {"gender", "gander", "sex"},
        "department": {"department", "depart", "dept"},
        "salary": {"salary", "basic salary", "net salary", "avance salary", "advance salary"},
        "sales": {"sales", "sale"},
        "revenue": {"revenue"},
        "profit": {"profit"},
        "quantity": {"quantity", "qty", "units"},
    }

    # Exact column-name match is strongest.
    exact = []
    for col in candidates:
        ct = _tokens(col)
        if ct and ct.issubset(q_tokens):
            exact.append((len(ct), col))
    if exact:
        exact.sort(reverse=True)
        return exact[0][1]

    # Score semantic matches instead of returning the first numeric column.
    scored = []
    for col in candidates:
        ct = _tokens(col)
        score = 0
        overlap = len(ct & q_tokens)
        score += overlap * 10

        for alias_name, alias_terms in aliases.items():
            if any(_clean_name(a) in q_clean for a in alias_terms):
                if alias_name in ct:
                    score += 30
                if alias_name == "salary" and "salary" in ct:
                    # For a generic salary request prefer NetSalary over
                    # BasicSalary/AvanceSalary as the employee's final pay.
                    if "net" in ct:
                        score += 12
                    elif "basic" in ct:
                        score += 8
                    elif "avance" in ct or "advance" in ct:
                        score -= 4

        if score:
            scored.append((score, col))

    if scored:
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][1]

    return None


def _period_from_question(
    question: str,
    date_series: pd.Series,
) -> Optional[tuple[pd.Timestamp, pd.Timestamp, str]]:
    clean = pd.to_datetime(date_series, errors="coerce").dropna()
    if clean.empty:
        return None

    # Use the latest date present in the uploaded dataset as the anchor. This
    # makes historical CSVs answer relative-period questions instead of returning
    # an empty result just because the file is older than today.
    anchor = clean.max().normalize()
    q = question.lower()

    if "last week" in q or "previous week" in q:
        this_monday = anchor - pd.Timedelta(days=anchor.weekday())
        start = this_monday - pd.Timedelta(days=7)
        end = this_monday - pd.Timedelta(days=1)
        return start, end, "last week"

    if "this week" in q:
        start = anchor - pd.Timedelta(days=anchor.weekday())
        return start, anchor, "this week"

    if "last 7 days" in q or "past 7 days" in q:
        return anchor - pd.Timedelta(days=6), anchor, "last 7 days"

    if "last month" in q or "previous month" in q:
        first_this = anchor.replace(day=1)
        end = first_this - pd.Timedelta(days=1)
        start = end.replace(day=1)
        return start, end, "last month"

    if "this month" in q:
        return anchor.replace(day=1), anchor, "this month"

    if "last year" in q or "previous year" in q:
        return (
            pd.Timestamp(anchor.year - 1, 1, 1),
            pd.Timestamp(anchor.year - 1, 12, 31),
            "last year",
        )

    if "this year" in q:
        return pd.Timestamp(anchor.year, 1, 1), anchor, "this year"

    return None


def _is_graph_request(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in GRAPH_WORDS)


def _is_time_question(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in TIME_WORDS)


def _build_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    temp = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "Value": _numeric_series(df[value_col]),
    }).dropna()
    if temp.empty:
        return pd.DataFrame(columns=["Date", value_col])
    daily = temp.assign(Date=temp["Date"].dt.normalize()).groupby("Date", as_index=False)["Value"].sum()
    daily = daily.rename(columns={"Value": value_col})
    return daily.sort_values("Date")


def _category_chart(df: pd.DataFrame, col: str) -> dict:
    counts = (
        df[col].fillna("Missing").astype(str).value_counts()
        .rename_axis(col).reset_index(name="Count")
    )
    return {
        "kind": "bar",
        "title": f"{col} distribution",
        "description": f"Record count by {col} across the complete dataset.",
        "x": col,
        "y": "Count",
        "data": counts,
    }


def build_professional_charts(df: pd.DataFrame, question: str = "") -> list[dict]:
    if df is None or df.empty:
        return []

    q = _clean_name(question)
    date_candidates = [c for c, _ in _candidate_date_columns(df)]
    value_candidates = _candidate_value_columns(df)
    category_candidates = _candidate_category_columns(df)
    charts: list[dict] = []

    # If the user named a metric, honor it.
    requested_value = _find_named_column(df, question, value_candidates)
    requested_category = _find_named_column(df, question, category_candidates)
    requested_date = _find_named_column(df, question, date_candidates)

    wants_time = _is_time_question(question) or any(k in q for k in ("sales", "revenue", "profit", "income", "amount", "salary", "net salary", "basic salary"))

    if date_candidates and value_candidates and wants_time:
        date_col = requested_date or date_candidates[0]
        value_col = requested_value or value_candidates[0]
        daily = _build_daily(df, date_col, value_col)
        if len(daily) >= 2:
            charts.append({
                "kind": "line",
                "title": f"{value_col} over time",
                "description": f"Daily {value_col} trend from the complete uploaded CSV.",
                "x": "Date",
                "y": value_col,
                "data": daily,
            })

    if _is_graph_request(question):
        # A generic graph request should show a useful categorical overview,
        # not an arbitrary ID-versus-date line.
        metric_words = {"sales", "sale", "revenue", "profit", "income", "amount", "salary", "price", "cost", "quantity", "qty", "units", "trend"}
        has_metric_request = bool(_tokens(question) & metric_words)

        if requested_category and not has_metric_request:
            charts = [_category_chart(df, requested_category)]
        elif has_metric_request and charts:
            pass
        elif requested_category:
            charts = [_category_chart(df, requested_category)]
        elif not has_metric_request and category_candidates:
            charts = [_category_chart(df, category_candidates[0])]
        elif charts:
            pass
        elif category_candidates:
            charts = [_category_chart(df, category_candidates[0])]
        elif value_candidates:
            col = requested_value or value_candidates[0]
            series = _numeric_series(df[col]).dropna()
            if not series.empty:
                bins = min(10, max(2, series.nunique()))
                counts, edges = np.histogram(series, bins=bins)
                hist = pd.DataFrame({
                    "Range": [f"{edges[i]:g} – {edges[i+1]:g}" for i in range(len(counts))],
                    "Count": counts.astype(int),
                })
                charts = [{
                    "kind": "bar",
                    "title": f"{col} distribution",
                    "description": f"Distribution of {col} across the complete dataset.",
                    "x": "Range",
                    "y": "Count",
                    "data": hist,
                }]

    # A non-explicit question such as "show sales last week" should still get
    # its relevant line chart through the time path above.
    return charts[:2]


def analyze_csv(frame: pd.DataFrame, question: str) -> Optional[dict]:
    if frame is None or frame.empty:
        return None

    df = frame.copy()
    graph_requested = _is_graph_request(question)
    charts = build_professional_charts(df, question)

    # Explicit graph request should never fall through to Gemini.
    if graph_requested and charts:
        return {
            "mode": "visualization",
            "file_rows": len(df),
            "charts": charts,
            "period": "full dataset",
        }

    date_candidates = [c for c, _ in _candidate_date_columns(df)]
    if not date_candidates:
        return None

    date_col = _find_named_column(df, question, date_candidates) or date_candidates[0]
    period = _period_from_question(question, df[date_col])
    if period is None:
        return None

    value_candidates = _candidate_value_columns(df)
    if not value_candidates:
        return None

    value_col = _find_named_column(df, question, value_candidates) or value_candidates[0]
    start, end, period_name = period

    work = df.copy()
    work["__date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["__value"] = _numeric_series(work[value_col])

    end_exclusive = end + pd.Timedelta(days=1)
    matching = work[(work["__date"] >= start) & (work["__date"] < end_exclusive)].copy()
    matching = matching.dropna(subset=["__date", "__value"])

    total = float(matching["__value"].sum())
    records = matching[[c for c in df.columns]].copy()
    daily = _build_daily(matching, "__date", "__value")
    if not daily.empty:
        daily = daily.rename(columns={"__value": value_col})

    calculation_chart = None
    if not daily.empty:
        calculation_chart = {
            "kind": "line",
            "title": f"{value_col} — {period_name}",
            "description": f"Daily {value_col} totals from {start.date()} to {end.date()}.",
            "x": "Date",
            "y": value_col,
            "data": daily,
        }

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
        "charts": [calculation_chart] if calculation_chart else [],
    }


def format_total(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"
