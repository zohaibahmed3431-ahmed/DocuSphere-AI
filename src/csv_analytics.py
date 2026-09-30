
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px


DATE_HINTS = ("date", "time", "datetime", "created", "updated", "due", "deadline", "month", "year")
AMOUNT_HINTS = ("amount", "sales", "sale", "revenue", "income", "expense", "cost", "price", "total", "debit", "credit", "profit", "budget", "value", "payment")
CATEGORY_HINTS = ("type", "status", "category", "department", "customer", "client", "project", "account", "party", "region", "route", "service", "vendor", "supplier", "owner", "assignee")
ID_HINTS = ("id", "no", "number", "code", "ref", "reference")


@dataclass
class AnalysisResult:
    title: str
    summary: str
    table: pd.DataFrame
    chart: object | None
    period: str | None = None
    value_column: str | None = None
    date_column: str | None = None
    records: pd.DataFrame | None = None


def load_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
    df.columns = [str(c).strip() for c in df.columns]
    return clean_dataframe(df)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return out


def find_date_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    best = None
    for col in df.columns:
        name = str(col).lower()
        if not any(h in name for h in DATE_HINTS):
            continue
        raw = df[col].dropna().astype(str).str.strip()
        # Reject columns that are clearly time-only, e.g. 19:37.2.
        time_only_ratio = raw.str.match(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$").mean() if len(raw) else 1.0
        if time_only_ratio >= 0.60:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="mixed")
        ratio = parsed.notna().mean()
        # A valid date column should represent more than one calendar day.
        if ratio >= 0.60 and parsed.dropna().dt.normalize().nunique() > 1:
            score = ratio + (0.15 if "date" in name or "datetime" in name else 0)
            if best is None or score > best[0]:
                best = (score, col, parsed)
    return (best[1], best[2]) if best else (None, None)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for c in df.columns:
        series=df[c]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().any():
                result.append(c)
                continue
        s = pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
            errors="coerce",
        )
        nonempty=series.notna().sum()
        if nonempty and s.notna().sum() / nonempty >= 0.60:
            result.append(c)
    return result


def choose_value_column(df: pd.DataFrame, query: str = "") -> str | None:
    nums = numeric_columns(df)
    if not nums:
        return None
    q = query.lower()
    # Exact semantic matches always win.
    for c in nums:
        cl = c.lower()
        if re.search(rf"\b{re.escape(cl)}\b", q):
            return c
    # Query synonyms.
    synonym_groups = {
        "sales": ("sales", "sale", "revenue", "income"),
        "expense": ("expense", "cost"),
        "profit": ("profit",),
        "budget": ("budget",),
        "payment": ("payment",),
        "debit": ("debit",),
        "credit": ("credit",),
        "amount": ("amount", "value", "total"),
    }
    for group, words in synonym_groups.items():
        if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            for c in nums:
                if c.lower() == group:
                    return c
            # Do not silently choose an unrelated numeric field for an explicit metric.
            matches=[c for c in nums if any(w in c.lower() for w in words)]
            if matches:
                return matches[0]
    ranked=[]
    for c in nums:
        cl=c.lower(); score=0
        for i,h in enumerate(AMOUNT_HINTS):
            if h in cl: score += 100-i
        ranked.append((score,c))
    ranked.sort(reverse=True)
    return ranked[0][1]

def choose_category_column(df: pd.DataFrame, query: str = "") -> str | None:
    q = query.lower()
    candidates = []
    for c in df.columns:
        if c in numeric_columns(df):
            continue
        cl = c.lower()
        nunique = df[c].nunique(dropna=True)
        if nunique < 2 or nunique > min(100, max(2, len(df) * 0.7)):
            continue
        score = 0
        if cl in q:
            score += 100
        for i, h in enumerate(CATEGORY_HINTS):
            if h in cl:
                score += 60 - i
        if cl in ID_HINTS:
            score -= 50
        candidates.append((score, c))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _period_mask(dates: pd.Series, query: str) -> tuple[pd.Series, str]:
    q = query.lower()
    now = pd.Timestamp.now().normalize()
    # explicit years/months
    years = re.findall(r"\b(20\d{2})\b", q)
    if years:
        y = int(years[0])
        return dates.dt.year.eq(y), str(y)

    month_map = {m.lower(): i for i, m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
    for name, num in month_map.items():
        if name in q:
            y = int(years[0]) if years else now.year
            return (dates.dt.year.eq(y) & dates.dt.month.eq(num)), f"{name} {y}"

    if "last year" in q:
        y = now.year - 1
        return dates.dt.year.eq(y), f"Last year ({y})"
    if "this year" in q:
        return dates.dt.year.eq(now.year), f"This year ({now.year})"
    if "last month" in q:
        start = (now.replace(day=1) - pd.offsets.MonthBegin(1))
        end = now.replace(day=1)
        return dates.ge(start) & dates.lt(end), f"Last month ({start.strftime('%B %Y')})"
    if "this month" in q:
        start = now.replace(day=1)
        return dates.ge(start) & dates.lt(now + pd.Timedelta(days=1)), f"This month ({now.strftime('%B %Y')})"
    if "last week" in q:
        start_this = now - pd.Timedelta(days=now.weekday())
        start = start_this - pd.Timedelta(days=7)
        end = start_this
        return dates.ge(start) & dates.lt(end), f"Last week ({start.date()} to {(end-pd.Timedelta(days=1)).date()})"
    if "this week" in q:
        start = now - pd.Timedelta(days=now.weekday())
        return dates.ge(start) & dates.lt(now + pd.Timedelta(days=1)), "This week"
    if "last 7 days" in q or "past 7 days" in q:
        start = now - pd.Timedelta(days=6)
        return dates.ge(start) & dates.lt(now + pd.Timedelta(days=1)), "Last 7 days"
    if "last 30 days" in q or "past 30 days" in q:
        start = now - pd.Timedelta(days=29)
        return dates.ge(start) & dates.lt(now + pd.Timedelta(days=1)), "Last 30 days"
    if "today" in q:
        return dates.dt.date.eq(now.date()), "Today"
    return pd.Series(True, index=dates.index), None


def analyze_csv(df: pd.DataFrame, query: str) -> AnalysisResult | None:
    q = query.lower()
    date_col, dates = find_date_column(df)
    value_col = choose_value_column(df, query)
    category_col = choose_category_column(df, query)

    asks_graph = any(k in q for k in ("graph", "chart", "plot", "visual", "trend", "diagram"))
    asks_total = any(k in q for k in ("total", "sum", "how much", "kitna", "kitni", "amount", "sales", "revenue", "expense", "debit", "credit"))
    asks_breakdown = any(k in q for k in ("by ", "per ", "wise", "breakdown", "distribution", "compare", "comparison"))
    asks_time = any(k in q for k in ("last week", "this week", "last month", "this month", "last year", "this year", "last 7 days", "last 30 days", "today", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")) or bool(re.search(r"\b20\d{2}\b", q))

    if asks_time and date_col is None:
        # Do not pretend time-only fields are dates.
        return AnalysisResult(
            title="Date-based analysis unavailable",
            summary=f"I found no reliable calendar-date column. `{date_col}` cannot be used for period calculations." if date_col else "I found no reliable calendar-date column in this CSV, so I cannot calculate last week/month/year reliably without guessing.",
            table=pd.DataFrame(),
            chart=None,
            period=None,
            value_column=value_col,
            date_column=date_col,
            records=work.drop(columns=[c for c in ["_parsed_date","_value"] if c in work.columns]).copy(),
        )

    if not (asks_graph or asks_total or asks_breakdown or asks_time):
        return None

    work = df.copy()
    if date_col:
        work["_parsed_date"] = dates
        mask, period = _period_mask(dates, q)
        if asks_time:
            work = work[mask].copy()
        else:
            period = None
    else:
        period = None

    if value_col:
        work["_value"] = pd.to_numeric(
            work[value_col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
            errors="coerce",
        )

    # Explicit multi-metric comparison (e.g. Debit vs Credit).
    metric_cols = []
    for c in numeric_columns(work):
        cl = c.lower()
        if re.search(rf"\b{re.escape(cl)}\b", q) and cl in {"debit","credit","sales","revenue","expense","cost","profit","budget","amount","total"}:
            metric_cols.append(c)
    if len(metric_cols) >= 2 and not asks_time:
        totals = pd.DataFrame({"Metric": metric_cols, "Total": [pd.to_numeric(work[c], errors="coerce").sum() for c in metric_cols]})
        totals["Total"] = totals["Total"].round(2)
        fig=px.bar(totals, x="Metric", y="Total", title="Metric comparison", text_auto=True)
        fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Total", hovermode="x unified")
        return AnalysisResult("Metric comparison", "Comparison of the requested numeric fields.", totals, fig, period, None, date_col)

    # Breakdown/category graph.
    if category_col and (asks_breakdown or asks_graph) and not asks_time:
        grouped = work.groupby(category_col, dropna=False)["_value"].agg(["sum", "count"]).reset_index()
        grouped = grouped.sort_values("sum", ascending=False).head(25)
        grouped["sum"] = grouped["sum"].round(2)
        fig = px.bar(grouped, x=category_col, y="sum", title=f"{value_col or 'Records'} by {category_col}", text_auto=True)
        fig.update_layout(template="plotly_white", xaxis_title=category_col, yaxis_title=value_col or "Value", hovermode="x unified")
        return AnalysisResult(
            title=f"{value_col or 'Records'} by {category_col}",
            summary=f"Showing the top {len(grouped)} {category_col} groups by {value_col or 'record value'}.",
            table=grouped,
            chart=fig,
            period=period,
            value_column=value_col,
            date_column=date_col,
            records=work.drop(columns=[c for c in ["_parsed_date","_value"] if c in work.columns]).copy(),
        )

    # Time trend.
    if date_col and (asks_graph or asks_time):
        if value_col:
            grouped = work.dropna(subset=["_parsed_date", "_value"]).copy()
            if grouped.empty:
                return AnalysisResult("No matching data", "No valid dated numeric records matched the requested period.", pd.DataFrame(), None, period, value_col, date_col)
            grouped["_period_date"] = grouped["_parsed_date"].dt.floor("D")
            trend = grouped.groupby("_period_date")["_value"].sum().reset_index()
            trend.columns = [date_col, value_col]
            fig = px.line(trend, x=date_col, y=value_col, markers=True, title=f"{value_col} trend{f' — {period}' if period else ''}")
            fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title=value_col, hovermode="x unified")
            summary = f"{len(work):,} matching rows; {value_col} total = {work['_value'].sum():,.2f}"
            return AnalysisResult(f"{value_col} trend", summary, trend, fig, period, value_col, date_col,
                                  work.drop(columns=["_parsed_date","_value"], errors="ignore").copy())
        else:
            counts = work.dropna(subset=["_parsed_date"]).groupby(work["_parsed_date"].dt.floor("D")).size().reset_index(name="Records")
            counts.columns = [date_col, "Records"]
            fig = px.line(counts, x=date_col, y="Records", markers=True, title=f"Record volume{f' — {period}' if period else ''}")
            fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Records")
            return AnalysisResult("Record volume", f"{len(work):,} matching records.", counts, fig, period, None, date_col,
                                  work.drop(columns=["_parsed_date"], errors="ignore").copy())

    # Simple total.
    if value_col and asks_total:
        total = work["_value"].sum()
        table = work.drop(columns=[c for c in ["_parsed_date","_value"] if c in work.columns]).head(100)
        return AnalysisResult(
            title=f"Total {value_col}",
            summary=f"**{value_col} total: {total:,.2f}** across {len(work):,} matching row(s).",
            table=table,
            chart=None,
            period=period,
            value_column=value_col,
            date_column=date_col,
            records=work.drop(columns=[c for c in ["_parsed_date","_value"] if c in work.columns]).copy(),
        )

    return None


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
