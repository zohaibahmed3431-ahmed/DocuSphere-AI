from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import plotly.express as px

DATE_HINTS = ("date", "datetime", "created", "updated", "due", "deadline", "start", "end")
AMOUNT_HINTS = ("amount", "sales", "sale", "revenue", "income", "expense", "cost", "price", "total", "debit", "credit", "profit", "budget", "value", "payment")
CATEGORY_HINTS = ("type", "status", "category", "department", "customer", "client", "project", "account", "party", "region", "route", "service", "vendor", "supplier", "owner", "assignee", "employee")
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
    kind: str = "answer"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    return out


def load_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
    return clean_dataframe(df)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                result.append(c)
                continue
        converted = pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce",
        )
        nonempty = s.notna().sum()
        if nonempty and converted.notna().sum() / nonempty >= 0.60:
            result.append(c)
    return result


def _parsed_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def find_date_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Find the most credible calendar-date column using both name and data shape.

    Time-only values, IDs and mostly-unparseable columns are rejected. This allows
    columns such as Timestamp/Transaction/Order to work even when they do not contain
    the word "date".
    """
    best = None
    for col in df.columns:
        name = str(col).lower().strip()
        raw = df[col].dropna().astype(str).str.strip()
        if len(raw) < 2:
            continue
        time_only_ratio = raw.str.match(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$").mean()
        if time_only_ratio >= 0.60:
            continue
        # Avoid treating obvious identifiers as dates.
        if any(h == name for h in ID_HINTS) and not any(h in name for h in DATE_HINTS):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="mixed")
        ratio = parsed.notna().mean()
        if ratio < 0.70:
            continue
        unique_dates = parsed.dropna().dt.normalize().nunique()
        if unique_dates < 2:
            continue
        name_score = 0.0
        if any(h in name for h in DATE_HINTS):
            name_score += 0.35
        if any(h in name for h in ("timestamp", "transaction", "order", "entry", "posted")):
            name_score += 0.20
        # Strongly prefer actual datetime dtype.
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            name_score += 0.25
        score = ratio + name_score
        if best is None or score > best[0]:
            best = (score, col, parsed)
    return (best[1], best[2]) if best else (None, None)

def choose_value_column(df: pd.DataFrame, query: str = "") -> str | None:
    nums = numeric_columns(df)
    if not nums:
        return None
    q = query.lower()

    # Explicit column mentioned by the user always wins.
    for c in nums:
        if re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", q):
            return c

    # If the user names a business metric, only use a matching field. Never
    # substitute an unrelated numeric field.
    metric_groups = {
        "sales": ("sales", "sale", "revenue", "income"),
        "expense": ("expense", "expenses", "cost"),
        "profit": ("profit", "profits"),
        "budget": ("budget",),
        "payment": ("payment", "payments"),
        "debit": ("debit",),
        "credit": ("credit",),
    }
    for words in metric_groups.values():
        if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            matches = [c for c in nums if any(w in c.lower() for w in words)]
            return matches[0] if matches else None

    if any(re.search(rf"\b{re.escape(w)}\b", q) for w in ("amount", "value", "total")):
        matches = [c for c in nums if any(w in c.lower() for w in ("amount", "value", "total"))]
        if matches:
            return matches[0]

    # For a generic graph/full-file request, prefer strong business/accounting
    # measures over helper fields such as BeforeExchangeAmount, ROE, IDs, etc.
    priority = (
        "sales", "revenue", "profit", "expense", "debit", "credit",
        "amount", "total", "payment", "cost", "budget", "quantity",
        "balance", "value",
    )
    ranked=[]
    for c in nums:
        cl=c.lower()
        score=0
        for i,h in enumerate(priority):
            if h in cl:
                score += 1000 - i*30
        if any(x in cl for x in ("beforeexchange", "exchange", "rate", "roe", "refno", "id", "no", "number")):
            score -= 500
        ranked.append((score, c))
    ranked.sort(key=lambda x: (-x[0], x[1].lower()))
    return ranked[0][1] if ranked else None

def choose_category_column(df: pd.DataFrame, query: str = "") -> str | None:
    q = query.lower()
    candidates = []
    nums = set(numeric_columns(df))
    for c in df.columns:
        if c in nums:
            continue
        cl = c.lower()
        nunique = df[c].nunique(dropna=True)
        if nunique < 2 or nunique > min(100, max(2, len(df) * 0.7)):
            continue
        score = 100 if cl in q else 0
        score += sum(60 - i for i, h in enumerate(CATEGORY_HINTS) if h in cl)
        if cl in ID_HINTS or any(h == cl for h in ID_HINTS):
            score -= 50
        candidates.append((score, c))
    return max(candidates, default=(0, None))[1]


def _period_mask(dates: pd.Series, query: str) -> tuple[pd.Series, str | None]:
    q = query.lower()
    now = pd.Timestamp.now().normalize()
    years = re.findall(r"\b(20\d{2})\b", q)
    if years:
        y = int(years[0])
        return dates.dt.year.eq(y), str(y)

    months = {m.lower(): i for i, m in enumerate([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ], 1)}
    for name, num in months.items():
        if name in q:
            y = now.year
            start = pd.Timestamp(y, num, 1)
            end = start + pd.offsets.MonthBegin(1)
            return dates.ge(start) & dates.lt(end), f"{name.title()} {y}"

    if "last year" in q:
        y = now.year - 1
        return dates.dt.year.eq(y), f"Last year ({y})"
    if "this year" in q:
        return dates.dt.year.eq(now.year), f"This year ({now.year})"
    if "last month" in q:
        start = now.replace(day=1) - pd.offsets.MonthBegin(1)
        end = now.replace(day=1)
        return dates.ge(start) & dates.lt(end), f"Last month ({start.strftime('%B %Y')})"
    if "this month" in q:
        start = now.replace(day=1)
        return dates.ge(start) & dates.lt(now + pd.Timedelta(days=1)), f"This month ({now.strftime('%B %Y')})"
    if "last week" in q:
        start_this = now - pd.Timedelta(days=now.weekday())
        start = start_this - pd.Timedelta(days=7)
        end = start_this
        return dates.ge(start) & dates.lt(end), f"Last week ({start.strftime('%d %b %Y')} – {(end-pd.Timedelta(days=1)).strftime('%d %b %Y')})"
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


def _style(fig, title: str, x_title: str = "", y_title: str = ""):
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_dark",
        height=460,
        margin={"l": 55, "r": 30, "t": 75, "b": 70},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        font={"size": 13},
    )
    fig.update_xaxes(title=x_title, showgrid=False, automargin=True)
    fig.update_yaxes(title=y_title, gridcolor="rgba(255,255,255,.10)", automargin=True)
    return fig


def _asks(q: str, words: tuple[str, ...]) -> bool:
    return any(w in q for w in words)


def analyze_csv(df: pd.DataFrame, query: str) -> AnalysisResult | None:
    q = query.lower().strip()
    date_col, dates = find_date_column(df)
    value_col = choose_value_column(df, query)
    category_col = choose_category_column(df, query)

    asks_graph = _asks(q, ("graph", "chart", "plot", "visual", "trend", "diagram", "visualize"))
    asks_total = _asks(q, ("total", "sum", "how much", "kitna", "kitni", "amount", "sales", "revenue", "expense", "debit", "credit", "profit", "budget"))
    asks_breakdown = _asks(q, ("by ", "per ", "wise", "breakdown", "distribution", "compare", "comparison", "top "))
    asks_count = _asks(q, ("how many", "kitne", "kitni entries", "count", "number of", "records"))
    asks_export = _asks(q, ("download", "export", "csv bana", "csv banao", "export csv", "make csv", "give me csv"))
    asks_growth = _asks(q, ("growth", "grew", "increase", "decrease", "change", "growth rate", "growth %", "percentage change", "percent change"))
    asks_time = _asks(q, ("last week", "this week", "last month", "this month", "last year", "this year", "last 7 days", "last 30 days", "today")) or bool(re.search(r"\b20\d{2}\b", q)) or any(m in q for m in ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"))

    # A date request must never be answered from a time-only field or invented date.
    if asks_time and date_col is None:
        return AnalysisResult(
            title="Date-based analysis needs a real date column",
            summary="This file does not contain a reliable calendar-date column, so I cannot calculate last week/month/year without guessing.",
            table=pd.DataFrame(), chart=None, value_column=value_col, date_column=None, kind="warning"
        )

    work = df.copy()
    period = None
    if date_col and dates is not None:
        work["__parsed_date"] = dates
        mask, period = _period_mask(dates, q)
        if asks_time:
            work = work[mask].copy()

    if value_col:
        work["__value"] = _parsed_numeric(work[value_col])

    # Count request.
    if asks_count and not asks_total and not asks_graph and not asks_breakdown and not asks_time:
        table = pd.DataFrame({"Metric": ["Records"], "Count": [len(work)]})
        return AnalysisResult("Record count", f"This file contains **{len(work):,} records**.", table, None, kind="answer")

    # Explicit two-metric comparison.
    metric_cols = [c for c in numeric_columns(work) if re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", q)]
    if len(metric_cols) >= 2:
        totals = pd.DataFrame({"Metric": metric_cols, "Total": [_parsed_numeric(work[c]).sum() for c in metric_cols]})
        totals["Total"] = totals["Total"].round(2)
        fig = px.bar(totals, x="Metric", y="Total", text_auto=True)
        _style(fig, "Metric comparison", "Metric", "Total")
        return AnalysisResult("Metric comparison", "Exact totals for the requested numeric fields.", totals, fig, period, kind="chart")

    # Breakdown / category chart.
    if category_col and (asks_breakdown or asks_graph) and not asks_time:
        if value_col:
            grouped = work.groupby(category_col, dropna=False)["__value"].sum().reset_index(name=value_col)
            grouped[value_col] = grouped[value_col].round(2)
            grouped = grouped.sort_values(value_col, ascending=False).head(20)
            fig = px.bar(grouped, x=category_col, y=value_col, text_auto=True)
            _style(fig, f"{value_col} by {category_col}", category_col, value_col)
            return AnalysisResult(f"{value_col} by {category_col}", f"Showing the top {len(grouped)} {category_col} groups by {value_col}.", grouped, fig, value_column=value_col, kind="chart")
        grouped = work[category_col].fillna("Unknown").value_counts().reset_index()
        grouped.columns = [category_col, "Records"]
        grouped = grouped.head(20)
        fig = px.bar(grouped, x=category_col, y="Records", text_auto=True)
        _style(fig, f"Records by {category_col}", category_col, "Records")
        return AnalysisResult(f"Records by {category_col}", f"Showing record counts by {category_col}.", grouped, fig, kind="chart")

    # A named metric must exist in the table before we calculate it.
    named_metric_request = asks_total or asks_growth
    if named_metric_request and value_col is None and not asks_count:
        return AnalysisResult(
            title="Requested metric not found",
            summary="I could not find a numeric column that matches the requested metric. I will not substitute an unrelated column or invent a value.",
            table=pd.DataFrame(), chart=None, period=period, date_column=date_col, kind="warning"
        )

    # Time series.
    if date_col and dates is not None and (asks_graph or asks_time):
        valid = work.dropna(subset=["__parsed_date"]).copy()
        if value_col:
            valid = valid.dropna(subset=["__value"])
            if valid.empty:
                return AnalysisResult("No matching data", "No valid numeric records matched the requested period.", pd.DataFrame(), None, period, value_col, date_col, kind="warning")
            valid["__period"] = valid["__parsed_date"].dt.floor("D")
            trend = valid.groupby("__period")["__value"].sum().reset_index()
            trend.columns = [date_col, value_col]
            trend[value_col] = trend[value_col].round(2)
            fig = px.line(trend, x=date_col, y=value_col, markers=True)
            _style(fig, f"{value_col} trend{f' — {period}' if period else ''}", "Date", value_col)
            total = float(valid["__value"].sum())
            summary = f"{len(valid):,} matching rows; **{value_col} total = {total:,.2f}**."
            if asks_growth and period:
                now = pd.Timestamp.now().normalize()
                previous_mask = None
                previous_label = None
                if "this month" in q:
                    start = now.replace(day=1)
                    prev_start = start - pd.offsets.MonthBegin(1)
                    previous_mask = dates.ge(prev_start) & dates.lt(start)
                    previous_label = prev_start.strftime("%B %Y")
                elif "last month" in q:
                    start = now.replace(day=1) - pd.offsets.MonthBegin(1)
                    prev_start = start - pd.offsets.MonthBegin(1)
                    previous_mask = dates.ge(prev_start) & dates.lt(start)
                    previous_label = prev_start.strftime("%B %Y")
                elif "this week" in q:
                    start = now - pd.Timedelta(days=now.weekday())
                    prev_start = start - pd.Timedelta(days=7)
                    previous_mask = dates.ge(prev_start) & dates.lt(start)
                    previous_label = "previous week"
                elif "last week" in q:
                    start = now - pd.Timedelta(days=now.weekday()) - pd.Timedelta(days=7)
                    prev_start = start - pd.Timedelta(days=7)
                    previous_mask = dates.ge(prev_start) & dates.lt(start)
                    previous_label = "week before last"
                elif "this year" in q:
                    start = pd.Timestamp(now.year, 1, 1)
                    prev_start = pd.Timestamp(now.year - 1, 1, 1)
                    previous_mask = dates.ge(prev_start) & dates.lt(start)
                    previous_label = str(now.year - 1)
                elif "last year" in q:
                    start = pd.Timestamp(now.year - 1, 1, 1)
                    prev_start = pd.Timestamp(now.year - 2, 1, 1)
                    previous_mask = dates.ge(prev_start) & dates.lt(start)
                    previous_label = str(now.year - 2)
                if previous_mask is not None:
                    previous_total = float(_parsed_numeric(df.loc[previous_mask, value_col]).sum())
                    if previous_total != 0:
                        growth = ((total - previous_total) / abs(previous_total)) * 100
                        summary += f" Previous period ({previous_label}) was **{previous_total:,.2f}**, so the change is **{growth:+.2f}%**."
                        trend.attrs["previous_period_total"] = previous_total
                        trend.attrs["growth_percent"] = growth
                    else:
                        summary += f" Previous period ({previous_label}) total was **0.00**, so a percentage growth rate is not mathematically defined."
            return AnalysisResult(f"{value_col} trend", summary, trend, fig, period, value_col, date_col, valid.drop(columns=["__parsed_date", "__value", "__period"], errors="ignore"), "chart")
        counts = valid.groupby(valid["__parsed_date"].dt.floor("D")).size().reset_index(name="Records")
        counts.columns = [date_col, "Records"]
        fig = px.line(counts, x=date_col, y="Records", markers=True)
        _style(fig, f"Record volume{f' — {period}' if period else ''}", "Date", "Records")
        return AnalysisResult("Record volume", f"{len(valid):,} matching records.", counts, fig, period, None, date_col, valid.drop(columns=["__parsed_date"], errors="ignore"), "chart")

    # Exact total, with deterministic period-over-period growth when requested.
    if value_col and asks_total:
        total = float(work["__value"].sum())
        table = pd.DataFrame({"Metric": [value_col], "Total": [round(total, 2)], "Records": [len(work)]})
        summary = f"**{value_col}: {total:,.2f}** across {len(work):,} matching row(s)."

        if asks_growth and date_col and dates is not None and period:
            now = pd.Timestamp.now().normalize()
            ql = q
            previous_mask = None
            previous_label = None
            if "this month" in ql:
                start = now.replace(day=1)
                prev_start = start - pd.offsets.MonthBegin(1)
                previous_mask = dates.ge(prev_start) & dates.lt(start)
                previous_label = prev_start.strftime("%B %Y")
            elif "last month" in ql:
                start = now.replace(day=1) - pd.offsets.MonthBegin(1)
                prev_start = start - pd.offsets.MonthBegin(1)
                previous_mask = dates.ge(prev_start) & dates.lt(start)
                previous_label = prev_start.strftime("%B %Y")
            elif "this week" in ql:
                start = now - pd.Timedelta(days=now.weekday())
                prev_start = start - pd.Timedelta(days=7)
                previous_mask = dates.ge(prev_start) & dates.lt(start)
                previous_label = "previous week"
            elif "last week" in ql:
                start = now - pd.Timedelta(days=now.weekday()) - pd.Timedelta(days=7)
                prev_start = start - pd.Timedelta(days=7)
                previous_mask = dates.ge(prev_start) & dates.lt(start)
                previous_label = "week before last"
            elif "this year" in ql:
                start = pd.Timestamp(now.year, 1, 1)
                prev_start = pd.Timestamp(now.year - 1, 1, 1)
                previous_mask = dates.ge(prev_start) & dates.lt(start)
                previous_label = str(now.year - 1)
            elif "last year" in ql:
                start = pd.Timestamp(now.year - 1, 1, 1)
                prev_start = pd.Timestamp(now.year - 2, 1, 1)
                previous_mask = dates.ge(prev_start) & dates.lt(start)
                previous_label = str(now.year - 2)

            if previous_mask is not None:
                previous_total = float(_parsed_numeric(df.loc[previous_mask, value_col]).sum())
                if previous_total != 0:
                    growth = ((total - previous_total) / abs(previous_total)) * 100
                    summary += f" Previous period ({previous_label}) was **{previous_total:,.2f}**, so the change is **{growth:+.2f}%**."
                    table["Previous period"] = [round(previous_total, 2)]
                    table["Growth %"] = [round(growth, 2)]
                else:
                    summary += f" Previous period ({previous_label}) total was **0.00**, so a percentage growth rate is not mathematically defined."
                    table["Previous period"] = [0.0]
                    table["Growth %"] = [None]

        return AnalysisResult(f"Total {value_col}", summary, table, None, period, value_col, date_col, work.drop(columns=["__parsed_date", "__value"], errors="ignore"), "export" if asks_export else "answer")

    # Export: return full dataset if user explicitly asks for a CSV and no narrower result exists.
    if asks_export:
        records = work.drop(columns=["__parsed_date", "__value"], errors="ignore").copy()
        return AnalysisResult("CSV export", f"Prepared **{len(records):,} records** for export.", records.head(100), None, period, value_col, date_col, records, "export")

    return None


def is_full_details_request(query: str) -> bool:
    q = query.lower().strip()
    phrases = (
        "full details", "all details", "complete details", "complete information",
        "full information", "file details", "file ki sari details",
        "sari details", "poori details", "puri details", "sara data",
        "complete overview", "full overview", "overview of this file",
        "describe this file", "analyze this file", "analyse this file",
        "file ka analysis", "file ka complete analysis",
        "full file", "all data", "show everything", "everything in this file",
        "file ki sari information", "file ki puri information",
        "file ka sara data", "file ka pura data", "file ka complete data",
        "details about this file", "details of this file", "give the details",
        "give me the details", "give details about", "give me details about",
        "information about this file", "information of this file",
        "tell me about this file", "tell me the details", "show me the details",
        "details and graph", "details with graph", "details plus graph",
    )
    return any(p in q for p in phrases)

def full_file_report(df: pd.DataFrame, filename: str) -> dict:
    """Build a deterministic, professional dashboard from the whole table.
    No LLM is required for this report, so a Gemini/API problem cannot turn a
    valid CSV analysis into a red-screen failure.
    """
    data = clean_dataframe(df)
    nums = numeric_columns(data)
    date_col, dates = find_date_column(data)
    charts = []

    # Financial / numeric overview: prefer real business/accounting measures.
    strong=[]
    priority=("sales","revenue","profit","expense","debit","credit","amount","total","payment","cost","budget","quantity","balance","value")
    for c in nums:
        cl=c.lower()
        score=sum(1000-i*30 for i,h in enumerate(priority) if h in cl)
        if any(x in cl for x in ("beforeexchange","exchange","rate","roe","refno","id","number")):
            score-=500
        strong.append((score,c))
    strong.sort(key=lambda x:(-x[0],x[1].lower()))
    positive=[c for score,c in strong if score > 0]
    metric_cols=positive[:6] if positive else nums[:6]
    metric_rows=[]
    for c in metric_cols:
        vals=_parsed_numeric(data[c])
        metric_rows.append({"Column":c,"Total":round(float(vals.sum()),2),"Average":round(float(vals.mean()),2) if vals.notna().any() else 0,"Minimum":round(float(vals.min()),2) if vals.notna().any() else 0,"Maximum":round(float(vals.max()),2) if vals.notna().any() else 0})
    metrics=pd.DataFrame(metric_rows)

    # If Debit/Credit (or another clearly paired business metric) exists, show
    # the comparison instead of an arbitrary helper field.
    paired=[c for c in metric_cols if c.lower() in {"debit","credit","sales","revenue","expense","profit"}]
    if len(paired)>=2:
        pair_df=metrics[metrics["Column"].isin(paired)].copy()
        fig=px.bar(pair_df,x="Column",y="Total",text_auto=".2s",title="Key financial metrics")
        _style(fig,"Key financial metrics","Metric","Total")
        charts.append(fig)
    elif metric_rows:
        fig=px.bar(metrics,x="Column",y="Total",text_auto=".2s",title="Key financial / numeric totals")
        _style(fig,"Key financial / numeric totals","Field","Total")
        charts.append(fig)

    # High-value categorical breakdown.
    cat_candidates = []
    for c in data.columns:
        if c in nums or c == date_col:
            continue
        nunique = data[c].nunique(dropna=True)
        if 2 <= nunique <= min(20, max(2, len(data) // 2)):
            hint_score = sum(1 for h in CATEGORY_HINTS if h in c.lower())
            cat_candidates.append((hint_score, -nunique, c))
    # Prefer concise, human-meaningful business categories for the first chart.
    exact_priority={"type":1000,"category":950,"status":900,"department":850,"region":800,"acc_title":780,"account":760}
    cat_candidates.sort(key=lambda item: (exact_priority.get(item[2].lower(), 0) + item[0]*10, item[1], item[2].lower()), reverse=True)
    for _, _, c in cat_candidates[:2]:
        counts = data[c].fillna("Unknown").astype(str).value_counts().reset_index()
        counts.columns = [c, "Records"]
        fig = px.bar(counts, x=c, y="Records", text_auto=True, title=f"Records by {c}")
        _style(fig, f"Records by {c}", c, "Records")
        charts.append(fig)

    # If a real date exists, show the primary numeric trend.
    if date_col and dates is not None and metric_cols:
        tmp = data.copy()
        tmp["__date"] = dates
        tmp["__value"] = _parsed_numeric(tmp[metric_cols[0]])
        tmp = tmp.dropna(subset=["__date", "__value"])
        if not tmp.empty:
            trend = tmp.groupby(tmp["__date"].dt.to_period("M").dt.to_timestamp())["__value"].sum().reset_index()
            trend.columns = ["Month", metric_cols[0]]
            fig = px.line(trend, x="Month", y=metric_cols[0], markers=True, title=f"Monthly {metric_cols[0]} trend")
            _style(fig, f"Monthly {metric_cols[0]} trend", "Month", metric_cols[0])
            charts.append(fig)

    # Column profile is useful even when no meaningful chart can be built.
    profile = pd.DataFrame({
        "Column": data.columns.astype(str),
        "Type": [str(data[c].dtype) for c in data.columns],
        "Non-empty": [int(data[c].notna().sum()) for c in data.columns],
        "Unique": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })

    summary_bits=[f"{len(data):,} rows and {len(data.columns):,} columns."]
    if date_col:
        summary_bits.append(f"Reliable calendar date column: {date_col}.")
    else:
        summary_bits.append("No reliable calendar-date column was detected; time-only fields are not treated as dates.")
    if metric_cols:
        summary_bits.append("Key numeric fields: " + ", ".join(metric_cols[:6]) + ".")
    categorical_summary=[]
    for c in [x[2] for x in cat_candidates[:3]]:
        categorical_summary.append(f"{c} ({data[c].nunique(dropna=True):,} values)")
    if categorical_summary:
        summary_bits.append("Useful categorical fields: " + ", ".join(categorical_summary) + ".")

    return {
        "filename": filename,
        "summary": " ".join(summary_bits),
        "rows": len(data),
        "columns": len(data.columns),
        "date_column": date_col,
        "metrics": metrics,
        "profile": profile,
        "charts": charts,
        "numeric_columns": nums,
        "category_columns": [c for _, _, c in cat_candidates],
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    import io
    if len(df) + 1 > 1_048_576:
        raise ValueError("This table exceeds Excel's 1,048,576-row worksheet limit; use CSV for the complete data.")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Data")
    return bio.getvalue()
