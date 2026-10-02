from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def _safe(value) -> str:
    if value is None:
        return ""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table(df: pd.DataFrame, max_rows: int = 80, max_cols: int = 12):
    if df is None or df.empty:
        return None
    view = df.copy().iloc[:max_rows, :max_cols]
    data = [[_safe(c) for c in view.columns]]
    for row in view.itertuples(index=False, name=None):
        data.append([_safe(v) for v in row])
    widths = [max(22*mm, min(48*mm, 180*mm / max(1, len(view.columns))))] * len(view.columns)
    t = Table(data, repeatRows=1, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263238")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f6f7")]),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t


def build_pdf_report(filename: str, *, dataframe: pd.DataFrame | None = None,
                     records: list[dict] | None = None, title: str | None = None,
                     summary: str | None = None, metrics: pd.DataFrame | None = None,
                     profile: pd.DataFrame | None = None, result_table: pd.DataFrame | None = None,
                     answer: str | None = None) -> bytes:
    """Create a deterministic PDF report for any supported uploaded file.

    For tabular files it contains schema, metrics, requested-result table and a
    bounded data preview. For documents/images it contains extracted content
    metadata and a text preview. It never invents facts.
    """
    landscape_mode = dataframe is not None and len(dataframe.columns) > 7
    pagesize = landscape(A4) if landscape_mode else A4
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=pagesize, rightMargin=12*mm, leftMargin=12*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = []
    report_title = title or f"DocuSphere AI Report — {filename}"
    story.append(Paragraph(_safe(report_title), styles["CenterTitle"]))
    story.append(Paragraph(f"Source file: <b>{_safe(filename)}</b> · Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Small"]))
    story.append(Spacer(1, 8))
    if summary:
        story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
        story.append(Paragraph(_safe(summary).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))
    if answer:
        story.append(Paragraph("<b>Requested answer</b>", styles["Heading2"]))
        story.append(Paragraph(_safe(answer).replace("\n", "<br/>"), styles["Small"]))
        story.append(Spacer(1, 8))

    if dataframe is not None:
        story.append(Paragraph("<b>File overview</b>", styles["Heading2"]))
        overview = pd.DataFrame({"Property": ["Rows", "Columns", "Column names"],
                                 "Value": [len(dataframe), len(dataframe.columns), ", ".join(map(str, dataframe.columns))]})
        t = _table(overview, max_rows=5, max_cols=2)
        if t: story.append(t)
        story.append(Spacer(1, 8))
        if metrics is not None and not metrics.empty:
            story.append(Paragraph("<b>Numeric / financial overview</b>", styles["Heading2"]))
            t = _table(metrics, max_rows=20, max_cols=8)
            if t: story.append(t)
            story.append(Spacer(1, 8))
        if result_table is not None and not result_table.empty:
            story.append(Paragraph("<b>Requested result</b>", styles["Heading2"]))
            t = _table(result_table, max_rows=120, max_cols=12)
            if t: story.append(t)
            story.append(Spacer(1, 8))
        if profile is not None and not profile.empty:
            story.append(Paragraph("<b>Column profile</b>", styles["Heading2"]))
            t = _table(profile, max_rows=100, max_cols=8)
            if t: story.append(t)
            story.append(PageBreak())
        story.append(Paragraph("<b>Data preview</b>", styles["Heading2"]))
        t = _table(dataframe, max_rows=100, max_cols=12)
        if t: story.append(t)
    else:
        recs = records or []
        story.append(Paragraph(f"<b>Extracted sections/chunks:</b> {len(recs):,}", styles["BodyText"]))
        story.append(Spacer(1, 8))
        text = "\n\n".join(str(r.get("text", "")) for r in recs)
        if text:
            # Keep report useful and deterministic without creating huge PDFs.
            preview = text[:50000]
            story.append(Paragraph("<b>Extracted content</b>", styles["Heading2"]))
            story.append(Paragraph(_safe(preview).replace("\n", "<br/>"), styles["Small"]))
        else:
            story.append(Paragraph("No text layer was available in the extracted source. The source was processed, but there is no textual content to place in this report.", styles["BodyText"]))

    doc.build(story)
    return out.getvalue()
