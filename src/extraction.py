
from __future__ import annotations
import io
from pathlib import Path
from typing import BinaryIO
import pandas as pd
from PIL import Image
import pytesseract
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook

TEXT_EXTENSIONS = {
    ".txt",".md",".py",".java",".cpp",".c",".h",".hpp",".js",".ts",".html",".css",".sql",".json",".xml"
}
IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".webp"}

def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")

def extract_pdf(data: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    pages=[]
    for number,page in enumerate(reader.pages,1):
        text=page.extract_text() or ""
        pages.append({"text":text,"page":number})
    return pages

def extract_docx(data: bytes) -> list[dict]:
    doc=Document(io.BytesIO(data))
    parts=[p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    text="\n".join(parts)
    return [{"text":text,"page":None}] if text else []

def extract_pptx(data: bytes) -> list[dict]:
    prs=Presentation(io.BytesIO(data)); out=[]
    for n,slide in enumerate(prs.slides,1):
        parts=[]
        for shape in slide.shapes:
            if hasattr(shape,"text") and shape.text.strip(): parts.append(shape.text.strip())
        text="\n".join(parts)
        if text: out.append({"text":text,"page":n})
    return out

def extract_xlsx(data: bytes) -> list[dict]:
    wb=load_workbook(io.BytesIO(data),read_only=True,data_only=True)
    out=[]
    for ws in wb.worksheets:
        rows=[]
        for row in ws.iter_rows(values_only=True):
            vals=["" if v is None else str(v) for v in row]
            if any(v.strip() for v in vals): rows.append(" | ".join(vals))
        if rows: out.append({"text":f"Sheet: {ws.title}\n"+"\n".join(rows),"page":None,"sheet":ws.title})
    return out

def extract_csv(data: bytes) -> list[dict]:
    bio=io.BytesIO(data)
    try: df=pd.read_csv(bio)
    except Exception:
        bio.seek(0); df=pd.read_csv(bio,encoding="latin1")
    df.columns=[str(c).strip() for c in df.columns]
    # Keep a readable representation for RAG; structured dataframe is handled separately in app.
    text=df.to_csv(index=False)
    return [{"text":text,"page":None,"sheet":None}]

def extract_image(data: bytes) -> list[dict]:
    image=Image.open(io.BytesIO(data))
    text=pytesseract.image_to_string(image)
    return [{"text":text,"page":1}] if text.strip() else [{"text":"","page":1}]

def extract_file(uploaded_file: BinaryIO, filename: str) -> list[dict]:
    data=uploaded_file.read()
    suffix=Path(filename).suffix.lower()
    if suffix==".pdf": return extract_pdf(data)
    if suffix==".docx": return extract_docx(data)
    if suffix==".pptx": return extract_pptx(data)
    if suffix==".xlsx": return extract_xlsx(data)
    if suffix==".csv": return extract_csv(data)
    if suffix in TEXT_EXTENSIONS:
        text=_decode_text(data); return [{"text":text,"page":None}] if text.strip() else []
    if suffix in IMAGE_EXTENSIONS: return extract_image(data)
    raise ValueError(f"Unsupported file type: {suffix}")
