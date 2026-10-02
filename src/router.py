from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestRoute:
    kind: str
    confidence: str = "normal"


def _q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower().strip())


def is_greeting(q: str) -> bool:
    return _q(q) in {
        "hi", "hello", "hey", "hiya", "salam", "aoa",
        "assalam o alaikum", "assalamualaikum",
        "good morning", "good afternoon", "good evening",
    }


def is_image_generation(q: str) -> bool:
    q = _q(q)
    return bool(
        re.search(r"\b(generate|create|make|draw|render|design|produce)\b.{0,80}\b(image|picture|photo|illustration|poster|artwork|visual|tasveer|pic)\b", q)
        or re.search(r"\b(image|picture|photo|illustration|poster|artwork|visual|tasveer|pic)\b.{0,60}\b(banao|bana do|generate|create|make|draw|dikhao|do|chahiye)\b", q)
        or any(x in q for x in ("tasveer banao", "tasveer bana do", "image banao", "image bana do", "picture banao", "picture bana do", "photo banao", "photo bana do", "pic banao", "pic bana do", "mujhe image chahiye", "mujhe picture chahiye", "mujhe pic chahiye"))
    )


def is_graph_request(q: str) -> bool:
    q = _q(q)
    return any(x in q for x in ("graph", "chart", "plot", "bar chart", "line chart", "pie chart"))


def is_graph_picture_request(q: str) -> bool:
    q = _q(q)
    return is_graph_request(q) and any(x in q for x in ("picture", "image", "photo", "pic", "tasveer", "visual", "banao", "bana do", "generate", "create", "make", "draw", "render"))


def is_pdf_request(q: str) -> bool:
    q = _q(q)
    return "pdf" in q and any(x in q for x in ("make", "create", "generate", "give", "send", "download", "export", "save", "bana", "banao", "chahiye", "do", "de do", "nikal", "report", "mein", "me"))


def is_reconciliation_request(q: str) -> bool:
    q = _q(q)
    return any(x in q for x in (
        "reconcile", "reconciliation", "reconcile bank", "match bank",
        "bank statement match", "bank ledger match", "software ledger match",
        "matched transactions", "unmatched transactions", "bank aur software",
        "bank and software", "bank statement aur software",
    ))


def is_full_file_request(q: str) -> bool:
    q = _q(q)
    phrases = (
        "full details", "all details", "complete details", "complete information",
        "file details", "file ki sari details", "sari details", "poori details", "puri details",
        "sara data", "complete overview", "full overview", "overview of this file",
        "describe this file", "analyze this file", "analyse this file", "file ka analysis",
        "file ka complete analysis", "full file", "all data", "show everything",
        "everything in this file", "file ki sari information", "file ki puri information",
        "file ka sara data", "file ka pura data", "file ka complete data",
        "details about this file", "details of this file", "give the details",
        "give me the details", "give details about", "give me details about",
        "information about this file", "information of this file", "tell me about this file",
        "tell me the details", "show me the details", "details and graph", "details with graph",
        "details plus graph",
    )
    return any(p in q for p in phrases)


def is_data_question(q: str) -> bool:
    q = _q(q)
    # Temporal words alone are NOT enough. This prevents an unrelated question such
    # as "what happened this week?" from being hijacked by an uploaded CSV.
    terms = (
        "sales", "revenue", "profit", "expense", "cost", "income", "payment", "debit", "credit",
        "amount", "total", "average", "sum", "count", "record", "records", "transaction", "transactions",
        "entry", "entries", "group by", "by month", "by day", "by category", "per day", "per month",
        "export", "download", "csv", "spreadsheet", "excel",
    )
    return any(t in q for t in terms) or is_graph_request(q)


def should_use_file_context(q: str) -> bool:
    q = _q(q)
    explicit = (
        "file", "document", "pdf", "csv", "xlsx", "excel", "spreadsheet", "uploaded",
        "upload", "page", "sheet", "row", "column", "source", "according to", "in the document",
        "in this document", "in the file", "from the file", "from this file", "what does it say",
        "what is written", "summarize this", "summary of this", "explain this document",
        "this file", "that file", "this document", "that document",
    )
    return any(x in q for x in explicit) or is_data_question(q) or is_reconciliation_request(q) or is_full_file_request(q) or is_pdf_request(q)


def should_use_vision(q: str) -> bool:
    q = _q(q)
    return bool(
        re.search(r"\b(image|picture|photo|pic|tasveer|ocr|visual|screenshot|scan|scanned)\b", q)
        or any(p in q for p in ("what is shown", "what does this show", "what do you see", "look at this", "read this image", "read the image", "describe the image", "describe this picture", "analyze this image"))
    )


def should_use_web(q: str) -> bool:
    q = _q(q)
    triggers = (
        "latest", "today", "current", "recent", "news", "search web", "search online",
        "internet", "price now", "right now", "as of today", "what happened",
        "who is the current", "live update", "currently", "this morning", "this week",
        "this month", "2026", "2025", "official website", "look up",
    )
    return any(x in q for x in triggers)


def route(q: str, has_files: bool = False, has_tables: bool = False) -> RequestRoute:
    # High-confidence tools first. Everything else remains normal AI conversation.
    if is_greeting(q):
        return RequestRoute("greeting", "high")
    if is_image_generation(q) and not is_graph_picture_request(q):
        return RequestRoute("image", "high")
    # Deterministic reconciliation must win over Gemini for any explicit
    # bank/software-ledger request, including wording such as "make a software
    # ledger and bank statement ledger of this file".
    if has_tables and (
        is_reconciliation_request(q)
        or (any(t in q for t in ("bank", "statement")) and any(t in q for t in ("software ledger", "software", "accounting ledger")))
        or ("bank ledger" in q and "ledger" in q)
    ):
        return RequestRoute("reconciliation", "high")
    if is_pdf_request(q) and has_files:
        return RequestRoute("pdf", "high")
    if is_graph_picture_request(q):
        return RequestRoute("graph_picture", "high")
    if is_full_file_request(q) and has_tables:
        return RequestRoute("full_file", "high")
    if is_data_question(q) and has_tables:
        return RequestRoute("data", "high")
    if should_use_web(q):
        return RequestRoute("web", "normal")
    if has_files and should_use_file_context(q):
        return RequestRoute("file", "normal")
    return RequestRoute("general", "normal")
