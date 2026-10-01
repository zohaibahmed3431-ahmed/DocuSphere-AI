from __future__ import annotations

import io
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ReconciliationResult:
    bank_name: str
    software_name: str
    matched: pd.DataFrame
    unmatched: pd.DataFrame
    summary: dict


def _norm_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _amount_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("(", "-", regex=False).str.replace(")", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def _find_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    cols = list(df.columns)
    normalized = {_norm_text(c).replace(" ", ""): c for c in cols}
    for alias in aliases:
        key = _norm_text(alias).replace(" ", "")
        if key in normalized:
            return normalized[key]
    for c in cols:
        nc = _norm_text(c).replace(" ", "")
        if any(_norm_text(a).replace(" ", "") in nc for a in aliases):
            return c
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    date_col = _find_col(df, ("date", "transaction date", "value date", "posting date", "entry date", "tran date", "trans date"))
    ref_col = _find_col(df, ("reference", "ref no", "refno", "transaction id", "transaction no", "txn id", "cheque no", "check no", "document no", "voucher no", "journal no"))
    desc_col = _find_col(df, ("description", "narration", "remarks", "particulars", "details", "memo", "transaction details", "payee", "beneficiary"))
    debit_col = _find_col(df, ("debit", "withdrawal", "withdrawals", "dr", "debit amount"))
    credit_col = _find_col(df, ("credit", "deposit", "deposits", "cr", "credit amount"))
    amount_col = _find_col(df, ("amount", "transaction amount", "value", "net amount", "total"))
    balance_col = _find_col(df, ("balance", "running balance", "closing balance", "available balance"))
    return {"date": date_col, "reference": ref_col, "description": desc_col, "debit": debit_col, "credit": credit_col, "amount": amount_col, "balance": balance_col}


def classify_ledger(name: str, df: pd.DataFrame) -> tuple[str, int]:
    text = _norm_text(name)
    score_bank = 0
    score_software = 0
    bank_terms = ("bank", "statement", "bankstatement", "mt940", "swift", "iban", "cheque", "check")
    software_terms = ("software", "ledger", "gl", "accounting", "erp", "book", "voucher", "journal")
    for t in bank_terms:
        if t in text:
            score_bank += 4
    for t in software_terms:
        if t in text:
            score_software += 3
    cols = " ".join(_norm_text(c) for c in df.columns)
    for t in ("bank", "cheque", "check", "value date", "posting date"):
        if t in cols:
            score_bank += 2
    for t in ("voucher", "journal", "account", "ledger", "acc title", "ref no"):
        if t in cols:
            score_software += 2
    if score_bank > score_software:
        return "bank", score_bank
    if score_software > score_bank:
        return "software", score_software
    return "unknown", 0


def _signed_amounts(df: pd.DataFrame, cols: dict) -> pd.Series:
    if cols["debit"] and cols["credit"]:
        debit = _amount_series(df[cols["debit"]]).fillna(0.0)
        credit = _amount_series(df[cols["credit"]]).fillna(0.0)
        return credit - debit
    if cols["amount"]:
        return _amount_series(df[cols["amount"]])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _parse_dates(s: pd.Series) -> pd.Series:
    # Keep date parsing conservative. A date column with almost no parseable values
    # is treated as unavailable instead of being guessed.
    a = pd.to_datetime(s, errors="coerce", dayfirst=False)
    b = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if a.notna().sum() >= b.notna().sum():
        out = a
    else:
        out = b
    return out.dt.normalize()


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _reference_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or (len(a) >= 5 and (a in b or b in a))


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy().reset_index(drop=True)
    cols = detect_columns(out)
    signed = _signed_amounts(out, cols)
    out["__recon_amount"] = signed.abs()
    if cols["debit"] and cols["credit"]:
        debit = _amount_series(out[cols["debit"]]).fillna(0.0)
        credit = _amount_series(out[cols["credit"]]).fillna(0.0)
        out["__recon_direction"] = np.where(debit.abs() > credit.abs(), "debit", np.where(credit.abs() > debit.abs(), "credit", "unknown"))
    elif cols["amount"]:
        raw = _amount_series(out[cols["amount"]])
        out["__recon_direction"] = np.where(raw < 0, "debit", np.where(raw > 0, "credit", "unknown"))
    else:
        out["__recon_direction"] = "unknown"
    out["__recon_date"] = _parse_dates(out[cols["date"]]) if cols["date"] else pd.NaT
    out["__recon_ref"] = out[cols["reference"]].map(_norm_text) if cols["reference"] else ""
    out["__recon_desc"] = out[cols["description"]].map(_norm_text) if cols["description"] else ""
    return out, cols


def _score_pair(bank: pd.Series, soft: pd.Series, tolerance: float, date_window: int) -> tuple[float, str, float | None]:
    ba, sa = bank["__recon_amount"], soft["__recon_amount"]
    if pd.isna(ba) or pd.isna(sa):
        return -1.0, "amount unavailable", None
    bdirection, sdirection = bank["__recon_direction"], soft["__recon_direction"]
    direction_state = "unknown"
    if bdirection != "unknown" and sdirection != "unknown":
        direction_state = "same" if bdirection == sdirection else "opposite"
    diff = abs(float(ba) - float(sa))
    scale = max(abs(float(ba)), abs(float(sa)), 1.0)
    if diff > max(tolerance, scale * 1e-9):
        return -1.0, "amount mismatch", None
    bd, sd = bank["__recon_date"], soft["__recon_date"]
    date_diff = None
    if pd.notna(bd) and pd.notna(sd):
        date_diff = abs((bd - sd).days)
        if date_diff > date_window:
            return -1.0, "date outside matching window", float(date_diff)
    ref = _reference_match(bank["__recon_ref"], soft["__recon_ref"])
    desc_sim = _text_similarity(bank["__recon_desc"], soft["__recon_desc"])
    score = 0.55
    basis = ["same amount"]
    if direction_state == "same":
        score += 0.05
        basis.append("same debit/credit direction")
    elif direction_state == "opposite":
        # Bank statements and software ledgers can legitimately use opposite
        # debit/credit conventions for the same bank-account movement.
        score -= 0.05
        basis.append("opposite debit/credit convention")
    if ref:
        score += 0.35
        basis.append("reference")
    elif desc_sim >= 0.82:
        score += 0.15
        basis.append("description")
    if date_diff is not None:
        score += max(0.0, 0.10 * (1.0 - min(date_diff / max(date_window, 1), 1.0)))
        basis.append(f"date ±{int(date_diff)}d")
    return min(score, 1.0), ", ".join(basis), date_diff


def reconcile_tables(bank_df: pd.DataFrame, software_df: pd.DataFrame, bank_name: str = "Bank", software_name: str = "Software", tolerance: float = 0.01, date_window: int = 3) -> ReconciliationResult:
    bank, bcols = _prepare(bank_df)
    soft, scols = _prepare(software_df)
    if bank["__recon_amount"].notna().sum() == 0 or soft["__recon_amount"].notna().sum() == 0:
        raise ValueError("Could not identify a usable amount/debit-credit field in both files. Reconciliation was stopped to avoid false matches.")

    candidates = []
    for bi, br in bank.iterrows():
        for si, sr in soft.iterrows():
            score, basis, dd = _score_pair(br, sr, tolerance, date_window)
            if score >= 0:
                # Strong references can match even with weak descriptions; amount remains mandatory.
                candidates.append((score, bi, si, basis, dd))
    candidates.sort(key=lambda x: (-x[0], x[4] if x[4] is not None else 9999, x[1], x[2]))

    used_b, used_s = set(), set()
    matches = []
    serial = 1
    for score, bi, si, basis, dd in candidates:
        if bi in used_b or si in used_s:
            continue
        # If there is no reference/description identity, duplicate amounts are ambiguous
        # even when their dates are identical. Never guess a one-to-one pairing.
        bcount = int(np.isclose(bank["__recon_amount"].astype(float), float(bank.loc[bi, "__recon_amount"]), atol=tolerance, rtol=0).sum())
        scount = int(np.isclose(soft["__recon_amount"].astype(float), float(soft.loc[si, "__recon_amount"]), atol=tolerance, rtol=0).sum())
        identity_available = bool(bank.loc[bi, "__recon_ref"] and soft.loc[si, "__recon_ref"]) or _text_similarity(bank.loc[bi, "__recon_desc"], soft.loc[si, "__recon_desc"]) >= 0.82
        if not identity_available and bcount > 1 and scount > 1:
            continue
        # Opposite debit/credit conventions are allowed only when there is
        # enough additional evidence. Never pair an amount-only opposite-sign
        # transaction merely because the amount happens to be unique.
        bdir = bank.loc[bi, "__recon_direction"]
        sdir = soft.loc[si, "__recon_direction"]
        if bdir != "unknown" and sdir != "unknown" and bdir != sdir:
            dd = bank.loc[bi, "__recon_date"]
            sd = soft.loc[si, "__recon_date"]
            exact_or_near_date = pd.notna(dd) and pd.notna(sd) and abs((dd - sd).days) <= 1
            if not identity_available and not exact_or_near_date:
                continue
        used_b.add(bi); used_s.add(si)
        matches.append((serial, bi, si, score, basis, dd))
        serial += 1

    def clean_row(row: pd.Series, prefix: str) -> dict:
        return {f"{prefix}_{c}": (None if pd.isna(v) else v) for c, v in row.items() if not str(c).startswith("__recon_")}

    matched_rows = []
    for sn, bi, si, score, basis, dd in matches:
        d = {"Serial No": f"REC-{sn:06d}", "Bank Serial No": f"REC-{sn:06d}", "Software Serial No": f"REC-{sn:06d}", "Match Status": "MATCHED", "Match Score": round(score * 100, 2), "Match Basis": basis, "Date Difference (days)": dd}
        d.update(clean_row(bank.loc[bi], "Bank")); d.update(clean_row(soft.loc[si], "Software"))
        matched_rows.append(d)
    matched = pd.DataFrame(matched_rows)

    unmatched_rows = []
    next_serial = serial
    for bi in bank.index:
        if bi not in used_b:
            d = {"Serial No": f"BANK-{next_serial:06d}", "Side": "Bank only", "Match Status": "UNMATCHED"}
            d.update(clean_row(bank.loc[bi], "Bank")); unmatched_rows.append(d); next_serial += 1
    for si in soft.index:
        if si not in used_s:
            d = {"Serial No": f"SOFT-{next_serial:06d}", "Side": "Software only", "Match Status": "UNMATCHED"}
            d.update(clean_row(soft.loc[si], "Software")); unmatched_rows.append(d); next_serial += 1
    unmatched = pd.DataFrame(unmatched_rows)

    summary = {
        "bank_rows": len(bank_df), "software_rows": len(software_df), "matched_rows": len(matched),
        "bank_unmatched": int((~bank.index.isin(used_b)).sum()), "software_unmatched": int((~soft.index.isin(used_s)).sum()),
        "bank_columns": bcols, "software_columns": scols, "tolerance": tolerance, "date_window_days": date_window,
    }
    return ReconciliationResult(bank_name, software_name, matched, unmatched, summary)


def looks_like_reconciliation_request(query: str, frames: dict) -> bool:
    q = _norm_text(query)
    terms = ("reconcile", "reconciliation", "bank reconciliation", "bank ledger", "bank statement", "software ledger", "match bank", "matching transactions", "matched transactions", "unmatched transactions")
    if any(t.replace(" ", "") in q.replace(" ", "") for t in terms):
        return len(frames) >= 2
    labels = " ".join(_norm_text(k) for k in frames)
    return len(frames) >= 2 and ("bank" in labels and ("ledger" in labels or "software" in labels))


def auto_select_two_ledgers(frames: dict) -> tuple[tuple[str, pd.DataFrame], tuple[str, pd.DataFrame]] | None:
    if len(frames) < 2:
        return None
    items = list(frames.items())
    scored = [(classify_ledger(name, df)[0], classify_ledger(name, df)[1], name, df) for name, df in items]
    banks = sorted([x for x in scored if x[0] == "bank"], key=lambda x: -x[1])
    softs = sorted([x for x in scored if x[0] == "software"], key=lambda x: -x[1])
    if banks and softs:
        return (banks[0][2], banks[0][3]), (softs[0][2], softs[0][3])
    # If both files are genuinely generic/ambiguous, never silently decide which
    # one is the bank statement and which one is the software ledger.
    return None


def reconciliation_excel(result: ReconciliationResult) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        summary = pd.DataFrame({"Metric": ["Bank file", "Software file", "Bank rows", "Software rows", "Matched", "Bank unmatched", "Software unmatched", "Amount tolerance", "Date matching window (days)"], "Value": [result.bank_name, result.software_name, result.summary["bank_rows"], result.summary["software_rows"], result.summary["matched_rows"], result.summary["bank_unmatched"], result.summary["software_unmatched"], result.summary["tolerance"], result.summary["date_window_days"]]})
        summary.to_excel(writer, sheet_name="Summary", index=False)
        result.matched.to_excel(writer, sheet_name="Matched", index=False)
        result.unmatched.to_excel(writer, sheet_name="Unmatched", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = min(max(len(str(cell.value or "")) for cell in col) + 2, 42)
                ws.column_dimensions[col[0].column_letter].width = width
    return bio.getvalue()
