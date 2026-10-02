from __future__ import annotations

import io
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

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
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True),
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
    return {
        "date": _find_col(df, ("date", "transaction date", "value date", "posting date", "entry date", "tran date", "trans date")),
        "reference": _find_col(df, ("reference", "ref no", "refno", "transaction id", "transaction no", "txn id", "cheque no", "check no", "document no", "voucher no", "journal no")),
        "description": _find_col(df, ("description", "narration", "remarks", "particulars", "details", "memo", "transaction details", "payee", "beneficiary")),
        "debit": _find_col(df, ("debit", "withdrawal", "withdrawals", "dr", "debit amount")),
        "credit": _find_col(df, ("credit", "deposit", "deposits", "cr", "credit amount")),
        "amount": _find_col(df, ("amount", "transaction amount", "value", "net amount", "total")),
        "balance": _find_col(df, ("balance", "running balance", "closing balance", "available balance")),
    }


def classify_ledger(name: str, df: pd.DataFrame) -> tuple[str, int]:
    text = _norm_text(name)
    score_bank = 0
    score_software = 0
    for t in ("bank", "statement", "bankstatement", "mt940", "swift", "iban", "cheque", "check"):
        if t in text:
            score_bank += 4
    for t in ("software", "ledger", "gl", "accounting", "erp", "book", "voucher", "journal"):
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
    if cols["credit"]:
        return _amount_series(df[cols["credit"]])
    if cols["debit"]:
        return -_amount_series(df[cols["debit"]])
    if cols["amount"]:
        return _amount_series(df[cols["amount"]])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _parse_dates(s: pd.Series) -> pd.Series:
    a = pd.to_datetime(s, errors="coerce", dayfirst=False, format="mixed")
    b = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    out = a if a.notna().sum() >= b.notna().sum() else b
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
    elif cols["credit"]:
        raw = _amount_series(out[cols["credit"]])
        out["__recon_direction"] = np.where(raw > 0, "credit", "unknown")
    elif cols["debit"]:
        raw = _amount_series(out[cols["debit"]])
        out["__recon_direction"] = np.where(raw > 0, "debit", "unknown")
    elif cols["amount"]:
        raw = _amount_series(out[cols["amount"]])
        out["__recon_direction"] = np.where(raw < 0, "debit", np.where(raw > 0, "credit", "unknown"))
    else:
        out["__recon_direction"] = "unknown"
    out["__recon_date"] = _parse_dates(out[cols["date"]]) if cols["date"] else pd.NaT
    out["__recon_ref"] = out[cols["reference"]].map(_norm_text) if cols["reference"] else ""
    out["__recon_desc"] = out[cols["description"]].map(_norm_text) if cols["description"] else ""
    return out, cols


def _candidate_score(bank: pd.Series, soft: pd.Series, tolerance: float, date_window: int) -> tuple[float, str, int | None]:
    ba, sa = bank["__recon_amount"], soft["__recon_amount"]
    if pd.isna(ba) or pd.isna(sa):
        return -1.0, "amount unavailable", None
    if abs(float(ba) - float(sa)) > tolerance:
        return -1.0, "amount mismatch", None

    bd, sd = bank["__recon_date"], soft["__recon_date"]
    date_diff = None
    if pd.notna(bd) and pd.notna(sd):
        date_diff = abs((bd - sd).days)
        if date_diff > date_window:
            return -1.0, "date outside matching window", date_diff

    ref = _reference_match(bank["__recon_ref"], soft["__recon_ref"])
    desc_sim = _text_similarity(bank["__recon_desc"], soft["__recon_desc"])
    bdir, sdir = bank["__recon_direction"], soft["__recon_direction"]
    same_direction = bdir != "unknown" and sdir != "unknown" and bdir == sdir
    opposite_direction = bdir != "unknown" and sdir != "unknown" and bdir != sdir

    # Safety rule: amount alone is NEVER enough evidence for a match.
    # A reference is strongest; otherwise description/date/direction must support it.
    identity = ref or desc_sim >= 0.82
    if not identity:
        return -1.0, "insufficient identity evidence", date_diff
    if opposite_direction and not (ref or (desc_sim >= 0.82 and date_diff is not None and date_diff <= 1)):
        return -1.0, "opposite direction without strong identity evidence", date_diff

    score = 0.50
    basis = ["same amount"]
    if ref:
        score += 0.40
        basis.append("reference")
    elif desc_sim >= 0.82:
        score += 0.18
        basis.append("description")
    if same_direction:
        score += 0.06
        basis.append("same debit/credit direction")
    elif opposite_direction:
        basis.append("opposite debit/credit convention")
    if date_diff is not None:
        score += 0.08 * (1.0 - min(date_diff / max(date_window, 1), 1.0))
        basis.append(f"date ±{int(date_diff)}d")
    return min(score, 1.0), ", ".join(basis), date_diff


def _amount_key(value: float, tolerance: float) -> int:
    # Integer bucket for fast candidate lookup. Callers must still verify the exact
    # absolute difference against ``tolerance``. Neighboring buckets are considered
    # where a boundary can otherwise hide a valid near-tolerance match.
    return int(np.floor(float(value) / max(tolerance, 0.01)))


def _amount_bucket_candidates(value: float, tolerance: float) -> tuple[int, ...]:
    key = _amount_key(value, tolerance)
    return (key - 1, key, key + 1)


def reconcile_tables(
    bank_df: pd.DataFrame,
    software_df: pd.DataFrame,
    bank_name: str = "Bank",
    software_name: str = "Software",
    tolerance: float = 0.01,
    date_window: int = 3,
) -> ReconciliationResult:
    """Conservative, deterministic, one-to-one bank/software reconciliation.

    Safety principles:
    - Source serials are never used as a requirement and are never overwritten.
    - Amount is mandatory, but amount alone can never produce a match.
    - Exact unique references are the strongest identity signal.
    - Unique amount + exact date + known same direction is allowed when the
      direction fields are genuinely available on both sides.
    - Description-based matching requires strong identity evidence and a date
      check; ambiguous candidates are left unmatched.
    - Every source row can be consumed at most once.
    - No greedy fuzzy match is accepted when the best candidate is too close to
      another candidate.
    """
    if tolerance < 0:
        raise ValueError("Amount tolerance cannot be negative.")
    if date_window < 0:
        raise ValueError("Date matching window cannot be negative.")

    bank, bcols = _prepare(bank_df)
    soft, scols = _prepare(software_df)
    if bank["__recon_amount"].notna().sum() == 0 or soft["__recon_amount"].notna().sum() == 0:
        raise ValueError(
            "Could not identify a usable amount/debit-credit field in both files. "
            "Reconciliation was stopped to avoid false matches."
        )

    used_b: set[int] = set()
    used_s: set[int] = set()
    matches: list[tuple[int, int, int, float, str, int | None]] = []
    next_recon_id = 1

    def pair_evidence(bi: int, si: int, allow_unique_signature: bool = False):
        """Validate a proposed pair and return (score, basis, date_diff), or None."""
        ba = bank.at[bi, "__recon_amount"]
        sa = soft.at[si, "__recon_amount"]
        if pd.isna(ba) or pd.isna(sa) or abs(float(ba) - float(sa)) > tolerance:
            return None

        bd = bank.at[bi, "__recon_date"]
        sd = soft.at[si, "__recon_date"]
        date_diff = None
        if pd.notna(bd) and pd.notna(sd):
            date_diff = abs(int((pd.Timestamp(bd) - pd.Timestamp(sd)).days))
            if date_diff > date_window:
                return None

        bref = str(bank.at[bi, "__recon_ref"])
        sref = str(soft.at[si, "__recon_ref"])
        bdesc = str(bank.at[bi, "__recon_desc"])
        sdesc = str(soft.at[si, "__recon_desc"])
        ref = _reference_match(bref, sref)
        desc_sim = _text_similarity(bdesc, sdesc)
        bdir = bank.at[bi, "__recon_direction"]
        sdir = soft.at[si, "__recon_direction"]
        known_direction = bdir != "unknown" and sdir != "unknown"
        same_direction = known_direction and bdir == sdir
        opposite_direction = known_direction and bdir != sdir

        # Amount-only evidence is never sufficient.
        strong_desc = desc_sim >= 0.88
        unique_signature = allow_unique_signature and date_diff == 0 and same_direction
        if not ref and not strong_desc and not unique_signature:
            return None

        # A known debit/credit conflict is accepted only with an exact reference,
        # or an exceptionally strong description match on the same day. This
        # handles systems whose debit/credit convention is reversed.
        if opposite_direction and not (ref or (desc_sim >= 0.94 and date_diff == 0)):
            return None

        score = 0.50
        basis = ["same amount"]
        if ref:
            score += 0.40
            basis.append("exact/strong reference")
        elif strong_desc:
            score += 0.18
            basis.append("strong description identity")
        elif unique_signature:
            score += 0.12
            basis.append("unique same amount + exact date + direction")

        if same_direction:
            score += 0.06
            basis.append("same debit/credit direction")
        elif opposite_direction:
            basis.append("opposite debit/credit convention")

        if date_diff is not None:
            score += 0.08 * (1.0 - min(date_diff / max(date_window, 1), 1.0))
            basis.append(f"date ±{date_diff}d")

        return min(score, 1.0), ", ".join(basis), date_diff

    def commit(bi: int, si: int, evidence) -> None:
        nonlocal next_recon_id
        if bi in used_b or si in used_s:
            return
        score, basis, date_diff = evidence
        used_b.add(bi)
        used_s.add(si)
        matches.append((next_recon_id, bi, si, score, basis, date_diff))
        next_recon_id += 1

    # ------------------------------------------------------------------
    # Pass 1: exact normalized reference, unique on both sides.
    # ------------------------------------------------------------------
    bank_refs = bank["__recon_ref"]
    soft_refs = soft["__recon_ref"]
    b_ref_counts = bank_refs[bank_refs.astype(bool)].value_counts()
    s_ref_counts = soft_refs[soft_refs.astype(bool)].value_counts()
    soft_ref_map = {
        ref: int(idx)
        for idx, ref in soft_refs.items()
        if ref and s_ref_counts.get(ref, 0) == 1
    }
    for bi, ref in bank_refs.items():
        if not ref or b_ref_counts.get(ref, 0) != 1:
            continue
        si = soft_ref_map.get(ref)
        if si is None:
            continue
        evidence = pair_evidence(int(bi), si)
        if evidence is not None:
            commit(int(bi), si, evidence)

    # ------------------------------------------------------------------
    # Pass 2: unique exact transaction signature.
    # This is the only pass that can match without reference/description.
    # It requires amount + exact date + known same direction, unique on both sides.
    # ------------------------------------------------------------------
    if (bcols["debit"] or bcols["credit"]) and (scols["debit"] or scols["credit"]):
        remaining_b = bank.loc[~bank.index.isin(used_b)].copy()
        remaining_s = soft.loc[~soft.index.isin(used_s)].copy()
        rb = remaining_b[remaining_b["__recon_amount"].notna() & remaining_b["__recon_date"].notna()]
        rs = remaining_s[remaining_s["__recon_amount"].notna() & remaining_s["__recon_date"].notna()]
        if not rb.empty and not rs.empty:
            rs_map: dict[tuple[str, str, int], list[int]] = {}
            for si, row in rs.iterrows():
                key_prefix = (row["__recon_date"].strftime("%Y-%m-%d"), row["__recon_direction"])
                for bucket in _amount_bucket_candidates(float(row["__recon_amount"]), tolerance):
                    rs_map.setdefault((*key_prefix, bucket), []).append(int(si))
            for bi, row in rb.iterrows():
                key_prefix = (row["__recon_date"].strftime("%Y-%m-%d"), row["__recon_direction"])
                candidates = []
                for bucket in _amount_bucket_candidates(float(row["__recon_amount"]), tolerance):
                    candidates.extend(rs_map.get((*key_prefix, bucket), []))
                candidates = list(dict.fromkeys(candidates))
                valid = []
                for si in candidates:
                    if si in used_s:
                        continue
                    evidence = pair_evidence(int(bi), si, allow_unique_signature=True)
                    if evidence is not None:
                        valid.append((si, evidence))
                # Exact signature matching must remain unique. If two software rows
                # satisfy the signature within tolerance, do not guess.
                if len(valid) == 1:
                    si, evidence = valid[0]
                    commit(int(bi), si, evidence)

    # ------------------------------------------------------------------
    # Pass 3: exact normalized description + amount, unique on both sides.
    # Description is only an identity signal here; the date window and amount
    # are still mandatory. This pass deliberately does not use fuzzy matching.
    # ------------------------------------------------------------------
    remaining_b = bank.loc[~bank.index.isin(used_b)].copy()
    remaining_s = soft.loc[~soft.index.isin(used_s)].copy()
    rb = remaining_b[remaining_b["__recon_desc"].astype(bool) & remaining_b["__recon_amount"].notna()]
    rs = remaining_s[remaining_s["__recon_desc"].astype(bool) & remaining_s["__recon_amount"].notna()]
    if not rb.empty and not rs.empty:
        rs_map: dict[tuple[str, int], list[int]] = {}
        for si, row in rs.iterrows():
            desc = row["__recon_desc"]
            for bucket in _amount_bucket_candidates(float(row["__recon_amount"]), tolerance):
                rs_map.setdefault((desc, bucket), []).append(int(si))
        for bi, row in rb.iterrows():
            desc = row["__recon_desc"]
            candidates = []
            for bucket in _amount_bucket_candidates(float(row["__recon_amount"]), tolerance):
                candidates.extend(rs_map.get((desc, bucket), []))
            candidates = list(dict.fromkeys(candidates))
            valid = []
            for si in candidates:
                if si in used_s:
                    continue
                evidence = pair_evidence(int(bi), si)
                if evidence is not None:
                    valid.append((si, evidence))
            if len(valid) == 1:
                si, evidence = valid[0]
                commit(int(bi), si, evidence)

    # ------------------------------------------------------------------
    # Pass 4: small-file fuzzy fallback.
    # A fuzzy match is accepted only when it has a strong score AND a clear
    # margin over the next candidate. No candidate means no match.
    # ------------------------------------------------------------------
    remaining_b = bank.loc[~bank.index.isin(used_b)].copy()
    remaining_s = soft.loc[~soft.index.isin(used_s)].copy()
    if len(remaining_b) <= 50_000 and len(remaining_s) <= 50_000:
        amount_index: dict[int, list[int]] = {}
        for si, row in remaining_s.iterrows():
            amount = row["__recon_amount"]
            if pd.notna(amount):
                amount_index.setdefault(_amount_key(float(amount), tolerance), []).append(int(si))

        proposals: list[tuple[float, float, int, int, str, int | None]] = []
        for bi, br in remaining_b.iterrows():
            amount = br["__recon_amount"]
            if pd.isna(amount):
                continue
            candidates = []
            for si in amount_index.get(_amount_key(float(amount), tolerance), []):
                if si in used_s:
                    continue
                evidence = _candidate_score(br, soft.loc[si], tolerance, date_window)
                score, basis, date_diff = evidence
                if score >= 0:
                    candidates.append((score, basis, date_diff, si))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (-x[0], x[2] if x[2] is not None else 9999, x[3]))
            best = candidates[0]
            second_score = candidates[1][0] if len(candidates) > 1 else -1.0
            # Require a meaningful score and a material margin. If candidates are
            # nearly tied, the correct professional result is UNMATCHED.
            if best[0] < 0.88:
                continue
            if len(candidates) > 1 and (best[0] - second_score) < 0.10:
                continue
            proposals.append((best[0], second_score, int(bi), int(best[3]), best[1], best[2]))

        # Resolve proposals globally by strongest evidence first. A later row is
        # never allowed to steal a source row from an already committed match.
        proposals.sort(key=lambda x: (-x[0], -x[2], x[3]))
        for score, _, bi, si, basis, date_diff in proposals:
            if bi in used_b or si in used_s:
                continue
            commit(bi, si, (score, basis, date_diff))

    # ------------------------------------------------------------------
    # Build outputs while preserving every original source column/value.
    # ------------------------------------------------------------------
    bank_internal = [c for c in bank.columns if not str(c).startswith("__recon_")]
    soft_internal = [c for c in soft.columns if not str(c).startswith("__recon_")]

    if matches:
        match_b_idx = [m[1] for m in matches]
        match_s_idx = [m[2] for m in matches]
        b_out = bank.loc[match_b_idx, bank_internal].reset_index(drop=True).add_prefix("Bank_")
        s_out = soft.loc[match_s_idx, soft_internal].reset_index(drop=True).add_prefix("Software_")
        meta = pd.DataFrame({
            "Reconciliation Serial No": [f"REC-{m[0]:06d}" for m in matches],
            "Match Status": ["MATCHED"] * len(matches),
            "Match Score": [round(m[3] * 100, 2) for m in matches],
            "Match Basis": [m[4] for m in matches],
            "Date Difference (days)": [m[5] for m in matches],
        })
        matched = pd.concat([meta, b_out, s_out], axis=1)
    else:
        matched = pd.DataFrame(columns=[
            "Reconciliation Serial No", "Match Status", "Match Score",
            "Match Basis", "Date Difference (days)"
        ])

    # Unmatched output keeps the original serial/reference fields verbatim under
    # source-prefixed column names. No artificial serial is created for an
    # unmatched source transaction because that could be mistaken for a source ID.
    unmatched_parts = []
    bank_unmatched_idx = [i for i in bank.index if i not in used_b]
    soft_unmatched_idx = [i for i in soft.index if i not in used_s]
    if bank_unmatched_idx:
        b_out = bank.loc[bank_unmatched_idx, bank_internal].reset_index(drop=True).add_prefix("Bank_")
        meta = pd.DataFrame({
            "Reconciliation Serial No": [pd.NA] * len(bank_unmatched_idx),
            "Side": ["Bank only"] * len(bank_unmatched_idx),
            "Match Status": ["UNMATCHED"] * len(bank_unmatched_idx),
            "Unmatched Reason": ["No safe counterpart identified"] * len(bank_unmatched_idx),
        })
        unmatched_parts.append(pd.concat([meta, b_out], axis=1))
    if soft_unmatched_idx:
        s_out = soft.loc[soft_unmatched_idx, soft_internal].reset_index(drop=True).add_prefix("Software_")
        meta = pd.DataFrame({
            "Reconciliation Serial No": [pd.NA] * len(soft_unmatched_idx),
            "Side": ["Software only"] * len(soft_unmatched_idx),
            "Match Status": ["UNMATCHED"] * len(soft_unmatched_idx),
            "Unmatched Reason": ["No safe counterpart identified"] * len(soft_unmatched_idx),
        })
        unmatched_parts.append(pd.concat([meta, s_out], axis=1))
    unmatched = (
        pd.concat(unmatched_parts, ignore_index=True)
        if unmatched_parts
        else pd.DataFrame(columns=["Reconciliation Serial No", "Side", "Match Status", "Unmatched Reason"])
    )

    summary = {
        "bank_rows": len(bank_df),
        "software_rows": len(software_df),
        "matched_rows": len(matched),
        "bank_unmatched": int((~bank.index.isin(used_b)).sum()),
        "software_unmatched": int((~soft.index.isin(used_s)).sum()),
        "bank_columns": bcols,
        "software_columns": scols,
        "tolerance": tolerance,
        "date_window_days": date_window,
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
    if not banks or not softs:
        return None
    # Require a clear classification margin so Book1/Book2 cannot be silently guessed.
    if banks[0][1] < 4 or softs[0][1] < 3:
        return None
    if len(banks) > 1 and banks[0][1] == banks[1][1]:
        return None
    if len(softs) > 1 and softs[0][1] == softs[1][1]:
        return None
    return (banks[0][2], banks[0][3]), (softs[0][2], softs[0][3])


def _style_excel(writer):
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for col in ws.columns:
            width = min(max(len(str(cell.value or "")) for cell in col) + 2, 42)
            ws.column_dimensions[col[0].column_letter].width = width


def reconciliation_excel(result: ReconciliationResult, which: str = "matched") -> bytes:
    """Return a clean, separate Excel workbook for matched or unmatched rows.

    Excel has a hard worksheet limit, so callers should fall back to CSV for larger
    results instead of silently truncating transactions.
    """
    max_rows = 1_048_575  # header consumes one row
    which = which.lower().strip()
    if which not in {"matched", "unmatched"}:
        raise ValueError("which must be 'matched' or 'unmatched'")
    df = result.matched if which == "matched" else result.unmatched
    if len(df) > max_rows:
        raise ValueError("This result exceeds Excel's worksheet row limit. Use the CSV download for the complete result.")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        summary = pd.DataFrame({
            "Metric": ["Bank file", "Software file", "Bank rows", "Software rows", "Matched", "Bank unmatched", "Software unmatched", "Amount tolerance", "Date matching window (days)"],
            "Value": [result.bank_name, result.software_name, result.summary["bank_rows"], result.summary["software_rows"], result.summary["matched_rows"], result.summary["bank_unmatched"], result.summary["software_unmatched"], result.summary["tolerance"], result.summary["date_window_days"]],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name=which.title(), index=False)
        _style_excel(writer)
    return bio.getvalue()
