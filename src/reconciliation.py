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
    """Find a column conservatively.

    Exact normalized names win. Fuzzy fallback is token based, never arbitrary
    substring matching; this prevents fields such as ``Address`` being mistaken
    for ``Debit`` merely because they contain the letters ``dr``.
    """
    cols = list(df.columns)
    normalized = {_norm_text(c).replace(" ", ""): c for c in cols}
    alias_keys = [_norm_text(a).replace(" ", "") for a in aliases]
    for key in alias_keys:
        if key in normalized:
            return normalized[key]

    def tokens(value: str) -> set[str]:
        return set(_norm_text(value).split())

    alias_tokens = [(k, tokens(a)) for k, a in zip(alias_keys, aliases)]
    for c in cols:
        ct = tokens(str(c))
        compact = _norm_text(c).replace(" ", "")
        for key, at in alias_tokens:
            if key in {"dr", "cr"}:
                if key in ct:
                    return c
            elif at and at.issubset(ct):
                return c
            elif len(key) >= 5 and key == compact:
                return c
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    return {
        "date": _find_col(df, ("date", "transaction date", "value date", "posting date", "entry date", "tran date", "trans date")),
        "reference": _find_col(df, ("reference", "ref", "ref no", "refno", "reference no", "transaction id", "transaction no", "txn id", "cheque no", "check no", "document no", "journal no")),
        "serial": _find_col(df, ("serial no", "serial number", "serial", "sr no", "sr#", "s no", "entry no", "entry number", "entry id", "record no", "record number", "line no", "line number", "voucher", "voucher no", "voucher number", "voucher id", "voucher code", "document no", "document number", "transaction no", "transaction number", "transaction id")),
        "description": _find_col(df, ("description", "narration", "remarks", "particulars", "details", "memo", "transaction details", "payee", "beneficiary")),
        "debit": _find_col(df, ("debit", "withdrawal", "withdrawals", "dr", "debit amount")),
        "credit": _find_col(df, ("credit", "deposit", "deposits", "cr", "credit amount")),
        "amount": _find_col(df, ("amount", "transaction amount", "value", "net amount", "total")),
        "balance": _find_col(df, ("balance", "running balance", "closing balance", "available balance")),
    }


def _schema_profile(name: str, df: pd.DataFrame) -> dict:
    """Describe transaction-table evidence without assuming source serial equality."""
    cols = detect_columns(df)
    normalized = {_norm_text(c) for c in df.columns}
    joined = " ".join(sorted(normalized))
    profile = {
        "date": bool(cols["date"]),
        "amount": bool(cols["amount"] or cols["debit"] or cols["credit"]),
        "direction": bool(cols["debit"] or cols["credit"]),
        "balance": bool(cols["balance"]),
        "reference": bool(cols["reference"]),
        "serial": bool(cols["serial"]),
        "description": bool(cols["description"]),
        "account": any(x in joined for x in ("account", "account title", "ledger", "gl code", "gl name")),
        "voucher": any(x in joined for x in ("voucher", "voucher no", "voucher number", "voucher id")),
        "journal": any(x in joined for x in ("journal", "journal no", "journal number")),
        "statement": any(x in joined for x in ("statement", "bank statement")),
        "bank_terms": any(x in joined for x in ("withdrawal", "deposit", "cheque", "check", "running balance", "available balance", "value date", "posting date")),
    }
    profile["transaction_ready"] = profile["date"] and profile["amount"]
    return profile


def classify_ledger(name: str, df: pd.DataFrame) -> tuple[str, int]:
    """Classify a table using filename, schema and transaction-field evidence."""
    if not isinstance(df, pd.DataFrame):
        return "unknown", 0
    text = _norm_text(name)
    p = _schema_profile(name, df)
    score_bank = 0
    score_software = 0

    for t in ("bank", "statement", "bankstatement", "mt940", "swift", "iban", "cheque", "check"):
        if t in text:
            score_bank += 6
    for t in ("software", "ledger", "gl", "accounting", "erp", "book", "voucher", "journal"):
        if t in text:
            score_software += 5

    # Core transaction schema is useful evidence, but is not itself enough to pick
    # a side. This lets generic Sheet1/Sheet2 work when their fields are informative.
    if p["transaction_ready"]:
        score_bank += 2
        score_software += 2
    if p["date"]: score_bank += 2; score_software += 2
    if p["direction"]: score_bank += 2; score_software += 2
    if p["balance"]: score_bank += 9
    if p["statement"]: score_bank += 8
    if p["bank_terms"]: score_bank += 6
    if p["reference"]: score_bank += 1; score_software += 1
    if p["description"]: score_bank += 1; score_software += 2
    if p["account"]: score_software += 8
    if p["voucher"]: score_software += 9
    if p["journal"]: score_software += 8

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
    # Exact normalized references only. Partial-reference matches are too risky
    # for financial reconciliation because prefixes/suffixes can collide.
    return bool(a and b and a == b)


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
    # Integer bucket for indexed lookup. Candidate retrieval checks neighboring
    # buckets as well, so values near a bucket boundary are never missed.
    return int(np.floor(float(value) / max(tolerance, 0.01)))


def reconcile_tables(
    bank_df: pd.DataFrame,
    software_df: pd.DataFrame,
    bank_name: str = "Bank",
    software_name: str = "Software",
    tolerance: float = 0.01,
    date_window: int = 3,
) -> ReconciliationResult:
    """Conservative one-to-one reconciliation designed for large ledgers.

    Fast paths are O(n) dictionary/index lookups. Source serial/reference values do
    not have to be equal. Amount is required; a unique same amount + date + explicit
    debit/credit direction can establish a safe match, while reference/description
    provide stronger identity evidence. Ambiguous rows remain unmatched instead of guessed.
    """
    bank, bcols = _prepare(bank_df)
    soft, scols = _prepare(software_df)
    if bank["__recon_amount"].notna().sum() == 0 or soft["__recon_amount"].notna().sum() == 0:
        raise ValueError("Could not identify a usable amount/debit-credit field in both files. Reconciliation was stopped to avoid false matches.")

    used_b: set[int] = set()
    used_s: set[int] = set()
    matches: list[tuple[int, int, int, float, str, int | None]] = []
    serial = 1
    signature_direction_available = bool((bcols["debit"] or bcols["credit"]) and (scols["debit"] or scols["credit"]))

    b_amt = bank["__recon_amount"].to_numpy()
    s_amt = soft["__recon_amount"].to_numpy()
    b_date = bank["__recon_date"].to_numpy()
    s_date = soft["__recon_date"].to_numpy()
    b_dir = bank["__recon_direction"].to_numpy()
    s_dir = soft["__recon_direction"].to_numpy()
    b_ref = bank["__recon_ref"].to_numpy()
    s_ref = soft["__recon_ref"].to_numpy()
    b_desc = bank["__recon_desc"].to_numpy()
    s_desc = soft["__recon_desc"].to_numpy()

    def try_pair(bi: int, si: int, identity_basis: str, allow_unique_signature: bool = False) -> bool:
        nonlocal serial
        if bi in used_b or si in used_s:
            return False
        ba, sa = b_amt[bi], s_amt[si]
        if pd.isna(ba) or pd.isna(sa) or abs(float(ba) - float(sa)) > tolerance:
            return False
        dd = None
        if not pd.isna(b_date[bi]) and not pd.isna(s_date[si]):
            dd = abs(int((pd.Timestamp(b_date[bi]) - pd.Timestamp(s_date[si])).days))
            if dd > date_window:
                return False
        ref = _reference_match(str(b_ref[bi]), str(s_ref[si]))
        desc_sim = _text_similarity(str(b_desc[bi]), str(s_desc[si]))
        same_direction = b_dir[bi] != "unknown" and s_dir[si] != "unknown" and b_dir[bi] == s_dir[si]
        if not ref and desc_sim < 0.82 and not (allow_unique_signature and signature_direction_available and dd == 0 and same_direction):
            return False
        opposite = b_dir[bi] != "unknown" and s_dir[si] != "unknown" and b_dir[bi] != s_dir[si]
        if opposite and not (ref or (desc_sim >= 0.82 and dd is not None and dd <= 1)):
            return False
        score = 0.50 + (0.40 if ref else (0.18 if desc_sim >= 0.82 else 0.12))
        basis = ["same amount"]
        if ref:
            basis.append("reference")
        elif desc_sim >= 0.82:
            basis.append("description")
        elif allow_unique_signature and signature_direction_available and dd == 0 and same_direction:
            basis.append("unique same amount + date + direction")
        if same_direction:
            score += 0.06; basis.append("same debit/credit direction")
        elif opposite:
            basis.append("opposite debit/credit convention")
        if dd is not None:
            score += 0.08 * (1.0 - min(dd / max(date_window, 1), 1.0)); basis.append(f"date ±{dd}d")
        used_b.add(bi); used_s.add(si)
        matches.append((serial, bi, si, min(score, 1.0), ", ".join(basis) or identity_basis, dd))
        serial += 1
        return True

    def _nearby_bucket_items(buckets: dict[int, list[int]], value: float) -> list[int]:
        if pd.isna(value):
            return []
        key = _amount_key(float(value), tolerance)
        out: list[int] = []
        for k in (key - 1, key, key + 1):
            out.extend(buckets.get(k, []))
        return out

    def _bucket_candidates(value: float) -> list[int]:
        """Return nearby software amount candidates without a Cartesian scan."""
        return _nearby_bucket_items(amount_buckets, value)

    # Fast path 1: exact normalized reference, unique on both sides.
    # Amount and date are still mandatory safety checks.
    soft_ref_counts = soft.loc[soft["__recon_ref"].astype(bool), "__recon_ref"].value_counts()
    bank_ref_counts = bank.loc[bank["__recon_ref"].astype(bool), "__recon_ref"].value_counts()
    soft_ref_map = {
        ref: int(idx)
        for idx, ref in soft.loc[soft["__recon_ref"].astype(bool), "__recon_ref"].items()
        if soft_ref_counts.get(ref, 0) == 1
    }
    for bi, ref in bank.loc[bank["__recon_ref"].astype(bool), "__recon_ref"].items():
        if bank_ref_counts.get(ref, 0) != 1:
            continue
        si = soft_ref_map.get(ref)
        if si is not None:
            try_pair(int(bi), si, "exact reference")

    # Build amount buckets once. Every later stage uses these indexed candidates.
    amount_buckets: dict[int, list[int]] = {}
    bank_amount_buckets: dict[int, list[int]] = {}
    for si, amount in enumerate(s_amt):
        if not pd.isna(amount):
            amount_buckets.setdefault(_amount_key(float(amount), tolerance), []).append(si)
    for bi, amount in enumerate(b_amt):
        if not pd.isna(amount):
            bank_amount_buckets.setdefault(_amount_key(float(amount), tolerance), []).append(bi)

    # Generic indexed candidate generator. It is deliberately bounded by amount,
    # so large files never perform a Cartesian bank×software comparison.
    def valid_candidates(bi: int, *, require_date: bool = False, require_direction: bool = False,
                         require_description: bool = False) -> list[int]:
        if bi in used_b or pd.isna(b_amt[bi]):
            return []
        candidates = []
        for si in _bucket_candidates(b_amt[bi]):
            if si in used_s:
                continue
            if pd.isna(s_amt[si]) or abs(float(b_amt[bi]) - float(s_amt[si])) > tolerance:
                continue
            bd, sd = b_date[bi], s_date[si]
            dd = None
            if not pd.isna(bd) and not pd.isna(sd):
                dd = abs(int((pd.Timestamp(bd) - pd.Timestamp(sd)).days))
                if dd > date_window:
                    continue
            elif require_date:
                continue
            if require_direction and not (
                b_dir[bi] != "unknown" and s_dir[si] != "unknown" and b_dir[bi] == s_dir[si]
            ):
                continue
            if require_description and not (
                b_desc[bi] and s_desc[si] and b_desc[bi] == s_desc[si]
            ):
                continue
            candidates.append(si)
        return candidates

    # Fast path 2: unique amount bucket + same calendar date + same explicit direction.
    # Build composite indexes once. This is O(n) and avoids scanning every duplicate
    # amount candidate for every row.
    def _date_key(value):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).date()

    soft_sig: dict[tuple[int, object, str], list[int]] = {}
    bank_sig: dict[tuple[int, object, str], list[int]] = {}
    for si, amount in enumerate(s_amt):
        dk = _date_key(s_date[si])
        direction = s_dir[si]
        if pd.isna(amount) or dk is None or direction == "unknown":
            continue
        soft_sig.setdefault((_amount_key(float(amount), tolerance), dk, direction), []).append(si)
    for bi, amount in enumerate(b_amt):
        dk = _date_key(b_date[bi])
        direction = b_dir[bi]
        if pd.isna(amount) or dk is None or direction == "unknown":
            continue
        bank_sig.setdefault((_amount_key(float(amount), tolerance), dk, direction), []).append(bi)

    def _indexed_unique(index: dict, amount: float, date_value, direction: str, used: set[int], other_amount: float | None = None):
        if pd.isna(amount) or pd.isna(date_value) or direction == "unknown":
            return None
        dk = _date_key(date_value)
        key = _amount_key(float(amount), tolerance)
        found = None
        count = 0
        for bucket in (key - 1, key, key + 1):
            for idx in index.get((bucket, dk, direction), []):
                if idx in used:
                    continue
                if other_amount is not None and (pd.isna(other_amount) or abs(float(other_amount) - float(amount)) > tolerance):
                    continue
                count += 1
                if count > 1:
                    return None
                found = idx
        return found if count == 1 else None

    for bi in range(len(bank)):
        if bi in used_b or pd.isna(b_amt[bi]):
            continue
        si = _indexed_unique(soft_sig, b_amt[bi], b_date[bi], b_dir[bi], used_s)
        if si is None:
            continue
        # Reciprocal uniqueness is mandatory.
        bj = _indexed_unique(bank_sig, s_amt[si], s_date[si], s_dir[si], used_b)
        if bj == bi:
            try_pair(int(bi), int(si), "unique amount + date + direction", allow_unique_signature=True)

    # Fast path 3: exact normalized description + amount, unique on both sides.
    # Composite indexes keep duplicate-heavy ledgers linear instead of quadratic.
    soft_desc: dict[tuple[int, str], list[int]] = {}
    bank_desc: dict[tuple[int, str], list[int]] = {}
    for si, amount in enumerate(s_amt):
        desc = s_desc[si]
        if pd.isna(amount) or not desc:
            continue
        soft_desc.setdefault((_amount_key(float(amount), tolerance), desc), []).append(si)
    for bi, amount in enumerate(b_amt):
        desc = b_desc[bi]
        if pd.isna(amount) or not desc:
            continue
        bank_desc.setdefault((_amount_key(float(amount), tolerance), desc), []).append(bi)

    def _description_candidates(index: dict, amount: float, desc: str, used: set[int], date_value, ref_value, other_dates, other_refs, side: str):
        if pd.isna(amount) or not desc:
            return []
        key = _amount_key(float(amount), tolerance)
        found = []
        for bucket in (key - 1, key, key + 1):
            for idx in index.get((bucket, desc), []):
                if idx in used:
                    continue
                # Resolve the opposite row's arrays according to side.
                idx_amount = (s_amt[idx] if side == "bank" else b_amt[idx])
                if pd.isna(idx_amount) or abs(float(amount) - float(idx_amount)) > tolerance:
                    continue
                idx_date = (s_date[idx] if side == "bank" else b_date[idx])
                if pd.notna(date_value) and pd.notna(idx_date):
                    if abs(int((pd.Timestamp(date_value) - pd.Timestamp(idx_date)).days)) > date_window:
                        continue
                idx_ref = (s_ref[idx] if side == "bank" else b_ref[idx])
                if ref_value and idx_ref and ref_value != idx_ref:
                    continue
                found.append(idx)
                if len(found) > 1:
                    return found
        return found

    for bi in range(len(bank)):
        if bi in used_b or not b_desc[bi] or pd.isna(b_amt[bi]):
            continue
        candidates = _description_candidates(soft_desc, b_amt[bi], b_desc[bi], used_s, b_date[bi], b_ref[bi], s_date, s_ref, "bank")
        if len(candidates) != 1:
            continue
        si = candidates[0]
        reverse = _description_candidates(bank_desc, s_amt[si], s_desc[si], used_b, s_date[si], s_ref[si], b_date, b_ref, "software")
        if len(reverse) == 1 and reverse[0] == bi:
            try_pair(int(bi), int(si), "exact description + amount")

    # Final conservative fuzzy stage. Only small remaining ledgers are eligible.
    # It requires: same amount, same direction, date within 1 day, no contradictory
    # non-empty references, a high description similarity, and reciprocal uniqueness.
    remaining_b = [i for i in range(len(bank)) if i not in used_b]
    remaining_s = [i for i in range(len(soft)) if i not in used_s]
    if len(remaining_b) <= 5_000 and len(remaining_s) <= 5_000:
        for bi in remaining_b:
            if bi in used_b or not b_desc[bi] or pd.isna(b_amt[bi]):
                continue
            candidates = []
            for si in _bucket_candidates(b_amt[bi]):
                if si in used_s or not s_desc[si] or pd.isna(s_amt[si]):
                    continue
                if abs(float(b_amt[bi]) - float(s_amt[si])) > tolerance:
                    continue
                if b_ref[bi] and s_ref[si] and b_ref[bi] != s_ref[si]:
                    continue
                if b_dir[bi] == "unknown" or s_dir[si] == "unknown" or b_dir[bi] != s_dir[si]:
                    continue
                if pd.isna(b_date[bi]) or pd.isna(s_date[si]):
                    continue
                dd = abs(int((pd.Timestamp(b_date[bi]) - pd.Timestamp(s_date[si])).days))
                if dd > 1:
                    continue
                sim = _text_similarity(b_desc[bi], s_desc[si])
                if sim >= 0.93:
                    candidates.append((sim, dd, si))
            candidates.sort(reverse=True)
            if not candidates:
                continue
            # One clear best candidate only.
            if len(candidates) > 1:
                if candidates[0][0] - candidates[1][0] < 0.08:
                    continue
            sim, dd, si = candidates[0]
            reverse = []
            for bj in _nearby_bucket_items(bank_amount_buckets, float(s_amt[si])):
                if bj in used_b or not b_desc[bj] or pd.isna(b_amt[bj]):
                    continue
                if abs(float(b_amt[bj]) - float(s_amt[si])) > tolerance:
                    continue
                if b_ref[bj] and s_ref[si] and b_ref[bj] != s_ref[si]:
                    continue
                if b_dir[bj] == "unknown" or s_dir[si] == "unknown" or b_dir[bj] != s_dir[si]:
                    continue
                if pd.isna(b_date[bj]) or pd.isna(s_date[si]):
                    continue
                if abs(int((pd.Timestamp(b_date[bj]) - pd.Timestamp(s_date[si])).days)) > 1:
                    continue
                rev_sim = _text_similarity(b_desc[bj], s_desc[si])
                if rev_sim >= 0.93:
                    reverse.append((rev_sim, bj))
            if len(reverse) == 1:
                try_pair(int(bi), int(si), "high-confidence description similarity")

    # Build output tables in vectorized pandas operations rather than one .loc call
    # per transaction. This matters for large reconciliations.
    bank_internal = [c for c in bank.columns if not str(c).startswith("__recon_")]
    soft_internal = [c for c in soft.columns if not str(c).startswith("__recon_")]
    if matches:
        match_b_idx = [m[1] for m in matches]
        match_s_idx = [m[2] for m in matches]
        b_out = bank.loc[match_b_idx, bank_internal].reset_index(drop=True).add_prefix("Bank_")
        s_out = soft.loc[match_s_idx, soft_internal].reset_index(drop=True).add_prefix("Software_")
        recon_ids = [f"REC-{m[0]:06d}" for m in matches]
        bank_original_serials = (
            bank.loc[match_b_idx, bcols["serial"]].reset_index(drop=True).tolist()
            if bcols["serial"] else [None] * len(matches)
        )
        software_original_serials = (
            soft.loc[match_s_idx, scols["serial"]].reset_index(drop=True).tolist()
            if scols["serial"] else [None] * len(matches)
        )
        meta = pd.DataFrame({
            # This is a NEW identifier created by DocuSphere for the matched pair.
            # It is intentionally repeated for both sides; original source serials
            # remain separate and untouched.
            "Reconciliation Serial No": recon_ids,
            "Bank Reconciliation Serial No": recon_ids,
            "Software Reconciliation Serial No": recon_ids,
            "Bank Original Serial No": bank_original_serials,
            "Software Original Serial No": software_original_serials,
            "Match Status": ["MATCHED"] * len(matches),
            "Match Score": [round(m[3] * 100, 2) for m in matches],
            "Match Basis": [m[4] for m in matches],
            "Date Difference (days)": [m[5] for m in matches],
        })
        matched = pd.concat([meta, b_out, s_out], axis=1)
    else:
        matched = pd.DataFrame(columns=[
            "Reconciliation Serial No", "Bank Reconciliation Serial No",
            "Software Reconciliation Serial No", "Bank Original Serial No",
            "Software Original Serial No", "Match Status", "Match Score",
            "Match Basis", "Date Difference (days)"
        ])

    bank_unmatched_idx = [i for i in bank.index if i not in used_b]
    soft_unmatched_idx = [i for i in soft.index if i not in used_s]
    unmatched_parts = []
    if bank_unmatched_idx:
        b_out = bank.loc[bank_unmatched_idx, bank_internal].reset_index(drop=True).add_prefix("Bank_")
        # Never invent or replace an original source serial. The unmatched report
        # must carry the exact original Bank serial when one exists.
        bank_original_serials = (
            bank.loc[bank_unmatched_idx, bcols["serial"]].reset_index(drop=True).tolist()
            if bcols["serial"] else [None] * len(bank_unmatched_idx)
        )
        meta = pd.DataFrame({
            "Reconciliation Serial No": [None] * len(bank_unmatched_idx),
            "Original Serial No": bank_original_serials,
            "Side": ["Bank only"] * len(bank_unmatched_idx),
            "Match Status": ["UNMATCHED"] * len(bank_unmatched_idx),
            "Unmatched Reason": ["No safe counterpart identified"] * len(bank_unmatched_idx),
        })
        unmatched_parts.append(pd.concat([meta, b_out], axis=1))
    if soft_unmatched_idx:
        s_out = soft.loc[soft_unmatched_idx, soft_internal].reset_index(drop=True).add_prefix("Software_")
        software_original_serials = (
            soft.loc[soft_unmatched_idx, scols["serial"]].reset_index(drop=True).tolist()
            if scols["serial"] else [None] * len(soft_unmatched_idx)
        )
        meta = pd.DataFrame({
            "Reconciliation Serial No": [None] * len(soft_unmatched_idx),
            "Original Serial No": software_original_serials,
            "Side": ["Software only"] * len(soft_unmatched_idx),
            "Match Status": ["UNMATCHED"] * len(soft_unmatched_idx),
            "Unmatched Reason": ["No safe counterpart identified"] * len(soft_unmatched_idx),
        })
        unmatched_parts.append(pd.concat([meta, s_out], axis=1))
    unmatched = pd.concat(unmatched_parts, ignore_index=True) if unmatched_parts else pd.DataFrame(columns=["Reconciliation Serial No", "Original Serial No", "Side", "Match Status", "Unmatched Reason"])

    summary = {"bank_rows": len(bank_df), "software_rows": len(software_df), "matched_rows": len(matched), "bank_unmatched": int((~bank.index.isin(used_b)).sum()), "software_unmatched": int((~soft.index.isin(used_s)).sum()), "bank_columns": bcols, "software_columns": scols, "tolerance": tolerance, "date_window_days": date_window}
    return ReconciliationResult(bank_name, software_name, matched, unmatched, summary)


def looks_like_reconciliation_request(query: str, frames: dict) -> bool:
    q = _norm_text(query)
    terms = ("reconcile", "reconciliation", "bank reconciliation", "bank ledger", "bank statement", "software ledger", "match bank", "matching transactions", "matched transactions", "unmatched transactions")
    if any(t.replace(" ", "") in q.replace(" ", "") for t in terms):
        return len(frames) >= 2
    labels = " ".join(_norm_text(k) for k in frames)
    return len(frames) >= 2 and ("bank" in labels and ("ledger" in labels or "software" in labels))


def auto_select_two_ledgers(frames: dict, allow_explicit_pair: bool = False) -> tuple[tuple[str, pd.DataFrame], tuple[str, pd.DataFrame]] | None:
    """Select exactly one bank/statement table and one software ledger table.

    A common real-world case is ONE XLSX containing two sheets named Sheet1/Sheet2.
    Filename-only classification cannot identify those sheets, so this function
    also scores the actual column schema. For an explicit reconciliation request
    with exactly two tabular sources, the two tables are a deterministic pair;
    role labels are assigned from the strongest schema evidence.
    """
    if not isinstance(frames, dict) or len(frames) < 2:
        return None
    # Only real DataFrames are eligible. Uploaded-file parsing failures or
    # unexpected session-state values must never crash the reconciliation path.
    items = [(str(name), df) for name, df in frames.items() if isinstance(df, pd.DataFrame)]
    if len(items) < 2:
        return None
    scored = []
    for name, df in items:
        role, score = classify_ledger(name, df)
        scored.append({"role": role, "score": score, "name": name, "df": df})

    banks = sorted([x for x in scored if x["role"] == "bank"], key=lambda x: -x["score"])
    softs = sorted([x for x in scored if x["role"] == "software"], key=lambda x: -x["score"])
    if banks and softs:
        # If multiple tables exist, require unique top candidates.
        if len(banks) > 1 and banks[0]["score"] == banks[1]["score"]:
            banks = []
        if len(softs) > 1 and softs[0]["score"] == softs[1]["score"]:
            softs = []
        if banks and softs and banks[0]["name"] != softs[0]["name"]:
            return (banks[0]["name"], banks[0]["df"]), (softs[0]["name"], softs[0]["df"])

    # Explicit reconciliation + exactly two tables: do not send the request to
    # Gemini just because the workbook used generic Sheet1/Sheet2 names. The
    # caller must explicitly opt into this deterministic pair fallback.
    # The reconciliation algorithm itself is symmetric; role assignment is based
    # on schema score, then filename/sheet-name hints, then stable input order.
    if len(items) == 2 and allow_explicit_pair:
        a, b = items
        role_a, score_a = classify_ledger(a[0], a[1])
        role_b, score_b = classify_ledger(b[0], b[1])
        if role_a == "bank" and role_b != "bank":
            return a, b
        if role_b == "bank" and role_a != "bank":
            return b, a
        if role_a == "software" and role_b != "software":
            return b, a
        if role_b == "software" and role_a != "software":
            return a, b
        if score_a != score_b:
            # Higher score is more likely the bank/statement only when its
            # features explicitly point to bank data; otherwise use software role.
            if score_a > score_b and role_a == "bank":
                return a, b
            if score_b > score_a and role_b == "bank":
                return b, a
        # Generic Sheet1/Sheet2 is common in real workbooks. If both tables have a
        # real transaction schema, use their field evidence to infer the roles rather
        # than failing just because the sheet names are generic. This is still a safety
        # gate: both sides must be transaction-ready and the inferred roles must differ.
        if role_a == role_b == "unknown":
            pa = _schema_profile(a[0], a[1])
            pb = _schema_profile(b[0], b[1])
            if pa["transaction_ready"] and pb["transaction_ready"]:
                # Bank statements commonly expose a running/closing balance or bank
                # transaction fields; software ledgers commonly expose account,
                # voucher or journal fields. Use these concrete schema signals.
                bank_a = int(pa["balance"])*8 + int(pa["bank_terms"])*5 + int(pa["statement"])*6
                bank_b = int(pb["balance"])*8 + int(pb["bank_terms"])*5 + int(pb["statement"])*6
                soft_a = int(pa["account"])*7 + int(pa["voucher"])*8 + int(pa["journal"])*6
                soft_b = int(pb["account"])*7 + int(pb["voucher"])*8 + int(pb["journal"])*6
                if bank_a > bank_b and soft_b > soft_a:
                    return a, b
                if bank_b > bank_a and soft_a > soft_b:
                    return b, a

        # If the source names/schemas genuinely do not provide enough evidence, do
        # not fabricate a role assignment. A professional financial tool must prefer
        # an explicit request for clarification over a false reconciliation.
        return None
    return None


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
