from __future__ import annotations

import io
import re
from typing import Optional

import pandas as pd

SERIAL_ALIASES = [
    "serial", "serial no", "serial number", "sr no", "sr. no", "s no", "s. no",
    "sl no", "sl. no", "voucher no", "voucher number", "voucher", "transaction id",
    "transaction no", "transaction number", "reference no", "ref no", "entry no",
    "document no", "doc no", "id", "entry id"
]
DATE_ALIASES = [
    "date", "transaction date", "transaction datetime", "txn date", "tran date", "posting date",
    "value date", "entry date", "effective date", "booking date", "document date"
]
AMOUNT_ALIASES = ["amount", "transaction amount", "txn amount", "value", "net amount", "total amount", "transaction value"]
DEBIT_ALIASES = ["debit", "dr", "withdrawal", "withdrawals", "debit amount", "paid out", "money out", "payment amount", "withdrawal amount"]
CREDIT_ALIASES = ["credit", "cr", "deposit", "deposits", "credit amount", "received", "money in", "receipt amount", "deposit amount"]
REF_ALIASES = [
    "reference", "reference no", "ref", "ref no", "transaction reference", "txn ref", "transaction id",
    "utr", "rrn", "trace no", "trace number", "cheque no", "check no", "instrument no", "bank reference"
]
DESC_ALIASES = [
    "description", "narration", "details", "particulars", "memo", "remarks", "transaction details",
    "purpose", "note", "payee", "beneficiary", "merchant", "transaction description"
]
DIRECTION_ALIASES = ["type", "transaction type", "dr/cr", "debit credit", "direction", "transaction direction"]


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def _find_col(df: pd.DataFrame, aliases: list[str], contains: bool = True) -> Optional[str]:
    normalized = {c: norm(c) for c in df.columns}
    alias_norm = [norm(a) for a in aliases]
    for a in alias_norm:
        for c, n in normalized.items():
            if n == a:
                return c
    if contains:
        # Conservative token matching prevents short aliases such as `type`
        # from accidentally matching `debit`, and `debit amount` from
        # accidentally resolving to a generic `amount` column.
        for a in alias_norm:
            at = set(a.split())
            if not at:
                continue
            for c, n in normalized.items():
                nt = set(n.split())
                if at.issubset(nt):
                    return c
    return None


def _numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    text = s.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
    text = text.str.replace(r"[^0-9.\-]+", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _date_series(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    # If the data clearly uses day-first dates, retry that interpretation.
    slash = text.str.contains(r"[/.-]", regex=True, na=False)
    parts = text.str.extract(r"^(\d{1,2})[/. -](\d{1,2})[/. -](\d{2,4})$")
    clear_day_first = pd.to_numeric(parts[0], errors="coerce").gt(12).fillna(False)
    if clear_day_first.any():
        alt = pd.to_datetime(text, errors="coerce", dayfirst=True)
        parsed.loc[clear_day_first & slash] = alt.loc[clear_day_first & slash]
    return parsed


def _clean_text(v) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v).strip()).lower()


def _identity_text(v) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(v))


def detect_schema(df: pd.DataFrame) -> dict:
    debit = _find_col(df, DEBIT_ALIASES)
    credit = _find_col(df, CREDIT_ALIASES)
    amount = _find_col(df, AMOUNT_ALIASES)
    date = _find_col(df, DATE_ALIASES)
    serial = _find_col(df, SERIAL_ALIASES)
    reference = _find_col(df, REF_ALIASES)
    description = _find_col(df, DESC_ALIASES)
    direction = _find_col(df, DIRECTION_ALIASES)
    scores = {"bank": 0, "software": 0}
    name_text = " ".join(norm(c) for c in df.columns)
    if debit or credit: scores["bank"] += 3; scores["software"] += 1
    if amount: scores["software"] += 2; scores["bank"] += 1
    if reference: scores["bank"] += 2; scores["software"] += 2
    if description: scores["bank"] += 1; scores["software"] += 2
    if any(x in name_text for x in ("bank", "statement", "withdrawal", "deposit")): scores["bank"] += 4
    if any(x in name_text for x in ("software", "voucher", "ledger", "invoice", "journal", "erp", "accounting")): scores["software"] += 4
    return {"serial": serial, "date": date, "amount": amount, "debit": debit, "credit": credit,
            "reference": reference, "description": description, "direction": direction, "role_scores": scores}


def _prepared(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dr = _numeric_series(df[schema["debit"]]) if schema["debit"] else pd.Series(float("nan"), index=df.index)
    cr = _numeric_series(df[schema["credit"]]) if schema["credit"] else pd.Series(float("nan"), index=df.index)
    amt = _numeric_series(df[schema["amount"]]) if schema["amount"] else pd.Series(float("nan"), index=df.index)
    signed = amt.copy()
    direction = pd.Series("", index=df.index, dtype="object")
    debit_mask = dr.notna() & dr.ne(0)
    credit_mask = ~debit_mask & cr.notna() & cr.ne(0)
    signed.loc[debit_mask] = -dr.loc[debit_mask].abs(); direction.loc[debit_mask] = "debit"
    signed.loc[credit_mask] = cr.loc[credit_mask].abs(); direction.loc[credit_mask] = "credit"
    if schema["direction"]:
        raw = df[schema["direction"]].map(_clean_text)
        debit_words = raw.str.contains(r"\b(?:dr|debit|withdrawal|payment|paid|paid out)\b", regex=True, na=False)
        credit_words = raw.str.contains(r"\b(?:cr|credit|deposit|receipt|received|money in)\b", regex=True, na=False)
        direction.loc[direction.eq("") & debit_words] = "debit"
        direction.loc[direction.eq("") & credit_words] = "credit"
        signed.loc[direction.eq("debit") & signed.notna()] = signed.loc[direction.eq("debit") & signed.notna()].abs() * -1
        signed.loc[direction.eq("credit") & signed.notna()] = signed.loc[direction.eq("credit") & signed.notna()].abs()
    out["_amount"] = signed.abs().round(2)
    out["_direction"] = direction
    out["_date"] = _date_series(df[schema["date"]]).dt.normalize() if schema["date"] else pd.NaT
    out["_ref"] = df[schema["reference"]].map(_identity_text) if schema["reference"] else ""
    out["_desc"] = df[schema["description"]].map(_identity_text) if schema["description"] else ""
    out["_row"] = range(len(df))
    return out.reset_index(drop=True)


def _date_compatible(bd, sd, tolerance: int) -> bool:
    if pd.isna(bd) or pd.isna(sd):
        return True
    return abs((bd - sd).days) <= tolerance


def _direction_compatible(bd: str, sd: str) -> bool:
    return not (bd and sd and bd != sd)


def _pair_allowed(brow: pd.Series, srow: pd.Series, tolerance: int) -> bool:
    if pd.isna(brow["_amount"]) or pd.isna(srow["_amount"]):
        return False
    if round(float(brow["_amount"]), 2) != round(float(srow["_amount"]), 2):
        return False
    if not _direction_compatible(brow["_direction"], srow["_direction"]):
        return False
    return _date_compatible(brow["_date"], srow["_date"], tolerance)


def _unique_pairs(bp: pd.DataFrame, sp: pd.DataFrame, keys: list[str], tolerance: int, matched: dict[int, int], used: set[int]):
    left = bp[~bp["_row"].isin(matched)].copy()
    right = sp[~sp["_row"].isin(used)].copy()
    if left.empty or right.empty:
        return
    for c in keys:
        if c == "_date":
            left = left[left[c].notna()]; right = right[right[c].notna()]
        else:
            left = left[left[c].astype(str).ne("")]; right = right[right[c].astype(str).ne("")]
    if left.empty or right.empty:
        return
    merged = left.merge(right, on=keys, suffixes=("_b", "_s"), how="inner", sort=False)
    if merged.empty:
        return

    # Vectorized safety checks. Do not use row-wise apply here: reconciliation
    # must remain practical for large ledgers.
    bd = merged.get("_direction_b", pd.Series("", index=merged.index))
    sd = merged.get("_direction_s", pd.Series("", index=merged.index))
    valid = ~(bd.ne("") & sd.ne("") & bd.ne(sd))
    if "_date_b" in merged.columns and "_date_s" in merged.columns:
        date_b = pd.to_datetime(merged["_date_b"], errors="coerce")
        date_s = pd.to_datetime(merged["_date_s"], errors="coerce")
        valid &= date_b.isna() | date_s.isna() | ((date_b - date_s).abs().dt.days <= tolerance)
    merged = merged[valid]
    if merged.empty:
        return

    counts_b = merged.groupby("_row_b")["_row_s"].transform("nunique")
    counts_s = merged.groupby("_row_s")["_row_b"].transform("nunique")
    pairs = merged[(counts_b == 1) & (counts_s == 1)].drop_duplicates(["_row_b", "_row_s"])
    for bi, si in zip(pairs["_row_b"].astype(int), pairs["_row_s"].astype(int)):
        if bi not in matched and si not in used:
            matched[bi] = si
            used.add(si)


def _unique_date_tolerance_pairs(bp: pd.DataFrame, sp: pd.DataFrame, tolerance: int, matched: dict[int, int], used: set[int]):
    if tolerance <= 0:
        return
    left = bp[(~bp["_row"].isin(matched)) & bp["_date"].notna()].copy()
    right = sp[(~sp["_row"].isin(used)) & sp["_date"].notna()].copy()
    if left.empty or right.empty:
        return
    left = left[left["_amount"].notna()]; right = right[right["_amount"].notna()]
    if left.empty or right.empty:
        return
    expanded = []
    for delta in range(-tolerance, tolerance + 1):
        x = right.copy()
        x["_join_date"] = x["_date"] + pd.Timedelta(days=delta)
        expanded.append(x)
    right = pd.concat(expanded, ignore_index=True)
    merged = left.merge(right, left_on=["_amount", "_date"], right_on=["_amount", "_join_date"],
                        suffixes=("_b", "_s"), how="inner", sort=False)
    if merged.empty:
        return
    bd = merged.get("_direction_b", pd.Series("", index=merged.index))
    sd = merged.get("_direction_s", pd.Series("", index=merged.index))
    valid = ~(bd.ne("") & sd.ne("") & bd.ne(sd))
    merged = merged[valid]
    if merged.empty:
        return
    counts_b = merged.groupby("_row_b")["_row_s"].transform("nunique")
    counts_s = merged.groupby("_row_s")["_row_b"].transform("nunique")
    pairs = merged[(counts_b == 1) & (counts_s == 1)].drop_duplicates(["_row_b", "_row_s"])
    for bi, si in zip(pairs["_row_b"].astype(int), pairs["_row_s"].astype(int)):
        if bi not in matched and si not in used:
            matched[bi] = si; used.add(si)


def reconcile(bank: pd.DataFrame, software: pd.DataFrame, bank_name="Bank", software_name="Software", date_tolerance_days: int = 1) -> dict:
    bank = bank.copy(); software = software.copy()
    bank.columns = [str(c).strip() for c in bank.columns]
    software.columns = [str(c).strip() for c in software.columns]
    bs, ss = detect_schema(bank), detect_schema(software)
    if bs["amount"] is None and not (bs["debit"] or bs["credit"]):
        raise ValueError("Bank ledger: no usable amount/debit/credit field was detected.")
    if ss["amount"] is None and not (ss["debit"] or ss["credit"]):
        raise ValueError("Software ledger: no usable amount/debit/credit field was detected.")
    bp, sp = _prepared(bank, bs), _prepared(software, ss)
    # Ignore non-transaction rows (blank/no amount) instead of exporting report titles/footers as transactions.
    bank_valid = bp["_amount"].notna(); soft_valid = sp["_amount"].notna()
    bp, sp = bp[bank_valid].copy(), sp[soft_valid].copy()
    matched: dict[int, int] = {}; used: set[int] = set()

    # Strongest evidence first. If both dates exist, every tier still enforces the date tolerance.
    _unique_pairs(bp, sp, ["_amount", "_date"], date_tolerance_days, matched, used)
    _unique_date_tolerance_pairs(bp, sp, date_tolerance_days, matched, used)
    _unique_pairs(bp, sp, ["_amount", "_ref"], date_tolerance_days, matched, used)
    _unique_pairs(bp, sp, ["_amount", "_desc"], date_tolerance_days, matched, used)

    # Build matched output with vectorized DataFrame operations rather than one
    # Python dict per row; this keeps large reconciliations responsive.
    pairs = sorted(matched.items())
    if pairs:
        bi = [p[0] for p in pairs]; si = [p[1] for p in pairs]
        mb = bank.iloc[bi].reset_index(drop=True).add_prefix("Bank_")
        ms = software.iloc[si].reset_index(drop=True).add_prefix("Software_")
        def serial_values(df, schema, indices, prefix):
            c = schema.get("serial")
            if c:
                vals = df.iloc[indices][c].reset_index(drop=True).astype(object)
                vals = vals.where(vals.notna() & vals.astype(str).str.strip().ne(""),
                                  pd.Series([f"{prefix}-{i + 1:06d}" for i in indices]))
                return vals.astype(str)
            return pd.Series([f"{prefix}-{i + 1:06d}" for i in indices])
        matched_df = pd.concat([
            pd.Series([f"REC-{n:06d}" for n in range(1, len(pairs) + 1)], name="Reconciliation_ID"),
            pd.Series(["MATCHED"] * len(pairs), name="Match_Status"),
            serial_values(bank, bs, bi, "BANK").rename("Bank_Original_Serial"),
            serial_values(software, ss, si, "SW").rename("Software_Original_Serial"),
            mb, ms
        ], axis=1)
    else:
        matched_df = pd.DataFrame(columns=["Reconciliation_ID", "Bank_Original_Serial", "Software_Original_Serial", "Match_Status"])

    matched_bank_idx = set(matched); matched_soft_idx = set(matched.values())
    bank_un = bank.loc[[i not in matched_bank_idx and bool(bank_valid.iloc[i]) for i in range(len(bank))]].copy()
    soft_un = software.loc[[i not in matched_soft_idx and bool(soft_valid.iloc[i]) for i in range(len(software))]].copy()
    bank_un.insert(0, "Source", bank_name); bank_un.insert(1, "Match_Status", "UNMATCHED")
    soft_un.insert(0, "Source", software_name); soft_un.insert(1, "Match_Status", "UNMATCHED")
    unmatched_df = pd.concat([bank_un, soft_un], ignore_index=True, sort=False)
    return {"matched": matched_df, "bank_unmatched": bank_un, "software_unmatched": soft_un, "unmatched": unmatched_df,
            "bank_schema": bs, "software_schema": ss, "bank_rows": len(bank_valid[bank_valid]), "software_rows": len(soft_valid[soft_valid]),
            "matched_rows": len(matched_df), "unmatched_rows": len(unmatched_df)}


EXCEL_MAX_ROWS = 1_048_576

def matched_excel_bytes(df: pd.DataFrame) -> bytes:
    if len(df) + 1 > EXCEL_MAX_ROWS:
        raise ValueError("Matched result exceeds Excel's 1,048,576-row worksheet limit; use CSV for the complete result.")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Matched")
    return bio.getvalue()


def unmatched_excel_bytes(bank_df: pd.DataFrame, software_df: pd.DataFrame) -> bytes:
    if max(len(bank_df), len(software_df), len(bank_df) + len(software_df)) + 1 > EXCEL_MAX_ROWS:
        raise ValueError("Unmatched result exceeds Excel's 1,048,576-row worksheet limit; use CSV for the complete result.")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        bank_df.to_excel(writer, index=False, sheet_name="Bank_Unmatched")
        software_df.to_excel(writer, index=False, sheet_name="Software_Unmatched")
        pd.concat([bank_df, software_df], ignore_index=True, sort=False).to_excel(writer, index=False, sheet_name="All_Unmatched")
    return bio.getvalue()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def classify_frames(frames: dict[str, pd.DataFrame]) -> tuple[Optional[tuple[str, pd.DataFrame]], Optional[tuple[str, pd.DataFrame]], list[str]]:
    ranked = []
    for label, df in frames.items():
        if df is None or df.empty or len(df.columns) < 1:
            continue
        schema = detect_schema(df)
        text = norm(label)
        bank_score, soft_score = schema["role_scores"]["bank"], schema["role_scores"]["software"]
        if any(w in text for w in ("bank", "statement", "banking")): bank_score += 8
        if any(w in text for w in ("software", "accounting", "ledger", "voucher", "erp", "journal")): soft_score += 8
        ranked.append((label, df, bank_score, soft_score))
    if len(ranked) < 2:
        return None, None, ["At least two usable transaction tables are required: one bank statement/ledger and one software ledger."]
    banks = sorted(ranked, key=lambda x: (-x[2], x[0].lower()))
    softs = sorted(ranked, key=lambda x: (-x[3], x[0].lower()))
    bank = banks[0]
    software = next((x for x in softs if x[0] != bank[0]), None)
    warnings = []
    if bank[2] < 4: warnings.append("Bank role could not be identified with strong evidence from the file name/columns.")
    if software is None or software[3] < 4: warnings.append("Software-ledger role could not be identified with strong evidence from the file name/columns.")
    if software is None: return None, None, warnings
    # If both roles are equally plausible, never silently guess.
    if bank[2] == software[2] and softs[0][3] == banks[0][3] and not any(k in norm(bank[0]) for k in ("bank", "statement")) and not any(k in norm(software[0]) for k in ("software", "accounting", "ledger", "voucher")):
        warnings.append("The two tables have similar schemas and generic names, so the bank/software roles are ambiguous.")
        return None, None, warnings
    return (bank[0], bank[1]), (software[0], software[1]), warnings
