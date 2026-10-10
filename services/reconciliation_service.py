def run_reconciliation(bank_df, ledger_df):
    bank_df = bank_df.copy()
    ledger_df = ledger_df.copy()
    counter = 0

    # Rule 1: Exact Reference + Amount + Direction
    for b_idx, bank in bank_df.iterrows():
        if bank_df.at[b_idx, "Status"] == "Matched":
            continue

        ref = str(bank["Reference"]).strip()
        if not ref:
            continue

        candidates = ledger_df[
            (ledger_df["Status"] == "Unmatched")
            & (ledger_df["Reference"].astype(str).str.strip() == ref)
            & (ledger_df["Amount"] == bank["Amount"])
            & (ledger_df["Direction"] == bank["Direction"])
        ]

        if len(candidates) != 1:
            continue

        l_idx = candidates.index[0]
        counter += 1
        match_id = f"REC-{counter:04d}"

        bank_df.at[b_idx, "Status"] = "Matched"
        bank_df.at[b_idx, "Match ID"] = match_id
        bank_df.at[b_idx, "Match Rule"] = "Exact Reference"

        ledger_df.at[l_idx, "Status"] = "Matched"
        ledger_df.at[l_idx, "Match ID"] = match_id
        ledger_df.at[l_idx, "Match Rule"] = "Exact Reference"

    # Rule 2: Exact Amount + Same Date (sirf unique candidate)
    for b_idx, bank in bank_df.iterrows():
        if bank_df.at[b_idx, "Status"] == "Matched":
            continue

        candidates = ledger_df[
            (ledger_df["Status"] == "Unmatched")
            & (ledger_df["Amount"] == bank["Amount"])
            & (ledger_df["Direction"] == bank["Direction"])
            & (ledger_df["Date"] == bank["Date"])
        ]

        if len(candidates) != 1:
            continue

        l_idx = candidates.index[0]
        counter += 1
        match_id = f"REC-{counter:04d}"

        bank_df.at[b_idx, "Status"] = "Matched"
        bank_df.at[b_idx, "Match ID"] = match_id
        bank_df.at[b_idx, "Match Rule"] = "Exact Amount + Date"

        ledger_df.at[l_idx, "Status"] = "Matched"
        ledger_df.at[l_idx, "Match ID"] = match_id
        ledger_df.at[l_idx, "Match Rule"] = "Exact Amount + Date"

    # Rule 3: Exact Amount + Date Window (3 din) — Suggested Match
    for b_idx, bank in bank_df.iterrows():
        if bank_df.at[b_idx, "Status"] == "Matched":
            continue

        bank_date = bank["Date"]

        candidates = ledger_df[
            (ledger_df["Status"] == "Unmatched")
            & (ledger_df["Amount"] == bank["Amount"])
            & (ledger_df["Direction"] == bank["Direction"])
            & (ledger_df["Date"].apply(
                lambda d: abs((d - bank_date).days) <= 3
            ))
        ]

        if len(candidates) != 1:
            continue

        l_idx = candidates.index[0]
        counter += 1
        match_id = f"REC-{counter:04d}"

        bank_df.at[b_idx, "Status"] = "Suggested Match"
        bank_df.at[b_idx, "Match ID"] = match_id
        bank_df.at[b_idx, "Match Rule"] = "Amount + Date Window (3 days)"

        ledger_df.at[l_idx, "Status"] = "Suggested Match"
        ledger_df.at[l_idx, "Match ID"] = match_id
        ledger_df.at[l_idx, "Match Rule"] = "Amount + Date Window (3 days)"

    bank_unmatched = bank_df[bank_df["Status"] == "Unmatched"]
    ledger_unmatched = ledger_df[ledger_df["Status"] == "Unmatched"]

    summary = {
        "Bank Transactions": len(bank_df),
        "Ledger Transactions": len(ledger_df),
        "Matched": len(bank_df) - len(bank_unmatched),
        "Bank Unmatched": len(bank_unmatched),
        "Ledger Unmatched": len(ledger_unmatched),
        "Status": "Needs Review" if len(bank_unmatched) or len(ledger_unmatched) else "Reconciled",
    }

    return bank_df, ledger_df, summary
