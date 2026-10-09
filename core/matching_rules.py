def match_by_reference(bank_df, ledger_df, counter):
    for bank_index, bank_row in bank_df.iterrows():
        if bank_df.at[bank_index, "Status"] == "Matched":
            continue

        reference = str(bank_row["Reference"]).strip()

        if not reference:
            continue

        candidates = ledger_df[
            (ledger_df["Status"] == "Unmatched")
            & (ledger_df["Reference"].astype(str).str.strip() == reference)
            & (ledger_df["Amount"] == bank_row["Amount"])
            & (ledger_df["Direction"] == bank_row["Direction"])
        ]

        if len(candidates) != 1:
            continue

        ledger_index = candidates.index[0]
        counter += 1
        match_id = f"REC-{counter:04d}"

        bank_df.at[bank_index, "Status"] = "Matched"
        ledger_df.at[ledger_index, "Status"] = "Matched"

        bank_df.at[bank_index, "Match ID"] = match_id
        ledger_df.at[ledger_index, "Match ID"] = match_id

        bank_df.at[bank_index, "Match Rule"] = "Exact Reference"
        ledger_df.at[ledger_index, "Match Rule"] = "Exact Reference"

    return bank_df, ledger_df, counter


def match_by_amount_and_date(bank_df, ledger_df, counter):
    for bank_index, bank_row in bank_df.iterrows():
        if bank_df.at[bank_index, "Status"] == "Matched":
            continue

        candidates = ledger_df[
            (ledger_df["Status"] == "Unmatched")
            & (ledger_df["Amount"] == bank_row["Amount"])
            & (ledger_df["Direction"] == bank_row["Direction"])
            & (ledger_df["Date"] == bank_row["Date"])
        ]

        if len(candidates) != 1:
            continue

        ledger_index = candidates.index[0]
        counter += 1
        match_id = f"REC-{counter:04d}"

        bank_df.at[bank_index, "Status"] = "Matched"
        ledger_df.at[ledger_index, "Status"] = "Matched"

        bank_df.at[bank_index, "Match ID"] = match_id
        ledger_df.at[ledger_index, "Match ID"] = match_id

        bank_df.at[bank_index, "Match Rule"] = "Exact Amount + Date"
        ledger_df.at[ledger_index, "Match Rule"] = "Exact Amount + Date"

    return bank_df, ledger_df, counter