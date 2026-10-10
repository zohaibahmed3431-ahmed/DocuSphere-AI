from datetime import timedelta


def run_reconciliation(
    bank_df,
    ledger_df,
    date_tolerance_days=0,
):
    bank_df = bank_df.copy()
    ledger_df = ledger_df.copy()

    counter = 0

    bank_df, ledger_df, counter = match_by_reference(
        bank_df,
        ledger_df,
        counter,
    )

    bank_df, ledger_df, counter = match_by_amount_and_date(
        bank_df,
        ledger_df,
        counter,
    )

    if date_tolerance_days > 0:
        bank_df, ledger_df, counter = match_by_amount_and_date_window(
            bank_df,
            ledger_df,
            counter,
            date_tolerance_days,
        )

    summary = create_summary(bank_df, ledger_df)

    return bank_df, ledger_df, summary


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

        bank_df.at[bank_index, "Confidence"] = "100%"
        ledger_df.at[ledger_index, "Confidence"] = "100%"

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

        bank_df.at[bank_index, "Match Rule"] = "Exact Amount + Same Date"
        ledger_df.at[ledger_index, "Match Rule"] = "Exact Amount + Same Date"

        bank_df.at[bank_index, "Confidence"] = "High"
        ledger_df.at[ledger_index, "Confidence"] = "High"

    return bank_df, ledger_df, counter


def match_by_amount_and_date_window(
    bank_df,
    ledger_df,
    counter,
    tolerance_days,
):
    for bank_index, bank_row in bank_df.iterrows():
        if bank_df.at[bank_index, "Status"] == "Matched":
            continue

        bank_date = bank_row["Date"]

        candidates = ledger_df[
            (ledger_df["Status"] == "Unmatched")
            & (ledger_df["Amount"] == bank_row["Amount"])
            & (ledger_df["Direction"] == bank_row["Direction"])
            & (
                ledger_df["Date"].apply(
                    lambda ledger_date: abs(
                        (ledger_date - bank_date).days
                    ) <= tolerance_days
                )
            )
        ]

        if len(candidates) != 1:
            continue

        ledger_index = candidates.index[0]
        counter += 1
        match_id = f"REC-{counter:04d}"

        bank_df.at[bank_index, "Status"] = "Suggested Match"
        ledger_df.at[ledger_index, "Status"] = "Suggested Match"

        bank_df.at[bank_index, "Match ID"] = match_id
        ledger_df.at[ledger_index, "Match ID"] = match_id

        bank_df.at[bank_index, "Match Rule"] = (
            f"Amount + Date Window ({tolerance_days} days)"
        )
        ledger_df.at[ledger_index, "Match Rule"] = (
            f"Amount + Date Window ({tolerance_days} days)"
        )

        bank_df.at[bank_index, "Confidence"] = "Medium"
        ledger_df.at[ledger_index, "Confidence"] = "Medium"

    return bank_df, ledger_df, counter


def create_summary(bank_df, ledger_df):
    bank_matched = bank_df[bank_df["Status"].isin(["Matched", "Suggested Match"])]
    ledger_matched = ledger_df[ledger_df["Status"].isin(["Matched", "Suggested Match"])]

    bank_unmatched = bank_df[bank_df["Status"] == "Unmatched"]
    ledger_unmatched = ledger_df[ledger_df["Status"] == "Unmatched"]

    bank_unmatched_inflow = bank_unmatched[
        bank_unmatched["Direction"] == "Inflow"
    ]["Amount"].sum()

    bank_unmatched_outflow = bank_unmatched[
        bank_unmatched["Direction"] == "Outflow"
    ]["Amount"].sum()

    ledger_unmatched_inflow = ledger_unmatched[
        ledger_unmatched["Direction"] == "Inflow"
    ]["Amount"].sum()

    ledger_unmatched_outflow = ledger_unmatched[
        ledger_unmatched["Direction"] == "Outflow"
    ]["Amount"].sum()

    return {
        "Bank Transactions": len(bank_df),
        "Ledger Transactions": len(ledger_df),
        "Matched Pairs": len(bank_matched),
        "Bank Unmatched": len(bank_unmatched),
        "Ledger Unmatched": len(ledger_unmatched),
        "Bank Unmatched Inflow": round(bank_unmatched_inflow, 2),
        "Bank Unmatched Outflow": round(bank_unmatched_outflow, 2),
        "Ledger Unmatched Inflow": round(ledger_unmatched_inflow, 2),
        "Ledger Unmatched Outflow": round(ledger_unmatched_outflow, 2),
        "Reconciliation Status": (
            "Reconciled"
            if len(bank_unmatched) == 0 and len(ledger_unmatched) == 0
            else "Needs Review"
        ),
    }