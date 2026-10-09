import pandas as pd


def run_reconciliation(bank_df, ledger_df):
    bank_df = bank_df.copy()
    ledger_df = ledger_df.copy()

    bank_df["Match Status"] = "Unmatched"
    ledger_df["Match Status"] = "Unmatched"

    bank_df["Match ID"] = ""
    ledger_df["Match ID"] = ""

    bank_df["Matched Ledger Row"] = ""
    ledger_df["Matched Bank Row"] = ""

    bank_df["Match Rule"] = ""
    ledger_df["Match Rule"] = ""

    bank_df["Confidence"] = ""
    ledger_df["Confidence"] = ""

    match_counter = 0

    bank_df, ledger_df, match_counter = match_by_reference(
        bank_df,
        ledger_df,
        match_counter,
    )

    bank_df, ledger_df, match_counter = match_by_amount_and_date(
        bank_df,
        ledger_df,
        match_counter,
    )

    summary = create_summary(bank_df, ledger_df)

    return bank_df, ledger_df, summary


def match_by_reference(bank_df, ledger_df, match_counter):
    for bank_index, bank_row in bank_df.iterrows():

        if bank_df.at[bank_index, "Match Status"] == "Matched":
            continue

        bank_reference = str(bank_row["Reference"]).strip()

        if bank_reference == "":
            continue

        possible_matches = ledger_df[
            (ledger_df["Match Status"] == "Unmatched")
            & (ledger_df["Reference"].astype(str).str.strip() == bank_reference)
            & (ledger_df["Amount"] == bank_row["Amount"])
            & (ledger_df["Direction"] == bank_row["Direction"])
        ]

        if len(possible_matches) != 1:
            continue

        ledger_index = possible_matches.index[0]

        match_counter += 1
        match_id = f"REC-{match_counter:04d}"

        bank_df.at[bank_index, "Match Status"] = "Matched"
        ledger_df.at[ledger_index, "Match Status"] = "Matched"

        bank_df.at[bank_index, "Match ID"] = match_id
        ledger_df.at[ledger_index, "Match ID"] = match_id

        bank_df.at[bank_index, "Matched Ledger Row"] = ledger_df.at[
            ledger_index,
            "Row",
        ]

        ledger_df.at[ledger_index, "Matched Bank Row"] = bank_df.at[
            bank_index,
            "Row",
        ]

        bank_df.at[bank_index, "Match Rule"] = "Exact Reference Match"
        ledger_df.at[ledger_index, "Match Rule"] = "Exact Reference Match"

        bank_df.at[bank_index, "Confidence"] = "100%"
        ledger_df.at[ledger_index, "Confidence"] = "100%"

    return bank_df, ledger_df, match_counter


def match_by_amount_and_date(bank_df, ledger_df, match_counter):
    for bank_index, bank_row in bank_df.iterrows():

        if bank_df.at[bank_index, "Match Status"] == "Matched":
            continue

        possible_matches = ledger_df[
            (ledger_df["Match Status"] == "Unmatched")
            & (ledger_df["Amount"] == bank_row["Amount"])
            & (ledger_df["Direction"] == bank_row["Direction"])
            & (ledger_df["Date"] == bank_row["Date"])
        ]

        if len(possible_matches) != 1:
            continue

        ledger_index = possible_matches.index[0]

        match_counter += 1
        match_id = f"REC-{match_counter:04d}"

        bank_df.at[bank_index, "Match Status"] = "Matched"
        ledger_df.at[ledger_index, "Match Status"] = "Matched"

        bank_df.at[bank_index, "Match ID"] = match_id
        ledger_df.at[ledger_index, "Match ID"] = match_id

        bank_df.at[bank_index, "Matched Ledger Row"] = ledger_df.at[
            ledger_index,
            "Row",
        ]

        ledger_df.at[ledger_index, "Matched Bank Row"] = bank_df.at[
            bank_index,
            "Row",
        ]

        bank_df.at[bank_index, "Match Rule"] = "Exact Amount + Same Date"
        ledger_df.at[ledger_index, "Match Rule"] = "Exact Amount + Same Date"

        bank_df.at[bank_index, "Confidence"] = "High"
        ledger_df.at[ledger_index, "Confidence"] = "High"

    return bank_df, ledger_df, match_counter


def create_summary(bank_df, ledger_df):
    bank_matched = bank_df[
        bank_df["Match Status"] == "Matched"
    ]

    ledger_matched = ledger_df[
        ledger_df["Match Status"] == "Matched"
    ]

    bank_unmatched = bank_df[
        bank_df["Match Status"] == "Unmatched"
    ]

    ledger_unmatched = ledger_df[
        ledger_df["Match Status"] == "Unmatched"
    ]

    bank_inflow = bank_unmatched[
        bank_unmatched["Direction"] == "Inflow"
    ]["Amount"].sum()

    bank_outflow = bank_unmatched[
        bank_unmatched["Direction"] == "Outflow"
    ]["Amount"].sum()

    ledger_inflow = ledger_unmatched[
        ledger_unmatched["Direction"] == "Inflow"
    ]["Amount"].sum()

    ledger_outflow = ledger_unmatched[
        ledger_unmatched["Direction"] == "Outflow"
    ]["Amount"].sum()

    summary = {
        "Bank Transactions": len(bank_df),
        "Ledger Transactions": len(ledger_df),
        "Matched Pairs": len(bank_matched),
        "Bank Unmatched": len(bank_unmatched),
        "Ledger Unmatched": len(ledger_unmatched),
        "Bank Unmatched Inflow": round(bank_inflow, 2),
        "Bank Unmatched Outflow": round(bank_outflow, 2),
        "Ledger Unmatched Inflow": round(ledger_inflow, 2),
        "Ledger Unmatched Outflow": round(ledger_outflow, 2),
        "Reconciliation Status": (
            "Reconciled"
            if len(bank_unmatched) == 0 and len(ledger_unmatched) == 0
            else "Needs Review"
        ),
    }

    return summary