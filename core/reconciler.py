from core.matching_rules import (
    match_by_reference,
    match_by_amount_and_date,
)


def run_reconciliation(bank_df, ledger_df):
    bank_df = bank_df.copy()
    ledger_df = ledger_df.copy()

    counter = 0

    bank_df, ledger_df, counter = match_by_reference(
        bank_df, ledger_df, counter
    )

    bank_df, ledger_df, counter = match_by_amount_and_date(
        bank_df, ledger_df, counter
    )

    summary = create_summary(bank_df, ledger_df)

    return bank_df, ledger_df, summary


def create_summary(bank_df, ledger_df):
    bank_unmatched = bank_df[bank_df["Status"] == "Unmatched"]
    ledger_unmatched = ledger_df[ledger_df["Status"] == "Unmatched"]

    return {
        "Bank Transactions": len(bank_df),
        "Ledger Transactions": len(ledger_df),
        "Matched": len(bank_df) - len(bank_unmatched),
        "Bank Unmatched": len(bank_unmatched),
        "Ledger Unmatched": len(ledger_unmatched),
        "Status": (
            "Needs Review"
            if len(bank_unmatched) or len(ledger_unmatched)
            else "Reconciled"
        ),
    }