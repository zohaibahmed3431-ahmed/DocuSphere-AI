import io

import pandas as pd


def create_reconciliation_excel(
    bank_df,
    ledger_df,
    summary,
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df = pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"],
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        bank_df.to_excel(
            writer,
            sheet_name="Bank Transactions",
            index=False,
        )

        ledger_df.to_excel(
            writer,
            sheet_name="Ledger Transactions",
            index=False,
        )

        bank_unmatched = bank_df[
            bank_df["Match Status"] == "Unmatched"
        ]

        ledger_unmatched = ledger_df[
            ledger_df["Match Status"] == "Unmatched"
        ]

        bank_unmatched.to_excel(
            writer,
            sheet_name="Bank Unmatched",
            index=False,
        )

        ledger_unmatched.to_excel(
            writer,
            sheet_name="Ledger Unmatched",
            index=False,
        )

        matched_pairs = pd.merge(
            bank_df[bank_df["Match Status"] == "Matched"],
            ledger_df[ledger_df["Match Status"] == "Matched"],
            on="Match ID",
            suffixes=("_Bank", "_Ledger"),
        )

        matched_pairs.to_excel(
            writer,
            sheet_name="Matched Pairs",
            index=False,
        )

    output.seek(0)

    return output