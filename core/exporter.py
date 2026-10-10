import io
import pandas as pd


def create_excel_report(bank_df, ledger_df, summary):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df = pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"],
        )

        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        bank_df.to_excel(writer, sheet_name="Bank Transactions", index=False)
        ledger_df.to_excel(writer, sheet_name="Ledger Transactions", index=False)

        bank_df[bank_df["Status"] == "Unmatched"].to_excel(
            writer,
            sheet_name="Bank Unmatched",
            index=False,
        )

        ledger_df[ledger_df["Status"] == "Unmatched"].to_excel(
            writer,
            sheet_name="Ledger Unmatched",
            index=False,
        )

        matched_bank = bank_df[
            bank_df["Status"].isin(["Matched", "Suggested Match"])
        ]

        matched_ledger = ledger_df[
            ledger_df["Status"].isin(["Matched", "Suggested Match"])
        ]

        matched_pairs = pd.merge(
            matched_bank,
            matched_ledger,
            on="Match ID",
            suffixes=("_Bank", "_Ledger"),
            how="inner",
        )

        matched_pairs.to_excel(
            writer,
            sheet_name="Matched Pairs",
            index=False,
        )

    output.seek(0)
    return output