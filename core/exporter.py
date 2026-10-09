import io

import pandas as pd


def create_excel_report(bank_df, ledger_df, summary):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame(
            summary.items(),
            columns=["Metric", "Value"],
        ).to_excel(writer, sheet_name="Summary", index=False)

        bank_df.to_excel(writer, sheet_name="Bank", index=False)
        ledger_df.to_excel(writer, sheet_name="Ledger", index=False)

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

    output.seek(0)

    return output