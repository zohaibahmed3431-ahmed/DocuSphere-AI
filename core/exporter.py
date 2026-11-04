import io
import pandas as pd

def export_reconciliation_report(bank_df, ledger_df, matched_df, summary):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False
        )
        if not matched_df.empty:
            matched_df.to_excel(writer, sheet_name="Matched Transactions", index=False)
        bank_df[bank_df["Status"] == "Unmatched"].to_excel(
            writer, sheet_name="Bank Unmatched", index=False
        )
        ledger_df[ledger_df["Status"] == "Unmatched"].to_excel(
            writer, sheet_name="Ledger Unmatched", index=False
        )
    output.seek(0)
    return output
