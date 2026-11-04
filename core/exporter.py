import io
import pandas as pd

def export_reconciliation_report(bank_df, ledger_df, matched_df, summary):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Formats
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#1E293B', 'font_color': '#FFFFFF', 'border': 1
        })
        
        # Sheet 1: Summary Dashboard
        summary_data = [
            {"Audit Metric": "Total Bank Transactions Processed", "Count / Value": summary.get("Bank Transactions", 0)},
            {"Audit Metric": "Total Ledger Transactions Processed", "Count / Value": summary.get("Ledger Transactions", 0)},
            {"Audit Metric": "Successfully Matched Transaction Pairs", "Count / Value": summary.get("Matched Pairs", 0)},
            {"Audit Metric": "Unmatched Bank Transactions (Outstandings)", "Count / Value": summary.get("Bank Unmatched", 0)},
            {"Audit Metric": "Unmatched Ledger Transactions (Unrecorded)", "Count / Value": summary.get("Ledger Unmatched", 0)},
            {"Audit Metric": "Final Reconciliation Status", "Count / Value": summary.get("Reconciliation Status", "Needs Review")}
        ]
        sum_df = pd.DataFrame(summary_data)
        sum_df.to_excel(writer, sheet_name="Reconciliation Summary", index=False)
        
        # Sheet 2: Matched Pairs (Full Details with Common IDs & Source Serials)
        if not matched_df.empty:
            matched_df.to_excel(writer, sheet_name="Matched Transactions", index=False)
        else:
            pd.DataFrame([{"Message": "No matched transactions found"}]).to_excel(writer, sheet_name="Matched Transactions", index=False)
            
        # Sheet 3: Bank Unmatched Items
        bank_unmatched = bank_df[bank_df["Status"] == "Unmatched"].copy() if "Status" in bank_df.columns else bank_df
        if not bank_unmatched.empty:
            bank_unmatched.to_excel(writer, sheet_name="Bank Unmatched", index=False)
        else:
            pd.DataFrame([{"Message": "Zero unmatched items in bank statement"}]).to_excel(writer, sheet_name="Bank Unmatched", index=False)

        # Sheet 4: Ledger Unmatched Items
        ledger_unmatched = ledger_df[ledger_df["Status"] == "Unmatched"].copy() if "Status" in ledger_df.columns else ledger_df
        if not ledger_unmatched.empty:
            ledger_unmatched.to_excel(writer, sheet_name="Ledger Unmatched", index=False)
        else:
            pd.DataFrame([{"Message": "Zero unmatched items in company ledger"}]).to_excel(writer, sheet_name="Ledger Unmatched", index=False)
            
        # Sheet 5: Audit Adjustment Recommendations
        adjustments = []
        for idx, row in bank_unmatched.iterrows():
            adjustments.append({
                "Source": "Bank",
                "Original Serial": row.get("Original_Serial", ""),
                "Date": row.get("Date", ""),
                "Amount": row.get("Amount", 0),
                "Description": row.get("Description", ""),
                "Recommended Action": "Post missing bank entry to General Ledger"
            })
        for idx, row in ledger_unmatched.iterrows():
            adjustments.append({
                "Source": "Ledger",
                "Original Serial": row.get("Original_Serial", ""),
                "Date": row.get("Date", ""),
                "Amount": row.get("Amount", 0),
                "Description": row.get("Description", ""),
                "Recommended Action": "Investigate uncredited ledger voucher / Bank timing difference"
            })
        pd.DataFrame(adjustments).to_excel(writer, sheet_name="Audit Adjustments Log", index=False)

    output.seek(0)
    return output
