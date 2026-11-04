import pandas as pd

def run_reconciliation(bank_df, ledger_df):
    bank_df = bank_df.copy()
    ledger_df = ledger_df.copy()
    
    match_counter = 1
    ledger_matched_indices = set()
    matched_records = []
    
    for b_idx, bank in bank_df.iterrows():
        b_amt = bank["Amount"]
        b_dir = bank["Direction"]
        b_ids = bank["Identifiers"]
        b_desc = bank["Description"].lower()
        
        found = False
        for l_idx, ledger in ledger_df.iterrows():
            if l_idx in ledger_matched_indices:
                continue
                
            l_amt = ledger["Amount"]
            l_dir = ledger["Direction"]
            
            if abs(b_amt - l_amt) > 0.01 or b_dir != l_dir:
                continue
                
            l_ids = ledger["Identifiers"]
            l_desc = ledger["Description"].lower()
            
            common_ids = b_ids.intersection(l_ids)
            
            if len(common_ids) > 0 or b_desc == l_desc or bank["Date"] == ledger["Date"]:
                rec_id = f"REC-{match_counter:06d}"
                match_counter += 1
                ledger_matched_indices.add(l_idx)
                
                bank_df.at[b_idx, "Status"] = "Matched"
                bank_df.at[b_idx, "Match ID"] = rec_id
                bank_df.at[b_idx, "Match Rule"] = "Exact Deterministic Match"
                
                ledger_df.at[l_idx, "Status"] = "Matched"
                ledger_df.at[l_idx, "Match ID"] = rec_id
                ledger_df.at[l_idx, "Match Rule"] = "Exact Deterministic Match"
                
                matched_records.append({
                    "Reconciliation_ID": rec_id,
                    "Date": bank["Date"],
                    "Bank_Original_Serial": bank["Original_Serial"],
                    "Ledger_Original_Serial": ledger["Original_Serial"],
                    "Direction": b_dir,
                    "Bank_Amount": b_amt,
                    "Ledger_Amount": l_amt,
                    "Bank_Description": bank["Description"],
                    "Ledger_Description": ledger["Description"]
                })
                found = True
                break
                
    matched_df = pd.DataFrame(matched_records)
    bank_unmatched = bank_df[bank_df["Status"] == "Unmatched"]
    ledger_unmatched = ledger_df[ledger_df["Status"] == "Unmatched"]
    
    summary = {
        "Bank Transactions": len(bank_df),
        "Ledger Transactions": len(ledger_df),
        "Matched Pairs": len(matched_records),
        "Bank Unmatched": len(bank_unmatched),
        "Ledger Unmatched": len(ledger_unmatched),
        "Status": "Reconciled" if len(bank_unmatched) == 0 and len(ledger_unmatched) == 0 else "Needs Review"
    }
    
    return bank_df, ledger_df, matched_df, summary
