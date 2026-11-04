def detect_document_type(sheets):
    text = ""
    for df in sheets.values():
        text += " ".join(
            df.astype(str).fillna("").values.flatten()[:4000]
        ).lower()
        
    if any(k in text for k in ["available balance", "bank statement", "stan", "withdrawal", "deposit"]):
        return "Bank Statement"
    if any(k in text for k in ["general ledger", "doc #", "voucher", "journal", "debit", "credit"]):
        return "Company Ledger"
    return "General Financial Data"
