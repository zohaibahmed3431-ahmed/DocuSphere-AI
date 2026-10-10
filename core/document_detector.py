def detect_document_type(sheets):
    text = ""

    for dataframe in sheets.values():
        text += " ".join(
            str(value).lower()
            for value in dataframe.astype(str).fillna("").values.flatten()[:5000]
        )

    if (
        "available balance" in text
        or "bank statement" in text
        or "stan" in text
    ):
        return "Bank Statement"

    if (
        "general ledger" in text
        or "doc #" in text
        or "voucher" in text
    ):
        return "Company Ledger"

    return "Unknown"