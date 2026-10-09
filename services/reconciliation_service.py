from core.file_loader import load_file
from core.document_detector import detect_document_type
from core.data_cleaner import clean_sheet, standardize
from core.reconciler import run_reconciliation


def process_uploaded_files(uploaded_files):
    bank_file = None
    ledger_file = None

    for uploaded_file in uploaded_files:
        sheets = load_file(uploaded_file)
        document_type = detect_document_type(sheets)

        if document_type == "Bank Statement":
            bank_file = {
                "name": uploaded_file.name,
                "type": document_type,
                "sheets": sheets,
            }

        elif document_type == "Company Ledger":
            ledger_file = {
                "name": uploaded_file.name,
                "type": document_type,
                "sheets": sheets,
            }

    if not bank_file or not ledger_file:
        raise ValueError("Bank statement aur company ledger dono upload karein.")

    bank_sheet_name = list(bank_file["sheets"].keys())[0]
    ledger_sheet_name = list(ledger_file["sheets"].keys())[0]

    bank_clean = clean_sheet(bank_file["sheets"][bank_sheet_name])
    ledger_clean = clean_sheet(ledger_file["sheets"][ledger_sheet_name])

    bank_df = standardize(bank_clean, "Bank Statement", bank_sheet_name)
    ledger_df = standardize(ledger_clean, "Company Ledger", ledger_sheet_name)

    bank_df, ledger_df, summary = run_reconciliation(bank_df, ledger_df)

    return {
        "bank": bank_df,
        "ledger": ledger_df,
        "summary": summary,
    }