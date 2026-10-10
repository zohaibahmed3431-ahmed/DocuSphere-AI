from core.file_loader import load_file
from core.data_cleaner import clean_sheet, standardize
from core.reconciler import run_reconciliation


def process_reconciliation(uploaded_files, date_tolerance_days=0):
    bank_file = None
    ledger_file = None

    for uploaded_file in uploaded_files:
        sheets = load_file(uploaded_file)

        if len(sheets) >= 2 and bank_file is None:
            sheet_names = list(sheets.keys())

            bank_file = {
                "name": uploaded_file.name,
                "bank_sheet": sheet_names[0],
                "ledger_sheet": sheet_names[1],
                "sheets": sheets,
            }

        elif bank_file is None:
            bank_file = {
                "name": uploaded_file.name,
                "bank_sheet": list(sheets.keys())[0],
                "ledger_sheet": None,
                "sheets": sheets,
            }

        elif ledger_file is None:
            ledger_file = {
                "name": uploaded_file.name,
                "bank_sheet": None,
                "ledger_sheet": list(sheets.keys())[0],
                "sheets": sheets,
            }

    if bank_file is None:
        raise ValueError("Bank statement file upload karein.")

    if ledger_file is None:
        raise ValueError("Company ledger file upload karein.")

    bank_sheet_name = bank_file["bank_sheet"]
    ledger_sheet_name = (
        ledger_file["ledger_sheet"]
        if ledger_file["ledger_sheet"]
        else ledger_file["bank_sheet"]
    )

    bank_clean = clean_sheet(bank_file["sheets"][bank_sheet_name])
    ledger_clean = clean_sheet(ledger_file["sheets"][ledger_sheet_name])

    bank_df = standardize(
        bank_clean,
        "Bank Statement",
        bank_sheet_name,
        bank_file["name"],
    )

    ledger_df = standardize(
        ledger_clean,
        "Company Ledger",
        ledger_sheet_name,
        ledger_file["name"],
    )

    bank_df, ledger_df, summary = run_reconciliation(
        bank_df,
        ledger_df,
        date_tolerance_days,
    )

    return {
        "bank": bank_df,
        "ledger": ledger_df,
        "summary": summary,
    }
