from core.file_loader import load_excel_or_csv
from core.document_detector import detect_document_type
from core.data_cleaner import clean_sheet, standardize
from core.reconciler import run_reconciliation

def process_reconciliation_files(uploaded_files):
    bank_sheet = None
    ledger_sheet = None
    
    for uploaded_file in uploaded_files:
        sheets = load_excel_or_csv(uploaded_file)
        doc_type = detect_document_type(sheets)
        sheet_names = list(sheets.keys())
        
        if len(sheets) >= 2:
            bank_sheet = standardize(clean_sheet(sheets[sheet_names[0]]), "Bank Statement", sheet_names[0])
            ledger_sheet = standardize(clean_sheet(sheets[sheet_names[1]]), "Company Ledger", sheet_names[1])
            break
        elif doc_type == "Bank Statement" and bank_sheet is None:
            bank_sheet = standardize(clean_sheet(sheets[sheet_names[0]]), "Bank Statement", sheet_names[0])
        elif doc_type == "Company Ledger" and ledger_sheet is None:
            ledger_sheet = standardize(clean_sheet(sheets[sheet_names[1]]), "Company Ledger", sheet_names[1])
            
    if bank_sheet is None or ledger_sheet is None:
        raise ValueError("Please upload both Bank Statement and Company Ledger files or a 2-sheet Excel file.")
        
    bank_df, ledger_df, matched_df, summary = run_reconciliation(bank_sheet, ledger_sheet)
    return bank_df, ledger_df, matched_df, summary
