APP_NAME = "Bank Reconciliation AI"

SUPPORTED_FILE_TYPES = ["xlsx", "xls", "csv"]

DEFAULT_DATE_TOLERANCE_DAYS = 0
DEFAULT_AMOUNT_TOLERANCE = 0.0

BANK_SHEET_HINTS = [
    "bank",
    "statement",
    "sheet1",
]

LEDGER_SHEET_HINTS = [
    "ledger",
    "general ledger",
    "sheet2",
]

DATE_COLUMN_HINTS = [
    "date",
    "transaction date",
    "posting date",
]

DESCRIPTION_COLUMN_HINTS = [
    "description",
    "narration",
    "particulars",
    "details",
]

DEBIT_COLUMN_HINTS = [
    "debit",
    "withdrawal",
    "money out",
]

CREDIT_COLUMN_HINTS = [
    "credit",
    "deposit",
    "money in",
]

REFERENCE_COLUMN_HINTS = [
    "reference",
    "ref",
    "stan",
    "slip",
    "cheque",
    "doc",
    "document",
]