APP_NAME = "Bank Reconciliation AI"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

MAX_FILE_SIZE_MB = 200

SUPPORTED_TYPES = [
    "xlsx", "xls", "csv",
    "pdf", "docx", "txt", "md",
    "png", "jpg", "jpeg", "tiff",
]

DEFAULT_DATE_TOLERANCE_DAYS = 0

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"