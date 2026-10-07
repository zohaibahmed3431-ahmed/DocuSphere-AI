import pandas as pd
import re

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all').dropna(how='all', axis=1)
    df.columns = [re.sub(r'\s+', '_', str(c).strip().lower()) for c in df.columns]
    return df

def extract_file_context(files) -> str:
    """Automatically extracts context as soon as files are uploaded"""
    if not files:
        return "No files currently uploaded."
    
    context = "=== UPLOADED DOCUMENTS CONTEXT ===\n\n"
    for idx, f in enumerate(files, 1):
        context += f"FILE #{idx}: {f.name}\n"
        try:
            f.seek(0)
            filename = f.name.lower()
            if filename.endswith(('.csv', '.xlsx', '.xls')):
                df = pd.read_csv(f) if filename.endswith('.csv') else pd.read_excel(f)
                clean_df = clean_column_names(df)
                context += f"Type: Tabular Data ({len(df)} rows, {len(df.columns)} columns)\n"
                context += f"Columns: {list(df.columns)}\n"
                context += f"Data Sample:\n{clean_df.head(5).to_string()}\n"
            else:
                content = f.read()
                try:
                    text = content.decode('utf-8', errors='ignore')
                    context += f"Content Preview:\n{text[:2500]}\n"
                except Exception:
                    context += "Binary / Image File loaded successfully.\n"
        except Exception as e:
            context += f"Read Error: {str(e)}\n"
        context += "-----------------------------------\n"
    return context