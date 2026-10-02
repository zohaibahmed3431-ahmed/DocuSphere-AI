import io
import pandas as pd
from src.reconciliation import detect_columns, auto_select_two_ledgers, reconcile_tables
from src.tabular import read_tabular


def test_v12_real_world_header_aliases_are_detected():
    df = pd.DataFrame({
        "Txn Dt": ["01/09/2026"],
        "UTR Number": ["UTR-100"],
        "Particular": ["ABC Traders"],
        "Paid Out": [2500],
        "Paid In": [0],
        "Closing Bal": [7500],
    })
    cols = detect_columns(df)
    assert cols["date"] == "Txn Dt"
    assert cols["reference"] == "UTR Number"
    assert cols["description"] == "Particular"
    assert cols["debit"] == "Paid Out"
    assert cols["credit"] == "Paid In"
    assert cols["balance"] == "Closing Bal"


def test_v12_single_side_by_side_bank_software_table_is_split():
    frame = pd.DataFrame({
        "Bank Date": ["2026-09-01"], "Bank Description": ["ABC Traders"], "Bank Debit": [2500], "Bank Credit": [0],
        "Software Date": ["2026-09-01"], "Software Account": ["ABC Traders"], "Software Debit": [2500], "Software Credit": [0],
    })
    pair = auto_select_two_ledgers({"reconciliation.xlsx::Sheet1": frame}, allow_explicit_pair=True)
    assert pair is not None
    result = reconcile_tables(pair[0][1], pair[1][1], pair[0][0], pair[1][0])
    assert len(result.matched) == 1


def test_v12_metadata_rows_above_excel_header_are_promoted(tmp_path):
    path = tmp_path / "ledger.xlsx"
    raw = pd.DataFrame([
        ["Company XYZ", None, None, None],
        ["Statement period", "September 2026", None, None],
        ["Txn Date", "Reference", "Description", "Amount"],
        ["2026-09-01", "TX-1", "ABC Traders", 2500],
    ])
    raw.to_excel(path, index=False, header=False)
    with open(path, "rb") as f:
        class Upload:
            name = "ledger.xlsx"
            def __init__(self, data): self.data = data
            def seek(self, n): pass
            def read(self): return self.data
        frames = read_tabular(Upload(path.read_bytes()))
    df = frames["ledger.xlsx::Sheet1"]
    assert "Txn Date" in list(df.columns)
    assert len(df) == 1

def test_v12_reconciliation_request_detects_single_tabular_source_for_side_by_side_data():
    from src.reconciliation import looks_like_reconciliation_request
    frame = pd.DataFrame({
        "Bank Date": ["2026-09-01"], "Bank Amount": [100],
        "Software Date": ["2026-09-01"], "Software Amount": [100],
    })
    assert looks_like_reconciliation_request("give me the reconciliation of this file", {"file.csv": frame})


def test_v12_unknown_schema_never_guesses_roles():
    frames = {
        "Sheet1": pd.DataFrame({"ColA": [1, 2], "ColB": [3, 4]}),
        "Sheet2": pd.DataFrame({"ColA": [1, 2], "ColB": [3, 4]}),
    }
    assert auto_select_two_ledgers(frames, allow_explicit_pair=True) is None
