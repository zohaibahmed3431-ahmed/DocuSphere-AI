import pandas as pd
from src.reconciliation import auto_select_two_ledgers, reconcile_tables


def test_v11_generic_two_sheet_workbook_is_role_detected_from_schema():
    frames = {
        "DATA.xlsx::Sheet1": pd.DataFrame({
            "Date": ["2026-09-01"], "Description": ["ABC Traders"],
            "Debit": [2500], "Credit": [0], "Balance": [7500],
        }),
        "DATA.xlsx::Sheet2": pd.DataFrame({
            "Entry Date": ["2026-09-01"], "Voucher No": ["SW-42"],
            "Account": ["ABC Traders"], "Debit": [2500], "Credit": [0],
        }),
    }
    pair = auto_select_two_ledgers(frames, allow_explicit_pair=True)
    assert pair is not None
    assert pair[0][0].endswith("Sheet1")
    assert pair[1][0].endswith("Sheet2")


def test_v11_language_preference_is_persistent_in_app():
    source = open("app.py", encoding="utf-8").read()
    assert "response_language" in source
    assert "detect_language_preference" in source
    assert "Reply in English" in source


def test_v11_reconciliation_keeps_different_source_serials_and_creates_common_id():
    bank = pd.DataFrame({
        "Serial No": ["BANK-987"], "Date": ["2026-09-01"],
        "Description": ["ABC Traders"], "Debit": [2500], "Credit": [0], "Balance": [7500],
    })
    software = pd.DataFrame({
        "Voucher No": ["SW-42"], "Entry Date": ["2026-09-01"],
        "Account": ["ABC Traders"], "Debit": [2500], "Credit": [0],
    })
    result = reconcile_tables(bank, software)
    assert len(result.matched) == 1
    row = result.matched.iloc[0]
    assert row["Bank Original Serial No"] == "BANK-987"
    assert row["Software Original Serial No"] == "SW-42"
    assert row["Reconciliation Serial No"] == "REC-000001"
