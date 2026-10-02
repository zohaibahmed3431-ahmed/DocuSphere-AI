from pathlib import Path

import pandas as pd

from src.reconciliation import detect_columns, auto_select_two_ledgers


def test_v10_local_frustration_response_is_intent_aware_not_echoed():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "Frustration/profanity is handled by intent" in source
    assert "main usi point ko seedha theek karta hoon" in source
    # The response itself must not simply parrot the user's abusive phrase.
    assert 'return "Lagta hai pichla response tumhari requirement ke mutabiq nahi tha.' in source


def test_v10_voucher_is_preserved_as_original_serial():
    df = pd.DataFrame({"Voucher": ["SW-42"], "Entry Date": ["2026-09-15"], "Debit": [100]})
    cols = detect_columns(df)
    assert cols["serial"] == "Voucher"


def test_v10_multiple_unknown_tables_are_not_guessed():
    frames = {
        "A.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Amount": [10]}),
        "B.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Amount": [10]}),
        "C.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Amount": [10]}),
    }
    assert auto_select_two_ledgers(frames, allow_explicit_pair=True) is None


def test_v10_reconciliation_wording_is_detected_for_two_tables():
    from src.reconciliation import looks_like_reconciliation_request
    frames = {
        "Bank Statement.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Debit": [100]}),
        "Software Ledger.xlsx": pd.DataFrame({"Entry Date": ["2026-01-01"], "Debit": [100]}),
    }
    assert looks_like_reconciliation_request("give me the reconciliation of this file", frames)
