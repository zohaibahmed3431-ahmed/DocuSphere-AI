import pandas as pd
from src.reconciliation import auto_select_two_ledgers, classify_ledger


def test_auto_select_ignores_non_dataframe_session_values():
    frames = {
        "Bank.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Debit": [100]}),
        "Software.xlsx": pd.DataFrame({"Entry Date": ["2026-01-01"], "Debit": [100]}),
        "bad": "not a dataframe",
    }
    pair = auto_select_two_ledgers(frames, allow_explicit_pair=True)
    assert pair is not None
    assert pair[0][0] == "Bank.xlsx"
    assert pair[1][0] == "Software.xlsx"


def test_auto_select_invalid_frames_returns_none_instead_of_typeerror():
    assert auto_select_two_ledgers(None, allow_explicit_pair=True) is None
    assert auto_select_two_ledgers({"bad": "not a dataframe"}, allow_explicit_pair=True) is None
    assert classify_ledger("bad", "not a dataframe") == ("unknown", 0)
