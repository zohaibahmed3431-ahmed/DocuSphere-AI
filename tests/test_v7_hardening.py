from pathlib import Path
import pandas as pd
from src.reconciliation import auto_select_two_ledgers


def test_truly_unknown_two_tables_are_not_blindly_assigned():
    frames = {
        "Sheet1": pd.DataFrame({"X": [1], "Y": [2]}),
        "Sheet2": pd.DataFrame({"X": [1], "Y": [2]}),
    }
    assert auto_select_two_ledgers(frames, allow_explicit_pair=True) is None


def test_bank_and_software_schema_still_auto_pair():
    frames = {
        "Sheet1": pd.DataFrame({"Date": ["2026-01-01"], "Debit": [100], "Credit": [0], "Balance": [900]}),
        "Sheet2": pd.DataFrame({"Entry Date": ["2026-01-01"], "Voucher No": ["V1"], "Debit": [100], "Credit": [0]}),
    }
    pair = auto_select_two_ledgers(frames, allow_explicit_pair=True)
    assert pair is not None
    assert pair[0][0] == "Sheet1"
    assert pair[1][0] == "Sheet2"


def test_llm_does_not_expose_raw_provider_error_payload():
    source = (Path(__file__).resolve().parents[1] / "src" / "llm.py").read_text()
    assert "Technical detail:" not in source
    assert "raw provider payloads" in source
    assert "FALLBACK_MODELS = [\"gemini-3.7-flash\", \"gemini-3.6-flash\"]" in source
    assert "retry_options=types.HttpRetryOptions(attempts=1)" in source
    assert "thinking_config=types.ThinkingConfig(thinking_level=thinking_level)" in source
