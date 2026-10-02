from pathlib import Path
import pandas as pd
from src.reconciliation import reconcile_tables


def test_provider_payload_is_not_exposed_by_image_error_path():
    source = (Path(__file__).resolve().parents[1] / "src" / "llm.py").read_text()
    assert 'raise RuntimeError(f"Gemini image generation failed: {message}")' not in source
    assert '503' in source
    assert 'Never expose raw provider payloads' in source


def test_ambiguous_duplicate_amount_date_direction_stays_unmatched():
    bank = pd.DataFrame({"Serial": ["B1", "B2"], "Date": ["2026-01-01", "2026-01-01"], "Debit": [100, 100]})
    software = pd.DataFrame({"Voucher": ["S1", "S2"], "Date": ["2026-01-01", "2026-01-01"], "Debit": [100, 100]})
    r = reconcile_tables(bank, software, "Bank Statement.xlsx", "Software Ledger.xlsx")
    assert len(r.matched) == 0
    assert len(r.unmatched) == 4


def test_exact_reference_can_match_across_debit_credit_convention():
    bank = pd.DataFrame({"Serial": ["B1"], "Date": ["2026-01-01"], "Reference": ["REF-1"], "Debit": [100]})
    software = pd.DataFrame({"Voucher No": ["S1"], "Date": ["2026-01-01"], "Reference": ["REF-1"], "Credit": [100]})
    r = reconcile_tables(bank, software, "Bank Statement.xlsx", "Software Ledger.xlsx")
    assert len(r.matched) == 1
    row = r.matched.iloc[0]
    assert row["Reconciliation Serial No"] == "REC-000001"
    assert row["Bank Original Serial No"] == "B1"
    assert row["Software Original Serial No"] == "S1"


def test_date_mismatch_does_not_match_even_when_reference_is_same():
    bank = pd.DataFrame({"Serial": ["B1"], "Date": ["2026-01-01"], "Reference": ["REF-1"], "Credit": [100]})
    software = pd.DataFrame({"Voucher No": ["S1"], "Date": ["2026-01-10"], "Reference": ["REF-1"], "Credit": [100]})
    r = reconcile_tables(bank, software, "Bank Statement.xlsx", "Software Ledger.xlsx")
    assert len(r.matched) == 0


def test_unique_amount_date_direction_is_allowed_when_no_identity_fields_exist():
    bank = pd.DataFrame({"Serial": ["B1"], "Date": ["2026-01-01"], "Debit": [100]})
    software = pd.DataFrame({"Voucher No": ["S1"], "Date": ["2026-01-01"], "Debit": [100]})
    r = reconcile_tables(bank, software, "Bank Statement.xlsx", "Software Ledger.xlsx")
    assert len(r.matched) == 1
