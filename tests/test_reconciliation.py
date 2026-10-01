import pandas as pd

from src.reconciliation import reconcile_tables, looks_like_reconciliation_request, auto_select_two_ledgers


def test_reconciliation_matches_and_serializes():
    bank = pd.DataFrame({
        "Date": ["2026-09-01", "2026-09-02", "2026-09-03"],
        "Reference": ["TXN100", "TXN200", "TXN300"],
        "Description": ["Customer A", "Customer B", "Bank fee"],
        "Debit": [0, 500, 50],
        "Credit": [1000, 0, 0],
    })
    software = pd.DataFrame({
        "Entry Date": ["2026-09-01", "2026-09-02", "2026-09-04"],
        "Ref No": ["TXN100", "TXN200", "TXN999"],
        "Remarks": ["Customer A", "Customer B", "Other"],
        "Debit": [0, 500, 25],
        "Credit": [1000, 0, 0],
    })
    r = reconcile_tables(bank, software, "Bank Statement.csv", "Software Ledger.csv")
    assert len(r.matched) == 2
    assert len(r.unmatched) == 2
    assert r.matched.iloc[0]["Reconciliation Serial No"] == "REC-000001"


def test_ambiguous_amount_only_is_not_guessed():
    bank = pd.DataFrame({"Date": ["2026-09-01", "2026-09-01"], "Amount": [100, 100]})
    software = pd.DataFrame({"Date": ["2026-09-01", "2026-09-01"], "Amount": [100, 100]})
    r = reconcile_tables(bank, software, "Bank.csv", "Software.csv")
    assert len(r.matched) == 0
    assert len(r.unmatched) == 4


def test_auto_detection():
    frames = {
        "Bank Statement.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Debit": [10]}),
        "Software Ledger.xlsx": pd.DataFrame({"Entry Date": ["2026-01-01"], "Debit": [10]}),
    }
    assert looks_like_reconciliation_request("reconcile these two files", frames)
    pair = auto_select_two_ledgers(frames)
    assert pair is not None


def test_opposite_debit_credit_convention_with_reference_matches():
    bank = pd.DataFrame({"Date": ["2026-09-10"], "Reference": ["PAY-1"], "Debit": [500], "Credit": [0]})
    software = pd.DataFrame({"Entry Date": ["2026-09-10"], "Ref No": ["PAY-1"], "Debit": [0], "Credit": [500]})
    r = reconcile_tables(bank, software, "Bank Statement.xlsx", "Software Ledger.xlsx")
    assert len(r.matched) == 1
    assert "opposite debit/credit convention" in r.matched.iloc[0]["Match Basis"]


def test_generic_two_files_are_not_silently_assigned():
    frames = {
        "Book1.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Amount": [10]}),
        "Book2.xlsx": pd.DataFrame({"Date": ["2026-01-01"], "Amount": [10]}),
    }
    assert auto_select_two_ledgers(frames) is None


def test_unique_amount_without_identity_is_not_auto_matched():
    bank = pd.DataFrame({"Date": ["2026-09-01"], "Amount": [123.45]})
    software = pd.DataFrame({"Date": ["2026-09-01"], "Amount": [123.45]})
    r = reconcile_tables(bank, software, "Bank Statement.csv", "Software Ledger.csv")
    assert len(r.matched) == 0
    assert len(r.unmatched) == 2


def test_reference_can_match_even_with_different_description():
    bank = pd.DataFrame({"Date": ["2026-09-01"], "Reference": ["TXN-ABC-123"], "Description": ["Customer payment"], "Credit": [500]})
    software = pd.DataFrame({"Entry Date": ["2026-09-01"], "Ref No": ["TXN-ABC-123"], "Remarks": ["Different wording"], "Debit": [0], "Credit": [500]})
    r = reconcile_tables(bank, software, "Bank Statement.csv", "Software Ledger.csv")
    assert len(r.matched) == 1
    assert r.matched.iloc[0]["Reconciliation Serial No"] == "REC-000001"


def test_source_serials_are_preserved_and_shared_id_is_new():
    bank = pd.DataFrame({
        "Serial No": ["BANK-987"], "Date": ["2026-09-15"],
        "Description": ["Customer payment 77"], "Credit": [2500],
    })
    software = pd.DataFrame({
        "Entry No": ["SW-42"], "Entry Date": ["2026-09-15"],
        "Remarks": ["Customer payment 77"], "Credit": [2500],
    })
    r = reconcile_tables(bank, software, "Bank Statement.csv", "Software Ledger.csv")
    assert len(r.matched) == 1
    assert r.matched.iloc[0]["Reconciliation Serial No"] == "REC-000001"
    assert r.matched.iloc[0]["Bank_Serial No"] == "BANK-987"
    assert r.matched.iloc[0]["Software_Entry No"] == "SW-42"


def test_unique_same_amount_date_and_direction_can_match_without_same_serial():
    bank = pd.DataFrame({
        "Serial": ["B-1"], "Date": ["2026-09-15"], "Debit": [750],
    })
    software = pd.DataFrame({
        "Voucher": ["V-999"], "Entry Date": ["2026-09-15"], "Debit": [750],
    })
    r = reconcile_tables(bank, software, "Bank Statement.csv", "Software Ledger.csv")
    assert len(r.matched) == 1
    assert r.matched.iloc[0]["Reconciliation Serial No"] == "REC-000001"
