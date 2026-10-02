from pathlib import Path
import unittest
import pandas as pd
from src.csv_analytics import analyze_csv, find_date_column


class RegressionTests(unittest.TestCase):
    def test_time_only_column_is_not_calendar_date(self):
        df = pd.DataFrame({
            "Entry_Date": ["19:37.2", "00:43.1", "11:12.3"],
            "Debit": [100, 200, 300],
            "Credit": [50, 100, 150],
        })
        self.assertIsNone(find_date_column(df)[0])
        result = analyze_csv(df, "last week sales growth")
        self.assertEqual(result.kind, "warning")
        self.assertIn("reliable calendar-date", result.summary)

    def test_graph_request_uses_real_numeric_metrics(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2026-09-01", "2026-09-02"]), "Debit": [10, 20], "Credit": [5, 15]})
        result = analyze_csv(df, "graph of debit and credit")
        self.assertEqual(result.title, "Metric comparison")
        self.assertEqual(set(result.table["Metric"]), {"Debit", "Credit"})
        self.assertIsNotNone(result.chart)

    def test_image_request_is_routed_before_retrieval(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        source = app.read_text()
        route_pos = source.index('request_route = route(')
        image_pos = source.index('if request_route.kind == "image":')
        retrieval_pos = source.index('results=st.session_state.retriever.search')
        self.assertLess(route_pos, image_pos)
        self.assertLess(image_pos, retrieval_pos)


if __name__ == "__main__":
    unittest.main()


def test_reconciliation_exports_are_separate_in_app():
    app = Path(__file__).resolve().parents[1] / "app.py"
    source = app.read_text()
    assert 'bank_reconciliation_matched.csv' in source
    assert 'bank_reconciliation_unmatched.csv' in source
    assert 'reconciliation_excel(reconciliation_result, "matched")' in source
    assert 'reconciliation_excel(reconciliation_result, "unmatched")' in source


def test_large_csv_exports_are_deferred_in_app():
    app = Path(__file__).resolve().parents[1] / "app.py"
    source = app.read_text()
    assert 'dataframe_to_csv_bytes(reconciliation_result.matched)' not in source
    assert 'dataframe_to_csv_bytes(reconciliation_result.unmatched)' not in source
    assert 'df.to_csv(index=False).encode("utf-8-sig")' in source


def test_two_generic_xlsx_sheets_are_reconciliation_pair():
    from src.reconciliation import auto_select_two_ledgers
    frames = {
        "4140 DATA.xlsx::Sheet1": pd.DataFrame({"Date": ["2026-01-01"], "Description": ["ABC"], "Debit": [100], "Credit": [0], "Balance": [900]}),
        "4140 DATA.xlsx::Sheet2": pd.DataFrame({"Date": ["2026-01-01"], "Voucher No": ["V-1"], "Account": ["ABC"], "Debit": [100], "Credit": [0]}),
    }
    pair = auto_select_two_ledgers(frames)
    assert pair is not None
    assert pair[0][0].endswith("Sheet1")
    assert pair[1][0].endswith("Sheet2")


def test_bank_software_wording_routes_to_deterministic_reconciliation():
    from src.router import route
    r = route("make a software ledger and bank or statement ledger of this file", has_files=True, has_tables=True)
    assert r.kind == "reconciliation"


def test_exact_tabular_request_does_not_require_gemini():
    app = Path(__file__).resolve().parents[1] / "app.py"
    source = app.read_text()
    assert 'elif analysis is not None and not needs_ai_interpretation(clean):' in source
    assert '# Exact tabular questions must not make a second Gemini request.' in source

def test_provider_error_message_is_not_the_old_generic_message():
    llm = Path(__file__).resolve().parents[1] / "src" / "llm.py"
    source = llm.read_text()
    assert "I couldn't generate a response right now" not in source
