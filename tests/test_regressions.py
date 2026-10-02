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
