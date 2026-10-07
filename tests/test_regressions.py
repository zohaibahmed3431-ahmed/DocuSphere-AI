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

    def test_image_request_is_handled_before_retrieval(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        source = app.read_text()
        image_pos = source.index('if should_generate_image(clean) and not is_graph_picture_request(clean):')
        self.assertLess(image_pos, source.index('with st.spinner("Analyzing your request…")'))

    def test_reconciliation_skips_semantic_retrieval_for_responsiveness(self):
        app = Path(__file__).resolve().parents[1] / "app.py"
        source = app.read_text()
        marker = '# Reconciliation does not need semantic retrieval.'
        self.assertIn(marker, source)
        self.assertIn('results=[]', source)


if __name__ == "__main__":
    unittest.main()
