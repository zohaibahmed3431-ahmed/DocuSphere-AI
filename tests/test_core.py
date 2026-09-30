import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import pandas as pd
from src.csv_analytics import analyze_csv, find_date_column, full_file_report, is_full_details_request


class CoreAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "OrderDate": pd.to_datetime(["2026-08-01", "2026-08-15", "2026-09-01", "2026-09-10"]),
            "Sales": [100, 200, 150, 250],
            "Status": ["Paid", "Paid", "Pending", "Paid"],
        })

    def test_date_detection(self):
        col, parsed = find_date_column(self.df)
        self.assertEqual(col, "OrderDate")
        self.assertEqual(parsed.notna().sum(), 4)

    def test_exact_sales_total(self):
        result = analyze_csv(self.df, "total sales")
        self.assertEqual(result.value_column, "Sales")
        self.assertEqual(float(result.table.loc[0, "Total"]), 700.0)

    def test_missing_named_metric_is_not_substituted(self):
        result = analyze_csv(self.df.drop(columns=["Sales"]), "total sales")
        self.assertEqual(result.kind, "warning")

    def test_full_report(self):
        report = full_file_report(self.df, "sales.csv")
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["columns"], 3)
        self.assertTrue(len(report["charts"]) >= 1)

    def test_full_details_detection(self):
        self.assertTrue(is_full_details_request("give me complete details of this file with graphs"))


if __name__ == "__main__":
    unittest.main()
