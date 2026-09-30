from pathlib import Path
import unittest


class ImageConfigRegressionTests(unittest.TestCase):
    def test_image_output_uses_jpeg(self):
        source = Path(__file__).resolve().parents[1]
        llm = (source / "src" / "llm.py").read_text()
        app = (source / "app.py").read_text()
        self.assertIn('"mime_type": "image/jpeg"', llm)
        self.assertIn('"docusphere_generated.jpg"', app)
        self.assertIn('"image/jpeg"', app)


if __name__ == "__main__":
    unittest.main()
