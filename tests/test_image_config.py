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

    def test_secrets_are_supported(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "src" / "llm.py").read_text()
        self.assertIn("st.secrets.get", source)
        self.assertIn('_setting("GEMINI_API_KEY")', source)


if __name__ == "__main__":
    unittest.main()


class ProductionConfigRegressionTests(unittest.TestCase):
    def test_gemini_38_config_does_not_use_removed_temperature(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "llm.py").read_text()
        self.assertNotIn("temperature=0.25", source)

    def test_embeddings_support_streamlit_secrets(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "embeddings.py").read_text()
        self.assertIn("st.secrets.get", source)
        self.assertIn("_api_key()", source)
