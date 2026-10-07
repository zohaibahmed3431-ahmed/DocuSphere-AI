import ast
import re
import unittest
from pathlib import Path


class TestLocalSmalltalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "local_smalltalk")
        module = ast.Module(body=[node], type_ignores=[])
        ns = {"re": re}
        exec(compile(module, str(source), "exec"), ns)
        cls.local_smalltalk = ns["local_smalltalk"]

    def test_hi_never_needs_gemini(self):
        self.assertEqual(type(self).local_smalltalk("hi"), "Hi! 👋 I’m DocuSphere AI. What would you like to work on?")

    def test_variants(self):
        self.assertIsNotNone(type(self).local_smalltalk("Assalamualaikum"))
        self.assertIsNotNone(type(self).local_smalltalk("good morning"))
        self.assertIsNotNone(type(self).local_smalltalk("thanks"))

    def test_real_question_is_not_smalltalk(self):
        self.assertIsNone(type(self).local_smalltalk("what is machine learning?"))
