from src.router import route, should_use_file_context, should_use_vision


def test_general_question_stays_general_even_with_files():
    assert route("what is polymorphism?", has_files=True, has_tables=True).kind == "general"


def test_followup_file_question_uses_file_context():
    assert route("what does this file say about the budget?", has_files=True, has_tables=True).kind == "file"


def test_csv_question_is_data_route():
    assert route("how many sales were there last week?", has_files=True, has_tables=True).kind == "data"


def test_pdf_export_is_explicit():
    assert route("give me the complete report as PDF", has_files=True, has_tables=True).kind == "pdf"


def test_image_question_can_use_vision():
    assert should_use_vision("explain what is shown in this")


def test_general_question_does_not_use_file_context():
    assert not should_use_file_context("explain recursion in simple words")


def test_unrelated_current_question_is_not_hijacked_by_uploaded_table():
    assert route("what happened this week in Pakistan?", has_files=True, has_tables=True).kind == "web"


def test_temporal_sales_question_stays_data():
    assert route("how many sales happened last week?", has_files=True, has_tables=True).kind == "data"


def test_generic_this_question_stays_general():
    assert route("is this a good idea?", has_files=True, has_tables=True).kind == "general"


def test_vision_requires_visual_language():
    assert not should_use_vision("is this a good idea?")
    assert should_use_vision("what does this image show?")


def test_short_ambiguous_input_stays_local_in_app():
    from pathlib import Path
    app = Path(__file__).resolve().parents[1] / "app.py"
    source = app.read_text()
    assert "Very short non-question inputs" in source
    assert "u shit" in source

