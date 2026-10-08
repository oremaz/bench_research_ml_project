"""Photo corrections, explicit saving, and Streamlit reruns without API calls."""

import copy
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.base import FoodAnalysisResult, FoodItem
from nutricoach.food_vision.review import photo_edit_rows, apply_photo_edits
from nutricoach.tools import save_photo_analysis
from shared.memory import MemoryManager


@pytest.fixture
def original():
    result = FoodAnalysisResult(method="vlm_single", food_items=[
        FoodItem("rice", 100, calories=130, protein_g=2.7, carbs_g=28.2, fat_g=0.3),
        FoodItem("egg", 50, calories=77.5, protein_g=6.5, carbs_g=0.55, fat_g=5.5),
    ])
    result.compute_totals()
    return result.to_dict()


def test_quantity_corrections_preserve_density_and_original(original):
    untouched = copy.deepcopy(original)
    rows = photo_edit_rows(original)
    rows[0]["quantity_grams"] = 200
    edited = apply_photo_edits(original, rows)
    assert edited.total_calories == 337.5
    assert edited.food_items[0].carbs_g == 56.4
    assert original == untouched


def test_rename_delete_and_add_with_explicit_sources(original):
    rows = photo_edit_rows(original)[:1]
    rows[0].update(name="salmon", nutrition_source="Database: salmon")
    rows.append({"name": "olive oil", "quantity_grams": 10, "nutrition_source": "Database: olive oil"})
    edited = apply_photo_edits(original, rows)
    assert [item.name for item in edited.food_items] == ["salmon", "olive oil"]
    assert edited.total_calories == pytest.approx(296.4)
    assert edited.total_fat_g == pytest.approx(23.4)


def test_manual_nutrients_are_per_100g(original):
    edited = apply_photo_edits(original, [{"name": "homemade dish", "quantity_grams": 250,
                                         "nutrition_source": "Manual", "calories": 100,
                                         "protein_g": 3, "carbs_g": 10, "fat_g": 4}])
    assert edited.total_calories == 250
    assert edited.total_protein_g == 7.5


@pytest.mark.parametrize("change", [
    {"quantity_grams": 0}, {"quantity_grams": float("nan")}, {"name": ""},
    {"name": "different food"}, {"nutrition_source": "Database: invented"},
    {"nutrition_source": "Manual", "calories": None},
    {"nutrition_source": "Manual", "protein_g": -1},
])
def test_invalid_corrections_are_rejected(original, change):
    rows = photo_edit_rows(original)
    rows[0].update(change)
    with pytest.raises((ValueError, TypeError)):
        apply_photo_edits(original, rows)


def test_empty_meal_and_new_photo_source_are_rejected(original):
    with pytest.raises(ValueError, match="at least one"):
        apply_photo_edits(original, [])
    with pytest.raises(ValueError, match="new ingredients"):
        apply_photo_edits(original, [{"name": "rice", "quantity_grams": 100,
                                     "nutrition_source": "Photo estimate"}])


def test_saving_corrected_meal_updates_existing_totals_and_isolates_users(original, tmp_path, monkeypatch):
    monkeypatch.setattr("nutricoach.tools.SECRETS_DIR", tmp_path)
    config = {"configurable": {"username": "alice"}}
    result = apply_photo_edits(original, photo_edit_rows(original))
    save_photo_analysis(result, config, meal_name="Lunch")
    save_photo_analysis(result, config, meal_name="Dinner")
    log = MemoryManager("alice", tmp_path).load_daily_log(date.today().isoformat())
    assert len(log.meals) == 2
    assert log.total_calories == 414
    assert log.total_protein_g == 18.4
    assert log.meals[0].description.startswith("Lunch: rice")
    assert MemoryManager("bob", tmp_path).load_todays_log() is None


def test_streamlit_edits_save_once_and_survive_reruns(original, tmp_path, monkeypatch):
    from nutricoach import app

    monkeypatch.setattr("nutricoach.tools.SECRETS_DIR", tmp_path)
    at = AppTest.from_string(
        'from nutricoach import app\napp.initialize_session_state()\napp.display_photo_review()'
    )
    at.session_state["username"] = "alice"
    at.session_state["photo_result"] = original
    at.session_state["photo_review_id"] = "test"
    at.run()
    assert not at.exception
    assert MemoryManager("alice", tmp_path).load_todays_log() is None
    edits = {
        "edited_rows": {0: {"name": "salmon", "quantity_grams": 200,
                            "nutrition_source": "Database: salmon"}},
        "deleted_rows": [1], "added_rows": [],
    }
    at.session_state["photo_editor_test"] = edits
    at.run()
    assert not at.exception
    assert any("416 kcal" in item.value for item in at.markdown)
    at.text_input(key="photo_name_test").set_value("Corrected lunch")
    # AppTest does not serialize data_editor edits when another widget changes.
    at.session_state["photo_editor_test"] = edits
    at.button(key="photo_save_test").click().run()
    assert not at.exception
    assert at.session_state["photo_logged"]
    at.run()
    log = MemoryManager("alice", tmp_path).load_todays_log()
    assert len(log.meals) == 1
    assert log.total_calories == 416
    assert log.meals[0].description.startswith("Corrected lunch: salmon (200g")
    assert not at.button
    assert any("416 kcal" in item.value for item in at.markdown)


def test_upload_analysis_is_not_repeated_by_editing_and_new_photo_clears_draft(original, tmp_path, monkeypatch):
    from nutricoach import app

    monkeypatch.setattr(app, "SECRETS_DIR", tmp_path)
    photo = SimpleNamespace(name="meal.jpg", getvalue=lambda: b"first photo")
    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: photo)
    monkeypatch.setattr(app.st, "image", lambda *args, **kwargs: None)
    invoke = MagicMock(return_value=original)
    monkeypatch.setattr(app, "analyze_food_image", SimpleNamespace(invoke=invoke))
    at = AppTest.from_string(
        'from nutricoach import app\napp.initialize_session_state()\napp.display_food_analysis()'
    ).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert invoke.call_count == 1
    assert not list(tmp_path.glob("*.jpg"))
    at.run()
    assert invoke.call_count == 1
    photo.getvalue = lambda: b"second photo"
    at.run()
    assert not at.exception
    assert "photo_result" not in at.session_state
    assert invoke.call_count == 1
