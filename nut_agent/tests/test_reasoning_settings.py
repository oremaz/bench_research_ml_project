"""Reasoning settings in both Streamlit entry points, without live API calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPENROUTER_REASONING_EFFORT, openrouter_reasoning


def test_default_and_invalid_effort():
    assert OPENROUTER_REASONING_EFFORT == "high"
    assert openrouter_reasoning() == {"reasoning": {"effort": "high"}}
    with pytest.raises(ValueError, match="Unsupported"):
        openrouter_reasoning("invalid")


def test_recipe_selector_and_cached_predictor(monkeypatch):
    from recipe_lab import app

    at = AppTest.from_string("from recipe_lab.app import main\nmain()").run()
    assert not at.exception
    assert at.selectbox(key="reasoning_effort").value == "high"
    at.selectbox(key="reasoning_effort").set_value("low").run()
    assert not at.exception
    assert at.session_state["reasoning_effort"] == "low"
    predictor = MagicMock(api_key="test", model_id="test/model", reasoning_effort="high")
    state = {"api_key": "test", "model_id": "test/model", "reasoning_effort": "low", "food_predictor": predictor}
    monkeypatch.setattr(app.st, "session_state", state)
    assert app.get_predictor() is predictor
    assert predictor.reasoning_effort == "low"


def test_nutricoach_selector_rebuilds_graph_and_keeps_thread(monkeypatch):
    from nutricoach import app

    for name in ("display_user_profile", "display_chat_interface", "display_food_analysis",
                 "display_nutrition_dashboard", "display_quick_actions", "display_daily_results"):
        monkeypatch.setattr(app, name, lambda: None)
    graph_builder = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(app, "build_nutricoach_graph", graph_builder)
    at = AppTest.from_string(
        'import streamlit as st\nfrom nutricoach import app\n'
        'st.session_state["thread_id"] = "test-thread"\n'
        'st.session_state["api_key"] = "test-key"\n'
        'app.initialize_session_state()\napp.main_app()\n'
        'st.session_state["test_config"] = app._thread_config()'
    ).run(timeout=15)
    assert not at.exception
    assert at.selectbox(key="reasoning_effort").value == "high"
    assert graph_builder.call_args.kwargs["reasoning_effort"] == "high"
    at.selectbox(key="reasoning_effort").set_value("low").run(timeout=15)
    assert not at.exception
    assert graph_builder.call_count == 2
    assert graph_builder.call_args.kwargs["reasoning_effort"] == "low"
    assert at.session_state["test_config"]["configurable"]["reasoning_effort"] == "low"
    assert at.session_state["thread_id"] == "test-thread"
