"""Tests for nutricoach.intent module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.intent import classify_intent, IntentSchema


def _make_mock_llm(intent: str, confidence: float = 0.9):
    """Create a mock LLM that returns a specific intent via with_structured_output."""
    mock_llm = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.invoke.return_value = IntentSchema(
        intent=intent,
        confidence=confidence,
        reasoning=f"Mock classified as {intent}",
    )
    mock_llm.with_structured_output.return_value = mock_classifier
    return mock_llm


class TestClassifyIntent:
    def test_no_profile_routes_to_profile_update(self):
        mock_llm = _make_mock_llm("general_chat")
        result = classify_intent(mock_llm, "Hello there!", has_profile=False)
        assert result.intent == "profile_update"
        assert result.confidence == 1.0
        # LLM should NOT be called when has_profile=False
        mock_llm.with_structured_output.assert_not_called()

    def test_general_chat(self):
        mock_llm = _make_mock_llm("general_chat")
        result = classify_intent(mock_llm, "Hello, how are you?")
        assert result.intent == "general_chat"

    def test_calculate_targets(self):
        mock_llm = _make_mock_llm("calculate_targets")
        result = classify_intent(mock_llm, "Calculate my daily nutrition targets")
        assert result.intent == "calculate_targets"

    def test_meal_planning(self):
        mock_llm = _make_mock_llm("meal_planning")
        result = classify_intent(mock_llm, "What should I eat for dinner tonight?")
        assert result.intent == "meal_planning"

    def test_log_daily(self):
        mock_llm = _make_mock_llm("log_daily")
        result = classify_intent(mock_llm, "I had oatmeal for breakfast and salad for lunch")
        assert result.intent == "log_daily"

    def test_analyze_progress(self):
        mock_llm = _make_mock_llm("analyze_progress")
        result = classify_intent(mock_llm, "How have I been doing this week?")
        assert result.intent == "analyze_progress"

    def test_profile_update(self):
        mock_llm = _make_mock_llm("profile_update")
        result = classify_intent(mock_llm, "I want to change my goal to lose weight")
        assert result.intent == "profile_update"

    def test_llm_error_falls_back_to_general_chat(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = Exception("API error")
        result = classify_intent(mock_llm, "Some message")
        assert result.intent == "general_chat"
        assert result.confidence == 0.5

    def test_prompt_contains_user_message(self):
        mock_llm = _make_mock_llm("general_chat")
        classify_intent(mock_llm, "I want tacos for lunch")
        # Verify the prompt was sent to the classifier
        mock_classifier = mock_llm.with_structured_output.return_value
        call_args = mock_classifier.invoke.call_args[0][0]
        assert "I want tacos for lunch" in call_args
