"""Tests for nutricoach.agent module (v2)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START

from nutricoach.agent import (
    NutriCoachState,
    SYSTEM_PROMPT,
    create_initial_state,
)
from nutricoach.tools import log_daily_intake
from nutricoach.tools import analyze_food_image
from nutricoach.food_vision.base import FoodAnalysisResult, FoodItem
from nutricoach.food_vision.vlm_analyzer import VLMAnalyzer, VLMAnalyzerSingleShot
from nutricoach.food_vision.rag_vlm_analyzer import RAGVLMAnalyzer
from nutricoach.food_vision.clip_analyzer import CLIPFoodAnalyzer
from shared.config import OPENROUTER_MODEL_ID


class TestCreateInitialState:
    def test_default_state(self):
        state = create_initial_state()
        assert state["messages"] == []

    def test_state_keys(self):
        state = create_initial_state()
        expected_keys = {"messages"}
        assert set(state.keys()) == expected_keys


class TestSystemPrompt:
    def test_prompt_mentions_tools(self):
        assert "log_daily_intake" in SYSTEM_PROMPT
        assert "get_progress_summary" in SYSTEM_PROMPT
        assert "calculate_personalized_nutrition_targets" in SYSTEM_PROMPT
        assert "analyze_food_image" in SYSTEM_PROMPT

    def test_prompt_has_guidelines(self):
        assert "Guidelines:" in SYSTEM_PROMPT
        assert "NutriCoach" in SYSTEM_PROMPT


class TestBuildGraph:
    """Test graph construction by mocking external dependencies."""

    @patch("nutricoach.agent._get_checkpointer", return_value=None)
    @patch("nutricoach.agent.ChatOpenAI")
    def test_custom_openrouter_model(self, mock_llm_cls, mock_cp):
        from nutricoach.agent import build_nutricoach_graph
        build_nutricoach_graph("fake-api-key", "testuser", model_id="anthropic/claude-sonnet-4.5")
        mock_llm_cls.assert_called_once_with(
            model="anthropic/claude-sonnet-4.5",
            api_key="fake-api-key",
            base_url="https://openrouter.ai/api/v1",
        )

    @patch("nutricoach.agent._get_checkpointer", return_value=None)
    @patch("nutricoach.agent.ChatOpenAI")
    @patch("nutricoach.agent.MemoryManager")
    def test_graph_compiles(self, mock_memory_cls, mock_llm_cls, mock_cp):
        """The graph should compile without errors."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm_cls.return_value = mock_llm

        from nutricoach.agent import build_nutricoach_graph
        graph = build_nutricoach_graph("fake-api-key", "testuser")
        assert graph is not None
        mock_llm_cls.assert_called_once_with(model="stealth/space-bunny-alpha", api_key="fake-api-key", base_url="https://openrouter.ai/api/v1")

    @patch("nutricoach.agent._get_checkpointer", return_value=None)
    @patch("nutricoach.agent.ChatOpenAI")
    @patch("nutricoach.agent.MemoryManager")
    def test_graph_has_expected_nodes(self, mock_memory_cls, mock_llm_cls, mock_cp):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm_cls.return_value = mock_llm

        from nutricoach.agent import build_nutricoach_graph
        graph = build_nutricoach_graph("fake-api-key", "testuser")

        node_names = set(graph.nodes.keys())
        # v2: no classify_intent node, just agent + tool_node
        assert "agent" in node_names
        assert "tool_node" in node_names

    @patch("nutricoach.agent._get_checkpointer", return_value=None)
    @patch("nutricoach.agent.ChatOpenAI")
    @patch("nutricoach.agent.MemoryManager")
    def test_llm_bind_tools_called(self, mock_memory_cls, mock_llm_cls, mock_cp):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm_cls.return_value = mock_llm

        from nutricoach.agent import build_nutricoach_graph
        build_nutricoach_graph("fake-api-key", "testuser")

        mock_llm_cls.assert_called_once_with(model="stealth/space-bunny-alpha", api_key="fake-api-key", base_url="https://openrouter.ai/api/v1")
        mock_llm.bind_tools.assert_called_once()
        tools_arg = mock_llm.bind_tools.call_args[0][0]
        assert len(tools_arg) >= 5  # 5 tools including analyze_food_image


class TestShouldUseToolsLogic:
    """Test the tool routing logic in isolation."""

    def test_message_with_tool_calls_routes_to_tools(self):
        msg = AIMessage(content="", tool_calls=[
            {"name": "log_daily_intake", "args": {"meals_description": "oatmeal"}, "id": "1"}
        ])
        state = {"messages": [msg]}

        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            result = "tool_node"
        else:
            result = "__end__"
        assert result == "tool_node"

    def test_message_without_tool_calls_routes_to_end(self):
        msg = AIMessage(content="Here is your meal plan!")
        state = {"messages": [msg]}

        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            result = "tool_node"
        else:
            result = "__end__"
        assert result == "__end__"


    def test_empty_messages_routes_to_end(self):
        state = {"messages": []}
        if not state["messages"]:
            result = "__end__"
        else:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                result = "tool_node"
            else:
                result = "__end__"
        assert result == "__end__"

    def test_human_message_routes_to_end(self):
        msg = HumanMessage(content="hello")
        state = {"messages": [msg]}
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            result = "tool_node"
        else:
            result = "__end__"
        assert result == "__end__"


def test_tool_node_isolates_users(tmp_path, monkeypatch):
    monkeypatch.setattr("nutricoach.tools.SECRETS_DIR", tmp_path)
    builder = StateGraph(NutriCoachState)
    builder.add_node("tools", ToolNode([log_daily_intake]))
    builder.add_edge(START, "tools")
    graph = builder.compile()
    for username in ("alice", "bob"):
        message = AIMessage(content="", tool_calls=[{
            "name": "log_daily_intake",
            "args": {"meals_description": username},
            "id": username,
        }])
        graph.invoke({"messages": [message]}, config={"configurable": {"username": username}})
    from shared.memory import MemoryManager
    assert MemoryManager("alice", tmp_path).load_todays_log().meals[0].description == "alice"
    assert MemoryManager("bob", tmp_path).load_todays_log().meals[0].description == "bob"


def test_photo_estimate_is_logged_only_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr("nutricoach.tools.SECRETS_DIR", tmp_path)
    result = FoodAnalysisResult(method="rag_vlm", food_items=[FoodItem("rice", 100, calories=130)])
    result.compute_totals()
    with patch("nutricoach.food_vision.rag_vlm_analyzer.RAGVLMAnalyzer") as analyzer:
        analyzer.return_value.analyze.return_value = result
        image = tmp_path / "meal.jpg"
        image.write_bytes(b"image")
        config = {"configurable": {"username": "alice", "openrouter_api_key": "vision-key", "vision_model_id": "test/vision"}}
        analyze_food_image.invoke({"image_path": str(image)}, config=config)
        analyzer.assert_called_with(api_key="vision-key", model="test/vision")
        from shared.memory import MemoryManager
        memory = MemoryManager("alice", tmp_path)
        assert memory.load_todays_log() is None
        analyze_food_image.invoke({"image_path": str(image), "log_meal": True}, config=config)
        assert memory.load_todays_log().meals[0].estimated_calories == 130


def test_vision_methods_default_to_shared_openrouter_model():
    assert VLMAnalyzer().model == OPENROUTER_MODEL_ID
    assert VLMAnalyzerSingleShot().model == OPENROUTER_MODEL_ID
    assert RAGVLMAnalyzer().model == OPENROUTER_MODEL_ID
    assert CLIPFoodAnalyzer().llm_model == OPENROUTER_MODEL_ID
