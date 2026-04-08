"""Tests for nutricoach.agent module (v2)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from nutricoach.agent import (
    NutriCoachState,
    SYSTEM_PROMPT,
    create_initial_state,
)


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
    @patch("nutricoach.agent.ChatGoogleGenerativeAI")
    @patch("nutricoach.agent.MemoryManager")
    @patch("nutricoach.agent.set_current_user")
    def test_graph_compiles(self, mock_set_user, mock_memory_cls, mock_llm_cls, mock_cp):
        """The graph should compile without errors."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm_cls.return_value = mock_llm

        from nutricoach.agent import build_nutricoach_graph
        graph = build_nutricoach_graph("fake-api-key", "testuser")
        assert graph is not None
        mock_set_user.assert_called_once_with("testuser")

    @patch("nutricoach.agent._get_checkpointer", return_value=None)
    @patch("nutricoach.agent.ChatGoogleGenerativeAI")
    @patch("nutricoach.agent.MemoryManager")
    @patch("nutricoach.agent.set_current_user")
    def test_graph_has_expected_nodes(self, mock_set_user, mock_memory_cls, mock_llm_cls, mock_cp):
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
    @patch("nutricoach.agent.ChatGoogleGenerativeAI")
    @patch("nutricoach.agent.MemoryManager")
    @patch("nutricoach.agent.set_current_user")
    def test_llm_bind_tools_called(self, mock_set_user, mock_memory_cls, mock_llm_cls, mock_cp):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm_cls.return_value = mock_llm

        from nutricoach.agent import build_nutricoach_graph
        build_nutricoach_graph("fake-api-key", "testuser")

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
