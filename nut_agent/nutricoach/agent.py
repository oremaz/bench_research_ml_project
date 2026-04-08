"""
NutriCoach LangGraph agent — v2 rewrite.

Changes from v1 (based on feedback):
- Dropped the expensive classify_intent LLM call (was "theater" — never routed)
- Removed ghost state fields (user_profile, nutrition_targets, todays_log)
- Uses SqliteSaver checkpointer for persistence (replaces ~80 lines of manual JSON)
- Kept StateGraph instead of create_react_agent for future conditional routing
"""

import os
import sqlite3
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from shared.config import GEMINI_MODEL, SECRETS_DIR
from shared.memory import MemoryManager
from nutricoach.tools import ALL_TOOLS, set_current_user


class NutriCoachState(TypedDict):
    """Minimal state — only what the graph actually uses."""
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are NutriCoach, a personalized AI nutritionist assistant.

Your role is to help users with their nutrition goals through:
- Calculating personalized nutrition targets (BMR, TDEE, macros)
- Creating meal plans based on their preferences and constraints
- Tracking daily food intake and progress
- Analyzing trends and providing coaching
- Analyzing food photos to estimate calories and macros

Guidelines:
- Use the available tools to perform calculations and log data — do NOT perform math yourself.
- Be encouraging but honest about progress.
- Base recommendations on the user's profile and goals from context.
- When the user reports what they ate, use the log_daily_intake tool.
- When asked about progress or trends, use the get_progress_summary tool.
- When nutrition targets are needed, use calculate_personalized_nutrition_targets.
- When the user shares a food photo, use the analyze_food_image tool.
- Keep responses concise and actionable.
"""


def _get_checkpointer(username: str):
    """Create a SqliteSaver checkpointer for persistent conversation state."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = SECRETS_DIR / f"{username}_checkpoints.db"
        SECRETS_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        # Fallback: no persistence (works in-memory only)
        return None


def build_nutricoach_graph(
    google_api_key: str,
    username: str,
    use_checkpointer: bool = True,
) -> Any:
    """
    Build and return the NutriCoach LangGraph agent.

    Args:
        google_api_key: Google API key for Gemini
        username: Current user's username (for memory access)
        use_checkpointer: Whether to use SqliteSaver for persistence

    Returns:
        Compiled LangGraph
    """
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Set user context for tools
    set_current_user(username)

    # Initialize LLM with tools bound
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Memory manager for context assembly
    memory = MemoryManager(username, SECRETS_DIR)

    # --- Node definitions ---

    def agent_node(state: NutriCoachState) -> dict:
        """
        Main agent node. Calls LLM with tools bound.
        Context from memory system is injected as a system message.
        """
        context = memory.assemble_context()

        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\n{context}"

        system_msg = SystemMessage(content=system_content)

        conversation_messages = [
            m for m in state.get("messages", [])
            if not isinstance(m, SystemMessage)
        ]

        all_messages = [system_msg] + conversation_messages
        response = llm_with_tools.invoke(all_messages)
        return {"messages": [response]}

    def should_use_tools(state: NutriCoachState) -> str:
        """Route to tool_node if the LLM wants to call tools, else END."""
        messages = state.get("messages", [])
        if not messages:
            return END

        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tool_node"
        return END

    # --- Build the graph ---

    tool_node = ToolNode(ALL_TOOLS)

    builder = StateGraph(NutriCoachState)

    builder.add_node("agent", agent_node)
    builder.add_node("tool_node", tool_node)

    # Simplified flow: START -> agent -> (tool_node -> agent)* -> END
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_use_tools)
    builder.add_edge("tool_node", "agent")

    # Compile with optional checkpointer
    checkpointer = _get_checkpointer(username) if use_checkpointer else None
    return builder.compile(checkpointer=checkpointer)


def create_initial_state() -> NutriCoachState:
    """Create the initial state for a new conversation."""
    return {
        "messages": [],
    }
