"""
Redesigned NutriCoach LangGraph agent.
Uses intent classification, proper tool calling via bind_tools/ToolNode,
and structured memory system.
"""

import os
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from shared.config import GEMINI_MODEL, MAX_RECURSION_LIMIT, SECRETS_DIR
from shared.memory import MemoryManager
from nutricoach.intent import classify_intent, IntentSchema
from nutricoach.tools import ALL_TOOLS, set_current_user


class NutriCoachState(TypedDict):
    """State for the NutriCoach agent."""
    messages: Annotated[list, add_messages]
    user_profile: Dict[str, Any]
    nutrition_targets: Dict[str, Any]
    todays_log: Dict[str, Any]
    intent: str
    profile_setup_complete: bool


SYSTEM_PROMPT = """You are NutriCoach, a personalized AI nutritionist assistant.

Your role is to help users with their nutrition goals through:
- Calculating personalized nutrition targets (BMR, TDEE, macros)
- Creating meal plans based on their preferences and constraints
- Tracking daily food intake and progress
- Analyzing trends and providing coaching

Guidelines:
- Use the available tools to perform calculations and log data — do NOT perform math yourself.
- Be encouraging but honest about progress.
- Base recommendations on the user's profile and goals from context.
- When the user reports what they ate, use the log_daily_intake tool.
- When asked about progress or trends, use the get_progress_summary tool.
- When nutrition targets are needed, use calculate_personalized_nutrition_targets.
- Keep responses concise and actionable.
"""


def build_nutricoach_graph(google_api_key: str, username: str) -> Any:
    """
    Build and return the NutriCoach LangGraph agent.

    Args:
        google_api_key: Google API key for Gemini
        username: Current user's username (for memory access)

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

    def classify_intent_node(state: NutriCoachState) -> dict:
        """Classify the user's intent from the latest message."""
        messages = state.get("messages", [])
        if not messages:
            return {"intent": "general_chat"}

        latest = messages[-1]
        if not isinstance(latest, HumanMessage):
            return {"intent": "general_chat"}

        has_profile = state.get("profile_setup_complete", False)
        intent_result = classify_intent(llm, latest.content, has_profile)
        return {"intent": intent_result.intent}

    def agent_node(state: NutriCoachState) -> dict:
        """
        Main agent node. Calls LLM with tools bound.
        Context from memory system is injected as a system message.
        """
        # Assemble context from structured memory
        context = memory.assemble_context()
        intent = state.get("intent", "general_chat")

        # Build system message with context
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\n{context}"
        system_content += f"\n\nUser's current intent: {intent}"

        system_msg = SystemMessage(content=system_content)

        # Get conversation messages (exclude any prior system messages)
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

    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tool_node", tool_node)

    # Flow: START -> classify_intent -> agent -> (tool_node -> agent)* -> END
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "agent")
    builder.add_conditional_edges("agent", should_use_tools)
    builder.add_edge("tool_node", "agent")  # Loop back after tool execution

    return builder.compile()


def create_initial_state(profile_complete: bool = False) -> NutriCoachState:
    """Create the initial state for a new conversation."""
    return {
        "messages": [],
        "user_profile": {},
        "nutrition_targets": {},
        "todays_log": {},
        "intent": "general_chat",
        "profile_setup_complete": profile_complete,
    }
