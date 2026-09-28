"""
NutriCoach LangGraph agent — v2 rewrite.

Changes from v1 (based on feedback):
- Dropped the expensive classify_intent LLM call (was "theater" — never routed)
- Removed ghost state fields (user_profile, nutrition_targets, todays_log)
- Uses SqliteSaver checkpointer for persistence (replaces ~80 lines of manual JSON)
- Kept StateGraph instead of create_react_agent for future conditional routing
"""

import sqlite3
import time
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from shared.config import OPENROUTER_MODEL_ID, SECRETS_DIR
from shared.memory import MemoryManager
from nutricoach.tools import ALL_TOOLS


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
- NEVER claim you logged or calculated something unless you actually called the tool
  in this conversation. If a tool call failed, say so.
- The USER CONTEXT section below already contains the user's profile, targets, and
  today's log when they exist; read it before asking the user for information.
- Be encouraging but honest about progress.
- Base recommendations on the user's profile and goals from context.
- When the user reports what they ate, use the log_daily_intake tool and include your
  calorie/macro estimates for the meal (use lookup_food_nutrition for grounded values).
- When the user reports drinking water, use log_water_intake.
- For "what can I still eat today?" questions, use get_remaining_daily_budget.
- When asked about progress or trends, use the get_progress_summary tool.
- For an end-of-week review, use generate_weekly_summary.
- When nutrition targets are needed, use calculate_personalized_nutrition_targets.
- When the user shares a food photo, use the analyze_food_image tool.
- After the user accepts a meal plan, save it with save_meal_plan. Use get_meal_plan
  to check today's planned meals or to build a grocery list from the plan.
- Food photo values are estimates. Do not log a photo analysis unless the user explicitly asks to log it.
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
    openrouter_api_key: str,
    username: str,
    use_checkpointer: bool = True,
    model_id: str = OPENROUTER_MODEL_ID,
) -> Any:
    """
    Build and return the NutriCoach LangGraph agent.

    Args:
        openrouter_api_key: OpenRouter API key
        username: Current user's username (for memory access)
        use_checkpointer: Whether to use SqliteSaver for persistence
        model_id: OpenRouter model identifier

    Returns:
        Compiled LangGraph
    """
    # Initialize LLM with tools bound
    llm = ChatOpenAI(
        model=model_id,
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
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
        human_indices = [i for i, m in enumerate(conversation_messages) if isinstance(m, HumanMessage)]
        if len(human_indices) > 12:
            conversation_messages = conversation_messages[human_indices[-12]:]

        all_messages = [system_msg] + conversation_messages
        # Free-tier providers intermittently return 500/429 in a 200 body,
        # which the SDK does not retry; retry those here.
        last_exc = None
        response = None
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(all_messages)
            except Exception as e:
                msg = str(e).lower()
                if any(tok in msg for tok in ("500", "429", "rate", "internal server")):
                    last_exc = e
                    # free-models-per-min caps need a window-sized backoff
                    time.sleep(20 * (attempt + 1))
                    continue
                raise
            # Some free-tier models return an empty completion; retry those too.
            if response.content or getattr(response, "tool_calls", None):
                return {"messages": [response]}
            time.sleep(1)
        if response is not None:
            if not response.content:
                response.content = (
                    "Done. Let me know if you want anything else logged or checked."
                )
            return {"messages": [response]}
        raise last_exc

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
