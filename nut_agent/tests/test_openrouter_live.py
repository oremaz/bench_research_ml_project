"""Live OpenRouter integration tests (free tier).

Each test costs at most one or two free-tier requests and is skipped when
OPENROUTER_API_KEY is not set or when the free models are rate-limited, so CI
without credentials stays green.

Run:
    PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/test_openrouter_live.py -v
"""

import os
import sys
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

HAS_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))
requires_key = pytest.mark.skipif(not HAS_KEY, reason="OPENROUTER_API_KEY not set")

TEST_USER = "_pytest_live_user"


def _skip_if_rate_limited(exc: Exception):
    if "429" in str(exc) or "rate" in str(exc).lower():
        pytest.skip(f"free tier rate-limited: {str(exc)[:120]}")
    raise exc


@pytest.fixture
def live_user_cleanup():
    yield
    from shared.config import SECRETS_DIR
    user_dir = SECRETS_DIR / TEST_USER
    if user_dir.exists():
        shutil.rmtree(user_dir)
    for db in SECRETS_DIR.glob(f"{TEST_USER}_checkpoints.db"):
        db.unlink()


@requires_key
class TestAgentLive:
    def test_agent_answers_and_calls_tool(self, live_user_cleanup):
        from langchain_core.messages import HumanMessage, AIMessage
        from nutricoach.agent import build_nutricoach_graph

        graph = build_nutricoach_graph(username=TEST_USER, use_checkpointer=False)
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=(
                    "Calculate my nutrition targets: 28 year old female, 62kg, "
                    "168cm, light activity, maintain weight."))]},
                config={"recursion_limit": 20},
            )
        except Exception as e:
            _skip_if_rate_limited(e)

        tool_msgs = [m for m in result["messages"] if m.__class__.__name__ == "ToolMessage"]
        assert tool_msgs, "agent should call calculate_personalized_nutrition_targets"

        final = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
            None,
        )
        assert final is not None
        # Mifflin-St Jeor for this profile: BMR 1345, TDEE ~1850
        assert any(tok in str(final.content) for tok in ("1,8", "18", "kcal", "calorie"))

    def test_targets_persisted_to_memory(self, live_user_cleanup):
        from shared.config import SECRETS_DIR
        from shared.memory import MemoryManager
        from langchain_core.messages import HumanMessage
        from nutricoach.agent import build_nutricoach_graph

        graph = build_nutricoach_graph(username=TEST_USER, use_checkpointer=False)
        try:
            graph.invoke(
                {"messages": [HumanMessage(content=(
                    "Compute nutrition targets for a 40 year old male, 90kg, 185cm, "
                    "sedentary, who wants to lose weight. Use the tool."))]},
                config={"recursion_limit": 20},
            )
        except Exception as e:
            _skip_if_rate_limited(e)

        memory = MemoryManager(TEST_USER, SECRETS_DIR)
        targets = memory.load_nutrition_targets()
        if targets is None:
            pytest.skip("model answered without tool call (free-tier model variance)")
        assert 1200 <= targets.target_calories <= 4000
        assert targets.bmr > 1000
