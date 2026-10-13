"""Opt-in real local Qwen chat and RGB inference on the benchmark VM."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(os.getenv("RUN_LOCAL_QWEN_TESTS") != "1",
                               reason="Set RUN_LOCAL_QWEN_TESTS=1 with cached Qwen weights")


@pytest.fixture(scope="module")
def meal_image():
    root = Path(__file__).parents[1] / "nutricoach/food_vision/data/nutrition5k_500"
    manifest = json.loads((root / "manifest.json").read_text())
    return root / "images" / f"{manifest['dish_ids'][0]}.png"


def test_local_single_shot_photo_without_api_credentials(meal_image, monkeypatch):
    from nutricoach.food_vision.vlm_analyzer import VLMAnalyzerSingleShot
    from nutricoach.food_vision.evaluate_nutrition5k import prediction_values
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = VLMAnalyzerSingleShot(provider="local", device="cuda").analyze(str(meal_image))
    assert not result.error, result.error
    assert prediction_values(result)["calories"] > 0


def test_local_nutricoach_chat_executes_nutrition_tool(tmp_path, monkeypatch):
    from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
    from nutricoach.agent import build_nutricoach_graph
    monkeypatch.setattr("nutricoach.agent.SECRETS_DIR", tmp_path)
    monkeypatch.setattr("nutricoach.tools.SECRETS_DIR", tmp_path)
    graph = build_nutricoach_graph("", "local_test", provider="local", local_device="cuda", use_checkpointer=False)
    result = graph.invoke({"messages": [HumanMessage(content=(
        "Use calculate_personalized_nutrition_targets for a 28 year old female, 62 kg, "
        "168 cm, light activity, maintain weight. Then report the calculated calorie target."))]},
        config={"recursion_limit": 10, "configurable": {"username": "local_test", "llm_provider": "local"}})
    assert any(isinstance(message, ToolMessage) and message.name == "calculate_personalized_nutrition_targets"
               for message in result["messages"])
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content


@pytest.mark.skipif(os.getenv("RUN_LOCAL_QWEN_4BIT_TEST") != "1", reason="Opt-in separate 4-bit memory test")
def test_local_4bit_photo_with_32_gib_gpu_limit(meal_image):
    import resource
    import torch
    from nutricoach.food_vision.vlm_analyzer import VLMAnalyzerSingleShot
    from nutricoach.food_vision.evaluate_nutrition5k import prediction_values
    torch.cuda.set_per_process_memory_fraction(32 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    result = VLMAnalyzerSingleShot(provider="local", device="cuda", quantization="4bit").analyze(str(meal_image))
    assert not result.error, result.error
    assert prediction_values(result)["calories"] > 0
    reserved = torch.cuda.max_memory_reserved()
    assert reserved <= 32 * 1024**3
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "peak_gpu_reserved_gib": reserved / 1024**3,
                      "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                      "calories": result.total_calories, "inference_seconds": result.elapsed_seconds}))
