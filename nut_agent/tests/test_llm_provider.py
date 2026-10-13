"""Provider isolation, local tool decoding and Streamlit selection."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_provider import create_client, request_settings, LOCAL_MODEL_ID
from shared.local_qwen import parse_tool_calls
from nutricoach.food_vision.vlm_analyzer import VLMAnalyzerSingleShot


def test_provider_credentials_and_urls_are_isolated(monkeypatch):
    import openai
    factory = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    create_client("groq")
    factory.assert_called_once_with(api_key="groq-key", base_url="https://api.groq.com/openai/v1")
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_client("groq")


def test_local_client_needs_no_api_key_and_is_cached(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    first = create_client("local")
    assert first is create_client("local")
    assert first.model is None
    assert VLMAnalyzerSingleShot(provider="local").reasoning_effort == "none"


def test_switching_local_precision_releases_previous_weights(monkeypatch):
    import torch
    import transformers
    from shared import local_qwen
    previous = local_qwen.LocalQwenClient(LOCAL_MODEL_ID, "test")
    previous.model = previous.processor = object()
    monkeypatch.setattr(local_qwen, "_ACTIVE_CLIENT", previous)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", MagicMock())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", MagicMock())
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock())
    selected = local_qwen.LocalQwenClient(LOCAL_MODEL_ID, "test", device="cpu")
    selected.load()
    assert previous.model is previous.processor is None
    assert selected.model is not None
    assert local_qwen._ACTIVE_CLIENT is selected


def test_groq_image_call_uses_native_settings(tmp_path):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = VLMAnalyzerSingleShot(provider="groq", api_key="test", reasoning_effort="none")
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [SimpleNamespace(message=SimpleNamespace(
        content=json.dumps({"items": [{"name": "rice", "quantity_grams": 100, "calories": 130}]})))]
    analyzer._client = client
    assert analyzer.analyze(str(image)).total_calories == 130
    settings = client.chat.completions.create.call_args.kwargs
    assert settings["model"] == "qwen/qwen3.8-27b"
    assert settings["max_completion_tokens"] == 8192
    assert settings["reasoning_effort"] == "none"
    assert settings["extra_body"] == {"reasoning_format": "hidden"}
    assert "max_tokens" not in settings
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        request_settings("groq", "xhigh")


TOOLS = [{"type": "function", "function": {"name": "calculate", "parameters": {
    "type": "object", "properties": {"weight": {"type": "number"}, "name": {"type": "string"}},
    "required": ["weight"]}}}]


def test_local_tool_arguments_preserve_schema_types():
    text = '<tool_call><function=calculate><parameter=weight>62.5</parameter><parameter=name>123</parameter></function></tool_call>'
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == ""
    assert json.loads(calls[0]["function"]["arguments"]) == {"weight": 62.5, "name": "123"}
    assert calls[0]["function"]["name"] == "calculate"


def test_local_chat_client_supports_langchain_tool_calls(monkeypatch):
    from langchain_openai import ChatOpenAI
    from openai.types.chat import ChatCompletion
    from shared.local_qwen import LocalQwenClient
    from shared.llm_provider import chat_settings
    client = LocalQwenClient(LOCAL_MODEL_ID, "test")
    completion = ChatCompletion.model_validate({"id": "test", "object": "chat.completion", "created": 0,
        "model": LOCAL_MODEL_ID, "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", "tool_calls": [{"id": "call_test", "type": "function",
                "function": {"name": "calculate", "arguments": '{"weight": 62.5}'}}]}}]})
    client.create = MagicMock(return_value=completion)
    monkeypatch.setattr("shared.llm_provider.create_client", lambda *args, **kwargs: client)
    response = ChatOpenAI(**chat_settings("local", "", LOCAL_MODEL_ID, "none")).bind_tools(TOOLS).invoke("Calculate")
    assert response.tool_calls[0]["args"] == {"weight": 62.5}
    client.create.assert_called_once()


@pytest.mark.parametrize("text", [
    '<tool_call><function=calculate><parameter=weight>62</parameter></function>',
    '<tool_call><function=unknown></function></tool_call>',
    '<tool_call><function=calculate></function></tool_call>',
    '<tool_call><function=calculate><parameter=weight>wrong</parameter></function></tool_call>',
    '<tool_call><function=calculate><parameter=weight>62</parameter><parameter=weight>65</parameter></function></tool_call>',
])
def test_malformed_local_tool_calls_are_rejected(text):
    with pytest.raises(ValueError):
        parse_tool_calls(text, TOOLS)


def test_streamlit_provider_switch_updates_chat_and_vision_models(monkeypatch):
    from nutricoach import app
    monkeypatch.setattr(app, "build_nutricoach_graph", MagicMock())
    at = AppTest.from_string(
        'from nutricoach import app\napp.initialize_session_state()\napp.display_llm_settings()', default_timeout=15
    ).run(timeout=15)
    assert not at.exception
    assert at.selectbox(key="llm_provider").value == "openrouter"
    at.selectbox(key="llm_provider").set_value("groq").run()
    assert not at.exception
    assert at.session_state["chat_model_id"] == "qwen/qwen3.8-27b"
    assert at.session_state["vision_model_id"] == "qwen/qwen3.8-27b"
    assert at.text_input(key="groq_api_key")
    at.selectbox(key="llm_provider").set_value("local").run()
    assert not at.exception
    assert at.session_state["chat_model_id"] == LOCAL_MODEL_ID
    assert at.session_state["vision_model_id"] == LOCAL_MODEL_ID
    assert at.session_state["reasoning_effort"] == "none"
    assert all("API Key" not in item.label for item in at.text_input)
