"""Food vision behavior without model downloads or live APIs."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.vlm_analyzer import VLMAnalyzerSingleShot
from nutricoach.food_vision.compare import get_available_methods, run_comparison
from nutricoach.tools import analyze_food_image


@pytest.mark.parametrize("method", ["rf_detr", "clip_vlm", "rag_vlm", "grounded_vlm", "geometry_vlm_db"])
def test_removed_methods_are_rejected(method):
    from nutricoach.food_vision.report_nutrition5k import REPORT_METHODS
    assert method not in REPORT_METHODS
    with pytest.raises(ValueError, match="Unknown food vision methods"):
        run_comparison("unused.png", methods=[method])


@pytest.mark.parametrize("effort", ["high", "low"])
def test_single_shot_makes_one_image_call(tmp_path, effort):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = VLMAnalyzerSingleShot(api_key="test", model="test/vision", reasoning_effort=effort)
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [SimpleNamespace(
        message=SimpleNamespace(content=json.dumps({"items": [{
            "name": "rice", "quantity_grams": 200, "calories": 260,
            "protein_g": 5.4, "carbs_g": 56.4, "fat_g": 0.6,
        }]}))
    )]
    analyzer._client = client
    result = analyzer.analyze(str(image))
    assert result.error is None
    assert result.total_calories == 260
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning": {"effort": effort}}
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192
    contents = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert contents[1]["type"] == "image_url"
    assert result.method == "vlm_single"
    assert "vlm_chain" not in get_available_methods()


def test_foodlmm_default_agent_dispatch(tmp_path, monkeypatch):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    from nutricoach.food_vision.supervised_analyzers import total_result
    analyzer = MagicMock()
    analyzer.analyze.return_value = total_result("foodlmm_4bit", {
        "calories": 260, "mass_g": 200, "fat_g": 1, "carbs_g": 56, "protein_g": 5,
    }, 0)
    monkeypatch.setattr("nutricoach.food_vision.foodlmm_analyzer.get_foodlmm_analyzer", lambda: analyzer)
    result = analyze_food_image.invoke(
        {"image_path": str(image)},
        config={"configurable": {"food_vision_method": "dpf_reproduction"}},
    )
    analyzer.analyze.assert_called_once_with(str(image))
    assert result["method"] == "foodlmm_4bit"
    assert result["total_calories"] == 260
    assert set(analyze_food_image.tool_call_schema.model_fields) == {"image_path", "log_meal", "method"}


def test_dpf_loads_once_and_applies_exif_orientation(tmp_path, monkeypatch):
    from PIL import Image
    from nutricoach.food_vision.dpf_analyzer import DPFAnalyzer
    from nutricoach.food_vision import evaluate_paper_system as paper

    image_path = tmp_path / "meal.jpg"
    image = Image.new("RGB", (1200, 600))
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)
    model = MagicMock()
    model.predict.return_value = ({"calories": 260, "mass_g": 200, "fat_g": 1,
                                   "carbs_g": 56, "protein_g": 5}, "")
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr(paper, "prepare_release", lambda method: tmp_path)
    monkeypatch.setattr(paper, "select_device", lambda device: "mps")
    monkeypatch.setattr(paper, "DPFReproduction", constructor)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    analyzer = DPFAnalyzer()
    for _ in range(2):
        result = analyzer.analyze(str(image_path))
        assert result.food_items[0].name == "Whole dish"
        assert result.food_items[0].quantity_grams == 200
        assert result.total_calories == 260
    constructor.assert_called_once_with(tmp_path, "mps")
    assert model.predict.call_args.args[0].size == (512, 1024)


def test_photo_ui_offers_foodlmm_and_dpf():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(
        "from nutricoach import app\napp.initialize_session_state()\napp.display_food_analysis()"
    ).run(timeout=15)
    assert not at.exception
    assert at.selectbox[0].options == ["FoodLMM (4-bit)", "DPF (reproduction)"]
    assert any("FoodLMM (4-bit)" in item.value for item in at.caption)
    assert not at.text_input


def test_photo_method_switch_preserves_tab_containers():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(
        "from nutricoach import app\n"
        "import streamlit as st\n"
        "app.initialize_session_state()\n"
        "chat, food, recipe = st.tabs(['Chat', 'Food Analysis', 'Recipe Lab'])\n"
        "with chat:\n    st.subheader('Chat with NutriCoach')\n"
        "with food:\n    app.display_food_analysis()\n"
        "with recipe:\n    st.subheader('Smart Recipe Lab')\n"
    ).run(timeout=15)
    at.selectbox(key="food_vision_method").select("dpf_reproduction").run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Chat", "Food Analysis", "Recipe Lab"]
    assert [item.value for item in at.tabs[0].subheader] == ["Chat with NutriCoach"]
    assert [item.value for item in at.tabs[1].subheader] == ["Analyze Food Photo"]
    assert [item.value for item in at.tabs[2].subheader] == ["Smart Recipe Lab"]
    assert at.tabs[1].selectbox[0].value == "dpf_reproduction"
    assert not at.tabs[0].selectbox
    assert not at.tabs[2].selectbox


def test_foodlmm_agent_dispatch(tmp_path, monkeypatch):
    from nutricoach.food_vision.supervised_analyzers import total_result
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = MagicMock()
    analyzer.analyze.return_value = total_result("foodlmm_4bit", {
        "calories": 260, "mass_g": 200, "fat_g": 1, "carbs_g": 56, "protein_g": 5,
    }, 0)
    monkeypatch.setattr("nutricoach.food_vision.foodlmm_analyzer.get_foodlmm_analyzer", lambda: analyzer)
    result = analyze_food_image.invoke({"image_path": str(image)},
        config={"configurable": {"food_vision_method": "foodlmm_4bit"}})
    analyzer.analyze.assert_called_once_with(str(image))
    assert result["method"] == "foodlmm_4bit"
    assert result["total_calories"] == 260


def test_app_photo_tool_dispatches_dpf(tmp_path, monkeypatch):
    from nutricoach.food_vision.supervised_analyzers import total_result
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = MagicMock()
    analyzer.analyze.return_value = total_result("dpf_reproduction", {
        "calories": 260, "mass_g": 200, "fat_g": 1, "carbs_g": 56, "protein_g": 5,
    }, 0)
    monkeypatch.setattr("nutricoach.food_vision.dpf_analyzer.get_dpf_analyzer", lambda: analyzer)
    result = analyze_food_image.invoke({"image_path": str(image), "method": "dpf_reproduction"},
                                       config={"configurable": {}})
    assert result["method"] == "dpf_reproduction"
    analyzer.analyze.assert_called_once_with(str(image))


def test_foodlmm_reuses_isolated_worker(tmp_path, monkeypatch):
    import io
    from nutricoach.food_vision.foodlmm_analyzer import FoodLMMAnalyzer
    payload = json.dumps({"values": {"calories": 260, "mass_g": 200, "fat_g": 1,
                                    "carbs_g": 56, "protein_g": 5}, "raw_response": "test"})
    process = MagicMock(stdin=io.StringIO(), stdout=io.StringIO((payload + "\n") * 2))
    process.poll.return_value = None
    factory = MagicMock(return_value=process)
    monkeypatch.setenv("FOODLMM_PYTHON", sys.executable)
    monkeypatch.setattr("nutricoach.food_vision.foodlmm_analyzer.subprocess.Popen", factory)
    analyzer = FoodLMMAnalyzer()
    try:
        for _ in range(2):
            result = analyzer.analyze(str(tmp_path / "meal.jpg"))
            assert result.method == "foodlmm_4bit"
            assert result.total_calories == 260
        factory.assert_called_once()
        assert factory.call_args.kwargs["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).absolute().parent)
        assert len(process.stdin.getvalue().splitlines()) == 2
    finally:
        analyzer.close()
    process.terminate.assert_called_once()


def test_foodlmm_missing_environment_is_actionable(tmp_path, monkeypatch):
    from nutricoach.food_vision.foodlmm_analyzer import FoodLMMAnalyzer
    monkeypatch.setenv("FOODLMM_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(RuntimeError, match="requirements_foodlmm.in"):
        FoodLMMAnalyzer().analyze("unused.jpg")


@pytest.mark.parametrize("packed", [True, False])
def test_foodlmm_worker_loads_packed_four_bit_model_once(tmp_path, monkeypatch, packed):
    import io
    from PIL import Image
    from nutricoach.food_vision import foodlmm_worker, evaluate_paper_system as paper
    image = tmp_path / "meal.jpg"
    Image.new("RGB", (1200, 600)).save(image)
    model = MagicMock()
    model.predict.return_value = ({"calories": 260, "mass_g": 200, "fat_g": 1,
                                   "carbs_g": 56, "protein_g": 5}, "test")
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr(paper, "FoodLMM", constructor)
    monkeypatch.setattr(paper, "prepare_release", lambda method: tmp_path)
    monkeypatch.setattr(paper, "select_device", lambda device: "mps")
    monkeypatch.setattr("huggingface_hub.snapshot_download", MagicMock())
    checkpoint = tmp_path / "data/foodlmm_4bit"
    checkpoint.mkdir(parents=True)
    if packed:
        (checkpoint / "configuration.json").write_text("{}")
        (checkpoint / "model.safetensors").touch()
    monkeypatch.delenv("FOODLMM_QUANTIZED_CHECKPOINT", raising=False)
    monkeypatch.setattr(foodlmm_worker, "__file__", str(tmp_path / "foodlmm_worker.py"))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "stdin", io.StringIO((json.dumps({"image_path": str(image)}) + "\n") * 2))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    foodlmm_worker.main()
    constructor.assert_called_once_with(tmp_path, "mps", quantization="4bit", quantized_checkpoint=checkpoint if packed else None)
    assert model.predict.call_count == 2
    assert model.predict.call_args.args[0].size == (1024, 512)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 2
    assert all(response["values"]["calories"] == 260 for response in responses)
    if packed:
        model.export_quantized.assert_not_called()
    else:
        model.export_quantized.assert_called_once_with(checkpoint)


def test_foodlmm_prediction_preserves_tensor_version_counters():
    import torch
    from nutricoach.food_vision.evaluate_paper_system import FoodLMM

    def generate(**kwargs):
        assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        tokens = torch.tensor([[1, 2, 3]])
        assert tokens._version == 0
        return SimpleNamespace(sequences=tokens)

    analyzer = FoodLMM.__new__(FoodLMM)
    analyzer.device = torch.device("cpu")
    analyzer.dtype = torch.float32
    analyzer.total_token_ids = [10, 11, 12, 13, 14]
    analyzer.processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 2, 2)})
    analyzer.conversation = MagicMock()
    analyzer.tokenize_image = MagicMock(return_value=torch.tensor([1, 2]))
    analyzer.tokenizer = MagicMock()
    analyzer.model = MagicMock()
    analyzer.model.generate.side_effect = generate
    analyzer.model.return_value.hidden_states = torch.ones(1, 2, 4)
    analyzer.model.model.text_hidden_fcs = [torch.nn.Identity()]
    analyzer.model.get_token_mask.return_value = [torch.tensor([[True, False]])] * 6
    for key in ("calorie", "mass", "fat", "carb", "protein"):
        getattr(analyzer.model, f"total_{key}_head").return_value = torch.tensor([1.0])
    values, _ = analyzer.predict(None)
    assert values == {"calories": 1., "mass_g": 1., "fat_g": 1., "carbs_g": 1., "protein_g": 1.}


def test_foodlmm_mps_unpack_preserves_packed_value_order():
    import torch
    from nutricoach.food_vision.evaluate_paper_system import quanto_unpack_context

    library = torch.library.Library("quanto", "FRAGMENT")
    try:
        torch.ops.quanto.unpack.default
    except AttributeError:
        library.define("unpack(Tensor self, int bits) -> Tensor")
    packed = torch.arange(256).to(torch.uint8).reshape(16, 16)
    for bits in (2, 4):
        expected = torch.cat([(packed.to(torch.int32) >> shift) & ((1 << bits) - 1)
                              for shift in range(0, 8, bits)]).to(torch.uint8)
        with quanto_unpack_context(torch.device("mps")):
            actual = torch.ops.quanto.unpack(packed, bits)
        torch.testing.assert_close(actual, expected)


def test_foodlmm_generation_stops_after_all_total_tokens():
    import torch
    from nutricoach.food_vision.evaluate_paper_system import nutrition_stopping_criteria

    stop = nutrition_stopping_criteria([10, 11, 12, 13, 14], 5)[0]
    prompt = [10, 11, 12, 13, 14]
    assert not stop(torch.tensor([prompt]), None)
    assert not stop(torch.tensor([prompt + [10, 11, 12, 13, 10]]), None)
    assert stop(torch.tensor([prompt + [10, 11, 12, 13, 10, 14]]), None)


@pytest.mark.parametrize("complete", [True, False])
def test_dpf_checkpoint_replaces_initialization_and_rejects_missing_weights(monkeypatch, capsys, complete):
    import torch
    from nutricoach.food_vision.evaluate_paper_system import DPFReproduction

    class DPFNet(torch.nn.Module):
        def __init__(self, tasks, pretrained):
            super().__init__()
            assert not pretrained
            self.rgb_stream = torch.nn.Linear(2, 2, bias=False)
            self.depth_stream = torch.nn.Linear(2, 2, bias=False)
            self.heads = torch.nn.Linear(2, 2, bias=False)
            self._load_food2k_or_log_fallback(None)

        def _load_food2k_or_log_fallback(self, path):
            print("Food2K ResNet101 path not provided; using random ResNet initialization.")

    monkeypatch.setitem(sys.modules, "nutrition5k_pkg.models.dpf_nutrition", SimpleNamespace(DPFNutritionNet=DPFNet))
    monkeypatch.setitem(sys.modules, "nutrition5k_pkg.data.transforms", SimpleNamespace(
        get_dpf_rgb_transform=lambda: None, get_dpf_depth_transform=lambda: None))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: "checkpoint.pt")
    monkeypatch.setattr("transformers.pipeline", MagicMock())
    weights = {name + ".weight": torch.full((2, 2), float(index + 1))
               for index, name in enumerate(("rgb_stream", "depth_stream", "heads"))}
    if not complete:
        del weights["rgb_stream.weight"]
    loader = MagicMock(return_value=weights)
    monkeypatch.setattr(torch, "load", loader)
    if complete:
        analyzer = DPFReproduction(None, "cpu")
        for name, expected in weights.items():
            torch.testing.assert_close(analyzer.model.state_dict()[name], expected)
    else:
        with pytest.raises(RuntimeError, match="Missing key"):
            DPFReproduction(None, "cpu")
    loader.assert_called_once_with("checkpoint.pt", map_location="cpu", weights_only=True)
    output = capsys.readouterr()
    assert "random ResNet" not in output.out
    assert ("complete released checkpoint loaded" in output.err) == complete
