"""Geometric units, calibration, overlap handling, and explicit fallbacks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.geometry_analyzer import GeometryVLMAnalyzer, estimate_volumes, pixel_box


def scene():
    depth = np.ones((100, 100), dtype=np.float32)
    masks = np.zeros((1, 100, 100), dtype=bool)
    masks[0, 40:60, 40:60] = True
    depth[masks[0]] = 0.9
    return depth, masks


def test_height_integration_and_cubic_plate_scale():
    depth, masks = scene()
    volumes, diagnostics = estimate_volumes(depth, masks, [0, 0, 1000, 1000], plate_diameter_cm=100)
    focal = 100 / (2 * np.tan(np.deg2rad(60) / 2))
    expected_ml = 400 * 0.1 / focal ** 2 * (1 / (100 / focal)) ** 3 * 1e6
    assert volumes[0] == pytest.approx(expected_ml, rel=1e-4)
    doubled, _ = estimate_volumes(depth, masks, [0, 0, 1000, 1000], plate_diameter_cm=200)
    assert doubled[0] == pytest.approx(volumes[0] * 8, rel=1e-5)
    assert diagnostics["support_plane_rmse_cm"] < 1e-5


def test_depth_global_scale_cancels_after_plate_calibration():
    depth, masks = scene()
    first, _ = estimate_volumes(depth, masks, [0, 0, 1000, 1000])
    second, _ = estimate_volumes(depth * 5, masks, [0, 0, 1000, 1000])
    assert second == pytest.approx(first, rel=1e-4)


def test_rectangular_plate_uses_width_reference():
    depth, masks = scene()
    volumes, diagnostics = estimate_volumes(depth, masks, [0, 0, 1000, 800], plate_shape="rectangle")
    assert volumes[0] > 0
    assert diagnostics["reference_shape"] == "rectangle"
    with pytest.raises(ValueError, match="shape"):
        estimate_volumes(depth, masks, [0, 0, 1000, 1000], plate_shape="unknown")


def test_overlap_is_not_counted_twice_and_flat_depth_has_zero_volume():
    depth, masks = scene()
    single, _ = estimate_volumes(depth, masks, [0, 0, 1000, 1000])
    duplicate, diagnostics = estimate_volumes(depth, np.repeat(masks, 2, axis=0), [0, 0, 1000, 1000])
    assert duplicate.sum() == pytest.approx(single.sum())
    assert diagnostics["overlap_pixels"] == 400
    flat, _ = estimate_volumes(np.ones_like(depth), masks, [0, 0, 1000, 1000])
    assert flat[0] == 0


def test_missing_exposed_plate_and_invalid_coordinates_are_rejected():
    depth, masks = scene()
    masks[:] = True
    with pytest.raises(ValueError, match="exposed"):
        estimate_volumes(depth, masks, [0, 0, 1000, 1000])
    for box in ([0, 0, -1, 1000], [0, 0, 1001, 1000], [0, 0, 0, 1000]):
        with pytest.raises(ValueError):
            pixel_box(box, 640, 480)


@pytest.mark.parametrize("surface", ["flat_plate", "bowl"])
def test_geometry_replaces_vlm_grams_or_records_fallback(tmp_path, surface):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = GeometryVLMAnalyzer(api_key="test", model="test")
    inventory = {"plate_reference": {"surface": surface, "bbox": [0, 0, 1000, 1000]},
                 "items": [{"name": "white rice", "bbox": [200, 200, 500, 500],
                            "density_g_per_ml": 0.74}]}
    matching = {"items": [{"item_id": 0, "db_name": "white rice", "quantity_grams": 200}]}
    analyzer._client = MagicMock()
    analyzer._client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        for payload in (inventory, matching)
    ]
    analyzer.geometry_estimator = MagicMock()
    analyzer.geometry_estimator.volumes.return_value = (np.array([100.0]), {})
    result = analyzer.analyze(str(image))
    assert result.error is None
    assert result.method == "geometry_vlm_db"
    if surface == "flat_plate":
        assert result.food_items[0].quantity_grams == 74
        assert result.total_calories == pytest.approx(96.2)
        assert "assumed 26cm" in result.food_items[0].portion_description
        assert json.loads(result.raw_response)["geometry"]["items"][0]["used_geometry"]
    else:
        assert result.food_items[0].quantity_grams == 200
        assert "geometry fallback" in result.food_items[0].portion_description
        analyzer.geometry_estimator.volumes.assert_not_called()
