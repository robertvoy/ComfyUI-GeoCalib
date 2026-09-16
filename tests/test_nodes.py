"""No-download unit tests plus real backend geometry tests (no learned inference)."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("geocalib_wrapper_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
nodes = sys.modules[SPEC.name + ".nodes"]
diag = sys.modules[SPEC.name + ".diagnostics"]


def fixture_result(pitch=-8.0):
    return {
        "camera": SimpleNamespace(vfov=torch.tensor([math.radians(60)]), f=torch.tensor([[100., 100.]])),
        "gravity": SimpleNamespace(rp=torch.tensor([[math.radians(-1), math.radians(pitch)]])),
        "pitch_uncertainty": torch.tensor([math.radians(4)]),
        "roll_uncertainty": torch.tensor([math.radians(2)]),
        "vfov_uncertainty": torch.tensor([math.radians(10)]),
        "focal_uncertainty": torch.tensor([20.]),
    }


class FakeModel:
    """Explicit test double: does not perform calibration."""
    def __init__(self, results=None, error=None):
        self.results = iter(results or [fixture_result()])
        self.error = error
        self.devices = []
        self.inputs = []

    def to(self, device):
        self.devices.append(str(device))
        return self

    def calibrate(self, image, camera_model):
        self.inputs.append((image.shape, camera_model))
        if self.error:
            raise self.error
        return next(self.results)


class DiagnosticsTests(unittest.TestCase):
    def test_radians_degrees_and_pitch_sign(self):
        d = diag.describe_result(fixture_result(-8))
        self.assertAlmostEqual(d["pitch_deg"], -8, places=5)
        self.assertAlmostEqual(d["camera_tilt_deg"], 8, places=5)
        self.assertAlmostEqual(d["pitch_uncertainty_deg"], 4, places=5)
        self.assertAlmostEqual(d["vfov_deg"], 60, places=5)

    def test_missing_uncertainty_is_not_zero(self):
        result = fixture_result()
        del result["pitch_uncertainty"]
        with self.assertRaises(KeyError):
            diag.describe_result(result)

    def test_nan_and_negative_uncertainties_fail(self):
        for val in (float("nan"), float("inf"), -1.):
            with self.subTest(value=val):
                result = fixture_result()
                result["pitch_uncertainty"] = torch.tensor([val])
                with self.assertRaises(ValueError):
                    diag.describe_result(result)

    def test_multiple_camera_results_rejected(self):
        result = fixture_result()
        result["gravity"].rp = torch.zeros((2, 2))
        with self.assertRaises(ValueError):
            diag.describe_result(result)

    def test_invalid_fov_and_focal_rejected(self):
        for field, value in (("vfov", torch.tensor([0.])), ("f", torch.tensor([[-1., 10.]]))):
            result = fixture_result()
            setattr(result["camera"], field, value)
            with self.assertRaises(ValueError):
                diag.describe_result(result)

    def test_horizontal_horizon_has_no_artificial_border(self):
        lat = np.broadcast_to(np.linspace(1, -1, 64)[:, None], (64, 80))
        mask = diag.horizon_mask(lat)
        self.assertEqual(np.flatnonzero(mask.any(axis=1)).tolist(), [31])
        self.assertTrue(mask[31].all())

    def test_out_of_frame_horizon_has_no_line(self):
        self.assertFalse(diag.horizon_mask(np.ones((64, 80))).any())
        self.assertFalse(diag.horizon_mask(-np.ones((64, 80))).any())

    def test_bad_latitude_rejected(self):
        with self.assertRaises(ValueError):
            diag.horizon_mask(np.full((64, 80), np.nan))

    def test_preview_preserves_source_and_shape(self):
        image = np.ones((64, 80, 3), np.float32) * 0.5
        original = image.copy()
        lat = np.broadcast_to(np.linspace(1, -1, 64)[:, None], (64, 80))
        preview, visible = diag.render_horizon(image, lat, diag.describe_result(fixture_result()))
        self.assertTrue(visible)
        self.assertEqual(preview.shape, image.shape)
        self.assertEqual(preview.dtype, np.float32)
        np.testing.assert_array_equal(image, original)


class NodeTests(unittest.TestCase):
    def run_node(self, image, model):
        lat = np.broadcast_to(np.linspace(1, -1, 64)[:, None], (64, 80))
        with patch.object(nodes, "_get_model", return_value=model), patch.object(nodes, "_device", return_value=torch.device("cpu")), patch.object(nodes, "_latitude_field", return_value=lat), patch.object(nodes, "_check_interrupted"), patch.object(nodes, "_release_cache"):
            return nodes.GeoCalibNode().analyze_image(image, "pinhole", "pinhole")

    def test_single_image_uses_declared_lists(self):
        model = FakeModel()
        out = self.run_node(torch.zeros((1, 64, 80, 3)), model)
        self.assertEqual(len(out), 7)
        self.assertTrue(all(isinstance(v, list) and len(v) == 1 for v in out))
        self.assertIsInstance(out[0][0], float)
        self.assertEqual(tuple(out[5][0].shape), (1, 64, 80, 3))
        self.assertEqual(json.loads(out[6][0])["frame_index"], 0)
        self.assertEqual(model.devices[-1], "cpu")

    def test_batch_order_and_scalar_pairing(self):
        model = FakeModel([fixture_result(-8), fixture_result(12)])
        out = self.run_node(torch.zeros((2, 64, 80, 3)), model)
        self.assertTrue(all(len(v) == 2 for v in out))
        self.assertAlmostEqual(out[3][0], 8, places=5)
        self.assertAlmostEqual(out[3][1], -12, places=5)
        self.assertEqual([json.loads(s)["frame_index"] for s in out[6]], [0, 1])
        self.assertTrue(all(type(v) is float for v in out[1]))

    def test_rgba_strips_alpha(self):
        model = FakeModel()
        self.run_node(torch.zeros((1, 64, 80, 4)), model)
        self.assertEqual(tuple(model.inputs[0][0]), (3, 64, 80))

    def test_invalid_input_never_loads_model(self):
        bad = [torch.zeros((0, 64, 80, 3)), torch.zeros((64, 80, 3)), torch.zeros((1, 8, 8, 3)), torch.zeros((1, 64, 80, 1)), torch.full((1, 64, 80, 3), float("nan")), torch.full((1, 64, 80, 3), 2.), torch.zeros((1, 64, 80, 3), dtype=torch.uint8)]
        for image in bad:
            with self.subTest(shape=image.shape), patch.object(nodes, "_get_model") as loader:
                with self.assertRaises(ValueError):
                    nodes.GeoCalibNode().analyze_image(image, "pinhole", "pinhole")
                loader.assert_not_called()

    def test_backend_error_propagates_and_model_offloads(self):
        model = FakeModel(error=RuntimeError("test-only backend failure"))
        with self.assertRaisesRegex(RuntimeError, "test-only backend failure"):
            self.run_node(torch.zeros((1, 64, 80, 3)), model)
        self.assertEqual(model.devices[-1], "cpu")

    def test_invalid_result_propagates_and_model_offloads(self):
        result = fixture_result()
        result["pitch_uncertainty"] = torch.tensor([float("nan")])
        model = FakeModel([result])
        with self.assertRaises(ValueError):
            self.run_node(torch.zeros((1, 64, 80, 3)), model)
        self.assertEqual(model.devices[-1], "cpu")

    def test_cancellation_does_not_load_model(self):
        with patch.object(nodes, "_check_interrupted", side_effect=RuntimeError("cancelled")), patch.object(nodes, "_get_model") as loader:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                nodes.GeoCalibNode().analyze_image(torch.zeros((1, 64, 80, 3)), "pinhole", "pinhole")
            loader.assert_not_called()


class OfficialGeometryTests(unittest.TestCase):
    def test_real_backend_pitch_sign_and_horizon(self):
        from geocalib.camera import Pinhole
        from geocalib.gravity import Gravity
        from geocalib.perspective_fields import get_latitude_field
        camera = Pinhole.from_dict({"width": 80., "height": 64., "f": 70.})
        for pitch in (-10., 0., 10.):
            with self.subTest(pitch=pitch):
                gravity = Gravity.from_rp(torch.tensor(0.), torch.tensor(math.radians(pitch)))
                lat = get_latitude_field(camera, gravity)[0, :, :, 0]
                self.assertAlmostEqual(math.degrees(lat[32, 40].item()), pitch, places=4)
                ys = np.nonzero(diag.horizon_mask(lat))[0]
                self.assertTrue(len(ys))
                if pitch < 0:
                    self.assertLess(ys.mean(), 32)
                elif pitch > 0:
                    self.assertGreater(ys.mean(), 32)

    def test_all_official_camera_models_generate_preview_field(self):
        from geocalib.camera import camera_models
        from geocalib.gravity import Gravity
        for name in nodes.GeoCalibNode.INPUT_TYPES()["required"]["camera_model"][0]:
            with self.subTest(model=name):
                camera = camera_models[name].from_dict({"width": 80., "height": 64., "f": 70., "k1": 0.01, "k2": 0.001}).unsqueeze(0)
                gravity = Gravity.from_rp(torch.tensor([0.]), torch.tensor([-0.1]))
                lat = nodes._latitude_field({"camera": camera, "gravity": gravity})
                self.assertEqual(lat.shape, (64, 80))
                self.assertTrue(np.isfinite(lat).all())


if __name__ == "__main__":
    unittest.main()
