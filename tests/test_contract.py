"""Dependency-free regression tests for the saved-workflow contract."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = "97b8968e7798a66bf04fcf791fb535624241bda7"


class ContractTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "nodes.py").read_text())
        self.node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GeoCalibNode")
        self.attrs = {}
        for statement in self.node.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        try:
                            self.attrs[target.id] = ast.literal_eval(statement.value)
                        except (ValueError, TypeError):
                            pass

    def test_legacy_outputs_keep_their_indices(self):
        self.assertEqual(self.attrs["RETURN_NAMES"][:3], ("roll", "pitch", "vfov"))
        self.assertEqual(self.attrs["RETURN_TYPES"][:3], ("FLOAT", "FLOAT", "FLOAT"))

    def test_appended_diagnostics(self):
        self.assertEqual(self.attrs["RETURN_NAMES"][3:7], (
            "camera_tilt_deg", "pitch_uncertainty_deg", "horizon_preview", "debug_json"
        ))
        self.assertEqual(self.attrs["RETURN_NAMES"][7:], (
            "focal_length_mm", "camera_rotation_x_degrees", "camera_rotation_y_degrees",
        ))
        self.assertEqual(self.attrs["RETURN_TYPES"][7:], ("FLOAT", "FLOAT", "FLOAT"))
        self.assertEqual(len(self.attrs["RETURN_TYPES"]), 10)

    def test_every_output_has_explicit_per_frame_list_semantics(self):
        self.assertEqual(self.attrs.get("OUTPUT_IS_LIST"), (True,) * 10)

    def test_official_backend_is_commit_pinned(self):
        requirements = (ROOT / "requirements.txt").read_text()
        self.assertIn("git+https://github.com/cvg/GeoCalib.git@" + PIN, requirements)
        self.assertNotIn("Antilopax/GeoCalib", requirements)

    def test_existing_input_order_preserved(self):
        method = next(n for n in self.node.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
        value = next(n for n in method.body if isinstance(n, ast.Return)).value
        assert value is not None
        schema = ast.literal_eval(value)
        self.assertEqual(list(schema["required"]), ["image", "weights", "camera_model"])


if __name__ == "__main__":
    unittest.main()
