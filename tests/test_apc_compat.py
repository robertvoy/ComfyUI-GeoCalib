"""APC lens/XYZ-Euler contract: no yaw inference and no production mesh fixture."""
import math
import unittest

import numpy as np
import torch
from geocalib.gravity import Gravity

from test_nodes import diag, nodes


class APCConversionTests(unittest.TestCase):
    def test_outputs_append_lens_and_xy_without_heading(self):
        self.assertEqual(nodes.GeoCalibNode.RETURN_NAMES[:7], (
            'roll', 'pitch', 'vfov', 'camera_tilt_deg', 'pitch_uncertainty_deg',
            'horizon_preview', 'debug_json'))
        self.assertEqual(nodes.GeoCalibNode.RETURN_NAMES[7:], (
            'focal_length_mm', 'camera_rotation_x_degrees', 'camera_rotation_y_degrees'))
        self.assertEqual(nodes.GeoCalibNode.RETURN_TYPES[7:], ('FLOAT', 'FLOAT', 'FLOAT'))
        self.assertEqual(nodes.GeoCalibNode.OUTPUT_IS_LIST, (True,) * 10)
        self.assertEqual(list(nodes.GeoCalibNode.INPUT_TYPES()['required']), ['image', 'weights', 'camera_model'])
        self.assertFalse(nodes.GeoCalibNode.INPUT_TYPES().get('optional'))
        self.assertNotIn('camera_rotation_z_degrees', nodes.GeoCalibNode.RETURN_NAMES)

    def test_full_frame_focal_uses_horizontal_pixels_and_is_scale_invariant(self):
        self.assertEqual(diag.focal_pixels_to_mm(1000., 2000), 18.)
        self.assertEqual(diag.focal_pixels_to_mm(2000., 4000), 18.)
        for focal, width in ((500., 1200), (1900., 800), (15., 64)):
            mm = diag.focal_pixels_to_mm(focal, width)
            photo_hfov = 2 * math.atan(width / (2 * focal))
            apc_hfov = 2 * math.atan(36. / (2 * mm))
            self.assertAlmostEqual(photo_hfov, apc_hfov, places=14)

    def test_focal_is_not_silently_clamped_to_widget_limits(self):
        self.assertEqual(diag.focal_pixels_to_mm(100., 3600), 1.)
        self.assertEqual(diag.focal_pixels_to_mm(1000., 100), 360.)

    def test_invalid_focal_or_width_fails(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    diag.focal_pixels_to_mm(value, 1000.)
                with self.assertRaises(ValueError):
                    diag.focal_pixels_to_mm(1000., value)

    def test_level_and_pitch_only_rotation_signs(self):
        for pitch in (-35., 0., 20.):
            gravity = Gravity.from_rp(torch.tensor(0., dtype=torch.float64), torch.tensor(math.radians(pitch), dtype=torch.float64))
            rx, ry = diag.gravity_to_apc_rotation_xy_degrees(gravity.vec3d)
            self.assertAlmostEqual(rx, 90. + pitch, places=12)
            self.assertAlmostEqual(ry, 0., places=12)
        for roll in (-25., 30.):
            gravity = Gravity.from_rp(torch.tensor(math.radians(roll), dtype=torch.float64), torch.tensor(0., dtype=torch.float64))
            rx, ry = diag.gravity_to_apc_rotation_xy_degrees(gravity.vec3d)
            self.assertAlmostEqual(rx, 90., places=12)
            self.assertAlmostEqual(ry, roll, places=12)

    def test_official_gravity_matches_apc_xyz_over_pose_grid(self):
        for pitch in (-90., -80., -45., -8.5, 0., 15., 75., 90.):
            for roll in (-180., -90., -45., -1., 0., 35., 90., 179.):
                with self.subTest(pitch=pitch, roll=roll):
                    gravity = Gravity.from_rp(torch.tensor(math.radians(roll), dtype=torch.float64), torch.tensor(math.radians(pitch), dtype=torch.float64))
                    rx, ry = map(math.radians, diag.gravity_to_apc_rotation_xy_degrees(gravity.vec3d))
                    # Third row of Rz @ Ry @ Rx: world up expressed in local
                    # Blender camera axes. It is independent of APC's retained Z.
                    up_blender = np.array([-math.sin(ry), math.cos(ry) * math.sin(rx), math.cos(ry) * math.cos(rx)])
                    up_cv = up_blender * np.array([1., -1., -1.])
                    np.testing.assert_allclose(up_cv, gravity.vec3d.numpy(), atol=1e-12, rtol=0)

    def test_combined_pitch_roll_is_not_naive_widget_copy(self):
        gravity = Gravity.from_rp(torch.tensor(math.radians(35.), dtype=torch.float64), torch.tensor(math.radians(-45.), dtype=torch.float64))
        rx, ry = diag.gravity_to_apc_rotation_xy_degrees(gravity.vec3d)
        self.assertGreater(abs(rx - 45.), 1.)
        self.assertGreater(abs(ry - 35.), 1.)

    def test_invalid_gravity_fails(self):
        for vector in ([0., 0., 0.], [0., float('nan'), 0.], [0., -1.], [[0., -1., 0.], [0., -1., 0.]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                diag.gravity_to_apc_rotation_xy_degrees(vector)

    def test_gravity_scale_does_not_change_rotations(self):
        vector = np.array([0.2, -0.8, -0.4])
        np.testing.assert_allclose(diag.gravity_to_apc_rotation_xy_degrees(vector),
                                   diag.gravity_to_apc_rotation_xy_degrees(vector * 5), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
