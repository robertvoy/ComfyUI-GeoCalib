"""Compare conversions with an installed APC camera_ops.py, without rendering.

Usage: python scripts/verify_apc_geometry.py --apc-camera-ops /path/to/apc/camera_ops.py
Does not copy/vendor APC source or download model weights. Imports the supplied
local module, so only pass a trusted installation you would normally execute.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from geocalib.gravity import Gravity


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apc-camera-ops", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    diag = load_module("geocalib_geometry_probe", Path(__file__).resolve().parents[1] / "diagnostics.py")
    apc = load_module("apc_geometry_reference", args.apc_camera_ops)
    rotations, lenses, gravity_error, focal_error = 0, 0, 0.0, 0.0
    for pitch in (-90., -80., -45., -8.5, 0., 15., 75., 90.):
        for roll in (-180., -90., -45., -1., 0., 35., 90., 179.):
            g = Gravity.from_rp(torch.tensor(math.radians(roll), dtype=torch.float64),
                                torch.tensor(math.radians(pitch), dtype=torch.float64)).vec3d
            rx, ry = diag.gravity_to_apc_rotation_xy_degrees(g)
            for z in (-135., 0., 31., 180.):
                rotation = apc.euler_xyz_degrees_to_matrix([rx, ry, z])
                recovered_cv = (rotation.T @ np.array([0., 0., 1.])) * [1., -1., -1.]
                error = float(np.max(np.abs(recovered_cv - g.numpy())))
                gravity_error = max(gravity_error, error)
                if error > 1e-12:
                    raise AssertionError((pitch, roll, z, error))
                rotations += 1
    for width in (64., 800., 1024., 2048., 8192.):
        for focal_px in (15., 100., 660., 1000., 5000.):
            mm = diag.focal_pixels_to_mm(focal_px, width)
            expected = apc.focal_px_to_horizontal_fov_degrees(focal_px, width)
            actual = apc.focal_length_mm_to_horizontal_fov_degrees(mm)
            error = abs(actual - expected)
            focal_error = max(focal_error, error)
            if error > 1e-12:
                raise AssertionError((width, focal_px, error))
            lenses += 1
    report = {
        "verification": "Installed APC functions vs official GeoCalib gravity; no learned inference",
        "rotation_cases": rotations,
        "focal_cases": lenses,
        "max_gravity_component_error": gravity_error,
        "max_horizontal_fov_error_degrees": focal_error,
        "apc_camera_ops_sha256": hashlib.sha256(args.apc_camera_ops.read_bytes()).hexdigest(),
        "heading_estimated": False,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
