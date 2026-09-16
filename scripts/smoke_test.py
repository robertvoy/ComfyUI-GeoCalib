"""Run real-model inference without a ComfyUI server; writes actual diagnostics.

Usage: python scripts/smoke_test.py IMAGE --output-dir OUTPUT [--weights pinhole]
A small unit-test suite is separate: python -m unittest discover -s tests -v
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=["pinhole", "distorted"], default="pinhole")
    parser.add_argument("--camera-model", choices=["pinhole", "simple_radial", "simple_divisional", "radial"], default="pinhole")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("geocalib_wrapper_smoke", root / "__init__.py", submodule_search_locations=[str(root)])
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    nodes = sys.modules[spec.name + ".nodes"]
    with Image.open(args.image) as loaded:
        rgb = np.asarray(ImageOps.exif_transpose(loaded).convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(rgb).unsqueeze(0).repeat(args.batch, 1, 1, 1)
    # Upstream NMF samples random bases even in eval mode. Warm the model
    # constructor first, then capture RNG state so adapter/API parity compares
    # the same stochastic inference rather than two different random draws.
    nodes._get_model(args.weights)
    torch.manual_seed(0)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    start = time.perf_counter()
    outputs = nodes.GeoCalibNode().analyze_image(image, args.weights, args.camera_model)
    elapsed = time.perf_counter() - start
    assert len(outputs) == 10 and all(len(o) == args.batch for o in outputs)
    reports = [json.loads(s) for s in outputs[6]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (preview, report) in enumerate(zip(outputs[5], reports)):
        assert tuple(preview.shape) == (1, *rgb.shape)
        assert torch.isfinite(preview).all()
        assert abs(outputs[3][index] + outputs[1][index]) < 1e-8
        assert outputs[4][index] >= 0
        assert report["frame_index"] == index
        for slot in (7, 8, 9):
            assert outputs[slot][index] == report[nodes.GeoCalibNode.RETURN_NAMES[slot]]
        pixels = np.rint(preview[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(args.output_dir / f"horizon_{index:02d}.png")
    # Compare against the current official API with the exact same source tensor.
    model = nodes._MODELS[args.weights]
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    device = nodes._device()
    try:
        model.to(device)
        with torch.inference_mode():
            direct = model.calibrate(image[0].permute(2, 0, 1).contiguous().to(device), camera_model=args.camera_model)
            reference = nodes.describe_result(direct)
            reference.update(nodes.apc_camera_parameters(
                reference["focal_length_px"]["fx"], rgb.shape[1], direct["gravity"].vec3d,
            ))
        del direct
    finally:
        model.to("cpu")
    errors = {k: abs(reports[0][k] - reference[k]) for k in (
        "roll_deg", "pitch_deg", "vfov_deg", "pitch_uncertainty_deg", "focal_length_mm",
        "camera_rotation_x_degrees", "camera_rotation_y_degrees",
    )}
    assert max(errors.values()) < 0.001, errors
    result = {"runtime": "real GeoCalib inference", "torch": torch.__version__, "device": str(device), "batch_size": args.batch, "elapsed_seconds": elapsed, "direct_api_absolute_errors": errors, "reports": reports}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
