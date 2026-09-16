"""ComfyUI adapter for the official, commit-pinned GeoCalib backend."""
import importlib.metadata
import json
import threading

import torch

from .diagnostics import describe_result, render_horizon

UPSTREAM_COMMIT = "97b8968e7798a66bf04fcf791fb535624241bda7"
_MODELS = {}
_MODEL_LOCK = threading.RLock()


def _get_model(weights):
    if weights not in _MODELS:
        try:
            from geocalib import GeoCalib
        except ImportError as exc:
            raise ImportError(
                "GeoCalib backend unavailable. Install this node's requirements.txt "
                "with the Python interpreter used by ComfyUI."
            ) from exc
        # GeoCalib downloads official v1.0 weights lazily to the Torch Hub cache.
        _MODELS[weights] = GeoCalib(weights=weights).eval().cpu()
    return _MODELS[weights]


def _device():
    try:
        import comfy.model_management as mm
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return mm.get_torch_device()


def _check_interrupted():
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def _release_cache():
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.soft_empty_cache()


def _backend_metadata():
    try:
        dist = importlib.metadata.distribution("geocalib")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        version = dist.version
    except (importlib.metadata.PackageNotFoundError, ValueError):
        version, commit = None, None
    return {"package_version": version, "installed_revision": commit, "expected_revision": UPSTREAM_COMMIT}


def _latitude_field(result):
    from geocalib.perspective_fields import get_latitude_field
    # Reproject the solved camera, including distortion. The network's raw
    # latitude_field need not agree exactly with the fitted camera/horizon.
    camera, gravity = result["camera"].cpu(), result["gravity"].cpu()
    return get_latitude_field(camera, gravity)[0, :, :, 0].detach().cpu().numpy()


def _validate_image(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("GeoCalib expects an IMAGE tensor shaped [B,H,W,C].")
    if image.shape[0] < 1 or min(image.shape[1:3]) < 32 or image.shape[3] not in (3, 4):
        raise ValueError("GeoCalib requires a nonempty batch of RGB/RGBA images at least 32 pixels per side.")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("GeoCalib IMAGE input must contain finite floating-point pixels.")
    if image.min().item() < 0 or image.max().item() > 1:
        raise ValueError("GeoCalib IMAGE pixels must be in [0,1].")


class GeoCalibNode:
    """Single-image calibration, explicitly mapped per frame for IMAGE batches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "weights": (["pinhole", "distorted"], {"default": "pinhole"}),
                "camera_model": (["pinhole", "simple_radial", "simple_divisional", "radial"], {"default": "pinhole"}),
            },
        }

    # Existing saved workflows keep their node ID, widget order and slots 0-2.
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("roll", "pitch", "vfov", "camera_tilt_deg", "pitch_uncertainty_deg", "horizon_preview", "debug_json")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True)
    FUNCTION = "analyze_image"
    CATEGORY = "GeoCalib"
    DESCRIPTION = (
        "Estimate roll/pitch/vertical FoV in degrees. Native pitch is positive UP; "
        "camera_tilt_deg is positive DOWN for ECHO/APC. Uncertainty is model-reported, "
        "not a guaranteed error bound. The preview overlays the fitted horizon without "
        "changing perspective. Each output maps over frames in input order."
    )

    def analyze_image(self, image, weights="pinhole", camera_model="pinhole"):
        _validate_image(image)
        schema = self.INPUT_TYPES()["required"]
        if weights not in schema["weights"][0] or camera_model not in schema["camera_model"][0]:
            raise ValueError("Unsupported GeoCalib weights or camera model.")
        outputs = ([], [], [], [], [], [], [])
        backend = _backend_metadata()
        # The backend mutates its optimizer's camera model, so protect cached
        # models against concurrent calls. Cache on CPU, not permanently on VRAM.
        with _MODEL_LOCK:
            _check_interrupted()
            model = _get_model(weights)
            try:
                device = _device()
                model.to(device)
                for index, frame in enumerate(image):
                    _check_interrupted()
                    rgb = frame[..., :3].detach().to(device="cpu", dtype=torch.float32)
                    tensor = rgb.permute(2, 0, 1).contiguous().to(device)
                    with torch.inference_mode():
                        result = model.calibrate(tensor, camera_model=camera_model)
                        report = describe_result(result)
                        preview, visible = render_horizon(rgb, _latitude_field(result), report)
                    report.update({
                        "schema_version": 1, "frame_index": index,
                        "image_size": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
                        "weights": weights, "camera_model": camera_model,
                        "horizon_visible": visible, "backend": backend,
                    })
                    values = (
                        report["roll_deg"], report["pitch_deg"], report["vfov_deg"],
                        report["camera_tilt_deg"], report["pitch_uncertainty_deg"],
                        torch.from_numpy(preview).unsqueeze(0),
                        json.dumps(report, indent=2, allow_nan=False),
                    )
                    for output, value in zip(outputs, values):
                        output.append(value)
                    del result, tensor
            finally:
                model.to("cpu")
                _release_cache()
        return outputs


NODE_CLASS_MAPPINGS = {"GeoCalibNode": GeoCalibNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeoCalibNode": "GeoCalib Estimator"}
