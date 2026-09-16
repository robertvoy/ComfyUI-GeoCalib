"""Validated camera diagnostics and a distortion-aware horizon preview."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def finite_scalar(value, name, nonnegative=False):
    array = as_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value; got {array.shape}.")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"GeoCalib returned invalid {name}: {result}.")
    return result


def describe_result(result):
    """Read real backend outputs; missing/invalid values never become zero tilt."""
    camera, gravity = result["camera"], result["gravity"]
    rp = as_numpy(gravity.rp).reshape(-1)
    if rp.size != 2:
        raise ValueError(f"Expected one roll/pitch pair, got {rp.shape}.")
    roll = math.degrees(finite_scalar(rp[0], "roll"))
    pitch = math.degrees(finite_scalar(rp[1], "pitch"))
    vfov = math.degrees(finite_scalar(camera.vfov, "vfov"))
    if not 0 < vfov < 180:
        raise ValueError(f"GeoCalib returned invalid vertical FoV: {vfov}.")
    report = {
        "roll_deg": roll,
        "pitch_deg": pitch,
        "vfov_deg": vfov,
        # Official latitude at the optical axis equals pitch. Positive latitude
        # looks above the horizon; APC/ECHO instead defines positive tilt DOWN.
        "camera_tilt_deg": -pitch,
        "conventions": {"pitch": "positive_up", "camera_tilt_deg": "positive_down"},
        "uncertainty_note": "Model-reported uncertainty, not measured error or a guaranteed bound.",
    }
    for name in ("roll", "pitch", "vfov"):
        report[f"{name}_uncertainty_deg"] = math.degrees(
            finite_scalar(result[f"{name}_uncertainty"], f"{name}_uncertainty", True)
        )
    focal = as_numpy(camera.f).reshape(-1)
    if focal.size != 2 or not np.isfinite(focal).all() or np.any(focal <= 0):
        raise ValueError("GeoCalib returned invalid focal lengths.")
    report["focal_length_px"] = {"fx": float(focal[0]), "fy": float(focal[1])}
    report["focal_uncertainty_px"] = finite_scalar(result["focal_uncertainty"], "focal_uncertainty", True)
    if hasattr(camera, "dist"):
        distortion = as_numpy(camera.dist).reshape(-1)
        if not np.isfinite(distortion).all():
            raise ValueError("GeoCalib returned invalid distortion coefficients.")
        report["distortion_coefficients"] = distortion.tolist()
    return report


def horizon_mask(latitude):
    """Zero crossings of the FITTED latitude field, not the neural field guess."""
    lat = as_numpy(latitude)
    if lat.ndim != 2 or min(lat.shape) < 2 or not np.isfinite(lat).all():
        raise ValueError("Expected a finite H x W fitted latitude field.")
    positive = lat >= 0
    mask = np.zeros(lat.shape, dtype=np.uint8)
    mask[:-1, :] |= positive[:-1, :] != positive[1:, :]
    mask[:, :-1] |= positive[:, :-1] != positive[:, 1:]
    return mask


def render_horizon(image, latitude, report):
    """Return a new float RGB preview at the original size; never warp input."""
    rgb = as_numpy(image)
    mask = horizon_mask(latitude)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError("Preview image and fitted latitude field must have matching RGB/HW shapes.")
    out = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    visible = bool(mask.any())
    width = max(1, round(min(mask.shape) / 450))
    core = cv2.dilate(mask, np.ones((width * 2 + 1, width * 2 + 1), np.uint8)) > 0
    outline = cv2.dilate(mask, np.ones((width * 2 + 5, width * 2 + 5), np.uint8)) > 0
    out[outline] = (0, 0, 0)
    out[core] = (0, 255, 190)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    text = (
        f"Tilt down {report['camera_tilt_deg']:+.2f} deg | "
        f"Pitch {report['pitch_deg']:+.2f} deg | "
        f"uncertainty +/-{report['pitch_uncertainty_deg']:.2f} deg\n"
        + ("Green: fitted horizon (0 deg latitude)" if visible else "Horizon is outside this frame")
    )
    box = draw.multiline_textbbox((8, 8), text)
    draw.rectangle((box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4), fill=(0, 0, 0))
    draw.multiline_text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(pil).astype(np.float32) / 255.0, visible
