# ComfyUI-GeoCalib

Single-image camera calibration in ComfyUI, based on [Antilopax's wrapper](https://github.com/Antilopax/ComfyUI-GeoCalib) and the **official [cvg/GeoCalib](https://github.com/cvg/GeoCalib) backend**.

This fork preserves `GeoCalibNode`, the original widget order, and all existing output slots. It adds uncertainty, a fitted-horizon preview, correctly declared per-frame outputs, positive-down tilt, and APC-compatible focal length and X/Y camera rotations. It does not estimate or add a heading/Z output.

## Install

While these changes are awaiting merge, select the `feat/official-geocalib-diagnostics` branch rather than this fork's unchanged `main`.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone --branch feat/official-geocalib-diagnostics https://github.com/robertvoy/ComfyUI-GeoCalib.git
/path/to/ComfyUI/.venv/bin/python -m pip install -r ComfyUI-GeoCalib/requirements.txt
```

Use **your ComfyUI instance's Python**, not a system Python. On a shared production environment, check existing dependencies first; do not downgrade or replace a working Torch/CUDA stack. Restart ComfyUI when no jobs are running, then refresh the browser.

Do not leave another copy of this node pack active: it uses the same `GeoCalibNode` class ID as the original.

The backend is pinned to official commit:

```text
97b8968e7798a66bf04fcf791fb535624241bda7
```

The first run of each weights choice downloads the corresponding official **GitHub release** checkpoint. GeoCalib stores it in the Torch Hub cache, normally `~/.cache/torch/hub/geocalib/pinhole.tar` or `distorted.tar` (subject to Torch Hub/TORCH_HOME configuration). No images are uploaded to a provider.

## Usage

1. Add **GeoCalib → GeoCalib Estimator**.
2. Connect an image from **Load Image**.
3. Start with `weights=pinhole`, `camera_model=pinhole` for a standard rectilinear photo.
4. Connect `horizon_preview` to **Preview Image** and `debug_json` to core **Preview as Text**.
5. Use the angle outputs to drive your workflow.

The supplied `workflows/GeoCalib.json` is the original author's example. Its original output connections retain their slot meanings; new diagnostics can be connected separately.

### Inputs

- `image`: a standard `[B,H,W,C]` floating-point IMAGE in `[0,1]`. RGB and RGBA are accepted; alpha is ignored. Images must be at least 32 pixels on each side.
- `weights`: `pinhole` or `distorted`. The latter is intended for strong lens distortion.
- `camera_model`: `pinhole`, `simple_radial`, `simple_divisional`, or `radial` (two distortion coefficients).

An estimated field of view is not proof that a more complex distortion model fits better. Start simple and inspect the horizon against genuine upright/horizontal scene structure.

### Outputs (fixed socket order)

0. **`roll` — FLOAT:** native GeoCalib roll, degrees.
1. **`pitch` — FLOAT:** native GeoCalib pitch, degrees; **positive means looking up**.
2. **`vfov` — FLOAT:** vertical field of view, degrees; not horizontal FOV.
3. **`camera_tilt_deg` — FLOAT:** `-pitch`; **positive means looking down**, zero means level.
4. **`pitch_uncertainty_deg` — FLOAT:** model-reported pitch uncertainty in degrees.
5. **`horizon_preview` — IMAGE:** input-size RGB preview with the fitted zero-latitude horizon and numerical labels. It does not rectify, crop or warp the input. Distorted camera models use their actual fitted latitude field, not a straight-line approximation.
6. **`debug_json` — STRING:** per-frame roll/pitch/FOV, uncertainties, pixel focal length, the three APC values and their conventions, fitted gravity vector, distortion coefficients where applicable, frame index, image dimensions and backend revision metadata.
7. **`focal_length_mm` — FLOAT:** 36 mm horizontal sensor-fit equivalent focal length, matching APC's lens input. This is not recovered physical lens/EXIF metadata.
8. **`camera_rotation_x_degrees` — FLOAT:** APC/Blender XYZ rotation X, solved jointly with Y to match the estimated pitch and roll. A level camera has X=90°.
9. **`camera_rotation_y_degrees` — FLOAT:** APC/Blender XYZ rotation Y from the same conversion. It is not simply native `roll` when the camera is tilted.

All outputs are declared lists, with one scalar/preview/report per input frame **in input order**. A single image produces a list of one which ComfyUI maps normally. Every preview item is a standard `[1,H,W,3]` IMAGE, allowing scalar and image outputs to stay paired downstream. This is independent per-image calibration, **not** shared-intrinsics or multi-camera-rig calibration.

### ECHO / APC camera wiring

For **APC: Interactive Mesh Camera**, connect the three new outputs to the inputs with exactly the same names:

```text
GeoCalib focal_length_mm             -> APC focal_length_mm
GeoCalib camera_rotation_x_degrees   -> APC camera_rotation_x_degrees
GeoCalib camera_rotation_y_degrees   -> APC camera_rotation_y_degrees
```

Convert those APC widgets to inputs if necessary. Leave `camera_rotation_z_degrees` under your own control; there is no heading input or output in GeoCalib.

**Viewport limitation:** the inspected APC version only mirrors its dedicated height/tilt override wires back into its interactive viewport. Direct lens/X/Y links drive its server-side render and camera outputs, but its 3D viewport can retain old widget values. Use the rendered IMAGE/camera outputs to verify the connection. This fork does not modify APC or its frontend.

For the estimated camera rather than an adjusted/automatic composition:

- Set APC `is_full_auto=false`.
- Set APC `camera_tilt_offset_deg=0` and `camera_fov_offset_deg=0`.
- **Leave APC's optional `camera_tilt_deg` disconnected.** It overrides the converted X value and breaks the combined pitch/roll mapping.
- Match APC's output aspect ratio to the calibrated image. Keep camera height/location separately controlled; GeoCalib does not estimate them.

The focal conversion uses the actual calibration-input width, not a hard-coded resolution or the preview height:

```text
focal_length_mm = fx_pixels * 36 / image_width_pixels
APC horizontal_FOV = 2 * atan(36 / (2 * focal_length_mm))
```

Rotations match the fitted gravity vector in APC's `Rz @ Ry @ Rx` convention after converting camera axes from `(right, down, forward)` to Blender's `(right, up, backward)`. This preserves the estimated pitch and roll for any independently selected Z rotation; it does not reconstruct absolute orientation. At nonzero roll, X need not equal `90 + pitch`. Geometry tests cover combined tilt/roll and Euler singularities.

APC is a pinhole camera: these wires do **not** transfer lens-distortion coefficients or undistort the image. Prefer `pinhole` for a directly matched rectilinear camera. The node never silently clamps focal length; JSON flags values outside APC's 10–150 mm widget range. APC separately clamps its effective FOV to 1–170°, so extreme estimates can still be limited there.

**Legacy pitch-only wiring:** use `camera_tilt_deg = -pitch` for a positive-down tilt input, instead of connecting the new X/Y pair. APC then sets `camera_rotation_x_degrees = 90 - camera_tilt_deg`. This preserves the old behavior but does not apply estimated roll. Do not mix the two wiring methods.

## Confidence and failure handling

- Reported uncertainty is **not measured error or a guaranteed confidence interval**. No arbitrary accept/reject threshold is imposed.
- Missing/non-finite results and invalid inputs raise an error instead of silently returning a level camera.
- `horizon_visible=false` means the fitted horizon is outside the image; it is not inherently a calibration failure.
- The backend assumes the principal point is at the image centre. Off-centre crops, perspective correction, weak scene cues and tilted vehicle geometry can bias calibration.
- Models are cached on CPU and returned to CPU after every call, including failure, rather than being retained indefinitely in VRAM. A lock protects the mutable backend optimiser from simultaneous calls.
- Upstream's NMF component samples random bases even in evaluation mode. Recomputed estimates can vary slightly; normal ComfyUI caching still applies. The integration test restores RNG state for a like-for-like comparison with official inference. No global seed is changed by the node.

## Verification

With the backend and node dependencies installed:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py /path/to/photo.png --output-dir outputs/smoke --batch 2
```

Optional, when APC is installed, compare against its actual camera math without model inference:

```bash
python scripts/verify_apc_geometry.py --apc-camera-ops /path/to/apc-pipeline/apc/camera_ops.py
```

Unit tests use explicit test doubles for node plumbing and the actual official geometry classes for sign/model checks; they do **not** claim to run learned inference. The smoke script runs the real model, checks per-frame outputs and source-size previews, and compares the wrapper's numbers against the official API using identical RNG state. It writes genuine `results.json` and horizon PNGs locally. Test images/results are not committed.

For an already-running ComfyUI instance, additionally verify `/object_info/GeoCalibNode` and complete a real prompt through `/history`; standalone tests do not prove the service reloaded.

## Credits and license

- Original ComfyUI wrapper: [Antilopax](https://github.com/Antilopax/ComfyUI-GeoCalib).
- GeoCalib: **Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, Marc Pollefeys**.
- [GeoCalib: Learning Single-image Calibration with Geometric Optimization (ECCV 2024)](https://arxiv.org/abs/2409.06704).
- Apache-2.0; original license retained.
