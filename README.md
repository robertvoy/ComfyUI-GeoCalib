# ComfyUI-GeoCalib

Single-image camera calibration in ComfyUI, based on [Antilopax's wrapper](https://github.com/Antilopax/ComfyUI-GeoCalib) and the **official [cvg/GeoCalib](https://github.com/cvg/GeoCalib) backend**.

This fork preserves `GeoCalibNode`, the original widget order, and output slots 0–2. It adds uncertainty, a fitted-horizon preview, correctly declared per-frame outputs, and an explicit positive-down tilt output.

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
6. **`debug_json` — STRING:** per-frame roll/pitch/FOV, their uncertainties, focal length and uncertainty, distortion coefficients where applicable, frame index, image dimensions and backend revision metadata.

All outputs are declared lists, with one scalar/preview/report per input frame **in input order**. A single image produces a list of one which ComfyUI maps normally. Every preview item is a standard `[1,H,W,3]` IMAGE, allowing scalar and image outputs to stay paired downstream. This is independent per-image calibration, **not** shared-intrinsics or multi-camera-rig calibration.

### ECHO / APC camera wiring

Connect **`camera_tilt_deg`**, not native `pitch`, to a positive-down camera tilt input:

```text
camera_tilt_deg = -pitch
camera_rotation_x_degrees = 90 - camera_tilt_deg
```

The sign is verified against the official backend's ray/latitude geometry. A negative native pitch puts the optical axis below the horizon: it is looking down.

Camera height is independent and is not estimated by this node. If a downstream camera also has an additive tilt offset or an automatic camera override, disable/reset those when the supplied tilt must remain authoritative. Roll is not applied automatically by a pitch-only connection.

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

Unit tests use explicit test doubles for node plumbing and the actual official geometry classes for sign/model checks; they do **not** claim to run learned inference. The smoke script runs the real model, checks per-frame outputs and source-size previews, and compares the wrapper's numbers against the official API using identical RNG state. It writes genuine `results.json` and horizon PNGs locally. Test images/results are not committed.

For an already-running ComfyUI instance, additionally verify `/object_info/GeoCalibNode` and complete a real prompt through `/history`; standalone tests do not prove the service reloaded.

## Credits and license

- Original ComfyUI wrapper: [Antilopax](https://github.com/Antilopax/ComfyUI-GeoCalib).
- GeoCalib: **Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, Marc Pollefeys**.
- [GeoCalib: Learning Single-image Calibration with Geometric Optimization (ECCV 2024)](https://arxiv.org/abs/2409.06704).
- Apache-2.0; original license retained.
