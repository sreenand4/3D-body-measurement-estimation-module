# Fitly BME Service (`bme-service`)

Body Measurement Estimation (BME) backend for Fitly. This service accepts 3 user photos (front, side, back) plus metadata, runs pose/segmentation analysis, and returns estimated body measurements.

Fitly Chrome extension listing: [Fitly - Virtual try on](https://chromewebstore.google.com/detail/fitly-virtual-try-on/hfeabhahdlpcanbapenbpipdaolljddm?utm_source=item-share-cb)

---

## 1) What this repo does in the Fitly system

`bme-service` is a standalone Python/Flask API used by frontend clients (`fitly-chrome`, potentially `fitly-web` and future `fitly-mobile`) to generate:

- `chest_circumference`
- `waist_circumference`
- `hip_circumference`

from uploaded user images.

At a high level:

1. Frontend collects 3 body photos + height (+ gender).
2. Frontend sends a JSON `POST /estimate`.
3. Service validates and decodes images.
4. Service runs MediaPipe pose detection + segmentation.
5. Service performs landmark visibility quality checks per view.
6. If valid, service computes circumferences and returns numeric measurements.
7. Service also returns landmark-overlay preview images for UI feedback.

---

## 2) Tech stack and runtime

- **Language**: Python 3.10
- **API framework**: Flask
- **CORS**: `flask-cors` enabled globally
- **CV/Pose**:
  - OpenCV (`opencv-python`)
  - MediaPipe Pose (`mediapipe`)
- **Numerics**: NumPy, SciPy
- **Memory inspection**: `psutil`
- **Container**: Docker (`python:3.10-slim`, exposes `8080`)

Primary entrypoint: `estimate_measurements.py`
Core measurement logic: `Scanner.py`

---

## 3) API contract

### Endpoint

- **Method**: `POST`
- **Path**: `/estimate`
- **Content-Type**: `application/json`

### Request body

Required fields:

- `front`: Base64 image string (data URL prefix allowed)
- `side`: Base64 image string
- `back`: Base64 image string
- `height`: Number (expected in centimeters)
- `gender`: String (`"F"` gets a chest correction factor; other values follow default path)

Example:

```json
{
  "front": "data:image/jpeg;base64,...",
  "side": "data:image/jpeg;base64,...",
  "back": "data:image/jpeg;base64,...",
  "height": 175,
  "gender": "M"
}
```

### Success response (`200`)

```json
{
  "status": "success",
  "measurements": {
    "chest_circumference": 96.2,
    "waist_circumference": 82.7,
    "hip_circumference": 94.9
  },
  "visualizations": {
    "front": "<base64 landmark overlay image>",
    "side": "<base64 landmark overlay image>",
    "back": "<base64 landmark overlay image>"
  }
}
```

### Error responses

#### `400 Bad Request` (validation/quality failures)

Returned for:

- Missing required request fields
- Any image fails Base64 decode into a valid image
- Pose landmark quality check failure for a specific view

Format:

```json
{
  "error": "<human-readable message>",
  "visualizations": {
    "front": "<optional>",
    "side": "<optional>",
    "back": "<optional>"
  }
}
```

Notes:

- `visualizations` is present for quality-check failures (useful for UI hints).
- Missing/invalid payload and decode failures currently return only `error`.

#### `500 Internal Server Error`

Unhandled exceptions return:

```json
{
  "error": "<exception string>"
}
```

---

## 4) Frontend state mapping (what UI should show)

This is the backend-to-frontend behavior contract developers should preserve.

### Recommended request lifecycle states

1. **Idle**: waiting for user upload/capture.
2. **Uploading/Analyzing**: in-flight `POST /estimate`.
3. **Quality Error (`400`)**:
   - Show `error` message directly to user.
   - If `visualizations` exists, display the affected view overlay so user can retake correctly.
4. **Success (`200`)**:
   - Render measurements.
   - Optionally expose overlays for confidence/debug UI.
5. **Server Error (`500`) or network failure**:
   - Show generic fallback ("Something went wrong, please retry").
   - Avoid exposing raw exception text in production UX.

### Important UX details linked to backend checks

- Frontend should guide users to capture **all required landmarks** per view, because server rejects low-visibility landmarks.
- Frontend should preserve view identity (`front`, `side`, `back`) exactly; backend logic and required landmarks are view-specific.
- Frontend should send height in **cm** unless backend is updated to support units.

---

## 5) Detailed request processing flow

Implemented in `estimate_measurements.py`.

1. Parse JSON body and check key presence (`front`, `side`, `back`, `height`).
2. Convert `height` to float.
3. Decode each Base64 image using OpenCV decode pipeline.
4. For each view:
   - Run MediaPipe pose inference.
   - Perform landmark visibility quality gate.
   - Generate and store landmark overlay visualization.
   - Fail fast with `400` if any view fails quality.
5. If all views pass:
   - Call `scanner.process_images(...)` to compute measurements.
6. Return `200` with measurements + all visualizations.
7. On exception, return `500`.

---

## 6) Quality gate and required landmarks (critical for frontend capture guidance)

Function: `check_landmark_quality(...)` in `estimate_measurements.py`.

Visibility threshold:

- Landmark must have `visibility >= 0.15`.

If one or more required landmarks are below threshold (or missing), request is rejected with:

- `400`
- error like:
  - `"Quality check failed for front view. Please re-take the image and make sure left-shoulder, right-shoulder, ... are visible."`

Required landmarks by view:

- **Front**:
  - left/right shoulder
  - left/right hip
  - left/right pinky
  - left/right index
- **Back**:
  - left/right shoulder
  - left/right hip
  - left/right thumb
  - left/right heel
- **Side**:
  - left/right shoulder
  - left/right hip
  - left/right heel

---

## 7) Measurement algorithm details

Core implementation: `Scanner.py`.

### Core approach

- Use MediaPipe landmarks + segmentation masks.
- Derive pixel measurements in front/side/back views.
- Convert pixels to centimeters using per-view scaling factors based on user height.
- Estimate circumference via ellipse approximation (Ramanujan formula) from width/depth pairs.

### Steps

1. **Per-view preprocessing**
   - RGB conversion
   - Pose + segmentation extraction
   - Store landmarks/segmentation maps per view

2. **Scale calibration**
   - Estimate person pixel height in each view
   - `scaling_factor = reference_height_cm / pixel_height`
   - Includes `1.03` correction during pixel height estimation

3. **Chest**
   - Front/back width sampled near upper torso (~17% from shoulder to hip)
   - Side depth sampled from segmentation mask at equivalent level
   - Circumference from ellipse formula
   - Additional female correction factor:
     - if `gender == "F"`: multiply chest by `1.05`

4. **Waist**
   - Front width approximated from pinky distance (`*0.98`)
   - Back width approximated from thumb distance (`*0.98`)
   - Side depth sampled around estimated waist line (`~20% above hip`)
   - Ellipse circumference with `*0.95` tightness correction

5. **Hip**
   - Front/back widths sampled from segmentation at hip center line
   - Side depth sampled similarly and corrected (`*0.85`)
   - Ellipse circumference with `*0.95` tightness correction

### Returned output units

- Measurements are returned in **centimeters**.

---

## 8) Error catalog for developers

### API-level errors

- `400`: Missing/invalid payload
  - Message:
    - `"Missing or invalid data: expected {'front': '...', 'side': '...', 'back': '...', 'height': ...}"`
- `400`: Image decode failure
  - Message:
    - `"Failed to decode one or more images"`
- `400`: Quality check failure
  - Message starts with:
    - `"Quality check failed for <view> view ..."`
  - Includes landmarks to fix.
- `500`: Unexpected runtime exception
  - Message:
    - `str(exception)` (internal detail; avoid surfacing directly in production UI)

### Notable internal exception sources (be aware when editing)

- Missing top-of-head detection in segmentation:
  - raises `ValueError("Could not find top of head in segmentation mask")`
- Missing pose results can cause downstream attribute access failures if not guarded.

---

## 9) Code map: where to edit

- `estimate_measurements.py`
  - API request/response schema
  - status codes and error payload shape
  - quality gate logic
  - landmark overlay generation

- `Scanner.py`
  - biometric estimation formulas
  - correction factors and heuristics
  - per-view scaling/segmentation/landmark mechanics

- `requirements.txt`
  - runtime dependencies

- `Dockerfile`
  - deployment image/runtime boot config

---

## 10) Cross-repo integration notes

Even though this repo is standalone, changes here affect client behavior in other repos:

- `fitly-chrome` (current primary client):
  - Depends on this response schema for BME UX and error rendering.
- `fitly-web`:
  - If web app consumes BME endpoint, it must align to same request and error contracts.
- `fitly-mobile` (future):
  - Should implement same state machine (`idle -> processing -> quality_error/success/server_error`) to stay consistent.

If you modify:

- field names (`measurements`, `visualizations`, `error`)
- status codes (`200/400/500`)
- landmark requirements
- correction factors

then update client-side parsing and UX guidance in downstream repos at the same time.

---

## 11) Local development

### Install and run (Python)

```bash
pip install -r requirements.txt
python estimate_measurements.py
```

Default port:

- `8080` (or `$PORT` if set)

### Run with Docker

```bash
docker build -t fitly-bme .
docker run -p 8080:8080 fitly-bme
```

---

## 12) Current limitations and future hardening

Current implementation is functional but has opportunities for production hardening:

- No explicit request schema validation library (manual key checks only).
- No structured error codes (string-only `error` messages).
- Broad `except Exception` can hide failure classes.
- No authentication/rate limiting at API layer.
- No formal API versioning.

When introducing these improvements, keep backward compatibility with client repos or version the endpoint.