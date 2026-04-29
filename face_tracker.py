import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile


def get_face_crop_path(video_path: str, target_w: int = 1080, target_h: int = 1920):

    cap = cv2.VideoCapture(video_path)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Crop dimensions (9:16 with zoom margin) ──────────────────────────────
    max_crop_h = src_h
    max_crop_w = int(src_h * target_w / target_h)

    if max_crop_w > src_w:
        max_crop_w = src_w
        max_crop_h = int(src_w * target_h / target_w)

    zoom_margin = 0.85
    crop_w = int(max_crop_w * zoom_margin)
    crop_h = int(max_crop_h * zoom_margin)

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_face       = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    )

    # ── EMA camera tracker ────────────────────────────────────────────────────
    # Pure exponential moving average — mathematically cannot overshoot or
    # oscillate, eliminating the spring-physics jitter from the old approach.
    #
    # Two alpha values:
    #   ALPHA_FAST  — used for first ~2s after detection while camera locks on.
    #                 Slightly more responsive so initial framing isn't sluggish.
    #   ALPHA_SLOW  — used once camera is settled; very lazy, only moves when
    #                 face actually drifts, ignoring micro-head-bobs.
    ALPHA_FAST    = 0.10    # ~2.5s settle, max 10px/frame
    ALPHA_SLOW    = 0.06    # ~4s settle, max 6px/frame — almost imperceptible

    # Headroom EMA: updated independently so vertical drift stays smooth
    ALPHA_HEADROOM = 0.08

    # Dead zone: don't move camera for tiny face shifts inside this radius.
    # 25px on the face-tracked crop (which is ~85% of the 9:16 inscribed rect).
    # This absorbs nodding and micro-movement without camera drift.
    DEAD_ZONE = 25          # px — was 45 (too large → visible lurches)

    # Warm-up period: use ALPHA_FAST for this many frames after first detection
    WARMUP_FRAMES = int(fps * 2.0)  # 2 seconds

    ema_cx       = None
    ema_cy       = None
    ema_headroom = None
    target_x     = None
    target_y     = None

    # Camera position — EMA directly on position, no velocity state needed
    cam_x = None
    cam_y = None
    frames_since_detection = 0
    first_detection_done   = False

    raw_path  = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        detected_cx = detected_cy = face_h_px = None

        if results.detections:
            det  = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
            )
            bbox = det.location_data.relative_bounding_box

            bx = bbox.xmin   * src_w
            by = bbox.ymin   * src_h
            bw = bbox.width  * src_w
            bh = bbox.height * src_h

            detected_cx = bx + bw / 2
            detected_cy = by + bh / 2
            face_h_px   = bh

        if detected_cx is not None:
            if not first_detection_done:
                # First detection: initialise everything at face position.
                # Camera snaps here but we post-smooth the whole path later,
                # so the snap becomes a fast ramp instead of an instant jump.
                ema_cx       = detected_cx
                ema_cy       = detected_cy
                ema_headroom = face_h_px * 0.55
                target_x     = ema_cx
                target_y     = ema_cy - ema_headroom
                cam_x        = target_x
                cam_y        = target_y
                first_detection_done   = True
                frames_since_detection = 0
            else:
                frames_since_detection += 1

                # EMA pre-filter on raw detection (smooths MediaPipe noise)
                ema_cx = 0.18 * detected_cx + 0.82 * ema_cx
                ema_cy = 0.18 * detected_cy + 0.82 * ema_cy

                raw_headroom = face_h_px * 0.55
                ema_headroom = ALPHA_HEADROOM * raw_headroom + (1 - ALPHA_HEADROOM) * ema_headroom
                proposed_y   = ema_cy - ema_headroom

                # Dead zone: only advance spring target when face moves meaningfully
                if abs(ema_cx   - target_x) > DEAD_ZONE:
                    target_x = ema_cx
                if abs(proposed_y - target_y) > DEAD_ZONE:
                    target_y = proposed_y

        if cam_x is None:
            # No face detected yet — park camera at frame centre
            cam_x = float(src_w // 2)
            cam_y = float(src_h // 2)
        else:
            # EMA camera movement: fast during warm-up, slow once settled
            alpha = ALPHA_FAST if frames_since_detection < WARMUP_FRAMES else ALPHA_SLOW
            cam_x = alpha * target_x + (1.0 - alpha) * cam_x
            cam_y = alpha * target_y + (1.0 - alpha) * cam_y

        # Clamp crop box to valid frame boundaries
        fx = cam_x - crop_w / 2
        fy = cam_y - crop_h / 2
        fx = max(0.0, min(fx, float(src_w - crop_w)))
        fy = max(0.0, min(fy, float(src_h - crop_h)))

        raw_path.append({"frame": frame_idx, "x": fx, "y": fy, "w": crop_w, "h": crop_h})
        frame_idx += 1

    cap.release()

    # ── Gaussian temporal smoothing (second pass) ─────────────────────────────
    # sigma_sec=0.70 (was 0.40) — wider kernel absorbs any remaining micro-jitter
    # and makes the path feel like a professional camera operator.
    crop_path = _smooth_path(raw_path, fps, sigma_sec=0.70)

    return crop_path, fps, src_w, src_h


def _smooth_path(raw_path: list[dict], fps: float, sigma_sec: float = 0.70) -> list[dict]:
    """Gaussian-weighted rolling average over x/y.  Operates in floats throughout."""
    if not raw_path:
        return raw_path

    xs = np.array([kf["x"] for kf in raw_path], dtype=float)
    ys = np.array([kf["y"] for kf in raw_path], dtype=float)

    sigma_frames = max(1.0, sigma_sec * fps)
    radius       = int(3 * sigma_frames)
    k            = np.arange(-radius, radius + 1)
    kernel       = np.exp(-0.5 * (k / sigma_frames) ** 2)
    kernel      /= kernel.sum()

    xs_s = np.convolve(xs, kernel, mode="same")
    ys_s = np.convolve(ys, kernel, mode="same")

    norm  = np.convolve(np.ones(len(raw_path)), kernel, mode="same")
    xs_s /= norm
    ys_s /= norm

    return [
        {"frame": kf["frame"], "x": float(xs_s[i]), "y": float(ys_s[i]),
         "w": kf["w"], "h": kf["h"]}
        for i, kf in enumerate(raw_path)
    ]


def export_crop_filter(crop_path, fps, src_w, src_h,
                       target_w=1080, target_h=1920,
                       min_keyframe_dist: float = 0.5,
                       script_path: str = None):
    """Write crop keyframe script.  Rounds x/y only here, at output time.
    
    Args:
        script_path: Optional custom path for script file. If not provided,
                    uses temp directory (which can cause command line length issues on Windows).
    """
    lines  = []
    prev_x = prev_y = None

    for kf in crop_path:
        x = round(kf["x"])
        y = round(kf["y"])
        w, h = kf["w"], kf["h"]

        if prev_x is not None:
            if abs(x - prev_x) + abs(y - prev_y) < min_keyframe_dist:
                continue

        t = kf["frame"] / fps

        lines.append(f"{t:.3f} [enter] crop x {x};")
        lines.append(f"{t:.3f} [enter] crop y {y};")
        lines.append(f"{t:.3f} [enter] crop w {w};")
        lines.append(f"{t:.3f} [enter] crop h {h};")

        prev_x, prev_y = x, y

    if script_path is None:
        script_path = os.path.join(tempfile.gettempdir(), "crop_script.txt")
    
    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    return script_path