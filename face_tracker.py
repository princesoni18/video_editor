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

    # ── Physics parameters ────────────────────────────────────────────────────
    stiffness = 0.07   # how eagerly camera chases target (lower = lazier)
    damping   = 0.82   # how quickly velocity bleeds off (higher = less overshoot)
    max_speed = 18     # px/frame speed cap

    # EMA pre-filter on raw detections (lower alpha = heavier smoothing)
    ema_alpha = 0.20
    ema_cx    = None   # initialised on first detection — no cold-start slam
    ema_cy    = None

    # Separate slow EMA for headroom so it doesn't drag the y target around
    ema_headroom       = None
    ema_headroom_alpha = 0.10

    dead_zone = 45     # px — ignore micro-movements inside this radius

    # Camera state — ALL FLOATS, never cast to int inside the loop
    smooth_x = float(src_w // 2)
    smooth_y = float(src_h // 2)
    vel_x    = 0.0
    vel_y    = 0.0
    target_x = smooth_x
    target_y = smooth_y

    first_detection_done = False

    raw_path  = []   # x/y stored as floats throughout
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

            # Keep as floats — no int() cast here
            bx = bbox.xmin   * src_w
            by = bbox.ymin   * src_h
            bw = bbox.width  * src_w
            bh = bbox.height * src_h

            detected_cx = bx + bw / 2
            detected_cy = by + bh / 2
            face_h_px   = bh

        if detected_cx is not None:
            # On the very first detection, snap everything to the face position
            # so the camera doesn't slam in from the frame centre
            if not first_detection_done:
                ema_cx           = detected_cx
                ema_cy           = detected_cy
                ema_headroom     = face_h_px * 0.55
                target_x         = ema_cx
                target_y         = ema_cy - ema_headroom
                smooth_x         = target_x
                smooth_y         = target_y
                first_detection_done = True

            # EMA pre-filter on position
            ema_cx = ema_alpha * detected_cx + (1 - ema_alpha) * ema_cx
            ema_cy = ema_alpha * detected_cy + (1 - ema_alpha) * ema_cy

            # Smooth headroom independently — apply to EMA cy, not smooth_y,
            # so it isn't double-filtered and doesn't cause vertical drift
            raw_headroom = face_h_px * 0.55
            ema_headroom = ema_headroom_alpha * raw_headroom + (1 - ema_headroom_alpha) * ema_headroom
            proposed_cy  = ema_cy - ema_headroom

            # Dead zone — only move spring target when face moves meaningfully
            if abs(ema_cx    - target_x) > dead_zone:
                target_x = ema_cx
            if abs(proposed_cy - target_y) > dead_zone:
                target_y = proposed_cy

        # ── Velocity-based spring physics ─────────────────────────────────────
        force_x = (target_x - smooth_x) * stiffness
        force_y = (target_y - smooth_y) * stiffness

        vel_x = vel_x * damping + force_x
        vel_y = vel_y * damping + force_y

        vel_x = max(-max_speed, min(max_speed, vel_x))
        vel_y = max(-max_speed, min(max_speed, vel_y))

        smooth_x += vel_x
        smooth_y += vel_y

        # Clamp crop box — stay in floats, never round here
        fx = smooth_x - crop_w / 2
        fy = smooth_y - crop_h / 2
        fx = max(0.0, min(fx, float(src_w - crop_w)))
        fy = max(0.0, min(fy, float(src_h - crop_h)))

        raw_path.append({"frame": frame_idx, "x": fx, "y": fy, "w": crop_w, "h": crop_h})
        frame_idx += 1

    cap.release()

    # ── Gaussian temporal smoothing over float positions ──────────────────────
    crop_path = _smooth_path(raw_path, fps, sigma_sec=0.40)

    return crop_path, fps, src_w, src_h


def _smooth_path(raw_path: list[dict], fps: float, sigma_sec: float = 0.40) -> list[dict]:
    """Gaussian-weighted rolling average over x/y.

    Operates entirely in floats — rounding is deferred to export_crop_filter
    so quantisation noise never enters the path data.
    """
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

    # Re-normalise edges (zero-padding in convolve biases the first/last frames)
    norm  = np.convolve(np.ones(len(raw_path)), kernel, mode="same")
    xs_s /= norm
    ys_s /= norm

    # Return floats — no rounding here
    return [
        {"frame": kf["frame"], "x": float(xs_s[i]), "y": float(ys_s[i]),
         "w": kf["w"], "h": kf["h"]}
        for i, kf in enumerate(raw_path)
    ]


def export_crop_filter(crop_path, fps, src_w, src_h,
                       target_w=1080, target_h=1920,
                       min_keyframe_dist: float = 0.5):
    """Write a crop keyframe script.

    x/y in crop_path are floats — rounded only here, once, at output time.
    min_keyframe_dist lowered to 0.5px: the path is already smooth so we
    want dense keyframes for the editor to interpolate, not sparse ones that
    create long linear segments.
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

    script_path = os.path.join(tempfile.gettempdir(), "crop_script.txt")
    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    return script_path