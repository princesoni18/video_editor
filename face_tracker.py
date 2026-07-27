import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
import logging

log = logging.getLogger(__name__)


def get_face_crop_path(
    video_path: str,
    target_w: int = 1080,
    target_h: int = 1920
):

    # ─────────────────────────────────────────
    # Open video
    # ─────────────────────────────────────────

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ─────────────────────────────────────────
    # Crop dimensions (9:16)
    # ─────────────────────────────────────────

    max_crop_h = src_h
    max_crop_w = int(src_h * target_w / target_h)

    if max_crop_w > src_w:

        max_crop_w = src_w
        max_crop_h = int(src_w * target_h / target_w)

    zoom_margin = 0.95

    crop_w = int(max_crop_w * zoom_margin)
    crop_h = int(max_crop_h * zoom_margin)

    # ─────────────────────────────────────────
    # MediaPipe Face Detector
    # ─────────────────────────────────────────

    mp_face = mp.solutions.face_detection

    face_detector = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    )

    # ─────────────────────────────────────────
    # TikTok/Reels Style Tracker Settings
    # ─────────────────────────────────────────

    # Raw detection smoothing
    DETECTION_ALPHA = 0.10

    # Separate responsiveness for X/Y
    TARGET_ALPHA_X = 0.20
    TARGET_ALPHA_Y = 0.30

    # Camera spring physics
    SPRING_STIFFNESS = 0.15
    SPRING_DAMPING = 0.85

    # Max camera speed
    MAX_VELOCITY = 50.0

    # Vertical framing responsiveness
    HEADROOM_ALPHA = 0.20

    # ─────────────────────────────────────────
    # Tracker State
    # ─────────────────────────────────────────

    ema_cx = None
    ema_cy = None
    ema_headroom = None

    target_x = None
    target_y = None

    cam_x = None
    cam_y = None

    vel_x = 0.0
    vel_y = 0.0

    first_detection_done = False

    raw_path = []
    frame_idx = 0

    # ─────────────────────────────────────────
    # Main Tracking Loop
    # ─────────────────────────────────────────

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detector.process(rgb)

        detected_cx = None
        detected_cy = None
        face_h_px = None

        # ─────────────────────────────────────
        # Detect largest face
        # ─────────────────────────────────────

        if results.detections:

            det = max(
                results.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
            )

            bbox = det.location_data.relative_bounding_box

            bx = bbox.xmin * src_w
            by = bbox.ymin * src_h
            bw = bbox.width * src_w
            bh = bbox.height * src_h

            detected_cx = bx + bw / 2
            detected_cy = by + bh / 2

            face_h_px = bh

        # ─────────────────────────────────────
        # Detection Filtering
        # ─────────────────────────────────────

        if detected_cx is not None:

            if not first_detection_done:

                ema_cx = detected_cx
                ema_cy = detected_cy

                ema_headroom = face_h_px * 0.55

                target_x = ema_cx
                target_y = ema_cy - ema_headroom

                cam_x = target_x
                cam_y = target_y

                first_detection_done = True

            else:

                # Smooth raw detections

                ema_cx = (
                    DETECTION_ALPHA * detected_cx
                    + (1.0 - DETECTION_ALPHA) * ema_cx
                )

                ema_cy = (
                    DETECTION_ALPHA * detected_cy
                    + (1.0 - DETECTION_ALPHA) * ema_cy
                )

                # Smooth headroom

                raw_headroom = face_h_px * 0.55

                ema_headroom = (
                    HEADROOM_ALPHA * raw_headroom
                    + (1.0 - HEADROOM_ALPHA) * ema_headroom
                )

                desired_y = ema_cy - ema_headroom

                # ─────────────────────────────
                # Smooth target movement
                # ─────────────────────────────

                target_x = (
                    TARGET_ALPHA_X * ema_cx
                    + (1.0 - TARGET_ALPHA_X) * target_x
                )

                target_y = (
                    TARGET_ALPHA_Y * desired_y
                    + (1.0 - TARGET_ALPHA_Y) * target_y
                )

        # ─────────────────────────────────────
        # Before first detection
        # ─────────────────────────────────────

        if cam_x is None:

            cam_x = float(src_w // 2)
            cam_y = float(src_h // 2)

        # ─────────────────────────────────────
        # Spring Camera Physics
        # ─────────────────────────────────────

        dx = target_x - cam_x
        dy = target_y - cam_y

        # Spring force

        force_x = dx * SPRING_STIFFNESS
        force_y = dy * SPRING_STIFFNESS

        # Apply force to velocity

        vel_x += force_x
        vel_y += force_y

        # Damping

        vel_x *= SPRING_DAMPING
        vel_y *= SPRING_DAMPING

        # Clamp max speed

        speed = (vel_x**2 + vel_y**2) ** 0.5

        if speed > MAX_VELOCITY:

            scale = MAX_VELOCITY / speed

            vel_x *= scale
            vel_y *= scale

        # Move camera

        cam_x += vel_x
        cam_y += vel_y

        # ─────────────────────────────────────
        # Clamp crop to frame boundaries
        # ─────────────────────────────────────

        fx = cam_x - crop_w / 2
        fy = cam_y - crop_h / 2

        fx = max(0.0, min(fx, float(src_w - crop_w)))
        fy = max(0.0, min(fy, float(src_h - crop_h)))

        raw_path.append({
            "frame": frame_idx,
            "x": fx,
            "y": fy,
            "w": crop_w,
            "h": crop_h
        })

        frame_idx += 1

    cap.release()

    # ─────────────────────────────────────────
    # Final temporal smoothing
    # ─────────────────────────────────────────

    crop_path = _smooth_path(
        raw_path,
        fps,
        sigma_sec=0.25
    )

    return crop_path, fps, src_w, src_h


def _smooth_path(
    raw_path: list[dict],
    fps: float,
    sigma_sec: float = 0.25
) -> list[dict]:

    if not raw_path:
        return raw_path

    xs = np.array(
        [kf["x"] for kf in raw_path],
        dtype=float
    )

    ys = np.array(
        [kf["y"] for kf in raw_path],
        dtype=float
    )

    sigma_frames = max(1.0, sigma_sec * fps)

    radius = int(3 * sigma_frames)

    k = np.arange(-radius, radius + 1)

    kernel = np.exp(
        -0.5 * (k / sigma_frames) ** 2
    )

    kernel /= kernel.sum()

    xs_s = np.convolve(xs, kernel, mode="same")
    ys_s = np.convolve(ys, kernel, mode="same")

    norm = np.convolve(
        np.ones(len(raw_path)),
        kernel,
        mode="same"
    )

    xs_s /= norm
    ys_s /= norm

    return [
        {
            "frame": kf["frame"],
            "x": float(xs_s[i]),
            "y": float(ys_s[i]),
            "w": kf["w"],
            "h": kf["h"]
        }
        for i, kf in enumerate(raw_path)
    ]


def export_crop_filter(
    crop_path,
    fps,
    src_w,
    src_h,
    target_w=1080,
    target_h=1920,
    min_keyframe_dist: float = 0.0,
    script_path: str = None
):

    lines = []

    for i, kf in enumerate(crop_path):

        x = kf["x"]
        y = kf["y"]

        w = kf["w"]
        h = kf["h"]

        t = kf["frame"] / fps

        if i < 30:
            log.info(f"[face_tracker] Emitting crop frame {i}: x={x:.2f}, y={y:.2f}, w={w}, h={h}")

        lines.append(f"{t:.3f} [enter] crop x {x:.2f};")
        lines.append(f"{t:.3f} [enter] crop y {y:.2f};")
        lines.append(f"{t:.3f} [enter] crop w {w};")
        lines.append(f"{t:.3f} [enter] crop h {h};")

    if script_path is None:

        script_path = os.path.join(
            tempfile.gettempdir(),
            "crop_script.txt"
        )

    with open(script_path, "w") as f:
        f.write("\n".join(lines))

    return script_path