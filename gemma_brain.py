"""
gemma_brain.py — Gemma 3n E4B via Ollama (multimodal)
Sends video keyframes + transcript → smart editing decisions as JSON

Gemma 3n E4B uses selective parameter activation:
  - 6.87B total params, ~4B effective VRAM cost
  - Multimodal: reads images + text together
  - 128K context window
  - Pulls ~7.5GB: ollama pull gemma4:e4b
"""

import json, re, base64, logging, cv2
from pathlib import Path
from dataclasses import dataclass

import requests
import numpy as np

log = logging.getLogger(__name__)

OLLAMA_URL  = "http://localhost:11434/api/chat"   # chat endpoint supports images
GEMMA_MODEL = "gemma4:e4b"

# How many keyframes to sample from the video and send to Gemma
# More = smarter decisions, slower inference. 6 is a good balance.
N_KEYFRAMES = 6




# ─────────────────────────────────────────────────────────────
#  DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class EditDecision:
    style: str           # "energetic" | "calm" | "educational" | "funny"
    hook: str            # punchy opener text shown at video start
    caption_color: str   # hex e.g. "#FFFF00"
    caption_font: str    # "bold" | "rounded" | "minimal"
    cut_points: list     # [float, ...] timestamps for cuts
    broll_prompts: list  # [{time, prompt, duration}, ...]
    energy_map: list     # [{time, energy: "low|high|peak"}, ...]
    suggested_trim: dict # {start: float, end: float}
    scene_description: str  # Gemma's visual read of the video
    visual_style: str       # Gemma's colour/aesthetic observation


# ─────────────────────────────────────────────────────────────
#  KEYFRAME EXTRACTION
# ─────────────────────────────────────────────────────────────

def _extract_keyframes(video_path: str, n: int = N_KEYFRAMES) -> list[str]:
    """
    Extract n evenly-spaced frames from the video.
    Returns list of base64-encoded JPEG strings (for Ollama image API).
    """
    cap      = cv2.VideoCapture(video_path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total / fps

    # Skip first/last 2s — usually boring intros/outros
    margin     = min(2.0, duration * 0.05)
    sample_pts = np.linspace(margin, duration - margin, n)

    frames_b64 = []
    for t in sample_pts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to 512px wide — enough for visual understanding, keeps token count low
        h, w = frame.shape[:2]
        scale = 512 / w
        frame = cv2.resize(frame, (512, int(h * scale)))

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()
    log.info(f"[gemma] Extracted {len(frames_b64)} keyframes")
    return frames_b64


# ─────────────────────────────────────────────────────────────
#  ROBUST JSON EXTRACTION
# ─────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """Extract JSON from Gemma's response even with extra text around it."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: truncate at last closing brace
    try:
        return json.loads(raw[:raw.rfind("}") + 1])
    except Exception:
        raise ValueError(f"Could not extract JSON from:\n{raw[:400]}")


# ─────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────

def analyze(transcript_words: list[dict], duration: float,
            video_path: str | None = None) -> EditDecision:
    """
    Send keyframes + transcript to Gemma 3n E4B for multimodal editing analysis.

    Args:
        transcript_words: [{word, start, end}, ...]
        duration:         video length in seconds
        video_path:       source video path for keyframe extraction (None = text-only)
    """

    # Build timed transcript
    timed_lines, chunk, chunk_start = [], [], None
    for w in transcript_words:
        if chunk_start is None:
            chunk_start = w["start"]
        chunk.append(w["word"])
        if len(chunk) >= 8:
            timed_lines.append(f"[{chunk_start:.1f}s] {' '.join(chunk)}")
            chunk, chunk_start = [], None
    if chunk:
        timed_lines.append(f"[{chunk_start:.1f}s] {' '.join(chunk)}")
    timed_transcript = "\n".join(timed_lines)

    # Extract keyframes
    frames_b64 = []
    if video_path:
        try:
            frames_b64 = _extract_keyframes(video_path)
        except Exception as e:
            log.warning(f"[gemma] Keyframe extraction failed: {e} — text-only mode")

    has_vision = len(frames_b64) > 0

    vision_note = (
        f"I am sending you {len(frames_b64)} keyframes from the video. "
        f"Use them to visually understand the scene, speaker energy, setting, "
        f"and colour tone. Let what you SEE directly inform your decisions — "
        f"especially caption colour (pick contrast against real background) "
        f"and b-roll prompts (make them visually complement the actual scene)."
        if has_vision else
        "No frames available — base all decisions on transcript only."
    )

    prompt = f"""You are a world-class YouTube Shorts editor. Analyze this content and return precise editing instructions.

{vision_note}

VIDEO DURATION: {duration:.1f}s

TIMED TRANSCRIPT:
{timed_transcript}

Return ONLY a valid JSON object — no markdown fences, no explanation, nothing before or after the JSON.

{{
  "scene_description": "1-2 sentences on what you actually see in the frames — speaker appearance, setting, props",
  "visual_style": "lighting + colour tone you observe, e.g. warm golden hour outdoor, cool blue studio, dark moody room",
  "style": "energetic|calm|educational|funny",
  "hook": "max 6 words — tease the most surprising or valuable moment",
  "caption_color": "#RRGGBB — colour with strong contrast against the actual video background you see",
  "caption_font": "bold|rounded|minimal",
  "cut_points": [floats — timestamps at natural pauses or energy shifts, max 8],
  "broll_prompts": [
    {{
      "time": 5.0,
      "prompt": "very specific scene that visually complements what speaker says at this moment. vertical 9:16, photorealistic, cinematic",
      "duration": 2.0
    }}
  ],
  "energy_map": [
    {{"time": 0.0, "energy": "low|high|peak"}}
  ],
  "suggested_trim": {{
    "start": 0.0,
    "end": {duration:.1f},
    "reason": "brief reason"
  }}
}}

Rules:
- scene_description + visual_style MUST reference what you actually see in the images
- caption_color: look at the actual background and pick a contrasting colour
- b-roll prompts: highly specific to content topic, NOT generic stock descriptions
- energy_map: one entry every ~8 seconds
- max 3 b-roll inserts for a Short
- if video under 60s keep full duration in suggested_trim; if longer find the best 45-55s window
"""

    message = {
        "role": "user",
        "content": prompt
    }
    # Prevent 500 Server Error by skipping multimodal images for text-only Gemma
    # if frames_b64:
    #     message["images"] = frames_b64

    payload = {
        "model":  GEMMA_MODEL,
        "stream": False,
        "messages": [message],
        "options": {"temperature": 0.25, "num_ctx": 8192, "top_p": 0.9},
    }

    log.info(f"[gemma] Querying {GEMMA_MODEL} "
             f"({'multimodal: ' + str(len(frames_b64)) + ' frames' if has_vision else 'text-only'})...")

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()

        raw  = resp.json()["message"]["content"].strip()
        data = _extract_json(raw)

        trim = data.get("suggested_trim", {"start": 0, "end": duration})

        log.info(f"[gemma] Style={data.get('style')} | Hook={data.get('hook')}")
        log.info(f"[gemma] Scene: {data.get('scene_description','')[:80]}")
        log.info(f"[gemma] Visual: {data.get('visual_style','')[:60]}")
        log.info(f"[gemma] Cuts: {data.get('cut_points',[])} | B-roll: {len(data.get('broll_prompts',[]))}")

        return EditDecision(
            style             = data.get("style", "energetic"),
            hook              = data.get("hook", ""),
            caption_color     = data.get("caption_color", "#FFFF00"),
            caption_font      = data.get("caption_font", "bold"),
            cut_points        = data.get("cut_points", []),
            broll_prompts     = data.get("broll_prompts", []),
            energy_map        = data.get("energy_map", []),
            suggested_trim    = {"start": trim.get("start", 0), "end": trim.get("end", duration)},
            scene_description = data.get("scene_description", ""),
            visual_style      = data.get("visual_style", ""),
        )

    except Exception as e:
        log.warning(f"[gemma] Failed: {e} — using safe defaults")
        return EditDecision(
            style="energetic", hook="You need to see this!",
            caption_color="#FFFF00", caption_font="bold",
            cut_points=[], broll_prompts=[], energy_map=[],
            suggested_trim={"start": 0, "end": duration},
            scene_description="", visual_style="",
        )