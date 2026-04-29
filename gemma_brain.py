"""
gemma_brain.py — Gemma 3:4B or Gemma 4:E4B via Ollama (multimodal)
Sends video keyframes + transcript → smart editing decisions as JSON

Dual-model support:
  - Gemma 3:4B (default): 4B parameters, ~3GB VRAM, faster (6144 context)
  - Gemma 4:E4B: 8B parameters, ~9GB VRAM, stronger (8192 context)
  
Set USE_GEMMA4 boolean to switch models at the top of this file.
"""

import json, re, base64, logging, cv2
from pathlib import Path
from dataclasses import dataclass, field

import requests
import numpy as np

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

# ── MODEL CONFIGURATION ─────────────────────────────────────
# Set USE_GEMMA4 = True to use gemma4:e4b (stronger, ~9GB, needs 8192+ context)
# Set USE_GEMMA4 = False to use gemma3:4b (faster, ~3GB, needs 6144 context)
USE_GEMMA4 = False

if USE_GEMMA4:
    GEMMA_MODEL = "gemma4:e4b"
    MODEL_CONTEXT = 8192
    MODEL_TIMEOUT = 300    # gemma4 is faster
else:
    GEMMA_MODEL = "gemma3:4b"
    MODEL_CONTEXT = 6144
    MODEL_TIMEOUT = 360    # gemma3:4b needs more time

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
    energy_map: list     # [{time, energy: "low|high|peak"}, ...]  (kept for compat)
    suggested_trim: dict # {start: float, end: float}
    scene_description: str
    visual_style: str

    # ── NEW ──────────────────────────────────────────────────
    # Gemma-directed zoom instructions.
    # Each entry:  { "time": float,   ← when to start the zoom (seconds, in trimmed timeline)
    #                "zoom": float,   ← target zoom level, e.g. 1.0 = normal, 1.06 = punch in
    #                "duration": float } ← how long to hold / transition (seconds)
    #
    # The list must be sorted by time.  pipeline.py passes this directly to
    # build_zoom_filter() in effects.py which converts it to an FFmpeg zoompan expression.
    zoom_cuts: list = field(default_factory=list)

    # Gemma-flagged word indices (into transcript word list) to show ALONE
    # as big emphasis captions. Sparse — max ~1 per 8–10 words.
    emphasis_word_indices: list = field(default_factory=list)

    # word_index → emoji string. Max 3 total across the whole video.
    emoji_map: dict = field(default_factory=dict)


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

    margin     = min(2.0, duration * 0.05)
    sample_pts = np.linspace(margin, duration - margin, n)

    frames_b64 = []
    for t in sample_pts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w   = frame.shape[:2]
        scale  = 512 / w
        frame  = cv2.resize(frame, (512, int(h * scale)))

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

    try:
        return json.loads(raw[:raw.rfind("}") + 1])
    except Exception:
        raise ValueError(f"Could not extract JSON from:\n{raw[:400]}")


# ─────────────────────────────────────────────────────────────
#  ZOOM CUTS VALIDATOR
# ─────────────────────────────────────────────────────────────

def _validate_zoom_cuts(raw_cuts: list, duration: float) -> list:
    """
    Sanitise Gemma's zoom_cuts output:
      - Clamp zoom to [1.0, 1.10]  (above 1.10 crops too aggressively on 9:16)
      - Clamp time to [0, duration]
      - Clamp duration to [0.3, 4.0]
      - Sort by time
      - Ensure gaps between cuts ≥ 1.5 s so zooms don't stack confusingly
      - Max 8 zoom cuts for a Short
    """
    if not isinstance(raw_cuts, list):
        return []

    validated = []
    for item in raw_cuts:
        try:
            t   = float(item.get("time", 0))
            z   = float(item.get("zoom", 1.0))
            dur = float(item.get("duration", 1.0))
        except (TypeError, ValueError):
            continue

        t   = max(0.0, min(t, duration - 0.5))
        z   = max(1.0, min(z, 1.10))
        dur = max(0.3, min(dur, 4.0))
        validated.append({"time": t, "zoom": z, "duration": dur})

    # Sort by time
    validated.sort(key=lambda x: x["time"])

    # Remove cuts that are too close to each other
    filtered = []
    for cut in validated:
        if filtered and cut["time"] - filtered[-1]["time"] < 1.5:
            continue
        filtered.append(cut)

    return filtered[:8]


# ─────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────

def analyze(transcript_words: list[dict], duration: float,
            video_path: str | None = None) -> EditDecision:
    """
    Send keyframes + transcript to Gemma 3:4B for multimodal editing analysis.
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
        "Use them to understand the scene, speaker energy, setting, and colour tone. "
        "Let what you SEE directly inform your decisions — especially caption colour "
        "(pick contrast against real background), b-roll prompts, and zoom moments "
        "(zoom in when the speaker makes a strong point or reacts visibly)."
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
  "zoom_cuts": [
    {{
      "time": 3.5,
      "zoom": 1.06,
      "duration": 1.2
    }}
  ],
  "emphasis_word_indices": [4, 17, 31],
  "emoji_map": {{"4": "⚡", "31": "🔥"}},
  "suggested_trim": {{
    "start": 0.0,
    "end": {duration:.1f},
    "reason": "brief reason"
  }}
}}

emphasis_word_indices rules:
- These are 0-based indices into the word list (word 0 = first word of transcript).
- An emphasis word is shown ALONE as a BIG SOLO caption pill — nothing else on screen.
- It creates a "SMASH CUT to single word" effect. Use it like a highlight reel moment.

WHAT TO PICK — only these types:
  * NOUNS that are the core topic: "AI", "pipeline", "agent", "schedule", "GPU", "deadline"
  * TECHNICAL TERMS the speaker is explaining: pick the term itself, not words around it
  * STRONG ADJECTIVES/VERBS with impact: "FREE", "WRONG", "FASTER", "BROKEN", "CRASHED"
  * Numbers/stats that shock: "100x", "ZERO", "1ms"
  * A word the speaker clearly stresses or pauses before/after (audible emphasis)

ABSOLUTE NEVER PICK (hard blacklist — English AND Hinglish fillers):
  English:  the, a, an, is, it, in, on, at, to, of, and, or, but, so, then, this, that, with, you, we, I
  Hinglish: toh, hai, kya, aur, bhi, yeh, ye, ek, jo, jisse, jab, ka, ki, ke, ko, na, hi, hoga, hota, tha, the, woh, wo, se, me, mein, ab, bas

SPARSITY IS CRITICAL — these are rare moments, but mark genuine highlights:
  - Pick approximately 1 emphasis word per 8–10 words of transcript (be more selective than usual)
  - For a 30-word transcript: 2–3 emphasis words is good
  - For 60 words: 4–6 emphasis words
  - For 90 words: 7–9 emphasis words
  - Never pick two consecutive word indices
  - Spread them evenly across the video — not all clustered at start or end
  - Prefer real semantic importance over perfect sparsity. Strong keywords beat perfect ratios.

emoji_map rules:
- Place 1–3 emojis total across the ENTIRE video. More = cluttered.
- Use only when the emoji genuinely adds meaning or humor to that exact word/moment.
- The emoji appears appended to that word's caption segment.
- Good uses: 🔥 on "amazing", ⚡ on "fast", ⏱️ on "time", 💡 on "idea", ❌ on "wrong"
- Bad uses: random emojis on every caption, emojis that don't relate to the word
- Key: emoji_map keys are STRINGS of the word index (JSON requirement).

zoom_cuts rules — read carefully:
- Pick moments where a zoom IN makes the video more engaging:
    * Speaker says something surprising, punchy, or emotionally charged
    * A reaction beat — a pause, a smirk, a head shake
    * A key word or phrase the viewer needs to feel, not just hear
    * Right after a cut point (gives the new segment energy)
- zoom value: 1.0 = no zoom (normal). Max is 1.08. Recommended: 1.04–1.07.
    * 1.03–1.04 = subtle pull-in (calm/educational content)
    * 1.05–1.06 = visible punch-in (energetic/motivational)
    * 1.07–1.08 = strong punch (peak moments only, use sparingly)
- duration: how long the zoom-in holds before easing back to 1.0 (0.5–3.0 s).
    * Short duration (0.5–1.0 s): snap zoom — punchy, exciting
    * Long duration (2.0–3.0 s): slow push — builds tension or intimacy
- Do NOT place zoom_cuts during b-roll inserts (those are separate clips).
- Spread zoom_cuts across the video — at least 1.5 s gap between any two.
- Target 4–7 zoom cuts for a typical Short. More is not better.
- zoom = 1.0 entries are valid and mean "ease back to normal" — use them
  after a punch-in if you want explicit control of the return timing.

Other rules:
- scene_description + visual_style MUST reference what you actually see in the images
- caption_color: look at the actual background and pick a contrasting colour
- b-roll prompts: highly specific to content topic, NOT generic stock descriptions
- energy_map: one entry every ~8 seconds
- max 3 b-roll inserts for a Short
- if video under 60s keep full duration in suggested_trim; if longer find the best 45-55s window
"""

    message = {"role": "user", "content": prompt}
    # images are attached if vision is available
    if frames_b64:
        message["images"] = frames_b64

    payload = {
        "model":  GEMMA_MODEL,
        "stream": False,
        "messages": [message],
        "options": {"temperature": 0.25, "num_ctx": MODEL_CONTEXT, "top_p": 0.9},
    }

    log.info(f"[gemma] Querying {GEMMA_MODEL} "
             f"({'multimodal: ' + str(len(frames_b64)) + ' frames' if has_vision else 'text-only'})...")

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=MODEL_TIMEOUT)
        resp.raise_for_status()

        raw  = resp.json()["message"]["content"].strip()
        data = _extract_json(raw)

        trim      = data.get("suggested_trim", {"start": 0, "end": duration})
        raw_zooms = data.get("zoom_cuts", [])
        zoom_cuts = _validate_zoom_cuts(raw_zooms, duration)

        # Parse emphasis word indices — validate they are ints in range
        raw_emphasis = data.get("emphasis_word_indices", [])
        n_words      = len(transcript_words)
        emphasis_word_indices = sorted(set(
            int(i) for i in raw_emphasis
            if str(i).lstrip("-").isdigit() and 0 <= int(i) < n_words
        ))

        # Parse emoji_map — keys are string word indices
        raw_emoji = data.get("emoji_map", {})
        emoji_map = {}
        for k, v in raw_emoji.items():
            try:
                wi = int(k)
                if 0 <= wi < n_words and isinstance(v, str) and len(v) <= 4:
                    emoji_map[wi] = v
            except (ValueError, TypeError):
                pass
        # Cap at 3 emojis
        if len(emoji_map) > 3:
            emoji_map = dict(list(emoji_map.items())[:3])

        log.info(f"[gemma] Style={data.get('style')} | Hook={data.get('hook')}")
        log.info(f"[gemma] Scene: {data.get('scene_description','')[:80]}")
        log.info(f"[gemma] Emphasis indices ({len(emphasis_word_indices)}): {emphasis_word_indices[:10]}")
        log.info(f"[gemma] Emoji map: {emoji_map}")
        log.info(f"[gemma] Cuts: {data.get('cut_points',[])} | B-roll: {len(data.get('broll_prompts',[]))}")
        log.info(f"[gemma] Zoom cuts ({len(zoom_cuts)}): {zoom_cuts}")

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
            zoom_cuts         = zoom_cuts,
            emphasis_word_indices = emphasis_word_indices,
            emoji_map         = emoji_map,
        )

    except requests.Timeout:
        log.warning(f"[gemma] Timeout after {MODEL_TIMEOUT}s (likely due to high GPU load or insufficient VRAM). "
                   f"Try: 1) Close other GPU apps, 2) Use gemma3:4b with text-only (no images), "
                   f"3) Reduce N_KEYFRAMES to 3 or 4")
        return EditDecision(
            style="energetic", hook="You need to see this!",
            caption_color="#FFFF00", caption_font="bold",
            cut_points=[], broll_prompts=[], energy_map=[],
            suggested_trim={"start": 0, "end": duration},
            scene_description="", visual_style="",
            zoom_cuts=[],
            emphasis_word_indices=[],
            emoji_map={},
        )
    except Exception as e:
        log.warning(f"[gemma] Failed: {e} — using safe defaults")
        return EditDecision(
            style="energetic", hook="You need to see this!",
            caption_color="#FFFF00", caption_font="bold",
            cut_points=[], broll_prompts=[], energy_map=[],
            suggested_trim={"start": 0, "end": duration},
            scene_description="", visual_style="",
            zoom_cuts=[],
            emphasis_word_indices=[],
            emoji_map={},
        )