"""
effects.py — Transitions and visual effects via FFmpeg
Matches reference YT Shorts style:
  - Yellow rounded-rect pill background behind caption text
  - All words in a group: same size, same weight (NO mixed sizing within a group)
  - Emphasis words: shown ALONE as their own big caption (single word, larger pill)
  - Emojis: Gemma-assigned, appended to relevant caption segment (1–2 max)
  - Clean cut transitions (no scale/fade animation that causes jitter)
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _safe(text: str) -> str:
    """Strip ASS-breaking characters."""
    return text.replace("{", "").replace("}", "").replace("\\", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  MAPS
# ─────────────────────────────────────────────────────────────────────────────

FONT_MAP = {
    "Montserrat": "Montserrat",
    "Arial":      "Arial",
    "Comic":      "Comic Sans MS",
    "bold":       "Arial Black",
    "rounded":    "Nunito-Bold",
    "minimal":    "Helvetica Neue",
}

SIZE_MAP = {"Small": 44, "Medium": 54, "Large": 64}

# ASS colours are &HAABBGGRR  (AA=alpha, BB=blue, GG=green, RR=red)
TEXT_COLOR = "&H00111111"   # near-black text inside pill

PILL_COLOR_MAP = {
    "Yellow": "&H0000FFFF",   # bright yellow
    "White":  "&H00FFFFFF",
    "Black":  "&H00000000",
    "Green":  "&H0050C820",
    "Blue":   "&H00FF5000",
    "Pink":   "&H00C060FF",
}


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION
#  Normal group : 2–3 words shown together in one pill, same size
#  Emphasis group: 1 word shown alone in a LARGER pill (Gemma-flagged index)
# ─────────────────────────────────────────────────────────────────────────────

def _segment_words(words: list[dict], emphasis_indices: set) -> list[dict]:
    """
    Returns list of segments:
    {
        "type":  "normal" | "emphasis",
        "words": [{word,start,end}, ...],
        "start": float,
        "end":   float,
        "emoji": None   # filled downstream
    }
    """
    WORDS_PER_GROUP = 3
    segments = []
    buf = []

    def flush_buf():
        if buf:
            segments.append({
                "type":  "normal",
                "words": buf[:],
                "start": buf[0]["start"],
                "end":   buf[-1]["end"],
                "emoji": None,
            })
            buf.clear()

    for i, w in enumerate(words):
        if i in emphasis_indices:
            flush_buf()
            segments.append({
                "type":  "emphasis",
                "words": [w],
                "start": w["start"],
                "end":   w["end"],
                "emoji": None,
            })
        else:
            buf.append(w)
            if len(buf) >= WORDS_PER_GROUP:
                flush_buf()

    flush_buf()
    return segments


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ASS BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_ass(words:            list[dict],
              c_color:          str,
              c_font:           str,
              c_size:           str,
              c_border:         str,
              emphasis_indices: set  = None,
              emoji_map:        dict = None,
              c_animation:      str  = "Pop",    # kept for API compat, ignored
              c_style:          str  = "Submagic",
              video_w:          int  = 1080,
              video_h:          int  = 1920) -> str:
    """
    Build ASS subtitle file in reference YT Shorts style.

    emphasis_indices : set[int] — word list indices Gemma flagged as emphasis.
                       These words are rendered ALONE in a larger pill.
    emoji_map        : dict[int, str] — word index → single emoji string.
                       The emoji is appended to whichever segment that word lands in.
                       Max 2 emojis across the whole video (Gemma controls this).
    """
    if emphasis_indices is None:
        emphasis_indices = set()
    if emoji_map is None:
        emoji_map = {}

    font      = FONT_MAP.get(c_font, c_font)
    base_size = SIZE_MAP.get(c_size, 58)
    size_n    = int(base_size * video_h / 1920)           # normal pill size
    size_e    = int(base_size * 1.32 * video_h / 1920)    # emphasis pill: 32% bigger

    pill      = PILL_COLOR_MAP.get(c_color, PILL_COLOR_MAP["Yellow"])
    margin_v  = int(video_h * 0.33) - 85

    # BorderStyle 4 = opaque background box (closest to a pill in ASS)
    # Outline value used as internal padding, Shadow=2 adds soft edge for rounded appearance
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Normal,{font},{size_n},{TEXT_COLOR},&H000000FF,{pill},{pill},-1,0,0,0,100,100,1,0,4,14,2,2,50,50,{margin_v},1
Style: Emph,{font},{size_e},{TEXT_COLOR},&H000000FF,{pill},{pill},-1,0,0,0,100,100,1,0,4,18,2,2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    segments = _segment_words(words, emphasis_indices)

    # Build reverse lookup: word index → segment index
    # We need to match words by their position in the original word list
    word_index_to_seg: dict[int, int] = {}
    wi_global = 0
    for seg_idx, seg in enumerate(segments):
        for _ in seg["words"]:
            word_index_to_seg[wi_global] = seg_idx
            wi_global += 1

    # Attach emojis to segments
    for wi, emoji in emoji_map.items():
        seg_idx = word_index_to_seg.get(wi)
        if seg_idx is not None and segments[seg_idx]["emoji"] is None:
            segments[seg_idx]["emoji"] = emoji

    events = []
    for seg in segments:
        txt   = " ".join(_safe(w["word"]) for w in seg["words"])
        style = "Emph" if seg["type"] == "emphasis" else "Normal"
        if seg["emoji"]:
            txt = f"{txt} {seg['emoji']}"
        events.append(
            f"Dialogue: 0,{_ass_time(seg['start'])},{_ass_time(seg['end'])},"
            f"{style},,0,0,0,,{txt}"
        )

    return header + "\n".join(events) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
#  WHIP PAN
# ─────────────────────────────────────────────────────────────────────────────

def build_whip_pan_filter(duration: float, cut_points: list[float],
                          w: int = 1080, h: int = 1920) -> list[dict]:
    if not cut_points:
        return []
    return [{"time": cp, "type": "slideleft", "duration": 0.12}
            for cp in sorted(cut_points)]


# ─────────────────────────────────────────────────────────────────────────────
#  ZOOM FILTER
# ─────────────────────────────────────────────────────────────────────────────

def build_zoom_filter(zoom_cuts: list[dict], duration: float, fps: float = 30) -> str:
    """
    Build a zoompan z= expression for FFmpeg.
    Each cut: ease-in → hold → ease-out, nested as properly-parenthesised ternaries.
    Bracket parity is guaranteed: outer if(RANGE, inner_if_chain, fallback).
    """
    if not zoom_cuts:
        return (
            "zoompan=z='1.0':d=1:s=1080x1920:fps=30"
            ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        )

    EASE_IN  = 6    # frames ~0.2 s
    EASE_OUT = 18   # frames ~0.6 s

    segs = []
    for cut in zoom_cuts:
        z    = cut["zoom"]
        f0   = int(cut["time"] * fps)
        f1   = f0 + EASE_IN                                         # peak start
        f2   = f1 + max(0, int(cut["duration"] * fps) - EASE_IN)   # peak end
        f3   = f2 + EASE_OUT                                         # ramp-down end
        segs.append((f0, f1, f2, f3, z))

    # Build outermost-first: last segment wraps earlier ones as its fallback.
    # Structure per segment (brackets balance to exactly +4 net opens):
    #   if(between(on,f0,f3),          ← outer guard  [1 open
    #     if(between(on,f0,f1-1),...,  ← ease-in      [1 open → 1 close at end of chain
    #     if(between(on,f1,f2),...,    ← hold          [1 open
    #     if(between(on,f2+1,f3),...,  ← ease-out      [1 open
    #     1.0))),                      ← 3 closes for inner ifs
    #   FALLBACK)                      ← 1 close for outer = 4 total
    expr = "1.0"
    for (f0, f1, f2, f3, z) in reversed(segs):
        dz    = z - 1.0
        inner = (
            f"if(between(on,{f0},{f1-1}),"
            f"1.0+{dz:.4f}*(on-{f0})/{EASE_IN},"
            f"if(between(on,{f1},{f2}),"
            f"{z:.4f},"
            f"if(between(on,{f2+1},{f3}),"
            f"{z:.4f}-{dz:.4f}*(on-{f2})/{EASE_OUT},"
            f"1.0)))"
        )
        expr = f"if(between(on,{f0},{f3}),{inner},{expr})"

    return (
        f"zoompan=z='{expr}':d=1:s=1080x1920:fps=30"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR GRADE
# ─────────────────────────────────────────────────────────────────────────────

GRADE_MAP = {
    "energetic":   "eq=contrast=1.08:brightness=0.03:saturation=1.3,colorchannelmixer=rr=1.06:gg=1.0:bb=0.94",
    "calm":        "eq=contrast=1.02:brightness=0.01:saturation=0.9,colorchannelmixer=rr=0.97:gg=1.0:bb=1.05",
    "educational": "eq=contrast=1.04:brightness=0.02:saturation=1.1",
    "funny":       "eq=contrast=1.1:brightness=0.04:saturation=1.4,colorchannelmixer=rr=1.08:gg=1.02:bb=0.95",
}

def color_grade_filter(style: str) -> str:
    return GRADE_MAP.get(style, GRADE_MAP["energetic"])


# ─────────────────────────────────────────────────────────────────────────────
#  HOOK TEXT OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def hook_drawtext_filter(hook: str, video_h: int = 1920) -> str:
    if not hook:
        return "null"
    safe_hook = hook.replace("'", "\\'").replace(":", "\\:")
    size = int(52 * video_h / 1920)
    return (
        f"drawtext=text='{safe_hook}'"
        f":fontfile='C\\:/Windows/Fonts/arialbd.ttf'"
        f":fontsize={size}:fontcolor=white"
        f":borderw=4:bordercolor=black"
        f":x=(w-text_w)/2:y=h*0.15"
        f":enable='between(t,0,2.5)'"
        f":alpha='if(lt(t,0.3),t/0.3,if(gt(t,2.0),(2.5-t)/0.5,1))'"
    )