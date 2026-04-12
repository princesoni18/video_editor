"""
effects.py — Transitions and visual effects via FFmpeg
Whip pan, zoom punch, dynamic face reframe, colour grade, captions
"""

import subprocess, logging, math, os
from pathlib import Path

log = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  ASS SUBTITLE BUILDER (word-by-word highlight)
# ─────────────────────────────────────────

def _ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"





def build_ass(words: list[dict], c_color: str, c_font: str, c_size: str, c_border: str,
              video_w: int = 1080, video_h: int = 1920) -> str:
    """
    Build ASS subtitle file with word-by-word highlight.
    """
    FONT_MAP = {
        "Arial": "Arial",
        "Comic": "Comic Sans MS",
        "Montserrat": "Montserrat",
        "bold": "Arial",
        "rounded": "Nunito-Bold",
        "minimal": "Helvetica-Neue",
    }
    # Force bold font behavior intrinsically natively if not explicitly matched
    font = FONT_MAP.get(c_font, c_font)

    SIZE_MAP = {"Small": 35, "Medium": 40, "Large": 54}
    base_size = SIZE_MAP.get(c_size, 50)
    size = int(base_size * video_h / 1920)

    # Convert hex or use strictly mapped colours -> ASS form (&H00BBGGRR)
    COLOR_MAP = {
        "White": "&H00FFFFFF",
        "Yellow": "&H0000FFFF",
        "Green": "&H0000FF00",
        "Cyan": "&H00FFFF00",
    }
    if c_color.startswith("#"):
        hx = c_color.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        ass_highlight = f"&H00{b:02X}{g:02X}{r:02X}"
    else:
        ass_highlight = COLOR_MAP.get(c_color, "&H0000FFFF")

    BORDER_MAP = {
        "Black": "&H00000000",
        "White": "&H00FFFFFF",
        "Red": "&H000000FF",
        "Blue": "&H00FF0000",
    }
    ass_border = BORDER_MAP.get(c_border, "&H00000000")

    margin_v  = int(video_h * 0.22)   # 22% from bottom

    # Force Bold mathematically (Bold=-1)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,{ass_border},&H90000000,-1,0,0,0,100,100,1.5,0,1,3,0,2,40,40,{margin_v},1
Style: Hi,{font},{size},{ass_highlight},&H000000FF,{ass_border},&H90000000,-1,0,0,0,100,100,1.5,0,1,3,0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    WORDS_PER_LINE = 5

    i = 0
    while i < len(words):
        group = words[i: i + WORDS_PER_LINE]
        group_end = group[-1]["end"]

        for j, w in enumerate(group):
            seg_start = w["start"]
            seg_end   = group[j + 1]["start"] if j + 1 < len(group) else group_end

            parts = []
            for k, gw in enumerate(group):
                txt = gw["word"].replace("{", "").replace("}", "").strip()
                if k == j:
                    parts.append(f"{{\\rHi}}{txt}{{\\rDefault}}")
                else:
                    parts.append(txt)
            line = " ".join(parts)
            events.append(
                f"Dialogue: 0,{_ass_time(seg_start)},{_ass_time(seg_end)},Default,,0,0,0,,{line}"
            )
        i += WORDS_PER_LINE

    return header + "\n".join(events) + "\n"


# ─────────────────────────────────────────
#  WHIP PAN TRANSITION
# ─────────────────────────────────────────

def build_whip_pan_filter(duration: float, cut_points: list[float],
                           w: int = 1080, h: int = 1920) -> str:
    """
    Returns FFmpeg xfade-style filter string for whip pan effect.
    A whip pan = fast horizontal blur + translate at cut point.
    We implement as: motion blur (minterpolate) + xfade with slideleft.
    """
    if not cut_points:
        return "null"

    # xfade transitions between segments at each cut point
    # We'll apply this in the main compositor after splitting
    # Returns the xfade chain description (assembled in pipeline.py)
    transitions = []
    for cp in sorted(cut_points):
        transitions.append({
            "time":       cp,
            "type":       "slideleft",   # whip pan feel
            "duration":   0.12,          # fast — 120ms
        })
    return transitions


# ─────────────────────────────────────────
#  ZOOM PUNCH
# ─────────────────────────────────────────

def zoom_punch_filter(energy_map: list[dict], fps: float = 30) -> str:
    """
    Build zoompan expression that punches in at high-energy moments.
    Returns FFmpeg zoompan z expression string.
    """
    if not energy_map:
        # Default: gentle zoom in from 1.0 to 1.03 over full video
        return "zoompan=z='1.0+0.03*on/duration':d=1:s=1080x1920:fps=30"

    # Build piecewise zoom expression based on energy
    # At "peak" moments → zoom to 1.08, at "low" → back to 1.0
    # FFmpeg zoompan z= expression uses 'on' (output frame number)
    segments = []
    for i, e in enumerate(energy_map):
        zoom = {"peak": 1.08, "high": 1.04, "low": 1.0}.get(e["energy"], 1.0)
        t_start = e["time"]
        t_end   = energy_map[i + 1]["time"] if i + 1 < len(energy_map) else 99999
        f_start = int(t_start * fps)
        f_end   = int(t_end * fps)
        segments.append(f"if(between(on,{f_start},{f_end}),{zoom},")

    # Close all ifs + final fallback
    expr = "".join(segments) + "1.0" + ")" * len(segments)
    return f"zoompan=z='{expr}':d=1:s=1080x1920:fps=30:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"


# ─────────────────────────────────────────
#  COLOUR GRADE (style-aware)
# ─────────────────────────────────────────

GRADE_MAP = {
    "energetic":   "eq=contrast=1.08:brightness=0.03:saturation=1.3,colorchannelmixer=rr=1.06:gg=1.0:bb=0.94",
    "calm":        "eq=contrast=1.02:brightness=0.01:saturation=0.9,colorchannelmixer=rr=0.97:gg=1.0:bb=1.05",
    "educational": "eq=contrast=1.04:brightness=0.02:saturation=1.1",
    "funny":       "eq=contrast=1.1:brightness=0.04:saturation=1.4,colorchannelmixer=rr=1.08:gg=1.02:bb=0.95",
}

def color_grade_filter(style: str) -> str:
    return GRADE_MAP.get(style, GRADE_MAP["energetic"])


# ─────────────────────────────────────────
#  HOOK TEXT OVERLAY
# ─────────────────────────────────────────

def hook_drawtext_filter(hook: str, video_h: int = 1920) -> str:
    """
    Animated hook text that appears at t=0, scales up, then fades out at t=2.5s.
    Uses FFmpeg drawtext with enable expression.
    """
    if not hook:
        return "null"

    safe_hook = hook.replace("'", "\\'").replace(":", "\\:")
    size = int(52 * video_h / 1920)

    return (
        f"drawtext=text='{safe_hook}'"
        f":fontfile='C\:/Windows/Fonts/arialbd.ttf'"
        f":fontsize={size}"
        f":fontcolor=white"
        f":borderw=4:bordercolor=black"
        f":x=(w-text_w)/2"
        f":y=h*0.15"
        f":enable='between(t,0,2.5)'"
        f":alpha='if(lt(t,0.3),t/0.3,if(gt(t,2.0),(2.5-t)/0.5,1))'"
    )
