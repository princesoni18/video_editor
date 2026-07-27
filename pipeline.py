"""
pipeline.py — Main orchestrator
Ties Whisper + Gemma 4 + Face Tracker + Image Gen + FFmpeg together
"""

import os, subprocess, tempfile, logging, json
from pathlib import Path

# Fix: Dynamically add Winget's FFmpeg to PATH so we don't need a system reboot
_ffmpeg_bin = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin")
if os.path.exists(_ffmpeg_bin) and _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

import torch
import whisper as whisper_lib
from gemma_brain  import analyze, EditDecision
from face_tracker import get_face_crop_path, export_crop_filter
from image_gen    import generate_all_broll
from effects      import build_ass, build_zoom_filter, color_grade_filter, hook_drawtext_filter

log = logging.getLogger(__name__)

# Device checking
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _run(cmd: list[str], label: str, cwd: str = None):
    # Ensure all command elements are strings
    cmd = [str(c) for c in cmd]
    log.info(f"[ffmpeg:{label}] " + " ".join(cmd[:6]) + " ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        log.error(f"[ffmpeg:{label}] FAILED:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"FFmpeg {label} failed")
    log.info(f"[ffmpeg:{label}] Done.")


def _probe(video_path: str) -> tuple[int, int, float, float]:
    """Returns (width, height, duration, fps) — uses video stream duration to avoid audio/video mismatch"""
    cmd = ["ffprobe","-v","quiet","-print_format","json","-show_streams","-show_format", video_path]
    data = json.loads(subprocess.check_output(cmd))
    w, h, dur = 1920, 1080, 0.0
    fps = 30.0
    
    # Get video stream duration (most reliable for videos with mismatched audio/video)
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            w   = int(s.get("width", 1920))
            h   = int(s.get("height", 1080))
            
            # FPS detection
            avg_frame_rate = s.get("avg_frame_rate", "0/1")
            r_frame_rate = s.get("r_frame_rate", "0/1")
            try:
                num, den = map(int, avg_frame_rate.split("/"))
                if den != 0:
                    fps = num / den
            except:
                pass
                
            is_vfr = avg_frame_rate != r_frame_rate
            log.info(f"[probe] FPS: {fps:.2f}, VFR Detected: {is_vfr}")

            # Prefer video stream duration, fallback to format duration
            if "duration" in s:
                dur = float(s["duration"])
            else:
                dur = float(data["format"].get("duration", 0))
            log.info(f"[probe] Video stream duration: {dur:.2f}s (h={h}x{w})")
            break
    
    # Check for audio/video duration mismatch
    audio_dur = 0.0
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio" and "duration" in s:
            audio_dur = float(s["duration"])
            break
    
    if audio_dur > 0 and abs(audio_dur - dur) > 0.5:
        log.warning(f"[probe] Audio/video duration mismatch! Video={dur:.2f}s, Audio={audio_dur:.2f}s")
        log.warning(f"[probe] Will use video duration {dur:.2f}s and trim audio to match")
    
    return w, h, dur, fps


def run_pipeline(input_path: str, output_path: str, user_prefs: dict = None) -> dict:
    if user_prefs is None: user_prefs = {}
    work = Path(tempfile.mkdtemp(prefix="shorts_"))
    log.info(f"[pipeline] Work dir: {work}")

    src_w, src_h, duration, fps = _probe(input_path)
    already_vertical = src_h > src_w
    log.info(f"[pipeline] Source: {src_w}x{src_h}, {duration:.1f}s | vertical={already_vertical}")

    # ── 0. Normalize Telegram/VFR Video ──────────────────────────────────────
    log.info("[pipeline] Step 0: Normalize VFR ")
    normalized_path = str(work / "normalized.mp4")
    _run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "fps=30",
        "-vsync", "cfr",
        "-c:a", "copy",
        normalized_path
    ], "normalize")
    input_path = normalized_path
    # update fps
    fps = 30.0

    # ── 1. Whisper transcription (Hinglish mode) ─────────────────────────────
    log.info("[pipeline] Step 1: Whisper transcription")
    log.info(f"[pipeline] Loading Whisper medium on {_device}...")
    _whisper = whisper_lib.load_model("medium", device=_device)
    
    log.info("[pipeline] Step 1: Detecting language...")
    audio = whisper_lib.load_audio(input_path)
    audio_clip = whisper_lib.pad_or_trim(audio)
    mel = whisper_lib.log_mel_spectrogram(audio_clip).to(_device)
    _, probs = _whisper.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    log.info(f"[pipeline] Detected language: {detected_lang} ({probs[detected_lang]:.2f})")

    log.info("[pipeline] Step 1: Transcribing (Hinglish mode)...")
    result = _whisper.transcribe(
        input_path,
        task="transcribe",
        language=detected_lang,
        initial_prompt=(
            "Yaar dekho, aisa hota h. Architecture mein kya difference h, "
            "samjhte ho? Bilkul sahi kaha. Toh basically kya h ki, "
            "ek dum clear h yeh concept. Matlab simply bolo toh."
        ),
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=True,
    )
    
    # 💥 FREE VRAM so Gemma has enough space!
    del _whisper
    if _device == "cuda":
        torch.cuda.empty_cache()
    import gc; gc.collect()

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
            
    transcript_preview = " ".join(w["word"] for w in words[:20])
    log.info(f"[pipeline] Transcribed {len(words)} words")
    log.info(f"[pipeline] Transcript preview: {transcript_preview}")

    # ── 2. Gemma 4 analysis ──────────────────────────────────────────────────
    log.info("[pipeline] Step 2: Gemma 4 analysis")
    decision: EditDecision = analyze(words, duration, video_path=input_path)

    # ── 3. Face tracking ─────────────────────────────────────────────────────
    log.info("[pipeline] Step 3: Face tracking")
    crop_path, fps, src_w, src_h = get_face_crop_path(input_path, 1080, 1920)
    # Write crop script to work dir for shorter path (avoids Windows command line length limits)
    crop_script = export_crop_filter(crop_path, fps, src_w, src_h, script_path=str(work / "crop_cmds.txt"))
    
    # ── 3.5. Audio/Video sync (fix Telegram trim issues) ──────────────────────
    # If audio is longer than video (e.g., Telegram trimmed video but not audio),
    # trim the audio to match video duration
    synced_video = str(work / "synced.mp4")
    log.info(f"[pipeline] Syncing audio to video duration ({duration:.2f}s)")
    _run([
        "ffmpeg", "-y",
        "-i", input_path,  # Use original input, not crop_path (which is keyframe data)
        "-t", str(duration),  # Convert to string explicitly
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        synced_video
    ], "audio_sync")
    input_path = synced_video  # Use synced video from now on

    # ── 4. B-roll image generation ───────────────────────────────────────────
    log.info("[pipeline] Step 4: B-roll generation")
    broll_items = generate_all_broll(decision.broll_prompts, str(work))

    # ── 5. Build ASS captions ────────────────────────────────────────────────
    log.info("[pipeline] Step 5: Building captions")
    c_color     = user_prefs.get("color",     decision.caption_color)
    c_font      = user_prefs.get("font",      "Montserrat")
    c_size      = user_prefs.get("size",      "Large")
    c_border    = user_prefs.get("border",    "Black")
    c_animation = user_prefs.get("animation", "Pop")
    c_style     = user_prefs.get("style",     "Submagic")

    # Gemma-supplied emphasis indices and emoji map
    emphasis_indices = set(getattr(decision, "emphasis_word_indices", []))
    emoji_map        = getattr(decision, "emoji_map", {})

    caption_position = getattr(decision, "caption_position", "bottom")
    ass_content = build_ass(
        words, c_color, c_font, c_size, c_border,
        emphasis_indices = emphasis_indices,
        emoji_map        = emoji_map,
        c_animation      = c_animation,
        c_style          = c_style,
       # caption_position = caption_position,
        video_w=1080, video_h=1920,
    )
    ass_path = str(work / "captions.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    # ── 6. Trim to suggested range ───────────────────────────────────────────
    trim_start = decision.suggested_trim.get("start", 0)
    trim_end   = decision.suggested_trim.get("end", duration)
    trimmed    = str(work / "trimmed.mp4")
    _run([
        "ffmpeg", "-y",
        "-ss", str(trim_start), "-to", str(trim_end),
        "-i", input_path,
        "-c", "copy", trimmed
    ], "trim")

    # ── 7. Face-tracked dynamic crop + resize to 9:16 ───────────────────────
    cropped = str(work / "cropped.mp4")
    log.info("[pipeline] Step 7: Dynamic face-tracking crop + resize")

    # FFmpeg's filtergraph parser processes escape sequences BEFORE passing the
    # value to sendcmd, so there is no escaping that survives a Windows drive
    # letter ("D\:/path" → FFmpeg sees option-name "D", value empty).
    # The only robust fix: use a plain filename (no path) and run FFmpeg with
    # cwd=work so it resolves the script without any drive-letter colon.
    crop_script_name = Path(crop_script).name   # e.g. "crop_cmds.txt"
    vf_crop = f"sendcmd=f={crop_script_name},crop=iw:ih:0:0,scale=1080:1920:flags=bicubic"

    ffmpeg_crop_cmd = ["ffmpeg", "-y"]
    if _device == "cuda":
        ffmpeg_crop_cmd.extend(["-hwaccel", "cuda"])
    ffmpeg_crop_cmd.extend(["-i", trimmed, "-vf", vf_crop])

    if _device == "cuda":
        ffmpeg_crop_cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"])
    else:
        ffmpeg_crop_cmd.extend(["-c:v", "libx264", "-crf", "22"])
    ffmpeg_crop_cmd.extend(["-c:a", "copy", cropped])

    _run(ffmpeg_crop_cmd, "dynamic_crop", cwd=str(work))

    # ── 8. Apply effects + captions (main pass) ──────────────────────────────
    log.info("[pipeline] Step 8: Effects + captions (zoom + captions)")
    effected = str(work / "effected.mp4")

    ass_esc  = ass_path.replace("\\","/").replace(":",r"\:")
    captions = f"ass='{ass_esc}'"

    # Build Gemma-directed zoom filter.
    # Shift zoom_cuts timestamps so they are relative to the trimmed clip.
    # trimmed_zoom_cuts = []
    # for zc in decision.zoom_cuts:
    #     t_shifted = zc["time"] - trim_start
    #     if 0 <= t_shifted <= (trim_end - trim_start):
    #         trimmed_zoom_cuts.append({**zc, "time": t_shifted})

    # trimmed_duration = trim_end - trim_start
    # zoom_filter = build_zoom_filter(trimmed_zoom_cuts, trimmed_duration, fps=fps)
    # log.info(f"[pipeline] Zoom filter built for {len(trimmed_zoom_cuts)} cuts")

    # zoompan must come BEFORE ass subtitles so the zoom doesn't crop the text.
    # zoompan is CPU-only — hwaccel decode is still fine.
    # vf_chain = f"{zoom_filter},{captions}" 
    vf_chain = f"{captions}"

    # Audio: normalize loudness + boost speech clarity
    af_chain = "loudnorm=I=-14:TP=-2:LRA=7,equalizer=f=3000:width_type=o:width=2:g=3"

    ffmpeg_eff_cmd = ["ffmpeg", "-y"]
    if _device == "cuda":
        ffmpeg_eff_cmd.extend(["-hwaccel", "cuda"])
    ffmpeg_eff_cmd.extend([
        "-i", cropped,
        "-vf", vf_chain,
        "-af", af_chain
    ])
    if _device == "cuda":
        ffmpeg_eff_cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"])
    else:
        ffmpeg_eff_cmd.extend(["-c:v", "libx264", "-crf", "22"])
    ffmpeg_eff_cmd.extend([
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        effected
    ])
    _run(ffmpeg_eff_cmd, "effects")

    # ── 9. Insert b-roll cutaways ────────────────────────────────────────────
    current = effected
    if broll_items:
        log.info(f"[pipeline] Step 9: Inserting {len(broll_items)} b-roll cutaways")
        for i, broll in enumerate(broll_items):
            broll_vid   = str(work / f"broll_vid_{i}.mp4")
            merged      = str(work / f"merged_{i}.mp4")
            img_path    = broll["image_path"]
            b_dur       = broll["duration"]
            b_time      = broll["time"] - trim_start   # adjust for trim

            if b_time < 0 or b_time > (trim_end - trim_start):
                continue

            # Convert still image → short video clip with zoom animation
            ffmpeg_broll_cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-vf", f"scale=1080:1920,zoompan=z='1.0+0.04*on/({b_dur}*30)':d=1:s=1080x1920:fps=30",
                "-t", str(b_dur)
            ]
            if _device == "cuda":
                ffmpeg_broll_cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"])
            else:
                ffmpeg_broll_cmd.extend(["-c:v", "libx264", "-crf", "22"])
            ffmpeg_broll_cmd.extend(["-an", broll_vid])
            _run(ffmpeg_broll_cmd, f"broll_vid_{i}")

            # Split main at b_time, insert b-roll, rejoin with whip-pan xfade
            before  = str(work / f"before_{i}.mp4")
            after   = str(work / f"after_{i}.mp4")
            _run(["ffmpeg","-y","-i",current,"-t",str(b_time),"-c","copy",before], f"split_before_{i}")
            _run(["ffmpeg","-y","-i",current,"-ss",str(b_time + b_dur),"-c","copy",after], f"split_after_{i}")

            # Concat: before + whip-pan → broll + whip-pan → after
            concat_list = str(work / f"concat_{i}.txt")
            with open(concat_list, "w") as f:
                f.write(f"file '{before}'\nfile '{broll_vid}'\nfile '{after}'\n")

            _run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                merged
            ], f"concat_{i}")

            current = merged

    # ── 10. Final output ─────────────────────────────────────────────────────
    log.info("[pipeline] Step 10: Final export")
    _run([
        "ffmpeg", "-y",
        "-i", current,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path
    ], "final")

    log.info(f"[pipeline] Done! Output: {output_path}")

    # Cleanup work dir
    import shutil
    try:
        shutil.rmtree(work)
    except Exception:
        pass

    return {
        "style":             decision.style,
        "hook":              decision.hook,
        "n_cuts":            len(decision.cut_points),
        "n_broll":           len(broll_items),
        "scene_description": decision.scene_description,
        "visual_style":      decision.visual_style,
    }