import os
import argparse
import logging
import torch
import whisper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        log.info(f"Loading Whisper 'medium' model on {_device}...")
        _whisper = whisper.load_model("medium", device=_device)
    except torch.cuda.OutOfMemoryError:
        log.warning(f"CUDA OutOfMemoryError: Not enough VRAM available for 'medium' model on {_device}.")
        log.info("Falling back to CPU...")
        _device = "cpu"
        _whisper = whisper.load_model("medium", device=_device)
    
    log.info("Detecting language...")
    audio = whisper.load_audio(input_path)
    audio_clip = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_clip).to(_device)
    _, probs = _whisper.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    log.info(f"Detected language: {detected_lang} ({probs[detected_lang]:.2f})")

    log.info(f"Transcribing (Hinglish mode) in {detected_lang}...")
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
    
    # Generate SRT content
    log.info("Generating SRT file...")
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

    log.info(f"Successfully saved transcript to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRT captions using Whisper")
    parser.add_argument("input", help="Path to the input video/audio file")
    parser.add_argument("-o", "--output", help="Optional output SRT file path")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    
    if not output_file:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}.srt"

    generate_srt(input_file, output_file)
