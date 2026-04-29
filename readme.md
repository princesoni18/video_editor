# 🎬 Smart AI Shorts Bot (Automated Video Editing Pipeline)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.1-red)
![FFmpeg](https://img.shields.io/badge/FFmpeg-NVENC-green)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_3-white)

An end-to-end, fully automated video editing pipeline that transforms standard videos into engaging, vertical, AI-edited Shorts/Reels/TikToks. Simply send a video to the Telegram bot, and the local AI models will collaborate to transcribe, script, track, generate b-roll, and render a highly produced short video.

---

## ⚡ What is the Pipeline Capable Of?

The pipeline acts as a fully autonomous video editor:

1. **🎙️ Smart Transcription (Whisper):** Transcribes audio perfectly, including complex mixed languages like **Hinglish**, while retaining Roman script outputs.
2. **🧠 AI Director (Gemma 3 via Ollama):** Analyzes the transcription to make creative decisions. It decides the video's mood, generates punchy hook text, suggests precise cut points, determines b-roll prompts, and maps out energy for zoom-ins.
3. **🎯 Active Face Tracking (MediaPipe):** Converts horizontal (16:9) video into vertical (9:16) format by actively tracking the speaker's face and keeping them perfectly centered on the screen.
4. **🎨 Dynamic B-Roll Generation (Stable Diffusion):** Automatically generates relevant contextual B-Roll images precisely when the AI Director requests them to fill gaps or enhance storytelling.
5. **🎬 Pro-Level VFX & Rendering (FFmpeg + CUDA):** Applies hardware-accelerated color grading, precise zoom-punches based on speech energy, burning active captions (with vibe-matching highlight colors), and snappy transitions.

---

## 📂 Project Structure

```text
shorts_bot/
├── bot.py           # Telegram bot UI & entry point
├── pipeline.py      # Main Orchestrator (ties all models & FFmpeg together)
├── gemma_brain.py   # AI Brain: Gemma 3 via Ollama for editing decisions
├── face_tracker.py  # MediaPipe based active speaker tracking
├── image_gen.py     # Stable Diffusion for B-roll generation
├── effects.py       # FFmpeg filters, transitions, caption assignments
├── whisper_srt.py   # Audio extraction & speech-to-text
└── requirements.txt # Dependencies & complete setup guide
```

---

## 💻 Hardware Requirements

Because the pipeline relies entirely on local AI processing, a dedicated NVIDIA GPU is highly recommended. 

| Component | VRAM Needed |
| :--- | :--- |
| **Whisper (Medium)** | ~3 GB |
| **Gemma 3 (12B)** | ~9 GB (via Ollama) |
| **Stable Diffusion** | ~4 GB (Loads/unloads as needed) |
| **FFmpeg NVENC** | ~1 GB |
| **Total Peak Minimum** | **~8 GB to 12 GB+ Recommended** |

*💡 **Tip:** If you have an 8GB or smaller card (e.g., GTX 1650/RTX 3050), change the constants in `requirements.txt` / `pipeline.py` to use `gemma3:4b` and Whisper `small`.*

---

## 🛠️ Installation & Setup

**Step 1: Install FFmpeg**
Install FFmpeg and ensure it is added to your system PATH.
```bash
winget install Gyan.FFmpeg
```

**Step 2: Install Python Dependencies**
Be sure to install the CUDA-enabled version of PyTorch!
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Step 3: Setup Ollama & Gemma**
Download and install [Ollama](https://ollama.com/download/windows). Then pull the Gemma model matching your hardware:
```bash
# High-end GPU
ollama pull gemma3:27b

# Mid-tier GPU (RTX 3060/3070/3080)
ollama pull gemma3:12b

# Budget GPU / Low VRAM
ollama pull gemma3:4b 
```

**Step 4: Configure Telegram Bot**
1. Message `@BotFather` on Telegram to create a new bot.
2. Copy your unique API Token.
3. Expose it in your environment:
```bash
set TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
```

---

## 🚀 How to Run

1. Start the main bot instance:
```bash
python bot.py
```
2. Open Telegram and navigate to your bot.
3. Send it a video clip (up to 50MB via Telegram).
4. **Sit back.** The pipeline will detect CUDA devices, transcribe the context, track the faces, generate the b-roll, edit the sequence based on Gemma's creative inputs, and send back a high-retention, fully edited Short.

---

## 🧠 What Gemma (The AI Brain) Decides:
The `gemma_brain.py` dynamically computes an `EditDecision` model that sets up the render pipeline context:
- **Style:** Energetic / Calm / Educational / Funny
- **Hook Script:** A hard-hitting text hook to show in the first 3 seconds
- **Color Grading & Captions:** Sets the text highlight color that perfectly fits the vibe.
- **B-Roll Timing:** Exactly when to show AI-generated images.
- **Energy Map:** Injects dramatic Zoom punches based on sentence impact and speech peaks.