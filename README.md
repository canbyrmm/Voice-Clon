# Academic Voice Cloning Application

This project presents an academic demonstration of **voice cloning from audio and video inputs**
using modern **Text-to-Speech (TTS)** and **Automatic Speech Recognition (ASR)** models.

## Features
- Audio-based voice cloning with custom text
- Video-based voice cloning using extracted speaker audio
- Automatic transcription using Whisper
- Optional audio-to-video merging
- Model selection:
  - XTTS v2 (zero-shot voice cloning)
  - Fine-tuned VITS (checkpoint-based)

## Models
- **XTTS v2** (Coqui TTS)
- **Whisper** (faster-whisper)

Fine-tuned models are expected to be provided externally
(e.g. Hugging Face Hub, Google Drive, S3).

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
