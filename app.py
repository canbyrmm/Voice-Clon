import os
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from TTS.api import TTS
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fine-tuned model (HF / S3 / Drive)
FT_MODEL_PATH = "checkpoints/vits_ft/best_model.pth"
FT_CONFIG_PATH = "checkpoints/vits_ft/config.json"

# =========================
# CACHED MODELS
# =========================
@st.cache_resource
def load_xtts():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2")

@st.cache_resource
def load_finetuned_vits():
    return TTS(model_path=FT_MODEL_PATH, config_path=FT_CONFIG_PATH)

@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")

# =========================
# UTILS
# =========================
def save_upload(uploaded_file, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.read())
    return path

def finetuned_ready():
    return os.path.exists(FT_MODEL_PATH) and os.path.exists(FT_CONFIG_PATH)

def extract_audio(video_path, wav_out):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "24000",
        wav_out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def replace_audio(video_path, wav_path, out_video):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        out_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================
# UI
# =========================
st.set_page_config(page_title="Academic Voice Cloning App", layout="wide")
st.title("🎙️ Academic Voice Cloning Application")

st.markdown("""
This application demonstrates **voice cloning from audio and video inputs**  
using **modern TTS and ASR models** in an academic setting.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input & Model Selection")

    input_mode = st.radio("Input Type", ["Audio", "Video"])
    model_choice = st.selectbox(
        "TTS Model",
        ["XTTS v2 (Zero-Shot)", "Fine-Tuned VITS"]
    )

    if model_choice == "Fine-Tuned VITS" and not finetuned_ready():
        st.warning("Fine-tuned model not found. XTTS is recommended.")

with col2:
    st.header("Processing & Output")

    # Load TTS safely
    tts = None
    if model_choice == "XTTS v2 (Zero-Shot)":
        tts = load_xtts()
    elif finetuned_ready():
        tts = load_finetuned_vits()

    if input_mode == "Audio":
        ref_audio = st.file_uploader("Reference Audio (.wav)", type=["wav"])
        text = st.text_area("Text to be synthesized", height=120)

        if st.button("Generate Speech") and ref_audio and text and tts:
            ref_path = save_upload(ref_audio, ".wav")
            out_wav = OUTPUT_DIR / "audio_output.wav"

            if model_choice.startswith("XTTS"):
                tts.tts_to_file(
                    text=text,
                    speaker_wav=ref_path,
                    language="tr",
                    file_path=str(out_wav)
                )
            else:
                tts.tts_to_file(text=text, file_path=str(out_wav))

            st.audio(str(out_wav))
            st.download_button("Download WAV", open(out_wav, "rb"), "output.wav")

    else:
        video = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])
        mode = st.radio("Video Mode", ["Read Transcript", "Read Custom Text"])
        merge_video = st.checkbox("Merge synthesized audio back to video", value=True)

        if video and tts:
            video_path = save_upload(video, ".mp4")
            ref_wav = tempfile.mktemp(suffix=".wav")
            extract_audio(video_path, ref_wav)

            if mode == "Read Transcript":
                whisper = load_whisper()
                segments, _ = whisper.transcribe(ref_wav, language="tr")
                text = " ".join([s.text for s in segments])
                st.text_area("Transcript", value=text, height=150)
            else:
                text = st.text_area("Custom Text", height=120)

            if st.button("Generate from Video"):
                out_wav = OUTPUT_DIR / "video_audio.wav"
                out_video = OUTPUT_DIR / "video_output.mp4"

                tts.tts_to_file(
                    text=text,
                    speaker_wav=ref_wav,
                    language="tr",
                    file_path=str(out_wav)
                )

                st.audio(str(out_wav))

                if merge_video:
                    replace_audio(video_path, str(out_wav), str(out_video))
                    st.video(str(out_video))
                    st.download_button("Download Video", open(out_video, "rb"), "output.mp4")
