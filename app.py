import os
import uuid
import gradio as gr
from typing import Optional, Tuple

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Lazy imports (heavy libs)
# ---------------------------
_whisper_model = None
_tts_model = None

def get_whisper(model_size: str = "small"):
    global _whisper_model
    if _whisper_model is None or getattr(_whisper_model, "model_size", None) != model_size:
        from faster_whisper import WhisperModel
        # device auto; Spaces GPU varsa "cuda" olur
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _whisper_model.model_size = model_size
    return _whisper_model

def get_tts(model_name: str):
    """
    model_name examples:
      - "tts_models/multilingual/multi-dataset/xtts_v2"
      - "tts_models/tr/common-voice/glow-tts"
    """
    global _tts_model
    if _tts_model is None or getattr(_tts_model, "_loaded_name", None) != model_name:
        from TTS.api import TTS
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        _tts_model = TTS(model_name=model_name).to(device)
        _tts_model._loaded_name = model_name
    return _tts_model

# ---------------------------
# Utility
# ---------------------------
def _new_path(ext: str) -> str:
    return os.path.join(OUT_DIR, f"{uuid.uuid4().hex}.{ext}")

def transcribe_video(video_path: str, whisper_size: str) -> str:
    """
    Extract audio via ffmpeg and transcribe.
    """
    import subprocess

    if not video_path:
        return ""

    wav_path = _new_path("wav")
    # extract mono 16k wav for ASR
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1", "-ar", "16000",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    model = get_whisper(whisper_size)
    segments, info = model.transcribe(wav_path, beam_size=5, language="tr")
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text

def synthesize_voice(
    speaker_wav: str,
    text: str,
    tts_model_name: str,
    language: str = "tr"
) -> str:
    """
    Clone voice from speaker_wav and read text.
    Works best with XTTSv2.
    """
    if not speaker_wav:
        raise ValueError("Speaker wav gerekli.")
    if not text or not text.strip():
        raise ValueError("Okunacak metin boş olamaz.")

    tts = get_tts(tts_model_name)
    out_wav = _new_path("wav")

    # XTTS v2 supports speaker_wav + language
    # Some models may not accept language param; we guard with try.
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=out_wav
        )
    except TypeError:
        # fallback for models that don't support language argument
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path=out_wav
        )

    return out_wav

def mux_audio_to_video(video_path: str, audio_path: str) -> str:
    """
    Replace video's audio track with generated audio.
    """
    import subprocess
    if not video_path or not audio_path:
        raise ValueError("Video ve ses gerekli.")

    out_mp4 = _new_path("mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        out_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_mp4

# ---------------------------
# UI Logic
# ---------------------------
def ui_mode_change(mode: str):
    """
    Show/Hide blocks depending on mode.
    """
    if mode == "Ses":
        return (
            gr.update(visible=True),   # audio block
            gr.update(visible=False),  # video block
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
        )

def handle_audio_flow(
    speaker_wav,
    input_text: str,
    tts_model_name: str
) -> Tuple[Optional[str], str]:
    """
    Audio selected => show textbox => synthesize audio
    """
    try:
        out_wav = synthesize_voice(
            speaker_wav=speaker_wav,
            text=input_text,
            tts_model_name=tts_model_name,
            language="tr"
        )
        return out_wav, "✅ Ses üretildi."
    except Exception as e:
        return None, f"❌ Hata: {e}"

def handle_video_transcribe(video_path: str, whisper_size: str) -> Tuple[str, str]:
    """
    Video selected => get transcript
    """
    try:
        text = transcribe_video(video_path, whisper_size)
        if not text:
            return "", "⚠️ Transkript boş döndü (ses yok veya ASR başarısız)."
        return text, "✅ Transkript hazır."
    except Exception as e:
        return "", f"❌ Hata: {e}"

def handle_video_flow(
    video_path: str,
    speaker_wav: str,
    transcript_text: str,
    custom_text: str,
    use_transcript: bool,
    return_audio_only: bool,
    return_muxed_video: bool,
    tts_model_name: str
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Video selected:
      Option 1: Use transcript -> clone from speaker_wav -> audio output
               optionally mux into video
      Option 2: Use custom text -> clone from speaker_wav -> audio output
               optionally mux into video
    Note: speaker_wav is required for cloning (we keep it explicit & academic).
    """
    try:
        if not video_path:
            raise ValueError("Video seçmelisin.")
        if not speaker_wav:
            raise ValueError("Klonlama için referans ses (speaker wav) seçmelisin.")

        text_to_read = transcript_text if use_transcript else custom_text
        if not text_to_read or not text_to_read.strip():
            raise ValueError("Okunacak metin boş. (Transkript veya custom text)")

        out_wav = synthesize_voice(
            speaker_wav=speaker_wav,
            text=text_to_read,
            tts_model_name=tts_model_name,
            language="tr"
        )

        out_video = None
        if return_muxed_video:
            out_video = mux_audio_to_video(video_path, out_wav)

        # If user wants only video, we still return audio too (academic traceability)
        msg = "✅ Video akışı tamamlandı."
        return out_wav if return_audio_only or True else None, out_video, msg

    except Exception as e:
        return None, None, f"❌ Hata: {e}"

# ---------------------------
# Gradio App
# ---------------------------
AVAILABLE_TTS_MODELS = [
    # Best for voice cloning:
    "tts_models/multilingual/multi-dataset/xtts_v2",
    # Turkish single-speaker-ish (not true cloning):
    "tts_models/tr/common-voice/glow-tts",
]

WHISPER_SIZES = ["tiny", "base", "small", "medium"]

with gr.Blocks(title="Akademik Voice Cloning Lab") as demo:
    gr.Markdown(
        """
# Akademik Voice Cloning Lab (TR)

**Amaç:**  
- *Ses modu*: Referans ses → metni aynı sesle okut (voice cloning / speaker adaptation).  
- *Video modu*: Video → transkript çıkar → seçilen metni klon sesle okut → istenirse videoya göm.

> Not: Fine-tune modelin hazır olduğunda, `Model seçimi` bölümünden **fine-tuned checkpoint yolunu** da ekleyeceğiz.
        """.strip()
    )

    with gr.Row():
        mode = gr.Radio(["Ses", "Video"], value="Ses", label="Giriş tipi")
        tts_model_name = gr.Dropdown(
            choices=AVAILABLE_TTS_MODELS,
            value=AVAILABLE_TTS_MODELS[0],
            label="Model seçimi (TTS)"
        )

    status = gr.Textbox(label="Durum", value="Hazır.", interactive=False)

    # -------------------- Audio Mode --------------------
    audio_block = gr.Group(visible=True)
    with audio_block:
        gr.Markdown("## 1) Ses modu: Referans ses + Metin")
        speaker_wav_audio = gr.Audio(type="filepath", label="Referans Ses (.wav) (Senin sesin)")
        input_text_audio = gr.Textbox(
            label="Okunacak metin",
            lines=5,
            placeholder="Buraya Türkçe metni yaz..."
        )
        run_audio = gr.Button("🎙️ Klonla ve Oku", variant="primary")
        out_audio = gr.Audio(type="filepath", label="Çıktı (Klonlanmış Ses)")

    # -------------------- Video Mode --------------------
    video_block = gr.Group(visible=False)
    with video_block:
        gr.Markdown("## 2) Video modu: Transkript + Klon + (Opsiyonel) Video Birleştirme")

        with gr.Row():
            video_in = gr.Video(label="Video yükle")
            whisper_size = gr.Dropdown(choices=WHISPER_SIZES, value="small", label="ASR model (Whisper)")

        get_tr = gr.Button("📝 Videodan Transkript Çıkar", variant="secondary")
        transcript_box = gr.Textbox(label="Transkript", lines=8)

        gr.Markdown("### Klonlama için referans ses")
        speaker_wav_video = gr.Audio(type="filepath", label="Referans Ses (.wav) (Senin sesin)")

        with gr.Row():
            use_transcript = gr.Checkbox(value=True, label="Transkripti okut")
            return_muxed_video = gr.Checkbox(value=True, label="Üretilen sesi videoya göm (mux)")
            return_audio_only = gr.Checkbox(value=True, label="Ayrıca sadece ses çıktısı ver")

        custom_text_video = gr.Textbox(
            label="Custom Text (transkript yerine)",
            lines=5,
            placeholder="Transkript yerine bunu okutmak istersen buraya yaz."
        )

        run_video = gr.Button("🎬 Klonla ve Üret", variant="primary")
        out_audio_video = gr.Audio(type="filepath", label="Çıktı Ses")
        out_video = gr.Video(label="Çıktı Video (Ses değiştirilmiş)")

    # ---------------------------
    # Events
    # ---------------------------
    mode.change(ui_mode_change, inputs=[mode], outputs=[audio_block, video_block])

    run_audio.click(
        handle_audio_flow,
        inputs=[speaker_wav_audio, input_text_audio, tts_model_name],
        outputs=[out_audio, status]
    )

    get_tr.click(
        handle_video_transcribe,
        inputs=[video_in, whisper_size],
        outputs=[transcript_box, status]
    )

    run_video.click(
        handle_video_flow,
        inputs=[
            video_in,
            speaker_wav_video,
            transcript_box,
            custom_text_video,
            use_transcript,
            return_audio_only,
            return_muxed_video,
            tts_model_name
        ],
        outputs=[out_audio_video, out_video, status]
    )

demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
