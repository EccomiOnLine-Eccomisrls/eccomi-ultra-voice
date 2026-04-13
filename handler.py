import os
import uuid
import tempfile
import traceback
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod
import torch
from TTS.api import TTS

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_INPUTS_BUCKET = os.getenv("SUPABASE_INPUTS_BUCKET", "inputs")

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔄 Carico XTTS v2 su {DEVICE}...", flush=True)
tts = TTS(MODEL_NAME).to(DEVICE)
print("✅ XTTS v2 pronto", flush=True)


def guess_extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()

    if path.endswith(".wav"):
        return ".wav"
    if path.endswith(".mp3"):
        return ".mp3"
    if path.endswith(".m4a"):
        return ".m4a"
    if path.endswith(".ogg"):
        return ".ogg"
    if path.endswith(".webm"):
        return ".webm"
    if path.endswith(".mp4"):
        return ".mp4"

    return ".bin"


def upload_to_supabase(local_path: str, object_path: str, content_type: str = "audio/wav"):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("ENV Supabase mancanti")

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_INPUTS_BUCKET}/{object_path}?upsert=true"

    with open(local_path, "rb") as f:
        data = f.read()

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": content_type,
    }

    print(f"☁️ Upload Supabase: {object_path}", flush=True)

    r = requests.put(upload_url, headers=headers, data=data, timeout=300)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload Supabase fallito: {r.status_code} {r.text}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_INPUTS_BUCKET}/{object_path}"
    print(f"✅ Upload completato: {public_url}", flush=True)
    return public_url


def download_file(url: str, out_path: str):
    print(f"⬇️ Download voice sample: {url}", flush=True)

    r = requests.get(url, stream=True, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Download fallito: {r.status_code}")

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if not os.path.exists(out_path):
        raise RuntimeError("Download completato ma file assente")

    print(f"✅ Download completato: {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def convert_audio_to_wav(input_path: str, output_path: str):
    print(f"🎚 Conversione audio -> WAV: {input_path}", flush=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "24000",
        "-ac", "1",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Conversione ffmpeg fallita.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError("Conversione WAV fallita: file output assente")

    if os.path.getsize(output_path) < 1000:
        raise RuntimeError("Conversione WAV fallita: file output troppo piccolo")

    print(f"✅ WAV pronto: {output_path} ({os.path.getsize(output_path)} bytes)", flush=True)


def handler(job):
    job_input = job.get("input", {}) or {}

    token = job_input.get("token") or str(uuid.uuid4())
    text = (job_input.get("text") or "").strip()
    voice_sample_url = (job_input.get("voice_sample_url") or "").strip()
    language = (job_input.get("language") or "it").strip()

    print(f"🚀 Job Ultra avviato | token={token}", flush=True)

    if not text:
        return {
            "ok": False,
            "error": "text mancante",
            "token": token
        }

    if not voice_sample_url:
        return {
            "ok": False,
            "error": "voice_sample_url mancante",
            "token": token
        }

    tmp_dir = tempfile.mkdtemp(prefix=f"xtts_{token}_")

    original_ext = guess_extension_from_url(voice_sample_url)
    downloaded_sample_path = str(Path(tmp_dir) / f"voice_sample_original{original_ext}")
    normalized_sample_path = str(Path(tmp_dir) / "voice_sample_normalized.wav")
    output_path = str(Path(tmp_dir) / "dubbed_audio.wav")

    try:
        download_file(voice_sample_url, downloaded_sample_path)

        if os.path.getsize(downloaded_sample_path) < 1000:
            return {
                "ok": False,
                "error": "campione voce non valido o troppo piccolo",
                "token": token
            }

        convert_audio_to_wav(downloaded_sample_path, normalized_sample_path)

        print("🎙 Genero audio clonato...", flush=True)

        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=normalized_sample_path,
            language=language,
            split_sentences=True
        )

        print("✅ Generazione XTTS completata", flush=True)

        if not os.path.exists(output_path):
            return {
                "ok": False,
                "error": "audio non creato",
                "token": token
            }

        if os.path.getsize(output_path) < 5000:
            return {
                "ok": False,
                "error": "audio creato troppo piccolo",
                "token": token
            }

        object_path = f"{token}/dubbed_audio.wav"
        dubbed_audio_url = upload_to_supabase(output_path, object_path, "audio/wav")

        return {
            "ok": True,
            "token": token,
            "dubbed_audio_url": dubbed_audio_url,
            "audio_url": dubbed_audio_url,
            "url": dubbed_audio_url
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ ERRORE ULTRA:", str(e), flush=True)
        print(tb, flush=True)

        return {
            "ok": False,
            "error": str(e),
            "traceback": tb,
            "token": token
        }


runpod.serverless.start({"handler": handler})
