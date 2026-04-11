import os
import uuid
import tempfile
from pathlib import Path

import requests
import runpod
from TTS.api import TTS

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_INPUTS_BUCKET = os.getenv("SUPABASE_INPUTS_BUCKET", "inputs")

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cuda"

print("🔄 Carico XTTS v2...")
tts = TTS(MODEL_NAME).to(DEVICE)
print("✅ XTTS v2 pronto")


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

    r = requests.put(upload_url, headers=headers, data=data, timeout=300)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload Supabase fallito: {r.status_code} {r.text}")

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_INPUTS_BUCKET}/{object_path}"


def download_file(url: str, out_path: str):
    r = requests.get(url, stream=True, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Download fallito: {r.status_code}")

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def handler(job):
    job_input = job.get("input", {}) or {}

    token = job_input.get("token") or str(uuid.uuid4())
    text = (job_input.get("text") or "").strip()
    voice_sample_url = (job_input.get("voice_sample_url") or "").strip()
    language = (job_input.get("language") or "it").strip()

    if not text:
        return {"error": "text mancante", "token": token}

    if not voice_sample_url:
        return {"error": "voice_sample_url mancante", "token": token}

    tmp_dir = tempfile.mkdtemp(prefix=f"xtts_{token}_")
    sample_path = str(Path(tmp_dir) / "voice_sample.wav")
    output_path = str(Path(tmp_dir) / "dubbed_audio.wav")

    try:
        download_file(voice_sample_url, sample_path)

        if not os.path.exists(sample_path) or os.path.getsize(sample_path) < 1000:
            return {"error": "campione voce non valido", "token": token}

        print("🎙 Genero audio clonato...")
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=[sample_path],
            language=language,
            split_sentences=True
        )

        if not os.path.exists(output_path):
            return {"error": "audio non creato", "token": token}

        if os.path.getsize(output_path) < 5000:
            return {"error": "audio creato troppo piccolo", "token": token}

        object_path = f"{token}/dubbed_audio.wav"
        dubbed_audio_url = upload_to_supabase(output_path, object_path, "audio/wav")

        return {
            "ok": True,
            "token": token,
            "dubbed_audio_url": dubbed_audio_url
        }

    except Exception as e:
        return {
            "error": str(e),
            "token": token
        }


runpod.serverless.start({"handler": handler})
