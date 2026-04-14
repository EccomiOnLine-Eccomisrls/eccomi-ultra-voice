import os
import re
import uuid
import tempfile
import traceback
import subprocess
import unicodedata
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

HTTP_TIMEOUT = 300
SILENCE_MS_BETWEEN_CHUNKS = 220

print(f"🔄 Carico XTTS v2 su {DEVICE}...", flush=True)
tts = TTS(MODEL_NAME).to(DEVICE)
print("✅ XTTS v2 pronto", flush=True)


# =====================================================
# STORAGE
# =====================================================

def upload_to_supabase(local_path: str, object_path: str, content_type: str = "audio/wav") -> str:
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

    print(f"☁️ Upload Supabase -> bucket={SUPABASE_INPUTS_BUCKET} path={object_path}", flush=True)

    r = requests.put(upload_url, headers=headers, data=data, timeout=HTTP_TIMEOUT)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload Supabase fallito: {r.status_code} {r.text}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_INPUTS_BUCKET}/{object_path}"
    print(f"✅ Upload OK: {public_url}", flush=True)
    return public_url


# =====================================================
# DOWNLOAD / AUDIO PREP
# =====================================================

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


def download_file(url: str, out_path: str):
    print(f"⬇️ Download voice sample: {url}", flush=True)

    r = requests.get(url, stream=True, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Download fallito: {r.status_code}")

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if not os.path.exists(out_path):
        raise RuntimeError("Download completato ma file assente")

    size = os.path.getsize(out_path)
    print(f"✅ Download completato: {out_path} ({size} bytes)", flush=True)


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


def make_silence_wav(output_path: str, duration_ms: int = 220):
    duration_s = max(duration_ms, 0) / 1000.0

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r=24000:cl=mono",
        "-t", str(duration_s),
        "-acodec", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Creazione pausa fallita.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError("File pausa non creato")


# =====================================================
# TEXT CLEANING FOR XTTS
# =====================================================

def normalize_text_for_xtts(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u00a0": " ",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": " ",
        "–": ",",
        "—": ",",
        ";": ",",
        ":": ",",
        "«": "",
        "»": "",
        "„": "",
        "‟": "",
        "•": ",",
        "|": ",",
        "(": ", ",
        ")": " ",
        "[": ", ",
        "]": " ",
        "{": ", ",
        "}": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t "
    )

    text = text.replace('"', "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[,]{2,}", ",", text)

    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"([,.;!?])([^\s])", r"\1 \2", text)

    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:4000]


def split_text_for_xtts(text: str, max_chars: int = 180) -> list[str]:
    text = normalize_text_for_xtts(text)

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            subparts = [p.strip() for p in part.split(",") if p.strip()]
            current = ""

            for sp in subparts:
                candidate = sp if not current else f"{current}, {sp}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        sentences.append(current.strip())
                    current = sp

            if current:
                sentences.append(current.strip())
        else:
            sentences.append(part)

    cleaned = []
    for s in sentences:
        s = s.strip()

        if not s:
            continue

        # toglie punteggiatura finale che XTTS a volte legge come parola
        s = re.sub(r"[.!?,:;]+$", "", s).strip()

        if s:
            cleaned.append(s)

    return cleaned


def synthesize_sentences_to_wav(sentences: list[str], speaker_wav: str, language: str, output_path: str):
    if not sentences:
        raise RuntimeError("Nessuna frase disponibile per XTTS")

    work_dir = Path(output_path).parent
    chunk_paths = []

    for idx, sentence in enumerate(sentences, start=1):
        chunk_path = str(work_dir / f"chunk_{idx:03d}.wav")
        print(f"🧩 XTTS chunk {idx}/{len(sentences)}: {sentence}", flush=True)

        tts_sentence = sentence.strip()
        tts_sentence = re.sub(r"[.!?,:;]+$", "", tts_sentence).strip()

        tts.tts_to_file(
            text=tts_sentence,
            file_path=chunk_path,
            speaker_wav=speaker_wav,
            language=language,
            split_sentences=False
        )

        if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 1000:
            raise RuntimeError(f"Chunk audio non valido: {chunk_path}")

        chunk_paths.append(chunk_path)

    concat_list_path = str(work_dir / "concat_list.txt")

    with open(concat_list_path, "w", encoding="utf-8") as f:
        for idx, p in enumerate(chunk_paths):
            safe_p = p.replace("\\", "/").replace("'", r"'\''")
            f.write(f"file '{safe_p}'\n")

            if idx < len(chunk_paths) - 1 and SILENCE_MS_BETWEEN_CHUNKS > 0:
                silence_path = str(work_dir / f"silence_{idx:03d}.wav")
                make_silence_wav(silence_path, SILENCE_MS_BETWEEN_CHUNKS)
                safe_silence = silence_path.replace("\\", "/").replace("'", r"'\''")
                f.write(f"file '{safe_silence}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:a", "pcm_s16le",
        "-ar", "24000",
        "-ac", "1",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Concatenazione audio fallita.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 5000:
        raise RuntimeError("Audio finale concatenato non valido")


# =====================================================
# HANDLER
# =====================================================

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

        clean_text = normalize_text_for_xtts(text)
        sentences = split_text_for_xtts(clean_text, max_chars=180)

        print(f"📝 Testo pulito XTTS: {clean_text}", flush=True)
        print(f"🧩 Frasi XTTS: {sentences}", flush=True)

        if not sentences:
            return {
                "ok": False,
                "error": "testo vuoto dopo pulizia XTTS",
                "token": token
            }

        print("🎙 Genero audio clonato a blocchi...", flush=True)

        synthesize_sentences_to_wav(
            sentences=sentences,
            speaker_wav=normalized_sample_path,
            language=language,
            output_path=output_path
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
