FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1
ENV TTS_HOME=/tmp/tts
ENV HF_HOME=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir "setuptools<81" && \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python", "-u", "handler.py"]
