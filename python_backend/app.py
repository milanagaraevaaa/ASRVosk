import os
import wave
import json
import logging
import subprocess
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Vosk ASR
from vosk import Model, KaldiRecognizer

# для преобразования форматов
import torchaudio
import librosa
import torch

# модель пунктуации
from deepmultilingualpunctuation import PunctuationModel

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Путь к Vosk-модели ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-ru-0.42")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {MODEL_PATH}")

asr_model = Model(MODEL_PATH)
logger.info("Loaded Vosk model")

# --- Модель пунктуации ---
# Будет загружена из HuggingFace при первом вызове
punct_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")
logger.info("Loaded punctuation model")

app = FastAPI()


class TranscriptionResponse(BaseModel):
    transcription: str


def convert_to_wav(input_path: str, output_path: str):
    """
    Конвертирует любой аудио-файл в WAV (моно, 16kHz, 16-bit).
    Сначала пытается через ffmpeg, если нет — через torchaudio/librosa.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return
    except Exception:
        logger.debug("ffmpeg failed, fallback to torchaudio/librosa")

    # fallback
    try:
        wav, sr = torchaudio.load(input_path)
    except:
        wav_np, sr = librosa.load(input_path, sr=16000)
        wav = torch.tensor(wav_np).unsqueeze(0)
        sr = 16000

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    pcm = (wav.squeeze(0).numpy() * 32767).astype("int16")
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())


def transcribe_with_vosk(audio_path: str) -> str:
    """
    Транскрибирует файл через Vosk, возвращает сырый текст без пунктуации.
    """
    tmp_wav = f"/tmp/{uuid4().hex}.wav"
    convert_to_wav(audio_path, tmp_wav)

    wf = wave.open(tmp_wav, "rb")
    rec = KaldiRecognizer(asr_model, wf.getframerate())
    rec.SetWords(True)
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        rec.AcceptWaveform(data)
    result = json.loads(rec.FinalResult())
    wf.close()
    os.remove(tmp_wav)
    return result.get("text", "")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    """
    Принимает аудио, возвращает транскрипцию с пунктуацией и заглавной буквой в начале.
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file uploaded")
    tmp_in = f"/tmp/{uuid4().hex}_{file.filename}"
    with open(tmp_in, "wb") as f:
        f.write(content)

    try:
        # 1) Сырой текст
        raw = transcribe_with_vosk(tmp_in)
        if not raw.strip():
            raise ValueError("No speech detected")

        # 2) Восстановление пунктуации
        punctuated = punct_model.restore_punctuation(raw)

        # 3) Убираем пробелы по краям и делаем первую букву заглавной
        text = punctuated.strip()
        if text:
            text = text[0].upper() + text[1:]
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(500, str(e))
    finally:
        os.remove(tmp_in)

    return TranscriptionResponse(transcription=text)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)