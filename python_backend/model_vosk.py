import os
import wave
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from vosk import Model, KaldiRecognizer
import torchaudio
import librosa
import uvicorn
import torch
from pydantic import BaseModel
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "/Users/milanagaraeva/PycharmProjects/Diplom v2/vosk-model-ru-0.42"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {MODEL_PATH}")

model = Model(MODEL_PATH)
logger.info("Loaded Vosk model")

app = FastAPI()

class TranscriptionResponse(BaseModel):
    transcription: str

def convert_to_wav(inp_path, out_path):
    try:
        wav, sr = torchaudio.load(inp_path)
    except Exception:
        wav_np, sr = librosa.load(inp_path, sr=16000)
        wav = torch.tensor(wav_np).unsqueeze(0)
        sr = 16000
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    pcm = (wav.squeeze(0).numpy() * 32767).astype("int16")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())

def transcribe_with_vosk(audio_path: str) -> str:
    tmp = f"/tmp/{uuid4().hex}.wav"
    convert_to_wav(audio_path, tmp)
    wf = wave.open(tmp, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    while True:
        data = wf.readframes(4000)
        if not data: break
        rec.AcceptWaveform(data)
    res = json.loads(rec.FinalResult())
    wf.close()
    os.remove(tmp)
    return res.get("text", "")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    # 1. Прочитать на диск
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file uploaded")
    tmp_in = f"/tmp/{uuid4().hex}_{file.filename}"
    with open(tmp_in, "wb") as f:
        f.write(content)

    # 2. Попытаться распознать
    try:
        text = transcribe_with_vosk(tmp_in)
    except Exception as e:
        logger.exception("Error in Vosk transcription")
        raise HTTPException(500, f"Vosk error: {e}")
    finally:
        os.remove(tmp_in)

    return TranscriptionResponse(transcription=text)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
