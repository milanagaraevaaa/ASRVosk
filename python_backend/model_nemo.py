import os
import sys
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import librosa
import speech_recognition as sr
import wave
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from jiwer import wer as compute_ter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from Levenshtein import distance as levenshtein_distance

# Загрузка данных NLTK
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# При работе на MPS можно отключить ограничение на использование памяти
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

######################################
# 1. Датасет для аудио без токенизации
######################################
class AudioDataset(Dataset):
    def __init__(self, tsv_file, audio_dir, transform=None):
        self.data = pd.read_csv(tsv_file, sep='\t')
        self.data = self.data.sample(frac=0.5).reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transform = transform
        self.data = self.data[self.data['path'].apply(lambda x: os.path.isfile(os.path.join(audio_dir, x)))]
        self.data = self.data.reset_index(drop=True)
        self.cache = {}
        print(f"Количество записей после фильтрации: {len(self.data)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_file = row['path']
        transcript = row['sentence']
        audio_path = os.path.join(self.audio_dir, audio_file)
        if audio_file in self.cache:
            waveform = self.cache[audio_file]
        else:
            try:
                waveform, sample_rate = torchaudio.load(audio_path, format="mp3")
            except RuntimeError as e:
                print(f"torchaudio.load не смог загрузить {audio_path}, пробуем через librosa: {e}")
                try:
                    waveform_np, sample_rate = librosa.load(audio_path, sr=16000)
                    waveform = torch.tensor(waveform_np).unsqueeze(0)
                except Exception as e_librosa:
                    raise RuntimeError(f"Ошибка загрузки {audio_path}: {e_librosa}")
            if self.transform:
                waveform = self.transform(waveform, sample_rate)
            self.cache[audio_file] = waveform
        return waveform, audio_path, transcript

######################################
# 2. Преобразование аудио для SpeechRecognition
######################################
def transform_eval_fn(waveform, sample_rate=16000):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform

def save_waveform_to_temp_file(waveform, temp_path):
    waveform = waveform.squeeze(0).numpy()
    waveform = (waveform * 32767).astype(np.int16)
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(waveform.tobytes())
    print(f"Создан WAV-файл: {temp_path}, размер: {os.path.getsize(temp_path)} байт")

######################################
# 3. Функция транскрибирования с SpeechRecognition
######################################
def transcribe_with_speechrecognition(audio_path, recognizer, sr_module, language="ru-RU"):
    if isinstance(audio_path, str):
        try:
            waveform, sr = torchaudio.load(audio_path)
            print(f"Загружено аудио: {audio_path}, форма: {waveform.shape}, частота: {sr}")
        except Exception as e:
            print(f"Ошибка torchaudio.load: {e}, пробуем через librosa")
            waveform_np, sr = librosa.load(audio_path, sr=16000)
            waveform = torch.tensor(waveform_np).unsqueeze(0)
            print(f"Загружено через librosa: форма: {waveform.shape}, частота: {sr}")
        waveform = transform_eval_fn(waveform, sr)
    else:
        waveform = audio_path
        print(f"Используется готовый waveform: форма: {waveform.shape}")

    temp_wav_path = "temp_audio.wav"
    save_waveform_to_temp_file(waveform, temp_wav_path)

    transcription = ""
    try:
        with sr_module.AudioFile(temp_wav_path) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio, language=language)
        print(f"Транскрипция: {transcription}")
    except sr_module.UnknownValueError:
        print("Google Speech Recognition не смог распознать аудио")
    except sr_module.RequestError as e:
        print(f"Ошибка запроса к Google Speech Recognition: {e}")
    except Exception as e:
        print(f"Общая ошибка транскрибирования: {e}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"Временный файл удалён: {temp_wav_path}")

    return transcription

######################################
# 4. Метрики для оценки качества транскрибирования
######################################
def compute_metrics(reference, hypothesis):
    # Постобработка текста: удаляем лишние пробелы и приводим к нижнему регистру
    reference = " ".join(reference.split()).lower()
    hypothesis = " ".join(hypothesis.split()).lower() if hypothesis else ""

    # Токенизация текстов
    ref_tokens = word_tokenize(reference)
    hyp_tokens = word_tokenize(hypothesis) if hypothesis else []

    # 1. BLEU с сглаживанием
    try:
        # Для коротких текстов (< 4 слов) используем упрощённый подход
        if len(ref_tokens) < 4 or len(hyp_tokens) < 4:
            # Если тексты идентичны после приведения к нижнему регистру, BLEU = 1.0
            if reference == hypothesis:
                bleu_score = 1.0
            else:
                # Используем только униграммы для коротких текстов
                bleu_score = sentence_bleu([ref_tokens], hyp_tokens, weights=(1.0, 0, 0, 0),
                                           smoothing_function=SmoothingFunction().method1)
        else:
            # Для более длинных текстов адаптируем веса
            max_n = min(4, max(len(ref_tokens), len(hyp_tokens)))
            weights = [1.0/max_n] * max_n
            bleu_score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights,
                                       smoothing_function=SmoothingFunction().method1)
    except ZeroDivisionError:
        bleu_score = 0.0

    # 2. ROUGE (отключаем стемминг, так как он не работает для русского языка)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    rouge_scores = scorer.score(reference, hypothesis)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    # 3. METEOR
    try:
        meteor = meteor_score([ref_tokens], hyp_tokens)
    except (TypeError, ValueError) as e:
        print(f"Ошибка при вычислении METEOR: {e}. Устанавливаем METEOR = 0.0")
        meteor = 0.0

    # 4. TER (эквивалентно WER)
    ter_score = compute_ter(reference, hypothesis)

    # 5. Косинусное расстояние
    vectorizer = CountVectorizer().fit_transform([reference, hypothesis])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]

    # 6. Расстояние Левенштейна
    lev_distance = levenshtein_distance(reference, hypothesis)

    # 7. Сходство Жаккара
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    jaccard_sim = len(ref_set.intersection(hyp_set)) / len(ref_set.union(hyp_set)) if ref_set.union(hyp_set) else 0.0

    # 8. Сходство Рэнда
    all_words = list(set(ref_tokens).union(set(hyp_tokens)))
    if not hyp_tokens:  # Если гипотеза пустая, сходство должно быть 0
        rand_sim = 0.0
    else:
        ref_binary = [1 if word in ref_tokens else 0 for word in all_words]
        hyp_binary = [1 if word in hyp_tokens else 0 for word in all_words]
        rand_sim = adjusted_rand_score(ref_binary, hyp_binary)

    return {
        "BLEU": bleu_score,
        "ROUGE-1": rouge1,
        "ROUGE-L": rougeL,
        "METEOR": meteor,
        "TER": ter_score,
        "Cosine Similarity": cosine_sim,
        "Levenshtein Distance": lev_distance,
        "Jaccard Similarity": jaccard_sim,
        "Rand Similarity": rand_sim
    }

######################################
# 5. Проверка чтения аудио
######################################
def test_audio_reading(dataset, num_samples=5):
    print("Проверка чтения аудио:")
    for i in range(num_samples):
        waveform, audio_path, _ = dataset[i]
        print(f"Образец {i}: путь = {audio_path}, форма аудио = {waveform.shape}")

######################################
# 6. Основная функция
######################################
def main():
    # Пути к данным
    tsv_file = "/Users/milanagaraeva/PycharmProjects/Diplom v2/data/dataset.tsv"
    audio_dir = "/Users/milanagaraeva/PycharmProjects/Diplom v2/data/clips"

    # Инициализация SpeechRecognition
    recognizer = sr.Recognizer()
    print("SpeechRecognition инициализирован.")

    # Если передан путь к аудиофайлу через аргументы командной строки
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.isfile(audio_path):
            print(f"Файл {audio_path} не найден.")
            return
        print(f"Транскрибирование файла: {audio_path}")
        transcription = transcribe_with_speechrecognition(audio_path, recognizer, sr)
        print("Результат транскрибирования:")
        print(transcription)
        return

    # Создаём датасет для проверки на Mozilla Common Voice
    eval_dataset = AudioDataset(tsv_file, audio_dir, transform=transform_eval_fn)
    test_audio_reading(eval_dataset, num_samples=5)

    # Транскрибируем несколько примеров из датасета
    num_samples = 10
    print(f"\nТранскрибирование {num_samples} примеров из датасета:")
    indices = torch.randperm(len(eval_dataset))[:num_samples].tolist()
    metrics_sums = {
        "BLEU": 0.0, "ROUGE-1": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0, "TER": 0.0,
        "Cosine Similarity": 0.0, "Levenshtein Distance": 0.0, "Jaccard Similarity": 0.0, "Rand Similarity": 0.0
    }
    for idx in indices:
        waveform, audio_path, target_text = eval_dataset[idx]
        print(f"\nПример {idx}: {audio_path}")
        print("Целевой текст:", target_text)
        transcription = transcribe_with_speechrecognition(waveform, recognizer, sr)
        print("Предсказанный текст:", transcription)
        # Вычисляем метрики
        metrics = compute_metrics(target_text, transcription)
        print("Метрики:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            metrics_sums[metric] += value

    # Вычисляем средние значения метрик
    print(f"\nСредние значения метрик на {num_samples} примерах:")
    for metric, total in metrics_sums.items():
        avg = total / num_samples
        print(f"Средняя {metric}: {avg:.4f}")

if __name__ == "__main__":
    main()