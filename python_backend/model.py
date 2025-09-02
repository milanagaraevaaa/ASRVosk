import os
import math
import time
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

# При работе на MPS можно отключить ограничение на использование памяти
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Ограничение длины аудио (число временных шагов мел-спектрограммы)
MAX_AUDIO_LEN = 800

# Флаг для аугментации (если требуется)
USE_AUGMENTATION = False

######################################
# 1. Позиционное кодирование
######################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=15000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # четные индексы
        pe[:, 1::2] = torch.cos(position * div_term)  # нечетные индексы
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        if x.dim() == 3 and x.shape[0] != self.pe.shape[0]:
            if x.shape[0] < self.pe.shape[0]:
                x = x + self.pe[:x.shape[0]]
            else:
                raise RuntimeError(f"Позиционное кодирование рассчитано на max_len={self.pe.shape[0]}, но получена последовательность длиной {x.shape[0]}")
        else:
            x = x.transpose(0, 1)
            if x.shape[0] > self.pe.shape[0]:
                raise RuntimeError(f"Позиционное кодирование рассчитано на max_len={self.pe.shape[0]}, но получена последовательность длиной {x.shape[0]}")
            x = x + self.pe[:x.shape[0]]
            x = x.transpose(0, 1)
        return self.dropout(x)

######################################
# 2. Символьный токенизатор с пунктуацией
######################################
class CharTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.special_tokens = ['<PAD>', '<SOS>', '<EOS>']
            letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя,.!?-:;()\"' ")
            self.vocab = self.special_tokens + letters
        else:
            self.vocab = vocab
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
    def __call__(self, text):
        text = text.lower()
        tokens = ['<SOS>'] + list(text) + ['<EOS>']
        return [self.token2id.get(token, self.token2id[' ']) for token in tokens]
    def vocab_size(self):
        return len(self.vocab)

def decode_tokens(token_ids, tokenizer):
    tokens = [tokenizer.id2token[i] for i in token_ids if i != tokenizer.token2id["<PAD>"]]
    if tokens and tokens[0] == "<SOS>":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "<EOS>":
        tokens = tokens[:-1]
    return "".join(tokens)

######################################
# 3. Датасет для аудио с кэшированием мел-спектрограммы
######################################
class AudioDataset(Dataset):
    def __init__(self, tsv_file, audio_dir, transform=None, tokenizer=None):
        self.data = pd.read_csv(tsv_file, sep='\t')
        # Выбираем случайным образом половину записей:
        self.data = self.data.sample(frac=0.5).reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transform = transform
        self.tokenizer = tokenizer
        # Оставляем только те записи, для которых существует аудиофайл
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
                waveform = self.transform(waveform)
                waveform = waveform.mean(dim=0)   # Усредняем по каналам
                waveform = waveform.transpose(0, 1)  # (T, mel)
                if MAX_AUDIO_LEN is not None and waveform.size(0) > MAX_AUDIO_LEN:
                    waveform = waveform[:MAX_AUDIO_LEN, :]
            self.cache[audio_file] = waveform
        token_ids = self.tokenizer(transcript)
        return waveform, torch.tensor(token_ids, dtype=torch.long)

######################################
# 4. Преобразования для обучения и оценки
######################################
def transform_train_fn(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)(waveform)
    db_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mean = db_spec.mean()
    std = db_spec.std() + 1e-5
    normalized = (db_spec - mean) / std
    if USE_AUGMENTATION:
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        normalized = time_masking(normalized)
        normalized = freq_masking(normalized)
    return normalized

def transform_eval_fn(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)(waveform)
    db_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mean = db_spec.mean()
    std = db_spec.std() + 1e-5
    normalized = (db_spec - mean) / std
    return normalized

######################################
# 5. Модель SpeechRecognitionTransformer
######################################
class SpeechRecognitionTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=256, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super(SpeechRecognitionTransformer, self).__init__()
        self.d_model = d_model
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.audio_proj = nn.Linear(256, d_model)
        self.audio_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=15000)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_norm = nn.LayerNorm(d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=15000)
        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt):
        src = src.transpose(1, 2)  # (batch, input_dim, src_seq_len)
        x = self.conv_block1(src)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.transpose(1, 2)  # (batch, T', 256)
        x = self.audio_proj(x)  # (batch, T', d_model)
        x = self.audio_norm(x)  # Нормировка аудио
        x = x.transpose(0, 1)  # (T', batch, d_model)
        x = self.pos_encoder(x)
        tgt_key_padding_mask = (tgt == 0).bool()
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.tgt_norm(tgt)  # Нормировка эмбеддингов
        tgt = tgt.transpose(0, 1)  # (tgt_seq_len, batch, d_model)
        tgt = self.pos_decoder(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device).bool()
        output = self.transformer(x, tgt, tgt_mask=tgt_mask,
                                  src_key_padding_mask=None,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=None)
        output = self.fc_out(output)
        return output.transpose(0, 1)  # (batch, tgt_seq_len, vocab_size)

######################################
# 6. collate_fn для пакетной загрузки
######################################
def collate_fn(batch):
    srcs = [item[0] for item in batch]
    tgts = [item[1] for item in batch]
    max_src_len = max(s.shape[0] for s in srcs)
    padded_srcs = []
    for s in srcs:
        pad_size = max_src_len - s.shape[0]
        if pad_size > 0:
            padding = torch.zeros(pad_size, s.shape[1])
            s = torch.cat([s, padding], dim=0)
        padded_srcs.append(s)
    padded_srcs = torch.stack(padded_srcs)
    max_tgt_len = max(t.shape[0] for t in tgts)
    padded_tgts = []
    for t in tgts:
        pad_size = max_tgt_len - t.shape[0]
        if pad_size > 0:
            padding = torch.zeros(pad_size, dtype=torch.long)
            t = torch.cat([t, padding], dim=0)
        padded_tgts.append(t)
    padded_tgts = torch.stack(padded_tgts)
    return padded_srcs, padded_tgts

######################################
# 7. Функция обучения с полным teacher forcing
######################################
def train_teacher_forcing(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training (Teacher Forcing)", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]   # Входная последовательность без последнего токена
        tgt_output = tgt[:, 1:]   # Целевая последовательность (сдвинутая на один токен)
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

######################################
# 8. Функция обучения для fine-tuning с Scheduled Sampling
######################################
def train_scheduled_sampling(model, dataloader, optimizer, criterion, device,
                             teacher_forcing_ratio=0.5, scheduled_sampling_prob=0.2):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Fine-tuning (Scheduled Sampling)", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        batch_size, seq_len = tgt.size()
        # Решаем, применять ли scheduled sampling для данного батча
        if torch.rand(1).item() < scheduled_sampling_prob:
            tgt_input_full = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output_full = model(src, tgt_input_full)
            pred_tokens = output_full.argmax(dim=-1)  # (batch, seq_len-1)
            mask = (torch.rand(pred_tokens.size(), device=device) >= teacher_forcing_ratio)
            mixed_tgt = torch.where(mask, pred_tokens, tgt[:, 1:])
            output = model(src, mixed_tgt)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        else:
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

######################################
# 9. Функция жадного декодирования для инференса
######################################
def greedy_decode(model, src, max_len, tokenizer, device):
    model.eval()
    with torch.no_grad():
        tgt_ids = torch.tensor([[tokenizer.token2id["<SOS>"]]], device=device)
        for _ in range(max_len):
            output = model(src, tgt_ids)
            next_token_logits = output[0, -1, :]
            next_token_id = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.token2id["<EOS>"]:
                break
    return tgt_ids[0].cpu().tolist()

######################################
# 10. Функция декодирования с использованием Beam Search для инференса
######################################
def beam_search_decode(model, src, tokenizer, device, max_len=100, beam_width=5, length_penalty=0.7):
    model.eval()
    with torch.no_grad():
        initial_token = tokenizer.token2id["<SOS>"]
        beams = [([initial_token], 0.0)]
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == tokenizer.token2id["<EOS>"]:
                    new_beams.append((seq, score))
                    continue
                tgt = torch.tensor([seq], device=device)
                output = model(src, tgt)
                logits = output[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                for log_prob, token_id in zip(topk_log_probs, topk_ids):
                    new_seq = seq + [token_id.item()]
                    new_score = score + log_prob.item()
                    new_score /= (len(new_seq) ** length_penalty)
                    new_beams.append((new_seq, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == tokenizer.token2id["<EOS>"] for seq, _ in beams):
                break
        best_seq, _ = beams[0]
        return best_seq

######################################
# 11. Функция декодирования токенов в строку
######################################
def decode_tokens(token_ids, tokenizer):
    tokens = [tokenizer.id2token[i] for i in token_ids if i != tokenizer.token2id["<PAD>"]]
    if tokens and tokens[0] == "<SOS>":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "<EOS>":
        tokens = tokens[:-1]
    return "".join(tokens)

######################################
# 12. Функция вычисления WER (Word Error Rate)
######################################
def compute_wer(reference, hypothesis):
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    r = len(ref_words)
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1,
                          d[i][j-1] + 1,
                          d[i-1][j-1] + cost)
    wer = d[len(ref_words)][len(hyp_words)] / float(r) if r > 0 else 0.0
    return wer

######################################
# 13. Функция оценки модели с WER
######################################
def evaluate(model, dataset, tokenizer, device, num_samples=100, decode_max_len=100, beam_width=5, decode_method='beam'):
    model.eval()
    print("Оценка модели на {} примерах:".format(num_samples))
    total_wer = 0.0
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    for idx in indices:
        waveform, target_tokens = dataset[idx]
        src = waveform.unsqueeze(0).to(device)
        if decode_method == 'beam' and beam_width > 1:
            pred_ids = beam_search_decode(model, src, tokenizer, device, max_len=decode_max_len, beam_width=beam_width)
        else:
            pred_ids = greedy_decode(model, src, max_len=decode_max_len, tokenizer=tokenizer, device=device)
        pred_text = decode_tokens(pred_ids, tokenizer)
        target_text = decode_tokens(target_tokens.tolist(), tokenizer)
        wer = compute_wer(target_text, pred_text)
        total_wer += wer
        print("Целевой текст: ", target_text)
        print("Предсказание:  ", pred_text)
        print("WER: {:.2f}".format(wer))
        print("-------------")
    avg_wer = total_wer / num_samples
    print("Средняя WER на {} примерах: {:.2f}".format(num_samples, avg_wer))

######################################
# 14. Функция проверки чтения аудио
######################################
def test_audio_reading(dataset, num_samples=5):
    print("Проверка чтения аудио:")
    for i in range(num_samples):
        waveform, _ = dataset[i]
        print(f"Образец {i}: форма аудио = {waveform.shape}")

######################################
# 15. Функция тестирования токенизатора
######################################
def test_tokenizer(tokenizer, text):
    token_ids = tokenizer(text)
    decoded_text = decode_tokens(token_ids, tokenizer)
    print("Исходный текст:   ", text)
    print("Список токенов:   ", token_ids)
    print("Декодированный текст: ", decoded_text)

######################################
# 16. Функция для построения графика loss
######################################
def plot_loss(loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("График Loss при обучении")
    plt.grid(True)
    plt.show()

######################################
# 17. Основная функция обучения и fine-tuning
######################################
def main():
    # Пути к данным
    tsv_file = "/Users/milanagaraeva/PycharmProjects/Diplom v2/data/dataset.tsv"
    audio_dir = "/Users/milanagaraeva/PycharmProjects/Diplom v2/data/clips"
    # Устанавливаем размер батча равным 300 для ускорения обучения (меньше итераций на эпоху)
    batch_size = 6
    learning_rate = 1e-4
    num_epochs_main = 200      # Основное обучение с Teacher Forcing
    num_epochs_finetune = 50   # Fine-tuning с Scheduled Sampling

    # Определяем устройство
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется CUDA:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size()
    sample_text = "Пример тестового текста для проверки токенизации."
    test_tokenizer(tokenizer, sample_text)

    # Создаем датасеты и проводим проверку аудио
    train_dataset = AudioDataset(tsv_file, audio_dir,
                                 transform=transform_train_fn if USE_AUGMENTATION else transform_eval_fn,
                                 tokenizer=tokenizer)
    eval_dataset = AudioDataset(tsv_file, audio_dir, transform=transform_eval_fn, tokenizer=tokenizer)
    test_audio_reading(train_dataset, num_samples=5)

    # DataLoader с persistent_workers для ускорения загрузки
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8, persistent_workers=True,
                              pin_memory=(device.type in ["cuda", "mps"]))

    input_dim = 64  # Размерность мел-спектрограммы
    model = SpeechRecognitionTransformer(input_dim, vocab_size, d_model=256, nhead=4,
                                         num_encoder_layers=2, num_decoder_layers=2,
                                         dim_feedforward=512, dropout=0.1).to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    loss_history = []
    print("Начало основного обучения с Teacher Forcing...")
    for epoch in range(num_epochs_main):
        start_time = time.time()
        loss = train_teacher_forcing(model, train_loader, optimizer, criterion, device)
        loss_history.append(loss)
        epoch_time = time.time() - start_time
        scheduler.step(loss)
        print(f"Основное обучение: Эпоха {epoch+1}/{num_epochs_main}, Loss: {loss:.4f}, Время: {epoch_time:.2f} сек")
    plot_loss(loss_history)
    torch.save(model, "trained_model.pth")
    print("Модель после основного обучения сохранена в trained_model.pth")

    print("Начало fine-tuning с Scheduled Sampling...")
    finetune_lr = 1e-5
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    loss_history_ft = []
    for epoch in range(num_epochs_finetune):
        start_time = time.time()
        loss = train_scheduled_sampling(model, train_loader, optimizer_ft, criterion, device,
                                        teacher_forcing_ratio=0.5, scheduled_sampling_prob=0.2)
        loss_history_ft.append(loss)
        epoch_time = time.time() - start_time
        print(f"Fine-tuning: Эпоха {epoch+1}/{num_epochs_finetune}, Loss: {loss:.4f}, Время: {epoch_time:.2f} сек")
    plot_loss(loss_history_ft)
    torch.save(model, "finetuned_model.pth")
    print("Модель после fine-tuning сохранена в finetuned_model.pth")

    # Оценка модели (смена decode_method: "beam", "sampling" или "greedy")
    evaluate(model, eval_dataset, tokenizer, device, num_samples=10, decode_max_len=100,
             beam_width=5, decode_method='beam')

if __name__ == "__main__":
    main()