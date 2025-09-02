// script.js

document.addEventListener('DOMContentLoaded', () => {
  const ALLOWED_EXT = ['.mp3','.wav','.ogg','.flac','.aac','.m4a','.wma'];

  const uploadArea         = document.getElementById('uploadArea');
  const fileInput          = document.getElementById('fileInput');
  const fileName           = document.getElementById('fileName');
  const recordBtn          = document.getElementById('recordBtn');
  const sendBtn            = document.getElementById('sendBtn');
  const audioSaveContainer = document.getElementById('audioSaveContainer');
  const saveAudioBtn       = document.getElementById('saveAudioBtn');
  const resultBox          = document.getElementById('resultBox');
  const resultText         = document.getElementById('resultText');
  const resultControls     = document.getElementById('resultControls');
  const copyBtn            = document.getElementById('copyBtn');
  const downloadWordBtn    = document.getElementById('downloadWordBtn');
  const howItems           = document.querySelectorAll('.how-item');

  let selectedFile = null;
  let mediaRecorder, audioChunks = [];
  let isProcessed = false;  // флаг, что результат уже показан

  function setActiveStep(n) {
    howItems.forEach(el => el.classList.toggle('active', el.dataset.step === String(n)));
  }

  function resetCopy() {
    copyBtn.disabled = true;
    downloadWordBtn.disabled = true;
    copyBtn.textContent = 'Копировать';
  }

  function resetAll() {
    selectedFile = null;
    isProcessed = false;
    fileName.textContent = 'Файл не выбран';
    sendBtn.textContent = 'Преобразовать в текст';
    sendBtn.disabled = true;
    resultBox.classList.remove('active');
    resultText.textContent = '';
    resultControls.style.display = 'none';
    resetCopy();
    setActiveStep(0);
    audioSaveContainer.style.display = 'none';
    saveAudioBtn.disabled = true;
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      recordBtn.textContent = 'Начать запись';
    }
  }

  function showError(msg) {
    resultText.textContent = msg;
    resultBox.classList.add('active');
    resultControls.style.display = 'none';
    setActiveStep(0);
  }

  function validateFormat(file) {
    const name = file.name.toLowerCase();
    if (!ALLOWED_EXT.some(ext => name.endsWith(ext))) {
      alert('Поддерживаются только аудио форматы: ' + ALLOWED_EXT.join(', '));
      return false;
    }
    return true;
  }

  function handleFile(file, isRecording) {
    if (!isRecording && !validateFormat(file)) return;

    selectedFile = file;
    isProcessed = false;
    fileName.textContent = file.name;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Преобразовать в текст';
    resultBox.classList.remove('active');
    resetCopy();
    setActiveStep(1);

    if (isRecording) {
      audioSaveContainer.style.display = 'flex';
      saveAudioBtn.disabled = false;
    } else {
      audioSaveContainer.style.display = 'none';
      saveAudioBtn.disabled = true;
    }
  }

  // Выбор файла и drag&drop
  uploadArea.addEventListener('click', () => fileInput.click());
  uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
  });
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, false);
  });
  fileInput.addEventListener('change', () => {
    const f = fileInput.files[0];
    if (f) handleFile(f, false);
  });

  // Запись с микрофона
  recordBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      recordBtn.textContent = 'Начать запись';
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const file = new File([blob], 'recording.webm', { type: blob.type });
        handleFile(file, true);
      };
      mediaRecorder.start();
      recordBtn.textContent = 'Остановить запись';
      setActiveStep(1);
      resultBox.classList.remove('active');
    } catch {
      showError('Не удалось получить доступ к микрофону');
    }
  });

  // Сохранить записанное
  saveAudioBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    const url = URL.createObjectURL(selectedFile);
    const a   = document.createElement('a');
    a.href    = url;
    a.download = selectedFile.name;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Отправка/сброс
  sendBtn.addEventListener('click', async () => {
    if (isProcessed) {
      resetAll();
      return;
    }

    // пустой файл
    if (!selectedFile || selectedFile.size === 0) {
      resultText.textContent = 'Файл пустой. Пожалуйста, загрузите или запишите аудио.';
      resultBox.classList.add('active');
      isProcessed = true;
      sendBtn.textContent = 'Загрузить новый файл';
      sendBtn.disabled = false;
      return;
    }

    setActiveStep(2);
    sendBtn.disabled = true;
    sendBtn.textContent = 'Обрабатываю…';
    resultText.textContent = '';
    resultControls.style.display = 'none';
    resetCopy();

    try {
      const form = new FormData();
      form.append('file', selectedFile);
      const resp = await fetch('http://127.0.0.1:8000/transcribe', {
        method: 'POST',
        body: form
      });

      if (!resp.ok) {
        const raw = await resp.text();
        // если модель вернула "No speech detected" — дружелюбно
        try {
          const j = JSON.parse(raw);
          if (j.detail === 'No speech detected') {
            resultText.textContent = 'Не удалось распознать аудио.';
            resultBox.classList.add('active');
            isProcessed = true;
            sendBtn.textContent = 'Загрузить новый файл';
            sendBtn.disabled = false;
            return;
          }
        } catch { /* не JSON или другая ошибка */ }
        throw new Error(raw);
      }

      const { transcription } = await resp.json();

      if (!transcription.trim()) {
        resultText.textContent = 'Не удалось распознать аудио.';
        resultBox.classList.add('active');
        isProcessed = true;
        sendBtn.textContent = 'Загрузить новый файл';
        sendBtn.disabled = false;
        return;
      }

      // Успешно показали текст
      resultText.textContent = transcription;
      resultBox.classList.add('active');
      setActiveStep(3);
      resultControls.style.display = 'flex';
      copyBtn.disabled = false;
      downloadWordBtn.disabled = false;

      isProcessed = true;
      sendBtn.textContent = 'Загрузить новый файл';
      sendBtn.disabled = false;

    } catch (err) {
      showError(`Ошибка: ${err.message}`);
      isProcessed = true;
      sendBtn.textContent = 'Загрузить новый файл';
      sendBtn.disabled = false;
    }
  });

  // Копировать в буфер
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(resultText.textContent);
      copyBtn.textContent = 'Скопировано!';
      setTimeout(() => copyBtn.textContent = 'Копировать', 1500);
    } catch {}
  });

  // Скачать как Word
  downloadWordBtn.addEventListener('click', () => {
    const text = resultText.textContent
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Транскрипция</title></head>
<body><pre style="font-family:Arial; font-size:14px;">${text}</pre></body>
</html>`;
    const blob = new Blob([html], { type: 'application/msword' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'transcription.doc';
    a.click();
    URL.revokeObjectURL(url);
  });

  // Инициализация
  resetAll();
  setActiveStep(0);
});