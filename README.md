# Offline/Online Speech-to-Text App (Electron + FastAPI)

## Описание
Приложение для распознавания речи в текст с возможностью работы **полностью оффлайн** или в **онлайн-режиме**.  
Интерфейс сделан на **Electron**, backend - на **FastAPI (Python)**.  
Движок распознавания: **Vosk**.

---

## Запуск проекта

### 1. Backend (Python)
Необходимо перейти в папку backend и установить зависимости:
```bash
cd python_backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 2. Frontend (Electron)
```bash
cd electron
npm install
npm start
```

### 3. Зависимости
Python: см. python_backend/requirements.txt

Node.js: см. electron/package.json

### 4. Модели

Модели не хранятся в репозитории.

Для Vosk: необходимо скачать модель vosk-model-ru-0.42 и положи в python_backend/vosk-model-ru-0.42/.


 ### Разработка
	•	IDE: PyCharm
	•	GitHub для версионирования
	•	Рекомендуемая ОС: macOS/Linux/Windows
