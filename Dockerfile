# У себя я использовал python версии 3.11, поэтому и здесь указал его же
FROM python:3.11-slim

# Устанавливаем зависимости для операционной системы
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файлы приложения в контейнер
COPY . .

# Команда запуска uvicorn при старте контейнера
CMD ["uvicorn", "classifier_fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
