# Dockerfile
FROM ubuntu:22.04 as builder

# Установка зависимостей для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем исходники
WORKDIR /app
COPY . .

# Сборка C++ части
WORKDIR /app/bpe_cpp/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) tokenizer_server_crow

# Финальный образ
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем собранный сервер и модели
COPY --from=builder /app/bpe_cpp/build/web/tokenizer_server_crow /app/
COPY --from=builder /app/bpe_cpp/models /app/models

# Копируем Python часть
COPY --from=builder /app/bpe_python /app/bpe_python
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/requirements.txt /app/

# Устанавливаем Python зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Открываем порт
EXPOSE 8080

# Запуск сервера
CMD ["./tokenizer_server_crow", "--port", "8080"]