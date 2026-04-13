# ============================================================================
# Dockerfile - C++ Code Generation Model + Tokenizer Server
# ============================================================================
# 
# @author Евгений П.
# @date 2026
# @version 3.5.0
# ============================================================================

# ----------------------------------------------------------------------------
# Этап 1: Сборка C++ токенизатора и Crow веб-сервера
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    cmake \
    make \
    git \
    libssl-dev \
    zlib1g-dev \
    libboost-system-dev \
    libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

# Клонируем ASIO (зависимость Crow)
WORKDIR /build
RUN git clone --depth 1 --branch asio-1-28-0 https://github.com/chriskohlhoff/asio.git asio

# Клонируем CrowCpp
RUN git clone --depth 1 --branch v1.2.0 https://github.com/CrowCpp/Crow.git crow

# Копируем исходники C++ токенизатора
COPY bpe_tokenizer_cpu/bpe_cpp/ /build/bpe_cpp/

# Копируем ASIO в include директорию Crow
RUN cp -r /build/asio/asio/include/asio.hpp /build/crow/include/ && \
    cp -r /build/asio/asio/include/asio /build/crow/include/

# Сборка C++ токенизатора И Crow веб-сервера
WORKDIR /build/bpe_cpp
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_WEB_SERVER=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_CROW_SERVER=ON \
        -DCROW_INCLUDE_DIR=/build/crow/include \
        -DUSE_OPENMP=ON \
        -DUSE_SIMD=ON \
    && make -j$(nproc) bpe_tokenizer_cpp tokenizer_server_crow

# Проверяем что собралось
RUN ls -la /build/bpe_cpp/build/web/

# ----------------------------------------------------------------------------
# Этап 2: Финальный образ
# ----------------------------------------------------------------------------
FROM python:3.12-slim

LABEL maintainer="Евгений П."
LABEL description="C++ Code Generation Model + Tokenizer Server with Crow API"
LABEL version="3.3.6"

WORKDIR /app

# Установка runtime зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgomp1 \
    libssl3 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Копируем Python зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копируем собранные C++ файлы
COPY --from=builder /build/bpe_cpp/build/bpe_tokenizer_cpp.so /app/bpe_tokenizer_cpp.so
COPY --from=builder /build/bpe_cpp/build/web/bpe_server_crow /app/tokenizer_server
RUN chmod +x /app/tokenizer_server

ENV PYTHONPATH=/app:$PYTHONPATH

# ----------------------------------------------------------------------------
# Копирование файлов модели
# ----------------------------------------------------------------------------
COPY cpp_code_model/tokenizers/ /app/cpp_code_model/tokenizers/
COPY bpe_tokenizer_cpu/bpe_python/*.py /app/bpe_tokenizer_cpu/bpe_python/
COPY bpe_tokenizer_cpu/bpe_python/models/ /app/bpe_tokenizer_cpu/bpe_python/models/
COPY bpe_tokenizer_cpu/bpe_cpp/models/ /app/bpe_tokenizer_cpu/bpe_cpp/models/
COPY cpp_code_model/checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt /app/cpp_code_model/checkpoints/medium/
COPY cpp_code_model/checkpoints/lora_medium/lora_weights_epoch_5.pt /app/cpp_code_model/checkpoints/lora_medium/lora_weights_best.pt
COPY cpp_code_model/configs/medium.json /app/cpp_code_model/configs/
COPY cpp_code_model/model/ /app/cpp_code_model/model/
COPY cpp_code_model/scripts/final_server.py /app/
COPY bpe_tokenizer_cpu/bpe_cpp/web/swagger.html /app/swagger.html

# ----------------------------------------------------------------------------
# Конфигурация supervisord
# ----------------------------------------------------------------------------
RUN mkdir -p /var/log/supervisor /var/run

RUN echo '[supervisord]\n\
nodaemon=true\n\
logfile=/var/log/supervisor/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
user=root\n\
\n\
[program:model_server]\n\
command=python -u /app/final_server.py\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
startsecs=5\n\
stopwaitsecs=10\n\
stdout_logfile=/var/log/supervisor/model_server.log\n\
stderr_logfile=/var/log/supervisor/model_server.log\n\
environment=PYTHONUNBUFFERED="1",PROJECT_ROOT="/app/cpp_code_model",PYTHONPATH="/app:/app/cpp_code_model:/app/bpe_tokenizer_cpu/bpe_python",MODEL_PATH="/app/cpp_code_model/checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt",LORA_PATH="/app/cpp_code_model/checkpoints/lora_medium/lora_weights_best.pt",CONFIG_PATH="/app/cpp_code_model/configs/medium.json"\n\
\n\
[program:tokenizer_server]\n\
command=/app/tokenizer_server --port 8081\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
startsecs=3\n\
stopwaitsecs=5\n\
stdout_logfile=/var/log/supervisor/tokenizer_server.log\n\
stderr_logfile=/var/log/supervisor/tokenizer_server.log\n\
environment=LD_LIBRARY_PATH="/app"\n' > /etc/supervisor/conf.d/servers.conf

ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app/cpp_code_model
ENV PYTHONPATH=/app:/app/cpp_code_model:/app/bpe_tokenizer_cpu/bpe_python:$PYTHONPATH
ENV MODEL_PATH=/app/cpp_code_model/checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt
ENV LORA_PATH=/app/cpp_code_model/checkpoints/lora_medium/lora_weights_best.pt
ENV CONFIG_PATH=/app/cpp_code_model/configs/medium.json

EXPOSE 8080 8081

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/servers.conf"]