FROM python:3.10-slim

# 1. Cài đặt dependencies - Thay execstack bằng patchelf
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 3. FIX LỖI BẢO MẬT BẰNG PATCHELF
# Lệnh --clear-execstack tương đương với execstack -c, dùng để tắt quyền thực thi trên stack
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime/capi/ -name "onnxruntime_pybind11_state.*.so" \
    -exec patchelf --clear-execstack {} \;

COPY . .

# Create data directory for persistent storage (can be mounted by docker-compose)
RUN mkdir -p /app/data && chmod 755 /app/data

RUN mkdir -p /root/.insightface

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]