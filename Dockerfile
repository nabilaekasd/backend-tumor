# Gunakan versi Python yang ringan (versi 3.10 sangat stabil untuk AI)
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Instal dependensi sistem
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements
COPY requirements.txt .

# Instal semua library Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin sisa kode backend
COPY . .

# ... (kode Dockerfile kamu sebelumnya) ...

RUN cp nnUNetTrainerV2_MedNeXt_M_ChooseOpt.py /usr/local/lib/python3.10/site-packages/nnunet/training/network_training/ 2>/dev/null || true
RUN cp nnUNetTrainerV2_MedNeXt_M_ChooseOpt.py /usr/local/lib/python3.10/site-packages/nnunet_mednext/training/network_training/ 2>/dev/null || true

# Ekspos port 8000 agar bisa diakses dari luar
EXPOSE 8000

# Perintah untuk menjalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]