import os
import time

# Setting agar tidak macet (Wajib di paling atas)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

print("1. Mulai Import Library...")
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
print("Library Terload.")

MODEL_PATH = "ai_models/Breast_Cancer.h5"

if not os.path.exists(MODEL_PATH):
    print("ERROR: File model tidak ditemukan!")
    exit()

print(f"2. Mencoba load model dari: {MODEL_PATH}")
print("   (Proses ini biasanya butuh 10-30 detik di CPU...)")

start_time = time.time()

try:
    # Kita coba load
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'KerasLayer': hub.KerasLayer},
        compile=False # Tips: compile=False bikin loading jauh lebih cepat
    )
    
    end_time = time.time()
    durasi = end_time - start_time
    print(f"🎉 SUKSES! Model berhasil dimuat dalam {durasi:.2f} detik.")
    
    # Tes Prediksi Dummy
    print("3. Mencoba Test Prediksi (Dummy Data)...")
    dummy_img = np.random.rand(1, 299, 299, 3).astype(np.float32)
    pred = model.predict(dummy_img)
    print(f"✅ Prediksi Berhasil! Output shape: {pred.shape}")

except Exception as e:
    print("\nTERJADI ERROR FATAL:")
    print(e)