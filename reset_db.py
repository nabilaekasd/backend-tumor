# reset_db.py
from database import engine
from models import Base
import models # Pastikan models terimport agar tabel terbaca

print("⏳ Sedang menghapus semua tabel lama...")
try:
    # Hapus semua tabel (Drop All)
    Base.metadata.drop_all(bind=engine)
    print("✅ Tabel lama berhasil dihapus.")
except Exception as e:
    print(f"⚠️ Warning saat hapus: {e}")

print("⏳ Sedang membuat tabel baru (dengan kolom catatan_dokter)...")
try:
    # Buat ulang tabel (Create All)
    Base.metadata.create_all(bind=engine)
    print("🎉 SUKSES! Database sudah diperbarui.")
except Exception as e:
    print(f"❌ Gagal membuat tabel: {e}")