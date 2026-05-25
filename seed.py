# seed.py
import time
import random
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import auth

models.Base.metadata.create_all(bind=engine)

def generate_rm_id():
    """Fungsi untuk meng-generate ID Rekam Medis otomatis (Contoh: RM-582910)"""
    return f"RM-{random.randint(100000, 999999)}"

def seed_database():
    db = SessionLocal()
    try:
        print("Memulai proses seeding data...")

        # 1. SEEDING USER STAFF
        staff_users = [
            {"username": "admin", "full_name": "Administrator Utama", "role": "admin", "password": "admin123"},
            {"username": "dokter", "full_name": "Dr. Andi (Sp.S)", "role": "dokter", "password": "password123"},
            {"username": "radiolog", "full_name": "Dr. Budi (Sp.Rad)", "role": "radiolog", "password": "password123"}
        ]

        for staff in staff_users:
            existing_user = db.query(models.User).filter(models.User.username == staff["username"]).first()
            if not existing_user:
                hashed_pw = auth.get_password_hash(staff["password"])
                new_user = models.User(
                    username=staff["username"],
                    full_name=staff["full_name"],
                    role=staff["role"],
                    hashed_password=hashed_pw
                )
                db.add(new_user)
                print(f"Akun Staff '{staff['username']}' ({staff['role']}) berhasil dibuat!")
            else:
                print(f"Akun Staff '{staff['username']}' sudah ada. Di-skip.")

        db.commit()

        # 2. SEEDING PASIEN (ID OTOMATIS)
        patients = [
            {"nama": "Anton", "tanggal_lahir": "1980-05-12", "status_pasien": "Aktif", "jenis_kelamin": "Laki-laki"},
            {"nama": "Siti", "tanggal_lahir": "1975-08-22", "status_pasien": "Aktif", "jenis_kelamin": "Perempuan"},
            {"nama": "Rudi", "tanggal_lahir": "1990-11-03", "status_pasien": "Aktif", "jenis_kelamin": "Laki-laki"},
            {"nama": "Ningsih", "tanggal_lahir": "1985-02-15", "status_pasien": "Aktif", "jenis_kelamin": "Perempuan"},
            {"nama": "Kevin", "tanggal_lahir": "2000-07-30", "status_pasien": "Tidak Aktif", "jenis_kelamin": "Laki-laki"}
        ]

        for pasien in patients:
            existing_patient = db.query(models.Patient).filter(models.Patient.nama == pasien["nama"]).first()
            
            if not existing_patient:
                generated_id = generate_rm_id() # Buat ID otomatis
                
                new_patient = models.Patient(
                    id_pasien_rs=generated_id,
                    nama=pasien["nama"],
                    tanggal_lahir=pasien["tanggal_lahir"],
                    status_pasien=pasien["status_pasien"],
                    jenis_kelamin=pasien["jenis_kelamin"]
                )
                db.add(new_patient)
                print(f"Data Pasien '{pasien['nama']}' berhasil ditambahkan dengan ID {generated_id}!")
            else:
                print(f"Data Pasien '{pasien['nama']}' sudah ada. Di-skip.")

        db.commit()
        
        print("Seeding Selesai! Semua data siap digunakan.")

    except Exception as e:
        print(f"Terjadi Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()