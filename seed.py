# seed.py
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import auth

# Pastikan tabel dibuat
models.Base.metadata.create_all(bind=engine)

def seed_database():
    db = SessionLocal()
    try:
        print("Memulai proses seeding data...")
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

        patients = [
            {"id_pasien_rs": "RM-2024-001", "nama": "Bapak Anton", "tanggal_lahir": "1980-05-12", "status_pasien": "Aktif", "jenis_kelamin": "Laki-laki"},
            {"id_pasien_rs": "RM-2024-002", "nama": "Ibu Siti", "tanggal_lahir": "1975-08-22", "status_pasien": "Aktif", "jenis_kelamin": "Perempuan"},
            {"id_pasien_rs": "RM-2024-003", "nama": "Bapak Rudi", "tanggal_lahir": "1990-11-03", "status_pasien": "Aktif", "jenis_kelamin": "Laki-laki"},
            {"id_pasien_rs": "RM-2024-004", "nama": "Ibu Ningsih", "tanggal_lahir": "1985-02-15", "status_pasien": "Aktif", "jenis_kelamin": "Perempuan"},
            {"id_pasien_rs": "RM-2024-005", "nama": "Saudara Kevin", "tanggal_lahir": "2000-07-30", "status_pasien": "Tidak Aktif", "jenis_kelamin": "Laki-laki"}
        ]

        for pasien in patients:
            existing_patient = db.query(models.Patient).filter(models.Patient.id_pasien_rs == pasien["id_pasien_rs"]).first()
            if not existing_patient:
                new_patient = models.Patient(
                    id_pasien_rs=pasien["id_pasien_rs"],
                    nama=pasien["nama"],
                    tanggal_lahir=pasien["tanggal_lahir"],
                    status_pasien=pasien["status_pasien"],
                    jenis_kelamin=pasien["jenis_kelamin"]
                )
                db.add(new_patient)
                print(f"Data Pasien '{pasien['nama']}' berhasil ditambahkan!")
            else:
                print(f"Data Pasien '{pasien['id_pasien_rs']}' sudah ada. Di-skip.")

        db.commit() # Simpan data pasien ke database
        
        print("Seeding Selesai! Semua data siap digunakan.")

    except Exception as e:
        print(f"Terjadi Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()