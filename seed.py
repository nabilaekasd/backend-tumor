# seed.py
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import auth

# Pastikan tabel dibuat
models.Base.metadata.create_all(bind=engine)

def create_super_admin():
    db = SessionLocal()
    try:
        existing_admin = db.query(models.User).filter(models.User.username == "admin").first()

        if existing_admin:
            print("User 'Admin' sudah ada di database.")
            return
        print("Sedang membuat user 'Admin'")

        # Enkripsi Password
        hashed_pw = auth.get_password_hash("admin123")

        admin = models.User(
            username="admin",
            full_name="Administrator Utama",
            role="admin",
            hashed_password=hashed_pw
        )
        db.add(admin)
        db.commit()

        print("User Admin berhasil dibuat!")
    
    except Exception as e:
        print(f"Terjasi Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_super_admin()