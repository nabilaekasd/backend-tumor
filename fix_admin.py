from database import SessionLocal
from models import User
import auth

USERNAME_TARGET = "admin"

def restore_admin():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == USERNAME_TARGET).first()
        if user:
            print(f"User ditemukan: {user.username}, Role saat ini: {user.role}")
            user.role = "admin" # Paksa jadi admin
            db.commit()
            print("SUKSES! User kembali menjadi ADMIN.")
        else:
            print("User tidak ditemukan!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    restore_admin()