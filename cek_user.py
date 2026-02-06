from database import SessionLocal
from models import User

def cek_semua_user():
    db = SessionLocal()
    try:
        # Ambil semua user
        users = db.query(User).all()
        print("-" * 50)
        print(f"{'ID':<5} {'USERNAME':<20} {'ROLE':<15} {'STATUS'}")
        print("-" * 50)
        
        found_target = False
        for user in users:
            print(f"{user.id:<5} {user.username:<20} {user.role:<15} {user.is_active}")
            
            # Auto-Fix jika ketemu yang namanya 'admin' tapi role 'dokter'
            # GANTI 'admin' DI BAWAH SESUAI USERNAME ANDA
            if user.username == "admin" and user.role != "admin":
                print(f"   >>> MEMPERBAIKI {user.username} MENJADI ADMIN...")
                user.role = "admin"
                found_target = True

        if found_target:
            db.commit()
            print("\n[SUKSES] Data berhasil diperbarui ke database!")
        else:
            print("\n[INFO] Tidak ada perubahan yang dilakukan.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    cek_semua_user()