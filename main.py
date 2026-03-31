# main.py
from fastapi.security import OAuth2AuthorizationCodeBearer, OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import JSONResponse, FileResponse
from jose import JWTError, jwt
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from urllib.parse import unquote
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import tensorflow_hub as hub
import auth
import models, schemas
import shutil
import uuid
import io
import numpy as np
from PIL import Image
import pydantic

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Memory Growth Activated: {len(gpus)} GPUS(s)")
    except RuntimeError as e:
        print(f"GPU Error: {e}")

# Create Database Table
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi Folder Upload
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Load Model AI
print("Sedang memuat model AI")
model_ai = None
try:
    MODEL_PATH = "ai_models/Breast_Cancer.h5"

    if os.path.exists(MODEL_PATH):

        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Ukuran: {file_size:.2f} MB")

        model_ai = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={
                'KerasLayer': hub.KerasLayer
            },
            compile=False,
            safe_mode=False
        )
        print("Model AI Berhasil Dimuat!")
    else:
        print(f"File model tidak ditemukan di: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal Memuat Model: {e}")

# Dependency Database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail= "Token tidak valid",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Bantuan Logging
def save_log(db: Session, username: str, role: str, activity: str, details: str = ""):
    try:
        new_log = models.ActivityLog(
            username=username,
            role=role,
            activity=activity,
            details=details
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        print(f"Gagal menyimpan log: {e}")

# ENDPOINT LOGIN
@app.post("/token/")
async def login_for_access_token(from_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == from_data.username).first()
    if not user or not auth.verify_password(from_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Username atau Password Salah")
    
    access_token = auth.create_access_token(
        data={"sub": user.username, "role": user.role, "id": user.id}
    )
    save_log(db, user.username, user.role, "Login", "Login Berhasil")
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

# ENDPOINT LOGOUT
@app.post("/logout/")
def logout(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    save_log(db, current_user.username, current_user.role, "Logout", "Logout dari sistem")
    return {"message": "Berhasil logout"}

# ENDPOINT ADMIN: MANAJEMEN USER
# Buat User Baru
@app.post("/users/", response_model=schemas.UserResponse)
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username sudah terdaftar")
    
    hashed_pw = auth.get_password_hash(user.password)

    new_user = models.User(
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        hashed_password=hashed_pw
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    save_log(db, current_user.username, current_user.role, "Create User", f"Membuat user: {user.username} ({user.role})")

    return new_user

# Edit User
@app.put("/users/{user_id}", response_model=schemas.UserResponse)
def update_user(user_id: int, user_update: schemas.UserCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):

    print(f"Data Mentah dari Flutter: {user_update.dict()}")

    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    
    db_user.username = user_update.username
    db_user.full_name = user_update.full_name
    db_user.role = user_update.role

    if user_update.password and user_update.password.strip():
        db_user.hashed_password = auth.get_password_hash(user_update.password)
    
    print(f"Avatar Lama di DB: {db_user.avatar}")
    print(f"Avatar Baru dari Flutter: {user_update.avatar}")

    if user_update.avatar is not None:
        db_user.avatar = user_update.avatar
        print("STATUS: Avatar Berhasil diupdate")
    else:
        print("STATUS: Avatar dari Flutter kosong(None).")
    
    db.commit()
    db.refresh(db_user)

    save_log(db, current_user.username, current_user.role, "Edit User", f"Mengedit user ID: {user_id}")

    return db_user

# Hapus User
@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    
    if db_user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Tidak dapat menghapus akun sendiri")
    
    target_username = db_user.username
    db.delete(db_user)
    db.commit()

    save_log(db, current_user.username, current_user.role, "Delete User", f"Menghapus user: {target_username}")

    return {"detail": "User berhasil dihapus"}

# User Me (Profile)
@app.get("/users/me/", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

# Lihat Semua User
@app.get("/users/", response_model=List[schemas.UserResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    users = db.query(models.User).offset(skip).limit(limit).all()

    return users

# ADMIN: MANAJEMEN DATA PASIEN
# Lihat Semua Pasien
@app.get("/patients/", response_model=List[schemas.PatientResponse])
def read_patient(db: Session = Depends(get_db)):
    patients = db.query(models.Patient).order_by(models.Patient.id.asc()).all()

    return patients

# Tambah Pasien Baru
@app.post("/patients/", response_model=schemas.PatientResponse)
def create_patient(patient: schemas.PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(models.Patient).filter(models.Patient.id_pasien_rs == patient.id_pasien_rs).first()
    if existing:
        raise HTTPException(status_code=400, detail="ID Pasien (RM) Sudah terdaftar")
    
    new_patient = models.Patient(
        id_pasien_rs=patient.id_pasien_rs,
        nama=patient.nama,
        tanggal_lahir=patient.tanggal_lahir,
        jenis_kelamin=patient.jenis_kelamin,
        status_pasien=patient.status_pasien
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    save_log(db, current_user.username, current_user.role, "Create Patient", f"Menambah pasien baru: {new_patient.nama} (RM: {new_patient.id_pasien_rs})")
    
    return new_patient

# Update Pasien
@app.put("/patients/{patient_id}", response_model=schemas.PatientResponse)
def update_patient(patient_id: int, patient_update: schemas.PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not db_patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan")
    
    old_name = db_patient.nama
    old_status = db_patient.status_pasien
    
    db_patient.id_pasien_rs = patient_update.id_pasien_rs
    db_patient.nama = patient_update.nama
    db_patient.tanggal_lahir = patient_update.tanggal_lahir
    db_patient.jenis_kelamin = patient_update.jenis_kelamin
    db_patient.status_pasien = patient_update.status_pasien

    db.commit()
    db.refresh(db_patient)

    detail_msg = f"Update data pasien: {db_patient.nama} (RM: {db_patient.id_pasien_rs})"

    if old_status != db_patient.status_pasien:
        detail_msg += f" | Status ubah: {old_status} -> {db_patient.status_pasien}"

    save_log(db, current_user.username, current_user.role, "Edit Patient", detail_msg)

    return db_patient

# ENDPOINT ADMIN: MONITORING LOG
@app.get("/logs/", response_model=List[schemas.LogResponse])
def get_logs(role: str = None, start_date: str = None, end_date: str = None, db: Session = Depends(get_db)):
    query = db.query(models.ActivityLog)

    if role and role != "Semua":
        query = query.filter(models.ActivityLog.role == role)

    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

            query = query.filter(models.ActivityLog.timestamp >= start)
            query = query.filter(models.ActivityLog.timestamp <= end)
        except ValueError:
            pass
    logs = query.order_by(models.ActivityLog.timestamp.desc()).all()
    return logs

# Upload Foto Profil
@app.post("/users/upload-avatar/")
async def upload_avatar(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):

    print(f"Filename: {file.filename}")
    print(f"Content-Type: {file.content_type}")
    
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print(f"ERROR Save: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {str(e)}")

    avatar_url = f"static/{unique_filename}"

    current_user.avatar = avatar_url
    db.commit()
    db.refresh(current_user)

    save_log(db, current_user.username, current_user.role, "Update Profile", "Mengganti foto profil")
    return {"message": "Foto profil berhasil diupdate", "url": avatar_url} 

# ENDPOINT UPLOAD MRI
@app.post("/upload-mri/")
async def upload_mri_smart(
    nama: str = Form(...),
    id_pasien: str = Form(...),
    tgl_lahir: str = Form(...),
    status: str = Form(...),
    jenis_mri: str = Form(...),
    catatan: str = Form(default="-"),
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # LOGIKA DATABASE PASIEN
    pasien_db = db.query(models.Patient).filter(models.Patient.id_pasien_rs == id_pasien).first()

    if not pasien_db:
        pasien_db = models.Patient(
            nama=nama,
            id_pasien_rs=id_pasien,
            tanggal_lahir=tgl_lahir,
            status_pasien=status
        )
        db.add(pasien_db)
        db.commit()
        db.refresh(pasien_db)

    else:
        pasien_db.nama = nama
        pasien_db.status_pasien = status
        db.commit()
    
    # SIMPAN FILE 
    original_name = file.filename

    clean_name = original_name.replace(" ", "_").replace("(", "").replace(")", "")

    filename_server = f"{pasien_db.id}_{int(datetime.now().timestamp())}_{clean_name}"

    print(f"File Asli: {original_name} > Jadi: {filename_server}")

    file_path = os.path.join(UPLOAD_DIR, filename_server)

    contents = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # SIMPAN DATA SCAN
    hasil_prediksi = "Belum Dianalisis"
    confidence_score = 0

    if model_ai:
        try:
            # Buka gambar dari memory & convert ke RGB
            img = Image.open(io.BytesIO(contents)).convert('RGB')

            # Resize ke 299 x 299
            img = img.resize((299, 299))

            # Ubah ke array & normalisasi
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model_ai.predict(img_array)
            score = float(prediction[0][0])

            print(f"Debug AI Score: {score}")

            # Jika Mendekati 1 = Cancer, mendekati 0 = Non-Cancer
            if score > 0.5:
                hasil_prediksi = "Cancer"
                confidence_score = int(score * 100)

            else:
                hasil_prediksi = "Non-Cancer"
                confidence_score = int((1 - score) * 100)
        
        except Exception as e:
            print(f"Error AI: {e}")
            hasil_prediksi = "Error Analisis AI"

    else:
        hasil_prediksi = "Model AI Tidak Siap"
    
    new_scan = models.MRIScan(
        patient_id=pasien_db.id,
        jenis_mri=jenis_mri,
        catatan_teknis=catatan,
        filepath=file_path,
        filename=filename_server,
        hasil_prediksi=hasil_prediksi,
        confidence=confidence_score,
    )
    db.add(new_scan)
    db.commit()

    save_log(db, current_user.username, current_user.role, "Upload MRI", f"Upload scan untuk pasien: {nama} ({hasil_prediksi})")

    return {"status": "sukses", "pesan": "Data tersimpan & Analisis selesai", "hasil_ai": hasil_prediksi, "Confidence": confidence_score}

# ENDPOINT RIWAYAT
@app.get("/riwayat-semua/")
def get_all_history(db: Session = Depends(get_db)):
    scans = db.query(models.MRIScan).order_by(models.MRIScan.upload_date.desc()).all()

    results = []
    for scan in scans:
        tgl_cantik = scan.upload_date.strftime("%d/%m/%Y")

        nama_pasien = scan.patient.nama if scan.patient else "Tanpa Nama"
        id_rm = scan.patient.id_pasien_rs if scan.patient else "-"

        results.append({
            "id": scan.id,
            "jenis_mri": scan.jenis_mri,
            "tanggal_periksa": tgl_cantik,
            "hasil_prediksi": scan.hasil_prediksi,
            "nama_pasien": nama_pasien,
            "id_rm": id_rm
        })
    return results

@app.get("/analisis/{analysis_id}")
def get_analysis_detail(analysis_id: int, db: Session = Depends(get_db)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    
    if not scan:
        raise HTTPException(status_code=404, detail="Data MRI tidak ditemukan")
    
    waktu_scan = scan.upload_date.strftime("%d/%m/%Y • %H:%M WIB")

    # 4. Return Data
    return {
        "id": scan.id,
        "image_url": f"/static/{scan.filename}", 
        "result": scan.hasil_prediksi,
        "confidence": scan.confidence,
        "waktu_scan": waktu_scan,
        "nama_pasien": scan.patient.nama if scan.patient else "Tanpa Nama",
        "id_rm": scan.patient.id_pasien_rs if scan.patient else "-",
        "tgl_lahir": scan.patient.tanggal_lahir if scan.patient else "-",
        "jenis_kelamin": scan.patient.jenis_kelamin if scan.patient else "-",
        "notes_radiolog": scan.catatan_teknis,
        "notes_dokter": getattr(scan, "catatan_dokter", "Belum ada catatan dokter") 
    }

@app.get("/get-image/{filename}")
async def get_image_manual(filename: str):

    file_path = os.path.join(UPLOAD_DIR, filename)
    
    print(f"Request Gambar: {filename}")

    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    
    print(f"File tidak ditemukan: {file_path}")
    return {"error": "File tidak ditemukan"}

# ENDPOINT UPDATE CATATAN DOKTER
@app.put("/analisis/{analysis_id}/update-notes/")
async def update_doctor_notes(analysis_id: int, data: dict, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):

    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()

    if not scan:
        raise HTTPException(status_code=404, detail="Data MRI tidak ditemukan")
    
    new_notes = data.get("notes_dokter", "")
    scan.catatan_dokter = new_notes

    db.commit()
    db.refresh(scan)
    save_log(db, current_user.username, current_user.role, "Update Notes", f"Update catatan dokter untuk Scan ID: {analysis_id}")
    return {"status": "sukses", "message": "Catatan berhasil diperbarui", "data": new_notes}

# ENDPOINT DASHBOARD SUMMARY
@app.get("/dashboard-summary/", response_model=schemas.DashboardSummary)
def get_summary(db: Session = Depends(get_db)):
    total_p = db.query(models.Patient).count()
    menunggu = db.query(models.MRIScan).filter(models.MRIScan.hasil_prediksi == "Belum Dianalisis").count()
    selesai = db.query(models.MRIScan).filter(models.MRIScan.hasil_prediksi != "Belum Dianalisis").count()

    return {
        "total_pasien": total_p,
        "total_menunggu": menunggu,
        "total_selesai": selesai
    }