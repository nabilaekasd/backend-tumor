# main.py
from fastapi.security import OAuth2AuthorizationCodeBearer, OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import JSONResponse, FileResponse
from jose import JWTError, jwt
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from urllib.parse import unquote
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from datetime import datetime
from sqlalchemy import func
import os
import zipfile
import auth
import models, schemas
import shutil
import uuid
import io
from PIL import Image
import pydantic
import subprocess
import nibabel as nib
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import gaussian_filter
import torch
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDNEXT_DIR = os.path.join(BASE_DIR, "backend_mednext")

os.environ["nnUNet_raw_data_base"] = os.path.join(MEDNEXT_DIR, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(MEDNEXT_DIR, "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = os.path.join(MEDNEXT_DIR, "nnUNet_results")
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

INPUT_DIR = os.path.join(os.environ["nnUNet_raw_data_base"], "temp_input")
OUTPUT_DIR = os.path.join(os.environ["RESULTS_FOLDER"], "temp_output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
def update_user(user_id: int, user_update: schemas.UserUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):

    print(f"Data Mentah dari Flutter: {user_update.dict()}")

    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    if user_update.username is not None:
        db_user.username = user_update.username
    
    if user_update.full_name is not None:
        db_user.full_name = user_update.full_name

    if user_update.role is not None:
        db_user.role = user_update.role
    
    if user_update.password and user_update.password.strip():
        db_user.hashed_password = auth.get_password_hash(user_update.password)

    if user_update.avatar is not None:
        db_user.avatar = user_update.avatar
    
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

# LOGIC 2D & 3D
def generate_visualizations_logic(mri_path_2d, mri_path_3d, pred_path, out_2d, out_3d, case_id):
    print(f"Mulai merender visualisasi untuk {case_id}")

    #Load Data 2D
    mri_2d = nib.load(mri_path_2d).get_fdata().astype(np.float32)
    pred_2d = nib.load(pred_path).get_fdata().astype(np.uint8)

    # Load Data 3D
    mri_3d = nib.as_closest_canonical(nib.load(mri_path_3d)).get_fdata().astype(np.float32)
    pred_3d = nib.as_closest_canonical(nib.load(pred_path)).get_fdata().astype(np.uint8)

    # Render 2D
    def norm01(x):
        v = x[np.isfinite(x)]
        lo, hi = np.percentile(v, [2, 98]) if v.size > 0 else (0.0, 1.0)
        if hi <= lo: hi = lo + 1e-8
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

    def best_slice(mask, axis):
        counts = np.sum(mask > 0, axis=tuple(i for i in range(3) if i != axis))
        return int(np.argmax(counts)) if counts.max() > 0 else mask.shape[axis] // 2

    def take_slice(vol, axis, idx):
        if axis == 0: return vol[idx, :, :] # sagittal
        if axis == 1: return vol[:, idx, :] # coronal
        if axis == 2: return vol[:, :, idx] # transversal

    views = {"Sagittal": 0, "Coronal": 1, "Transversal": 2}
    colors_hex = {
        1: "#e41a1c", # NETC
        2: "#377eb8", # SNFH
        3: "#4daf4a", # ET
        4: "#984ea3", # RC
    }
    mask_cmap = ListedColormap(["none"] + [colors_hex[1], colors_hex[2], colors_hex[3], colors_hex[4]])

    fig_2d, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=150)

    for i, (name, axis) in enumerate(views.items()):
        idx = best_slice(pred_2d, axis)
        mri_s = norm01(take_slice(mri_2d, axis, idx))
        pred_s = take_slice(pred_2d, axis, idx)

        axs[i].imshow(mri_s, cmap="gray", interpolation="nearest")
        axs[i].imshow(pred_s, cmap=mask_cmap, alpha=0.65, vmin=0, vmax=4, interpolation="nearest")
        axs[i].set_title(f"Prediction {name}")
        axs[i].axis("off")

    plt.tight_layout(pad=0.6, w_pad=0.2, h_pad=0.4)
    plt.savefig(out_2d, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_2d)

    # Render 3D
    ds = 2
    mri01 = norm01(mri_3d)
    mri_ds = mri01[::ds, ::ds, ::ds]

    colors_3d = {1:("NETC","#e41a1c"), 2:("SNFH","#377eb8"), 3:("ET","#4daf4a"), 4:("RC","#984ea3")}
    op_map = {1:0.95, 2:0.05, 3:0.90, 4:0.15}
    draw_order = [2, 4, 3, 1]

    fig_3d = go.Figure()

    # Otak Transparan
    brain = mri_ds > 0.1
    if brain.sum() > 0:
        brain_smooth = gaussian_filter(brain.astype(float), sigma=1.2)
        v, f, _, _ = measure.marching_cubes(brain_smooth, level=0.5)
        fig_3d.add_trace(go.Mesh3d(
            x=v[:,0] * ds, y=v[:,1] * ds, z=v[:,2] * ds, i=f[:,0], j=f[:,1], k=f[:,2],
            color="lightgray", opacity=0.75, name="Brain", hoverinfo="skip"
        ))

    # Tumor (Tumpukan)
    for lbl in draw_order:
        name, col = colors_3d[lbl]
        bin_vol = (pred_3d == lbl).astype(np.uint8)
        if bin_vol.sum() > 0:
            v, f, _, _ = measure.marching_cubes(bin_vol, level=0.5)
            fig_3d.add_trace(go.Mesh3d(
                x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
                color=col, opacity=op_map[lbl], name=name
            ))
    
    axis_config = dict(
        showgrid=True,
        gridcolor='#444444',
        zerolinecolor='#444444',
        color='white',
        showbackground=False,
        title_font=dict(color='white'),
    )

    fig_3d.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        scene=dict(
            aspectmode="data",
            xaxis=dict(**axis_config, title="X"), 
            yaxis=dict(**axis_config, title="Y"), 
            zaxis=dict(**axis_config, title="Z"),
            bgcolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig_3d.write_html(out_3d, full_html=True, include_plotlyjs='cdn')
    print(f"Visualisasi 2D dan 3D untuk {case_id} berhasil disimpan!")

# MEDNEXT & VISUALISASI
def process_mri_ai(scan_id: int, input_dir: str, case_id: str):
    db = SessionLocal()
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == scan_id).first()

    if not scan:
        db.close()
        return

    print(f"[PROSES MULAI] AI sedang membedah pasien: {case_id}...")

    command = [
        "nnUNet_predict",
        "-i", input_dir,
        "-o", OUTPUT_DIR,
        "-t", "Task001_BraTS2024",
        "-m", "3d_fullres",
        "-f", "4",
        "-tr", "nnUNetTrainerV2_MedNeXt_M_ChooseOpt",
        "-p", "nnUNetPlansv2.1_trgSp_1x1x1",
        "--mode", "normal"
    ]

    try:
        print("Menunggu AI memproses gambar 3D")
        subprocess.run(command, check=True)
        print(f"AI Selesai Menebak {case_id}!")

        print(f"Membuat visualisasi 3D")

        pred_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")
        mri_path_2d = os.path.join(INPUT_DIR, f"{case_id}_0002.nii.gz")
        mri_path_3d = os.path.join(INPUT_DIR, f"{case_id}_0003.nii.gz")

        try:
            # Baca Hasil Prediksi
            pred_data = nib.load(pred_path).get_fdata()

            # Cari angka unik di dalamnya
            unique_labels = np.unique(pred_data).astype(int).tolist()

            # Hapus angka 0 karena 0 bukan tumor
            if 0 in unique_labels:
                unique_labels.remove(0)

            scan.detected_regions = json.dumps(unique_labels)
            print(f"[INFO WARNA] Region tumor yang terdeteksi: {unique_labels}")
        except Exception as e_color:
            print(f"[WARNING] Gagal mendeteksi warna dinamis: {e_color}")
            scan.detected_regions = json.dumps([1, 2, 3, 4])

        # Nama File untuk Flutter
        fname_2d = f"result_{scan_id}_2d.png"
        fname_3d = f"result_{scan_id}_3d.html"

        # Simpan ke folder uploads
        path_2d = os.path.join(UPLOAD_DIR, fname_2d)
        path_3d = os.path.join(UPLOAD_DIR, fname_3d)

        generate_visualizations_logic(mri_path_2d, mri_path_3d, pred_path, path_2d, path_3d, case_id)

        scan.hasil_prediksi = "Tumor Terdeteksi" #Nanti dibuat otomatis baca hasil
        scan.filepath_2d = f"static/{fname_2d}" # Path URL untuk Flutter
        scan.filepath_3d = f"static/{fname_3d}"

        # Kirim Notifikasi
        nama_pasien = scan.patient.nama if scan.patient else "Tanpa Nama"
        target_roles = ["Dokter", "Radiolog"]

        for target in target_roles:
            new_notif = models.Notification(
                target_role=target,
                title="Analisis AI Selesai!",
                message=f"Hasil pemindaian tumor 3D untuk pasien {nama_pasien} sudah keluar. Silakan cek riwayat.",
                analysis_id=scan.id
            )
            db.add(new_notif)
        db.commit()

    except Exception as e:
        print(f"[ERROR AI] Gagal memproses: {e}")
        scan.hasil_prediksi = "Gagal Diproses AI"
        db.commit()
    finally:
        db.close()

# ENDPOINT UPLOAD MRI
@app.post("/upload-mri/")
async def upload_mri_smart(
    background_tasks: BackgroundTasks,
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

    # ZIP File
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Harus file .zip yang berisi 4 modalitas MRI!")

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
    
    # EKSTRAK ZIP KE FOLDER MEDNEXT
    zip_path = os.path.join(MEDNEXT_DIR, "pasien_temp.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    for f in os.listdir(INPUT_DIR):
        os.remove(os.path.join(INPUT_DIR, f))

    # Ekstrak ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(INPUT_DIR)
    os.remove(zip_path)

    # CARI ID OTOMATIS DARI ISI ZIP
    extracted_files = os.listdir(INPUT_DIR)
    t1c_file = next((f for f in extracted_files if f.endswith('_0000.nii.gz')), None)

    if not t1c_file:
        raise HTTPException(status_code=400, detail="Format salah! File _0000.nii.gz tidak ditemukan di dalam ZIP.")

    case_id = t1c_file.replace('_0000.nii.gz', '')
    print(f"Pasien MedNeXt terdeteksi: {case_id}")

    status_ai = "Sedang Dianalisis..."
    
    new_scan = models.MRIScan(
        patient_id=pasien_db.id,
        jenis_mri="MRI Otak (MedNeXt)",
        catatan_teknis=catatan,
        filepath_raw=INPUT_DIR,
        filepath_2d=None,
        filepath_3d=None,
        hasil_prediksi=status_ai,
        confidence=0,
    )
    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    new_notif = models.Notification(
        target_role="Dokter",
        title="Data MRI Otak Diterima",
        message=f"File ZIP pasien {nama} berhasil diekstrak dan masuk antrean AI.",
        analysis_id=new_scan.id
    )
    db.add(new_notif)
    db.commit()

    print(f"Notifikasi disimpan: Untuk dokter, pasien: {nama} | ID scan: {new_scan.id}")

    save_log(db, current_user.username, current_user.role, "Upload MRI", f"Upload scan untuk pasien: {nama} (Masuk Antrean AI)")

    background_tasks.add_task(process_mri_ai, new_scan.id, INPUT_DIR, case_id)

    return {"status": "sukses", "pesan": "ZIP terekstrak & masuk antrean AI", "scan_id": new_scan.id}

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

    detected_list = [1, 2, 3, 4]
    if scan.detected_regions:
        try:
            detected_list = json.loads(scan.detected_regions)
        except Exception:
            pass
    
    waktu_scan = scan.upload_date.strftime("%d/%m/%Y • %H:%M WIB")

    # Return Data
    return {
        "id": scan.id,
        "image_url": scan.filepath_2d if scan.filepath_2d else "", 
        "result": scan.hasil_prediksi,
        "confidence": scan.confidence,
        "waktu_scan": waktu_scan,
        "nama_pasien": scan.patient.nama if scan.patient else "Tanpa Nama",
        "id_rm": scan.patient.id_pasien_rs if scan.patient else "-",
        "tgl_lahir": scan.patient.tanggal_lahir if scan.patient else "-",
        "jenis_kelamin": scan.patient.jenis_kelamin if scan.patient else "-",
        "notes_radiolog": scan.catatan_teknis,
        "notes_dokter": getattr(scan, "catatan_dokter", "Belum ada catatan dokter"),
        "detected_regions": detected_list 
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

    nama_pasien = scan.patient.nama if scan.patient else "Tanpa Nama"
    new_notif = models.Notification(
        target_role="Radiolog",
        title="Catatan Dokter",
        message=f"Dokter telah menambahkan catatan untuk pasien {nama_pasien}.",
        analysis_id=scan.id
    )

    db.add(new_notif)
    db.commit()

    save_log(db, current_user.username, current_user.role, "Update Notes", f"Update catatan dokter untuk Scan ID: {analysis_id}")
    return {"status": "sukses", "message": "Catatan berhasil diperbarui", "data": new_notes}

# ENDPOINT DASHBOARD SUMMARY
@app.get("/dashboard-summary/", response_model=schemas.DashboardSummary)
def get_summary(db: Session = Depends(get_db)):
    total_p = db.query(models.Patient).count()
    menunggu = db.query(models.MRIScan).filter(models.MRIScan.hasil_prediksi == "Sedang Dianalisis...").count()
    selesai = db.query(models.MRIScan).filter(models.MRIScan.hasil_prediksi != "Sedang Dianalisis...").count()

    return {
        "total_pasien": total_p,
        "total_menunggu": menunggu,
        "total_selesai": selesai
    }

# ENDPOINT NOTIFIKASI
@app.get("/notifications/")
def get_notifications(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    notifs = db.query(models.Notification).filter(func.lower(models.Notification.target_role) == func.lower(current_user.role)).order_by(models.Notification.created_at.desc()).all()

    print(f"Jumlah notifikasi ditemukan: {len(notifs)}")

    results = []
    for n in notifs:
        results.append({
            "id": n.id,
            "title": n.title,
            "message": n.message,
            "analysis_id": n.analysis_id,
            "is_read": n.is_read,
            "created_at": n.created_at.strftime("%d/%m/%Y • %H:%M WIB")
        })
    return results

@app.put("/notifications/{notif_id}/read")
def mark_notification_read(notif_id: int, db: Session = Depends(get_db)):
    notif = db.query(models.Notification).filter(models.Notification.id == notif_id).first()
    if notif:
        notif.is_read = True
        db.commit()
    return {"status": "sukses"}