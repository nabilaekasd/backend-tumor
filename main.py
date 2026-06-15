# main.py
from fastapi.security import OAuth2AuthorizationCodeBearer, OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from jose import JWTError, jwt
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from urllib.parse import unquote
from typing import List, Optional
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
from datetime import datetime, timezone
import pytz

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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Token tidak valid", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None: raise credentials_exception
    return user

def save_log(db: Session, username: str, role: str, activity: str, details: str = ""):
    try:
        new_log = models.ActivityLog(username=username, role=role, activity=activity, details=details)
        db.add(new_log)
        db.commit()
    except Exception as e:
        print(f"Gagal menyimpan log: {e}")

# ENDPOINT AUTH & USER
@app.post("/token/")
async def login_for_access_token(from_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == from_data.username).first()
    if not user or not auth.verify_password(from_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Username atau Password Salah")
    access_token = auth.create_access_token(data={"sub": user.username, "role": user.role, "id": user.id})
    save_log(db, user.username, user.role, "Login", "Login Berhasil")
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.post("/logout/")
def logout(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    save_log(db, current_user.username, current_user.role, "Logout", "Logout dari sistem")
    return {"message": "Berhasil logout"}

@app.post("/users/", response_model=schemas.UserResponse)
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user: raise HTTPException(status_code=400, detail="Username sudah terdaftar")
    hashed_pw = auth.get_password_hash(user.password)
    new_user = models.User(username=user.username, full_name=user.full_name, role=user.role, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    save_log(db, current_user.username, current_user.role, "Create User", f"Membuat user: {user.username} ({user.role})")
    return new_user

@app.put("/users/{user_id}", response_model=schemas.UserResponse)
def update_user(user_id: int, user_update: schemas.UserUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="User tidak ditemukan")
    if user_update.username is not None: db_user.username = user_update.username
    if user_update.full_name is not None: db_user.full_name = user_update.full_name
    if user_update.role is not None: db_user.role = user_update.role
    if user_update.password and user_update.password.strip():
        db_user.hashed_password = auth.get_password_hash(user_update.password)
    if user_update.avatar is not None: db_user.avatar = user_update.avatar
    if user_update.is_active is not None:
        db_user.is_active = user_update.is_active
    db.commit()
    db.refresh(db_user)
    status_text = "Aktif" if db_user.is_active else "Nonaktif"
    save_log(db, current_user.username, current_user.role, "Edit User", f"Mengedit user ID: {user_id} - Status: {status_text}")
    return db_user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="User tidak ditemukan")
    if db_user.id == current_user.id: raise HTTPException(status_code=400, detail="Tidak dapat menghapus akun sendiri")
    target_username = db_user.username
    db_user.is_active = False
    db.commit()
    save_log(db, current_user.username, current_user.role, "Deactivate User", f"Menonaktifkan user: {target_username}")
    return {"detail": "User berhasil dinonaktifkan"}

@app.get("/users/me/", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.get("/users/", response_model=List[schemas.UserResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return db.query(models.User).offset(skip).limit(limit).all()

@app.post("/users/upload-avatar/")
async def upload_avatar(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {str(e)}")
    avatar_url = f"static/{unique_filename}"
    current_user.avatar = avatar_url
    db.commit()
    db.refresh(current_user)
    save_log(db, current_user.username, current_user.role, "Update Profile", "Mengganti foto profil")
    return {"message": "Foto profil berhasil diupdate", "url": avatar_url} 

# ENDPOINT DATA PASIEN
@app.get("/patients/", response_model=List[schemas.PatientResponse])
def read_patient(db: Session = Depends(get_db)):
    return db.query(models.Patient).order_by(models.Patient.id.asc()).all()

@app.post("/patients/", response_model=schemas.PatientResponse)
def create_patient(patient: schemas.PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(models.Patient).filter(models.Patient.id_pasien_rs == patient.id_pasien_rs).first()
    if existing: raise HTTPException(status_code=400, detail="ID Pasien (RM) Sudah terdaftar")
    new_patient = models.Patient(id_pasien_rs=patient.id_pasien_rs, nama=patient.nama, tanggal_lahir=patient.tanggal_lahir, jenis_kelamin=patient.jenis_kelamin, status_pasien=patient.status_pasien)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    save_log(db, current_user.username, current_user.role, "Create Patient", f"Menambah pasien baru: {new_patient.nama} (RM: {new_patient.id_pasien_rs})")
    return new_patient

@app.put("/patients/{patient_id}", response_model=schemas.PatientResponse)
def update_patient(patient_id: int, patient_update: schemas.PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not db_patient: raise HTTPException(status_code=404, detail="Pasien tidak ditemukan")
    old_status = db_patient.status_pasien
    db_patient.id_pasien_rs = patient_update.id_pasien_rs
    db_patient.nama = patient_update.nama
    db_patient.tanggal_lahir = patient_update.tanggal_lahir
    db_patient.jenis_kelamin = patient_update.jenis_kelamin
    db_patient.status_pasien = patient_update.status_pasien
    db.commit()
    db.refresh(db_patient)
    detail_msg = f"Update data pasien: {db_patient.nama} (RM: {db_patient.id_pasien_rs})"
    if old_status != db_patient.status_pasien: detail_msg += f" | Status ubah: {old_status} -> {db_patient.status_pasien}"
    save_log(db, current_user.username, current_user.role, "Edit Patient", detail_msg)
    return db_patient

# HELPER LOGIC AI: METRIK & 3D RENDERER
def recall_specificity(pred, gt, cls, eps=1e-8):
    pred_c = (pred == cls)
    gt_c = (gt == cls)
    TP = np.logical_and(pred_c, gt_c).sum()
    FP = np.logical_and(pred_c, ~gt_c).sum()
    TN = np.logical_and(~pred_c, ~gt_c).sum()
    FN = np.logical_and(~pred_c, gt_c).sum()
    recall = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    return round(float(recall), 4), round(float(specificity), 4)

def load_u8(path):
    nii = nib.as_closest_canonical(nib.load(path))
    return np.rint(nii.get_fdata()).astype(np.uint8)

def calculate_metrics_if_gt_exists(pred_path, case_id):
    gt_path = os.path.join(MEDNEXT_DIR, f"{case_id}_GT.nii.gz")
    if not os.path.exists(gt_path):
        return None
    try:
        pred = load_u8(pred_path)
        gt = load_u8(gt_path)
        classes = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
        metrics = {}
        for cls, name in classes.items():
            r, s = recall_specificity(pred, gt, cls)
            metrics[name] = {"recall": r, "specificity": s}
        return json.dumps(metrics)
    except Exception as e:
        print(f"Error saat menghitung metrik: {e}")
        return None

def norm01(x):
    v = x[np.isfinite(x)]
    lo, hi = np.percentile(v, [2, 98]) if v.size > 0 else (0.0, 1.0)
    if hi <= lo: hi = lo + 1e-8
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

def generate_single_3d(mri_ds, pred_ds, out_path, target_label, ds=2, show_brain=True):
    colors_3d = {1:("NETC","#e41a1c"), 2:("SNFH","#377eb8"), 3:("ET","#4daf4a"), 4:("RC","#984ea3")}
    op_map = {1:0.95, 2:0.18, 3:0.55, 4:0.35} 
    fig_3d = go.Figure()

    if show_brain:
        brain = mri_ds > 0.1
        if brain.sum() > 0:
            brain_smooth = gaussian_filter(brain.astype(float), sigma=1.2)
            v, f, _, _ = measure.marching_cubes(brain_smooth, level=0.5)
            fig_3d.add_trace(go.Mesh3d(
                x=v[:,0] * ds, y=v[:,1] * ds, z=v[:,2] * ds, # Otak dikali ds
                i=f[:,0], j=f[:,1], k=f[:,2],
                color="lightgray", opacity=0.4, lighting=dict(ambient=0.6, diffuse=0.8),
                name="Brain", hoverinfo="skip"
            ))

    labels_to_draw = [2, 4, 3, 1] if target_label == 0 else [target_label]

    for lbl in labels_to_draw:
        name, col = colors_3d[lbl]
        bin_vol = (pred_ds == lbl).astype(np.uint8)
        if bin_vol.sum() > 0:
            v, f, _, _ = measure.marching_cubes(bin_vol, level=0.5, allow_degenerate=True)
            fig_3d.add_trace(go.Mesh3d(
                x=v[:,0] * ds, y=v[:,1] * ds, z=v[:,2] * ds,
                i=f[:,0], j=f[:,1], k=f[:,2],
                color=col, opacity=1.0 if target_label != 0 else op_map[lbl], name=name
            ))
            
    axis_config = dict(showgrid=True, gridcolor='#444444', zerolinecolor='#444444', color='white', showbackground=False, title_font=dict(color='white'))
    fig_3d.update_layout(
        paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white'),
        scene=dict(aspectmode="data", xaxis=dict(**axis_config, title="X"), yaxis=dict(**axis_config, title="Y"), zaxis=dict(**axis_config, title="Z"), bgcolor='black'),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig_3d.write_html(out_path, full_html=True, include_plotlyjs='cdn')

# AI PROCESSOR (BACKGROUND TASK)
def process_mri_ai(scan_id: int, input_dir: str, output_dir: str, case_id: str, gt_file_path: str):
    db = SessionLocal()
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == scan_id).first()
    if not scan:
        db.close()
        return

    print(f"[PROSES MULAI] AI sedang membedah pasien: {case_id}...")

    command = [
        "nnUNet_predict", 
        "-i", input_dir, 
        "-o", output_dir,
        "-t", "Task001_BraTS2024", 
        "-m", "3d_fullres", 
        "-f", "4",
        "-tr", "nnUNetTrainerV2_MedNeXt_M_ChooseOpt", 
        "-p", "nnUNetPlansv2.1_trgSp_1x1x1", 
        "--mode", "normal"
    ]

    try:
        subprocess.run(command, check=True)
        pred_path = os.path.join(output_dir, f"{case_id}.nii.gz")
        mri_path_3d = os.path.join(input_dir, f"{case_id}_0003.nii.gz")

        pred_data = nib.load(pred_path).get_fdata()
        unique_labels = np.unique(pred_data).astype(int).tolist()
        if 0 in unique_labels: unique_labels.remove(0)
        scan.detected_regions = json.dumps(unique_labels)

        metrics_json = calculate_metrics_if_gt_exists(pred_path, case_id)

        scan.catatan_teknis = json.dumps({
            "metrics": json.loads(metrics_json) if metrics_json else None,
            "shape": pred_data.shape,
            "case_id": case_id
        })

        mri_3d = nib.as_closest_canonical(nib.load(mri_path_3d)).get_fdata().astype(np.float32)
        pred_3d = nib.as_closest_canonical(nib.load(pred_path)).get_fdata().astype(np.uint8)

        ds = 2
        mri01 = norm01(mri_3d)
        mri_ds = mri01[::ds, ::ds, ::ds]

        pred_ds = pred_3d[::ds, ::ds, ::ds]
        path_3d_db = {}
        base_labels = {"all": 0, "netc": 1, "snfh": 2, "et": 3, "rc": 4}

        for key, lbl in base_labels.items():
            # 1. Render Versi Dengan Otak
            fname_3d_with = f"result_{scan_id}_3d_{key}.html"
            out_path_with = os.path.join(UPLOAD_DIR, fname_3d_with)
            generate_single_3d(mri_ds, pred_ds, out_path_with, lbl, show_brain=True)
            path_3d_db[key] = f"static/{fname_3d_with}"

            # 2. Render Versi TANPA Otak
            fname_3d_no = f"result_{scan_id}_3d_{key}_nobrain.html"
            out_path_no = os.path.join(UPLOAD_DIR, fname_3d_no)
            generate_single_3d(mri_ds, pred_ds, out_path_no, lbl, show_brain=False)
            path_3d_db[f"{key}_nobrain"] = f"static/{fname_3d_no}"

        scan.filepath_3d = json.dumps(path_3d_db)
        scan.filepath_2d = "dynamic" 
        scan.hasil_prediksi = "Tumor Terdeteksi" if unique_labels else "Normal"

        target_roles = ["Dokter", "Radiolog"]
        for target in target_roles:
            db.add(models.Notification(target_role=target, title="Analisis AI Selesai!", message="Hasil pemindaian 3D dan Slice 2D dapat dilihat.", analysis_id=scan.id))
        db.commit()
    
    except Exception as e:
        print(f"[ERROR AI] Gagal memproses: {e}")
        scan.hasil_prediksi = "Gagal Diproses AI"
        db.commit()
    finally:
        db.close()

# ENDPOINT ANALISIS & FILE
@app.post("/upload-mri/")
async def upload_mri_smart(
    background_tasks: BackgroundTasks, nama: str = Form(...), id_pasien: str = Form(...), tgl_lahir: str = Form(...),
    status: str = Form(...), jenis_mri: str = Form(...), catatan: str = Form(default="-"), file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    if not file.filename.endswith('.zip'): raise HTTPException(status_code=400, detail="Harus file .zip yang berisi 4 modalitas MRI!")

    pasien_db = db.query(models.Patient).filter(models.Patient.id_pasien_rs == id_pasien).first()
    if not pasien_db:
        pasien_db = models.Patient(nama=nama, id_pasien_rs=id_pasien, tanggal_lahir=tgl_lahir, status_pasien=status)
        db.add(pasien_db)
        db.commit()
        db.refresh(pasien_db)
    else:
        pasien_db.nama = nama
        pasien_db.status_pasien = status
        db.commit()
    
    new_scan = models.MRIScan(patient_id=pasien_db.id, jenis_mri="MRI Otak", catatan_teknis=catatan, filepath_raw="", hasil_prediksi="Sedang Dianalisis...")
    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    # Folder Scan
    scan_folder = os.path.join(UPLOAD_DIR, f"scan_{new_scan.id}")
    input_dir = os.path.join(scan_folder, "input")
    output_dir = os.path.join(scan_folder, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Ekstrak ZIP
    zip_path = os.path.join(scan_folder, "temp.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(input_dir)
    os.remove(zip_path)

    extracted_files = os.listdir(input_dir)
    t1c_file = next((f for f in extracted_files if f.endswith('_0000.nii.gz')), None)
    if not t1c_file: 
        raise HTTPException(status_code=400, detail="Format salah! File _0000.nii.gz tidak ditemukan di dalam ZIP.")
    case_id = t1c_file.replace('_0000.nii.gz', '')


    valid_modalities = [
        f"{case_id}_0000.nii.gz", f"{case_id}_0001.nii.gz", 
        f"{case_id}_0002.nii.gz", f"{case_id}_0003.nii.gz",
    ]

    gt_file_path = os.path.join(scan_folder, f"{case_id}_GT.nii.gz")

    for f in extracted_files:
        if f.endswith('.nii.gz') and f not in valid_modalities:
            shutil.move(os.path.join(input_dir, f), gt_file_path)
            print(f"File GT ditemukan dan diamankan: {f}")

    new_scan.filepath_raw = scan_folder
    db.commit()

    new_notif = models.Notification(target_role="Dokter", title="Data MRI Otak Diterima", message=f"File ZIP Pasien {nama} berhasil diekstrak dan masuk antrean AI.", analysis_id=new_scan.id)
    db.add(new_notif)
    db.commit()

    save_log(db, current_user.username, current_user.role, "Upload MRI", f"Upload scan untuk pasien: {nama} (Masuk Antrean AI)")

    background_tasks.add_task(process_mri_ai, new_scan.id, input_dir, output_dir, case_id, gt_file_path)
    return {"status": "sukses", "pesan": "ZIP terekstrak dan masuk antrean AI", "scan_id": new_scan.id}

@app.get("/analisis/{analysis_id}/slice")
def get_mri_slice(analysis_id: int, axis: int = 2, idx: int = 75, label: str = "all", db: Session = Depends(get_db)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan tidak ditemukan")
    try:
        meta = json.loads(scan.catatan_teknis)
        case_id = meta.get("case_id")
    except: raise HTTPException(status_code=400, detail="Data belum siap")

    scan_folder = scan.filepath_raw
    input_dir = os.path.join(scan_folder, "input")
    output_dir = os.path.join(scan_folder, "output")

    pred_path = os.path.join(output_dir, f"{case_id}.nii.gz")
    mri_path_2d = os.path.join(input_dir, f"{case_id}_0002.nii.gz") 

    try:
        mri_vol = nib.load(mri_path_2d).get_fdata().astype(np.float32)
        pred_vol = nib.load(pred_path).get_fdata().astype(np.uint8)

        def take_slice(vol, axis, idx):
            if axis == 0: return vol[idx, :, :]
            if axis == 1: return vol[:, idx, :] 
            if axis == 2: return vol[:, :, idx]

        mri_s = norm01(take_slice(mri_vol, axis, idx))
        pred_s = take_slice(pred_vol, axis, idx)

        if label == "netc": pred_s = np.where(pred_s == 1, 1, 0)
        elif label == "snfh": pred_s = np.where(pred_s == 2, 2, 0)
        elif label == "et": pred_s = np.where(pred_s == 3, 3, 0)
        elif label == "rc": pred_s = np.where(pred_s == 4, 4, 0)

        colors_hex = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}
        mask_cmap = ListedColormap(["none", colors_hex[1], colors_hex[2], colors_hex[3], colors_hex[4]])

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(np.rot90(mri_s), cmap="gray")
        ax.imshow(np.rot90(pred_s), cmap=mask_cmap, alpha=0.6, vmin=0, vmax=4)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/riwayat-semua/")
def get_all_history(db: Session = Depends(get_db)):
    scans = db.query(models.MRIScan).order_by(models.MRIScan.upload_date.desc()).all()
    
    tz_jkt = pytz.timezone('Asia/Jakarta')
    results = []
    
    for scan in scans:
        if not scan.upload_date:
            tgl_cantik = "-"
        else:
            if scan.upload_date.tzinfo is None:
                waktu_utc = scan.upload_date.replace(tzinfo=pytz.utc)
                waktu_lokal = waktu_utc.astimezone(tz_jkt)
            else:
                waktu_lokal = scan.upload_date.astimezone(tz_jkt)
            
            tgl_cantik = waktu_lokal.strftime("%d/%m/%Y")

        results.append({
            "id": scan.id, "jenis_mri": scan.jenis_mri, "tanggal_periksa": tgl_cantik,
            "hasil_prediksi": scan.hasil_prediksi, "nama_pasien": scan.patient.nama if scan.patient else "Tanpa Nama",
            "id_rm": scan.patient.id_pasien_rs if scan.patient else "-"
        })
    return results

@app.get("/analisis/{analysis_id}")
def get_analysis_detail(analysis_id: int, db: Session = Depends(get_db)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    if not scan: raise HTTPException(status_code=404, detail="Data MRI tidak ditemukan")

    detected_list = []
    if scan.detected_regions:
        try: detected_list = json.loads(scan.detected_regions)
        except: pass
    
    meta = {}
    if scan.catatan_teknis and "{" in scan.catatan_teknis:
        try: meta = json.loads(scan.catatan_teknis)
        except: pass

    paths_3d = {}
    if scan.filepath_3d and "{" in scan.filepath_3d:
        try: paths_3d = json.loads(scan.filepath_3d)
        except: paths_3d = {"all": scan.filepath_3d}
    else:
        paths_3d = {"all": scan.filepath_3d}

    # Konversi Zona Waktu
    tz_jkt = pytz.timezone('Asia/Jakarta')
    if scan.upload_date:
        if scan.upload_date.tzinfo is None:
            waktu_utc = scan.upload_date.replace(tzinfo=pytz.utc)
            waktu_lokal = waktu_utc.astimezone(tz_jkt)
        else:
            waktu_lokal = scan.upload_date.astimezone(tz_jkt)
        waktu_scan_cantik = waktu_lokal.strftime("%d/%m/%Y • %H:%M WIB")
    else:
        waktu_scan_cantik = "-"

    return {
        "id": scan.id,
        "image_url": "dynamic", 
        "paths_3d": paths_3d,
        "result": scan.hasil_prediksi,
        "confidence": scan.confidence,
        "waktu_scan": waktu_scan_cantik,
        "nama_pasien": scan.patient.nama if scan.patient else "-",
        "id_rm": scan.patient.id_pasien_rs if scan.patient else "-",
        "tgl_lahir": scan.patient.tanggal_lahir if scan.patient else "-",
        "jenis_kelamin": scan.patient.jenis_kelamin if scan.patient else "-",
        "notes_radiolog": meta.get("catatan", "-"), 
        "notes_dokter": getattr(scan, "catatan_dokter", "Belum ada catatan dokter"),
        "detected_regions": detected_list,
        "metrics": meta.get("metrics"),
        "shape": meta.get("shape", [155, 240, 240]) 
    }

@app.get("/get-image/{filename}")
async def get_image_manual(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"})
    return {"error": "File tidak ditemukan"}

@app.put("/analisis/{analysis_id}/update-notes/")
async def update_doctor_notes(analysis_id: int, data: dict, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    if not scan: raise HTTPException(status_code=404, detail="Data MRI tidak ditemukan")
    
    new_notes = data.get("notes_dokter", "")
    scan.catatan_dokter = new_notes
    db.commit()
    db.refresh(scan)

    nama_pasien = scan.patient.nama if scan.patient else "Tanpa Nama"
    db.add(models.Notification(target_role="Radiolog", title="Catatan Dokter", message=f"Dokter telah menambahkan catatan untuk pasien {nama_pasien}.", analysis_id=scan.id))
    db.commit()

    save_log(db, current_user.username, current_user.role, "Update Notes", f"Update catatan dokter untuk Scan ID: {analysis_id}")
    return {"status": "sukses", "message": "Catatan berhasil diperbarui", "data": new_notes}

# ENDPOINT SUMMARY
@app.get("/dashboard-summary/", response_model=schemas.DashboardSummary)
def get_summary(db: Session = Depends(get_db)):
    total_p = db.query(models.Patient).count()
    menunggu = db.query(models.MRIScan).filter(models.MRIScan.hasil_prediksi == "Sedang Dianalisis...").count()
    selesai = db.query(models.MRIScan).filter(models.MRIScan.hasil_prediksi != "Sedang Dianalisis...").count()
    return {"total_pasien": total_p, "total_menunggu": menunggu, "total_selesai": selesai}

@app.get("/logs/", response_model=List[schemas.LogResponse])
def get_logs(role: str = None, start_date: str = None, end_date: str = None, db: Session = Depends(get_db)):
    query = db.query(models.ActivityLog)
    if role and role != "Semua": query = query.filter(models.ActivityLog.role == role)
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(models.ActivityLog.timestamp >= start)
            query = query.filter(models.ActivityLog.timestamp <= end)
        except ValueError: pass
        
    logs = query.order_by(models.ActivityLog.timestamp.desc()).all()
    tz_jkt = pytz.timezone('Asia/Jakarta')
    results = []
    
    for log in logs:
        if log.timestamp:
            if log.timestamp.tzinfo is None:
                waktu_utc = log.timestamp.replace(tzinfo=pytz.utc)
                waktu_lokal = waktu_utc.astimezone(tz_jkt)
            else:
                waktu_lokal = log.timestamp.astimezone(tz_jkt)
        else:
            waktu_lokal = None

        results.append({
            "id": log.id,
            "username": log.username,
            "role": log.role,
            "activity": log.activity,
            "details": log.details,
            "timestamp": waktu_lokal
        })
        
    return results

@app.get("/notifications/")
def get_notifications(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    notifs = db.query(models.Notification).filter(func.lower(models.Notification.target_role) == func.lower(current_user.role)).order_by(models.Notification.created_at.desc()).all()
    
    tz_jkt = pytz.timezone('Asia/Jakarta')
    results = []
    
    for n in notifs:
        if not n.created_at:
            tgl_cantik = "-"
        else:
            if n.created_at.tzinfo is None:
                waktu_utc = n.created_at.replace(tzinfo=pytz.utc)
                waktu_lokal = waktu_utc.astimezone(tz_jkt)
            else:
                waktu_lokal = n.created_at.astimezone(tz_jkt)
            
            tgl_cantik = waktu_lokal.strftime("%d/%m/%Y • %H:%M WIB")

        results.append({
            "id": n.id, 
            "title": n.title, 
            "message": n.message, 
            "analysis_id": n.analysis_id, 
            "is_read": n.is_read, 
            "created_at": tgl_cantik
        })
        
    return results

@app.put("/notifications/{notif_id}/read")
def mark_notification_read(notif_id: int, db: Session = Depends(get_db)):
    notif = db.query(models.Notification).filter(models.Notification.id == notif_id).first()
    if notif:
        notif.is_read = True
        db.commit()
    return {"status": "sukses"}