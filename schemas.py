# schemas.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# USER SCHEMAS
# Schema tambah user baru (input)
class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str
    role: str
    avatar: Optional[str] = None

# Schema menampilkan data user (output)
class UserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    is_active: bool
    avatar: Optional[str] = None

    class Config:
        from_attributes = True

# PATIENT SCHEMAS
# Base schema
class PatientBase(BaseModel):
    nama: str
    id_pasien_rs: str
    tanggal_lahir: str
    status_pasien: str = "Aktif"
    jenis_kelamin: str 

# Input (create/update)
class PatientCreate(PatientBase):
    pass

# Output (response)
class PatientResponse(PatientBase):
    id: int

    class Config:
        from_attributes = True

# ACTIVITY LOG SCHEMAS
class LogResponse(BaseModel):
    id: int
    username: str
    role: str
    activity: str
    details: str
    timestamp: datetime

    class Config:
        from_attributes = True

# RIWAYAT & DASHBOARD
# Format list riwayat
class RiwayatResponse(BaseModel):
    id: int
    jenis_mri: str
    tanggal_periksa: str
    hasil_prediksi: str
    
    class Config:
        from_attributes = True

# 2. Format dashboard summary (Radiolog)
class DashboardSummary(BaseModel):
    total_pasien: int
    total_menunggu: int
    total_selesai: int

class UserUpdate(BaseModel):
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = None
    avatar: Optional[str] = None
    is_active: Optional[bool] = None
    old_password: Optional[str] = None