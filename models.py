# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    role = Column(String)
    is_active = Column(Boolean, default=True)
    avatar = Column(String, nullable=True)

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    # id_pasien_rs adalah ID yang diketik radiolog (misal: RM-2024-001)
    id_pasien_rs = Column(String, unique=True, index=True) 
    nama = Column(String, index=True)
    tanggal_lahir = Column(String)
    status_pasien = Column(String) # 'Aktif' / 'Tidak Aktif'
    jenis_kelamin = Column(String, default="Laki-laki")
    
    # Mencatat kapan terakhir data ini diupdate/dibuat
    created_at = Column(DateTime, default=datetime.now)
    
    # Relasi ke Scan
    scans = relationship("MRIScan", back_populates="patient", cascade="all, delete-orphan")

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    role = Column(String)
    activity = Column(String)
    details = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

class MRIScan(Base):
    __tablename__ = "mri_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    
    jenis_mri = Column(String) # T1, T2, FLAIR
    catatan_teknis = Column(Text, nullable=True)
    catatan_dokter = Column(Text, nullable=True)
    
    filepath = Column(String)
    filename = Column(String)
    upload_date = Column(DateTime, default=datetime.now)
    
    # Hasil Analisis AI
    hasil_prediksi = Column(String, default="Belum Dianalisis") 
    confidence = Column(Integer, default=0)
    
    patient = relationship("Patient", back_populates="scans")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    target_role = Column(String) 
    title = Column(String)
    message = Column(String)

    analysis_id = Column(Integer, ForeignKey("mri_scans.id"), nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)