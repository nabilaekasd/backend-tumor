from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# KONFIGURASI POSTGRESQL
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:passwordnabilaesd@localhost/tumordb"

# Membuat mesin database (Postgres tidak butuh check_same_thread seperti SQLite)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()