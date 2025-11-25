from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship

# ==========================================
# 1. DATABASE MODELS (THIẾT KẾ CSDL)
# ==========================================

# --- Bảng Môn học (Subjects) ---
class Subject(SQLModel, table=True):
    __table_args__ = {'extend_existing': True} # Thêm dòng này để fix lỗi redefine table
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True) 

# --- Bảng Trường (Schools) ---
class School(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True) 
    name: str 

# --- Bảng Ngành học (Majors) ---
class Major(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True) 
    name: str

# --- Bảng Tổ hợp môn (Combinations) ---
class Combination(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True) 
    
    subject1_id: int = Field(foreign_key="subject.id")
    subject2_id: int = Field(foreign_key="subject.id")
    subject3_id: int = Field(foreign_key="subject.id")

# --- Bảng Liên kết: Trường - Ngành (SchoolMajors) ---
class SchoolMajor(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    school_id: int = Field(foreign_key="school.id")
    major_id: int = Field(foreign_key="major.id")

# --- Bảng Điểm chuẩn (AdmissionScores) ---
class AdmissionScore(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    school_major_id: int = Field(foreign_key="schoolmajor.id")
    combination_id: int = Field(foreign_key="combination.id")
    year: int
    score: float

# ==========================================
# 2. DATABASE CONFIG
# ==========================================
import os
# Lấy đường dẫn thư mục hiện tại của file api_service.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sqlite_file_name = "admission_db.sqlite"
# Nối đường dẫn để đảm bảo tìm thấy file db
sqlite_url = f"sqlite:///{os.path.join(BASE_DIR, sqlite_file_name)}"

# check_same_thread=False cần thiết cho SQLite khi dùng với FastAPI đa luồng
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# ==========================================
# 3. FASTAPI APP
# ==========================================
app = FastAPI(title="Admission Data API", version="1.0")

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# --- API ENDPOINTS (Giữ nguyên như cũ) ---

@app.get("/schools/", response_model=List[School])
def read_schools(session: Session = Depends(get_session)):
    schools = session.exec(select(School)).all()
    return schools

@app.post("/schools/", response_model=School)
def create_school(school: School, session: Session = Depends(get_session)):
    session.add(school)
    session.commit()
    session.refresh(school)
    return school

@app.get("/data-export/")
def export_data(session: Session = Depends(get_session)):
    statement = (
        select(
            School.name.label("school_name"),
            Major.name.label("major_name"),
            Major.code.label("major_code"),
            Combination.code.label("combination_code"),
            AdmissionScore.year,
            AdmissionScore.score
        )
        .join(SchoolMajor, SchoolMajor.id == AdmissionScore.school_major_id)
        .join(School, School.id == SchoolMajor.school_id)
        .join(Major, Major.id == SchoolMajor.major_id)
        .join(Combination, Combination.id == AdmissionScore.combination_id)
    )
    
    results = session.exec(statement).all()
    
    data = []
    for row in results:
        data.append({
            "Trường": row.school_name,
            "Tên ngành": row.major_name,
            "Mã ngành": row.major_code,
            "Tổ hợp môn": row.combination_code,
            "Năm": row.year,
            "Điểm chuẩn": row.score
        })
    return data