import streamlit as st
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from sqlmodel import Session, select, Field, SQLModel, col

# Import các models từ service
# Giả sử file fast_api_service.py nằm cùng thư mục
try:
    from fast_api_service import engine, School, Major, SchoolMajor, Combination, Subject, AdmissionScore
except ImportError:
    pass

# Tắt cảnh báo
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. DATABASE MODELS BỔ SUNG
# ==========================================
class PredictionData(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    school_major_id: int = Field(foreign_key="schoolmajor.id")
    combination_id: int = Field(foreign_key="combination.id")
    target_year: int 
    pred_ensemble: float      
    pred_rf: Optional[float] = None
    pred_xgb: Optional[float] = None
    pred_cat: Optional[float] = None
    actual_score: Optional[float] = None 
    error_ensemble: Optional[float] = None

# ==========================================
# 2. DATA CLASSES
# ==========================================
@dataclass
class StudentInput:
    university: str
    major_name: str
    major_code: str
    combination: str
    subject_scores: Dict[str, float]
    priority_score: float

    @property
    def total_score(self) -> float:
        return sum(self.subject_scores.values()) + self.priority_score
    
    @property
    def raw_score(self) -> float:
        return sum(self.subject_scores.values())

@dataclass
class PredictionResult:
    university: str
    major_name: str
    major_code: str
    combination: str
    predictions: Dict[str, float]
    best_prediction: float
    student_score: float
    is_passed: bool
    margin: float

# ==========================================
# 3. CLASS DỰ ĐOÁN (Logic Chính)
# ==========================================
class AdmissionPredictor:
    """
    Class quản lý việc lấy dữ liệu dự báo và lịch sử từ Database
    """

    def get_historical_cutoffs(self, university_code: str, major_code: str, combination_code: str) -> Dict[int, float]:
        """
        Lấy lịch sử điểm chuẩn các năm trước (từ bảng AdmissionScore)
        để vẽ biểu đồ xu hướng.
        """
        history = {}
        with Session(engine) as session:
            # Join các bảng để lấy đúng điểm của Trường-Ngành-Tổ hợp đó
            statement = select(AdmissionScore.year, AdmissionScore.score)\
                .join(SchoolMajor, AdmissionScore.school_major_id == SchoolMajor.id)\
                .join(School, SchoolMajor.school_id == School.id)\
                .join(Major, SchoolMajor.major_id == Major.id)\
                .join(Combination, AdmissionScore.combination_id == Combination.id)\
                .where((School.code == university_code) | (School.name == university_code))\
                .where(Major.code == major_code)\
                .where(Combination.code == combination_code)\
                .order_by(AdmissionScore.year)
            
            results = session.exec(statement).all()
            for year, score in results:
                if score > 0:
                    history[year] = float(score)
        return history

    def predict(self, student: StudentInput) -> PredictionResult:
        """
        Truy vấn kết quả dự báo năm 2025 từ bảng PredictionData
        """
        with Session(engine) as session:
            # 1. Tìm School (theo Code hoặc Tên)
            school = session.exec(select(School).where(School.code == student.university)).first()
            if not school:
                 school = session.exec(select(School).where(School.name == student.university)).first()
            
            # 2. Tìm Major
            major = session.exec(select(Major).where(Major.code == student.major_code)).first()
            
            # 3. Tìm Combination
            combo = session.exec(select(Combination).where(Combination.code == student.combination)).first()
            
            # Nếu thiếu thông tin cơ bản -> Return dummy
            if not school or not major or not combo:
                return self._create_dummy_result(student, 0.0)
            
            # 4. Tìm liên kết SchoolMajor
            sm = session.exec(select(SchoolMajor).where(SchoolMajor.school_id == school.id, SchoolMajor.major_id == major.id)).first()
            if not sm:
                return self._create_dummy_result(student, 0.0)

            # 5. Lấy dữ liệu Dự Báo (Năm 2025)
            pred_data = session.exec(select(PredictionData).where(
                PredictionData.school_major_id == sm.id,
                PredictionData.combination_id == combo.id,
                PredictionData.target_year == 2025
            )).first()

            predictions = {}
            best_pred = 0.0

            if pred_data:
                # Ưu tiên Ensemble -> RF -> XGB
                best_pred = pred_data.pred_ensemble
                if best_pred == 0: best_pred = pred_data.pred_rf
                if best_pred == 0 and pred_data.pred_xgb: best_pred = pred_data.pred_xgb

                predictions = {
                    "Ensemble": pred_data.pred_ensemble,
                    "RandomForest": pred_data.pred_rf,
                    "XGBoost": pred_data.pred_xgb
                }
            else:
                # Fallback: Lấy điểm chuẩn 2024
                hist = session.exec(select(AdmissionScore).where(
                    AdmissionScore.school_major_id == sm.id,
                    AdmissionScore.combination_id == combo.id,
                    AdmissionScore.year == 2024
                )).first()
                if hist:
                     best_pred = hist.score
                     predictions["Điểm chuẩn 2024"] = best_pred

            # Tính toán kết quả
            student_total = student.total_score
            margin = student_total - best_pred

            return PredictionResult(
                university=school.name,
                major_name=major.name,
                major_code=major.code,
                combination=combo.code,
                predictions=predictions,
                best_prediction=round(best_pred, 2),
                student_score=round(student_total, 2),
                is_passed=(margin >= 0),
                margin=round(margin, 2)
            )

    def _create_dummy_result(self, student, pred_val):
        return PredictionResult(
            university=student.university,
            major_name=student.major_name,
            major_code=student.major_code,
            combination=student.combination,
            predictions={},
            best_prediction=pred_val,
            student_score=student.total_score,
            is_passed=False,
            margin=0.0
        )

# ==========================================
# 4. UI HELPERS (Dữ liệu cho Dropdown)
# ==========================================

def get_all_schools():
    """Lấy danh sách tất cả các trường"""
    with Session(engine) as session:
        return session.exec(select(School).order_by(School.code)).all()

def get_majors_by_school(school_id: int):
    """Lấy danh sách ngành theo ID trường"""
    with Session(engine) as session:
        return session.exec(select(Major).join(SchoolMajor).where(SchoolMajor.school_id == school_id).distinct()).all()

def get_combinations_by_school_major(school_id: int, major_id: int):
    """
    Lấy tổ hợp môn. Ưu tiên lấy từ bảng PredictionData.
    """
    with Session(engine) as session:
        sm = session.exec(select(SchoolMajor).where(SchoolMajor.school_id == school_id, SchoolMajor.major_id == major_id)).first()
        if not sm: return []
        
        # 1. Tìm trong bảng Dự báo
        combos = session.exec(
            select(Combination)
            .join(PredictionData)
            .where(PredictionData.school_major_id == sm.id)
            .distinct()
        ).all()
        
        # 2. Fallback: Tìm trong bảng Điểm chuẩn
        if not combos:
             combos = session.exec(
                select(Combination)
                .join(AdmissionScore)
                .where(AdmissionScore.school_major_id == sm.id)
                .distinct()
            ).all()
            
        return combos

def get_subjects_of_combination(combo_id: int):
    """Trả về tên 3 môn học"""
    with Session(engine) as session:
        c = session.get(Combination, combo_id)
        if not c: return []
        
        s1 = session.get(Subject, c.subject1_id)
        s2 = session.get(Subject, c.subject2_id)
        s3 = session.get(Subject, c.subject3_id)
        
        names = []
        if s1: names.append(s1.name)
        if s2: names.append(s2.name)
        if s3: names.append(s3.name)
        return names

# ==========================================
# 5. RECOMMENDATION FUNCTIONS
# ==========================================

def get_recommendations_by_specific_combo(combo_code: str, user_score: float, current_school_id: Optional[int] = None, limit: int = 5):
    """
    Gợi ý ngành dựa trên tổ hợp và điểm.
    """
    results_list = []
    
    with Session(engine) as session:
        combo = session.exec(select(Combination).where(Combination.code == combo_code)).first()
        if not combo: return pd.DataFrame()

        query = select(
            School.name.label("school_name"),
            Major.name.label("major_name"),
            PredictionData.pred_ensemble,
            PredictionData.pred_rf
        ).join(SchoolMajor, PredictionData.school_major_id == SchoolMajor.id)\
         .join(School, SchoolMajor.school_id == School.id)\
         .join(Major, SchoolMajor.major_id == Major.id)\
         .where(PredictionData.combination_id == combo.id)\
         .where(PredictionData.target_year == 2025)

        if current_school_id:
            query = query.where(School.id == current_school_id)

        data = session.exec(query).all()

        for row in data:
            pred = row.pred_ensemble if row.pred_ensemble > 0 else (row.pred_rf or 0)
            if pred == 0: continue

            diff = user_score - pred
            
            # Lọc bớt kết quả quá xa vời (User thiếu > 3 điểm)
            if diff > -3.0: 
                results_list.append({
                    "Trường": row.school_name,
                    "Ngành": row.major_name,
                    "Tổ hợp": combo_code,
                    "2023": 0,
                    "2024": 0,
                    "Dự báo 2025": round(pred, 2),
                    "Chênh lệch": round(diff, 2)
                })

    df = pd.DataFrame(results_list)
    if df.empty: return df

    return df.sort_values(by="Chênh lệch", ascending=False).head(limit)

def get_lowest_score_majors(limit=5):
    """Lấy danh sách ngành điểm thấp nhất (2024)"""
    with Session(engine) as session:
        query = select(School.name.label("Trường"), Major.name.label("Ngành"), Combination.code.label("Tổ hợp"), AdmissionScore.score)\
            .join(SchoolMajor, AdmissionScore.school_major_id == SchoolMajor.id)\
            .join(School, SchoolMajor.school_id == School.id)\
            .join(Major, SchoolMajor.major_id == Major.id)\
            .join(Combination, AdmissionScore.combination_id == Combination.id)\
            .where(AdmissionScore.year == 2024, AdmissionScore.score > 12)\
            .order_by(AdmissionScore.score).limit(limit)
        
        results = session.exec(query).all()
        data = [{"Trường": r[0], "Ngành": r[1], "Tổ hợp": r[2], "2023": 0, "2024": r[3], "Dự báo 2025": r[3], "Chênh lệch": "+ Dư dả"} for r in results]
            
        return pd.DataFrame(data)