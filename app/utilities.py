import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, List, Optional
from dataclasses import dataclass
from sqlmodel import Session, select
from fast_api_service import engine, School, Major, SchoolMajor, Combination, Subject, AdmissionScore
import pickle
from pathlib import Path

# --- CONSTANTS & PATHS ---
MODEL_RF_PATH = '../best_model.pkl'
MODEL_XGB_PATH = 'admission_model_xgb.pkl'
MODEL_CAT_PATH = 'admission_model_cat.pkl'

# --- DATA CLASSES (INPUT/OUTPUT) ---
@dataclass
class StudentInput:
    """Input từ thí sinh"""
    university: str      # Mã trường (VD: QHS)
    major_name: str      # Tên ngành (để hiển thị)
    major_code: str      # Mã ngành (VD: 7480201)
    combination: str     # Mã tổ hợp (VD: A00)
    subject_scores: Dict[str, float]  # {'Toán': 8.5, 'Văn': 7.0}
    priority_score: float  # Điểm ưu tiên

    @property
    def total_score(self) -> float:
        """Tổng điểm = điểm các môn + điểm ưu tiên"""
        return sum(self.subject_scores.values()) + self.priority_score

    @property
    def raw_score(self) -> float:
        """Điểm thô"""
        return sum(self.subject_scores.values())

@dataclass
class PredictionResult:
    """Kết quả dự đoán"""
    university: str
    major_name: str
    major_code: str
    combination: str
    predictions: Dict[str, float]  # {'Baseline': 25.0, 'RandomForest': 24.5}
    best_prediction: float
    student_score: float
    is_passed: bool
    margin: float  # Chênh lệch điểm

# --- PREDICTOR CLASS (SQLITE VERSION) ---
class AdmissionPredictor:
    def __init__(self):
        # Không cần load df vào RAM nữa
        self.models = self._load_models()

    def _load_models(self) -> Dict:
        models = {}
        # Load Random Forest using pickle
        rf_path = Path(MODEL_RF_PATH)
        if rf_path.exists():
            try:
                with open(rf_path, "rb") as f:
                    loaded_object = pickle.load(f)
                
                # Check if it's a dictionary (artifacts) or the model itself
                if isinstance(loaded_object, dict) and 'model' in loaded_object:
                    print(f"   ✓ Loaded RandomForest artifacts from {rf_path}")
                    models['RandomForest'] = loaded_object['model']
                    # You might also want to load encoders here if needed:
                    # self.le_major = loaded_object.get('le_major')
                    # self.le_combo = loaded_object.get('le_combo')
                else:
                    print(f"   ✓ Loaded RandomForest model from {rf_path}")
                    models['RandomForest'] = loaded_object
                    
            except Exception as e:
                print(f"Error loading RandomForest model: {e}")
        
        return models

    def get_historical_cutoffs(self, university_code: str, major_code: str, combination_code: str) -> Dict[int, float]:
        """
        Lấy điểm chuẩn lịch sử từ SQLite Database.
        Input: Mã trường (VD: QHS), Mã ngành (VD: 7480201), Mã tổ hợp (VD: A00)
        """
        history = {}
        
        with Session(engine) as session:
            # 1. Lấy School
            school = session.exec(select(School).where(School.code == university_code)).first()
            if not school: return {}

            # 2. Lấy Major
            major = session.exec(select(Major).where(Major.code == major_code)).first()
            if not major: return {}

            # 3. Tìm liên kết SchoolMajor
            sm_link = session.exec(select(SchoolMajor).where(
                SchoolMajor.school_id == school.id,
                SchoolMajor.major_id == major.id
            )).first()
            if not sm_link: return {}

            # 4. Lấy Combination ID
            combo = session.exec(select(Combination).where(Combination.code == combination_code)).first()
            if not combo: return {}

            # 5. Lấy điểm chuẩn
            scores = session.exec(select(AdmissionScore).where(
                AdmissionScore.school_major_id == sm_link.id,
                AdmissionScore.combination_id == combo.id
            )).all()

            # Convert to dict {2023: 25.5, 2024: 26.0}
            for s in scores:
                if 2017 <= s.year <= 2025 and s.score > 0:
                    history[s.year] = float(s.score)
                    
        return history

    def _create_features(self, history: Dict[int, float], combination: str) -> Optional[List[float]]:
        """Tạo features từ history cho ML models"""
        if not history or len(history) < 1:
            return None

        years = sorted(history.keys())
        cutoffs = [history[y] for y in years]

        features = []

        # 1. Lag features (Điểm các năm trước)
        features.append(cutoffs[-1])  # lag_1
        features.append(cutoffs[-2] if len(cutoffs) >= 2 else 0)  # lag_2
        features.append(cutoffs[-3] if len(cutoffs) >= 3 else 0)  # lag_3

        # 2. Aggregate features
        features.append(np.mean(cutoffs))  # avg_all
        features.append(np.mean(cutoffs[-3:]) if len(cutoffs) >= 3 else np.mean(cutoffs))  # avg_3y
        features.append(np.std(cutoffs[-3:]) if len(cutoffs) >= 3 else 0)  # std_3y
        features.append(max(cutoffs))  # max
        features.append(min(cutoffs))  # min

        # 3. Trend features
        if len(years) >= 2:
            try:
                x_norm = np.arange(len(years))
                coeffs = np.polyfit(x_norm, cutoffs, deg=1)
                features.append(coeffs[0])  # trend_coef
            except:
                features.append(0)
        else:
            features.append(0)

        # 4. Momentum
        if len(years) >= 2:
            features.append(cutoffs[-1] - cutoffs[-2])
        else:
            features.append(0)

        # 5. Combination Encoding
        combo_code = hash(combination) % 100
        features.append(combo_code)

        # 6. Padding to expected length
        # Điều chỉnh số này khớp với model bạn train (ví dụ 13 hoặc 73)
        # Ở đây tôi để dynamic, nếu model yêu cầu số feature cố định thì code dưới sẽ pad thêm 0
        rf_model = self.models.get('RandomForest')
        expected_len = 13 # Mặc định
        if rf_model and hasattr(rf_model, "n_features_in_"):
            expected_len = rf_model.n_features_in_

        while len(features) < expected_len:
            features.append(0)
            
        # Nếu features dài hơn expected (do logic code thay đổi), cắt bớt
        return features[:expected_len]

    def predict(self, student: StudentInput) -> PredictionResult:
        """Thực hiện dự báo cho 1 học sinh"""
        
        # 1. Lấy dữ liệu lịch sử từ DB
        history = self.get_historical_cutoffs(
            student.university, 
            student.major_code, 
            student.combination
        )
        
        predictions = {}
        
        # --- Baseline ---
        if history:
            years = sorted(history.keys())
            s_last = history[years[-1]]
            if len(years) >= 2:
                s_prev = history[years[-2]]
                trend = s_last - s_prev
                baseline = s_last + (trend * 0.5)
            else:
                baseline = s_last
            predictions['Baseline'] = min(max(baseline, 13.0), 29.9)
        else:
            predictions['Baseline'] = 0.0

        # --- Random Forest ---
        rf_model = self.models.get('RandomForest')
        if rf_model and history:
            try:
                feats = self._create_features(history, student.combination)
                if feats:
                    # Reshape: (1, n_features)
                    pred = rf_model.predict([feats])[0]
                    predictions['RandomForest'] = float(pred)
            except Exception as e:
                print(f"RF Prediction Error: {e}")
                predictions['RandomForest'] = predictions['Baseline']
        
        if 'RandomForest' not in predictions:
            predictions['RandomForest'] = predictions['Baseline']

        # Kết quả cuối cùng
        best_pred = predictions['RandomForest']
        margin = student.total_score - best_pred
        
        return PredictionResult(
            university=student.university,
            major_name=student.major_name,
            major_code=student.major_code,
            combination=student.combination,
            predictions=predictions,
            best_prediction=round(best_pred, 2),
            student_score=round(student.total_score, 2),
            is_passed=(margin >= 0),
            margin=round(margin, 2)
        )

# --- HELPER FUNCTIONS CHO UI (GIỮ NGUYÊN) ---
@st.cache_resource
def load_all_models():
    # Để tương thích ngược với các file khác nếu cần
    return AdmissionPredictor().models

def get_all_schools():
    with Session(engine) as session:
        return session.exec(select(School).order_by(School.code)).all()

def get_majors_by_school(school_id):
    with Session(engine) as session:
        return session.exec(select(Major).join(SchoolMajor).where(SchoolMajor.school_id == school_id).distinct()).all()

def get_combinations_by_school_major(school_id, major_id):
    with Session(engine) as session:
        sm = session.exec(select(SchoolMajor).where(SchoolMajor.school_id == school_id, SchoolMajor.major_id == major_id)).first()
        if not sm: return []
        return session.exec(select(Combination).join(AdmissionScore).where(AdmissionScore.school_major_id == sm.id).distinct()).all()

def get_subjects_of_combination(combo_id):
    with Session(engine) as session:
        c = session.get(Combination, combo_id)
        if not c: return []
        return [session.get(Subject, c.subject1_id).name, session.get(Subject, c.subject2_id).name, session.get(Subject, c.subject3_id).name]

def get_history_dict(school_id, major_id, combo_id):
    # Wrapper để lấy lịch sử cho UI (vẽ biểu đồ)
    with Session(engine) as session:
        sm = session.exec(select(SchoolMajor).where(SchoolMajor.school_id == school_id, SchoolMajor.major_id == major_id)).first()
        if not sm: return {}
        scores = session.exec(select(AdmissionScore).where(AdmissionScore.school_major_id == sm.id, AdmissionScore.combination_id == combo_id)).all()
        return {str(s.year): s.score for s in scores}

# --- RECOMMENDATION LOGIC (Đã tích hợp Predictor mới) ---
def get_recommendations_by_specific_combo(combo_code, user_total_score, current_school_id=None, limit=5):
    predictor = AdmissionPredictor()
    rf_model = predictor.models.get('RandomForest')

    with Session(engine) as session:
        query = (
            select(
                School.code.label("MaTruong"),
                School.name.label("Trường"),
                Major.code.label("MaNganh"),
                Major.name.label("Tên ngành"),
                Combination.code.label("Tổ hợp môn"),
                AdmissionScore.year,
                AdmissionScore.score
            )
            .join(SchoolMajor, AdmissionScore.school_major_id == SchoolMajor.id)
            .join(School, SchoolMajor.school_id == School.id)
            .join(Major, SchoolMajor.major_id == Major.id)
            .join(Combination, AdmissionScore.combination_id == Combination.id)
            .where(Combination.code == combo_code)
            .where(AdmissionScore.year.in_([2022, 2023, 2024]))
        )
        
        if current_school_id:
            query = query.where(School.id == current_school_id)
            
        results = session.exec(query).all()
        
        # Group data by (School, Major) to create history dict for prediction
        grouped_data = {}
        for r in results:
            key = (r.MaTruong, r.MaNganh)
            if key not in grouped_data:
                grouped_data[key] = {
                    "info": r, 
                    "history": {}
                }
            grouped_data[key]["history"][r.year] = r.score

        final_list = []
        
        for (s_code, m_code), data in grouped_data.items():
            history = data["history"]
            info = data["info"]
            
            # Dự báo dùng class mới (hoặc logic feature cũ)
            # Để nhanh, dùng trực tiếp logic feature extraction
            pred_2025 = 0
            if rf_model:
                try:
                    feats = predictor._create_features(history, combo_code)
                    if feats:
                        pred_2025 = float(rf_model.predict([feats])[0])
                except: pass
            
            # Fallback Baseline
            if pred_2025 == 0:
                s24 = history.get(2024, 0)
                s23 = history.get(2023, 0)
                pred_2025 = s24 + (s24 - s23)*0.5 if (s24>0 and s23>0) else s24

            # So sánh
            if user_total_score >= pred_2025 - 0.5:
                final_list.append({
                    "Trường": info.Trường,
                    "Ngành": getattr(info, "Tên ngành"), # Xử lý label có dấu cách
                    "Tổ hợp": getattr(info, "Tổ hợp môn"),
                    "2023": history.get(2023, 0),
                    "2024": history.get(2024, 0),
                    "Dự báo 2025": round(pred_2025, 2),
                    "Chênh lệch": round(user_total_score - pred_2025, 2)
                })

        df = pd.DataFrame(final_list)
        if df.empty: return df
        return df.sort_values(by='Dự báo 2025', ascending=False).head(limit)

def get_lowest_score_majors(limit=5):
    # Logic cũ giữ nguyên
    with Session(engine) as session:
        # ... (Query lấy top thấp nhất 2024)
        statement = (
            select(School.name, Major.name, Combination.code, AdmissionScore.score)
            .join(SchoolMajor, AdmissionScore.school_major_id == SchoolMajor.id)
            .join(School, SchoolMajor.school_id == School.id)
            .join(Major, SchoolMajor.major_id == Major.id)
            .join(Combination, AdmissionScore.combination_id == Combination.id)
            .where(AdmissionScore.year == 2024, AdmissionScore.score > 12)
            .order_by(AdmissionScore.score)
            .limit(limit)
        )
        results = session.exec(statement).all()
        data = []
        for r in results:
            data.append({
                "Trường": r[0], "Ngành": r[1], "Tổ hợp": r[2],
                "2023": 0, "2024": r[3], "Dự báo 2025": r[3], "Chênh lệch": "+"
            })
        return pd.DataFrame(data)

def get_status_text(diff):
    if diff >= 1.0: return "✅ Đậu An Toàn"
    elif diff >= 0: return "🟢 Đậu Sát Nút"
    elif diff >= -0.5: return "⚠️ Nguy Cơ"
    else: return "❌ Khó Đậu"