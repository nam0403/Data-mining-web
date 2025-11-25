import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sqlmodel import Session, select, col
from fast_api_service import engine, School, Major, SchoolMajor, Combination, Subject, AdmissionScore

# --- CONSTANTS & PATHS ---
MODEL_RF_PATH = 'admission_model_v2.pkl'
MODEL_XGB_PATH = 'admission_model_xgb.pkl'
MODEL_CAT_PATH = 'admission_model_cat.pkl'

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_all_models():
    models = {}
    if os.path.exists(MODEL_RF_PATH):
        try:
            models['RandomForest'] = joblib.load(MODEL_RF_PATH)
        except: pass
    return models

# --- 2. LOGIC DỰ BÁO ---
def predict_multimodel(row_data, models):
    results = {}
    s23 = float(row_data.get('2023', 0))
    s24 = float(row_data.get('2024', 0))
    
    # Baseline
    if s24 > 0 and s23 > 0:
        trend = s24 - s23
        baseline = s24 + (trend * 0.5)
    elif s24 > 0:
        baseline = s24
    else:
        baseline = 0
    results['Baseline'] = min(max(baseline, 13.0), 29.9)

    # Random Forest
    if 'RandomForest' in models:
        artifacts = models['RandomForest']
        pred_rf = predict_score_advanced(row_data, artifacts)
        results['RandomForest'] = pred_rf
    else:
        results['RandomForest'] = results['Baseline']

    return results

def predict_score_advanced(row, artifacts):
    try:
        s1 = float(row.get('2022', 0))
        s2 = float(row.get('2023', 0))
        s3 = float(row.get('2024', 0))
        
        if s3 == 0: return 0.0
        
        if artifacts:
            model = artifacts['model']
            le_major = artifacts['le_major']
            le_combo = artifacts['le_combo']
            major_name = row.get('Tên ngành', '')
            combo_name = row.get('Tổ hợp môn', '')
            
            if (major_name in le_major.classes_) and (combo_name in le_combo.classes_):
                major_code = le_major.transform([major_name])[0]
                combo_code = le_combo.transform([combo_name])[0]
                input_vec = np.array([[major_code, combo_code, s1, s2, s3]])
                pred = model.predict(input_vec)[0]
                return float(pred)
        return s3
    except:
        return 0.0

# --- 3. TRUY VẤN DATABASE ---
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
    with Session(engine) as session:
        sm = session.exec(select(SchoolMajor).where(SchoolMajor.school_id == school_id, SchoolMajor.major_id == major_id)).first()
        if not sm: return {}
        scores = session.exec(select(AdmissionScore).where(AdmissionScore.school_major_id == sm.id, AdmissionScore.combination_id == combo_id)).all()
        return {str(s.year): s.score for s in scores}

# --- 4. RECOMMENDATION ENGINE (SPECIFIC COMBO) ---

def get_recommendations_by_specific_combo(combo_code, user_total_score, current_school_id=None, limit=5):
    """
    Tìm kiếm và Dự báo điểm 2025 cho TẤT CẢ các ngành có cùng tổ hợp (combo_code).
    So sánh với điểm tổng của thí sinh.
    """
    models = load_all_models()
    rf_model = models.get('RandomForest')

    with Session(engine) as session:
        # 1. Tìm tất cả các ngành xét tuyển bằng tổ hợp này (combo_code)
        # Lấy kèm dữ liệu điểm 2024 để lọc sơ bộ cho nhanh
        query = (
            select(
                School.name.label("Trường"),
                Major.name.label("Tên ngành"),
                Combination.code.label("Tổ hợp môn"),
                SchoolMajor.id.label("sm_id"),
                Combination.id.label("c_id"),
                AdmissionScore.score
            )
            .join(SchoolMajor, AdmissionScore.school_major_id == SchoolMajor.id)
            .join(School, SchoolMajor.school_id == School.id)
            .join(Major, SchoolMajor.major_id == Major.id)
            .join(Combination, AdmissionScore.combination_id == Combination.id)
            .where(Combination.code == combo_code)  # CHỈ LỌC THEO TỔ HỢP NÀY
            .where(AdmissionScore.year == 2024)
            # Lọc sơ bộ: Điểm chuẩn 2024 không được quá cao so với điểm thí sinh (biên độ 3 điểm)
            .where(AdmissionScore.score <= user_total_score + 3.0)
            .where(AdmissionScore.score >= user_total_score - 5.0)
        )
        
        # Nếu cần tìm trong trường cụ thể (Kịch bản 1, 2a)
        if current_school_id:
            query = query.where(School.id == current_school_id)
            
        candidates = session.exec(query).all()
        
        final_results = []
        
        # 2. Duyệt qua từng ứng viên -> Lấy lịch sử -> Chạy Model -> So sánh
        for item in candidates:
            # Lấy lịch sử điểm 3 năm gần nhất
            scores = session.exec(select(AdmissionScore).where(
                AdmissionScore.school_major_id == item.sm_id,
                AdmissionScore.combination_id == item.c_id,
                AdmissionScore.year.in_([2022, 2023, 2024])
            )).all()
            
            score_map = {str(s.year): s.score for s in scores}
            
            # --- CHẠY DỰ BÁO AI ---
            row_data = score_map.copy()
            row_data.update({'Tên ngành': item[1], 'Tổ hợp môn': item[2]}) # col2=Major, col3=Combo
            
            if rf_model:
                pred_2025 = predict_score_advanced(row_data, rf_model)
            else:
                # Fallback Baseline
                s23 = score_map.get('2023', 0)
                s24 = score_map.get('2024', 0)
                pred_2025 = s24 + (s24 - s23)*0.5 if s23 > 0 else s24

            # --- SO SÁNH VỚI ĐIỂM THÍ SINH ---
            # Logic: Chỉ recommend nếu Điểm thí sinh >= Dự báo (hoặc thiếu chút xíu <= 0.5)
            print(f"Recommend: {item[0]} - {item[1]} | Predicted 2025: {pred_2025} | User Score: {user_total_score}")
            if user_total_score >= pred_2025 - 0.5:
                
                final_results.append({
                    "Trường": item[0], # School Name
                    "Ngành": item[1],  # Major Name
                    "Tổ hợp": item[2], # Combo Code
                    "2023": round(score_map.get('2023', 0), 2),
                    "2024": round(score_map.get('2024', 0), 2),
                    "Dự báo 2025": round(pred_2025, 2),
                    "Chênh lệch": round(user_total_score - pred_2025, 2)
                })
        
        df = pd.DataFrame(final_results)
        if df.empty: return df
        
        # Sắp xếp: Ưu tiên ngành điểm cao nhất mà thí sinh đậu (gần với năng lực nhất)
        df = df.sort_values(by='Dự báo 2025', ascending=False)
        
        return df.head(limit)

def get_lowest_score_majors(limit=5):
    with Session(engine) as session:
        statement = (
            select(
                School.name.label("Trường"),
                Major.name.label("Ngành"),
                Combination.code.label("Tổ hợp"),
                AdmissionScore.score.label("2024")
            )
            .join(SchoolMajor, AdmissionScore.school_major_id == SchoolMajor.id)
            .join(School, SchoolMajor.school_id == School.id)
            .join(Major, SchoolMajor.major_id == Major.id)
            .join(Combination, AdmissionScore.combination_id == Combination.id)
            .where(AdmissionScore.year == 2024)
            .where(AdmissionScore.score > 12) 
            .order_by(AdmissionScore.score) 
            .limit(limit)
        )
        results = session.exec(statement).all()
        data = []
        for r in results:
            s = round(r[3], 2)
            data.append({
                "Trường": r[0], "Ngành": r[1], "Tổ hợp": r[2],
                "2023": 0, "2024": s, "Dự báo 2025": s
            })
        return pd.DataFrame(data)

def get_status_text(diff):
    if diff >= 1.0: return "✅ Đậu An Toàn"
    elif diff >= 0: return "🟢 Đậu Sát Nút"
    elif diff >= -0.5: return "⚠️ Nguy Cơ"
    else: return "❌ Khó Đậu"