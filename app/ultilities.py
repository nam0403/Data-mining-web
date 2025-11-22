import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib

# --- CONSTANTS ---
KHOI_THI_MAPPING = {
    'A00': ['Toán', 'Lý', 'Hóa'],
    'A01': ['Toán', 'Lý', 'Anh'],
    'B00': ['Toán', 'Hóa', 'Sinh'],
    'C00': ['Văn', 'Sử', 'Địa'],
    'D01': ['Toán', 'Văn', 'Anh'],
    'D04': ['Toán', 'Văn', 'Tiếng Trung'],
    'D14': ['Văn', 'Sử', 'Anh'],
    'D78': ['Văn', 'KHXH', 'Anh'],
    # Thêm các khối khác nếu cần
}
MODEL_FILE_V2 = 'admission_model_v2.pkl'

# --- 1. HÀM LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_FILE_V2):
        try:
            return joblib.load(MODEL_FILE_V2)
        except Exception as e:
            st.error(f"Lỗi load model: {e}")
    return None

# --- 2. HÀM DỰ BÁO NÂNG CAO ---
def predict_score_advanced(row, artifacts):
    """
    Dự báo dựa trên: Ngành + Tổ hợp + Điểm 3 năm (2022, 2023, 2024)
    """
    try:
        s1 = float(row.get('2022', 0))
        s2 = float(row.get('2023', 0))
        s3 = float(row.get('2024', 0))
        
        # Nếu thiếu dữ liệu lịch sử -> Fallback về trung bình
        if s3 == 0: return 0.0
        if s2 == 0: return s3
        
        if artifacts:
            model = artifacts['model']
            le_major = artifacts['le_major']
            le_combo = artifacts['le_combo']
            
            major_name = row['Tên ngành']
            combo_name = row['Tổ hợp môn']
            
            # Kiểm tra xem Ngành/Tổ hợp này có trong lúc train không?
            # Nếu có mới encode được, không thì dùng thuật toán fallback
            if (major_name in le_major.classes_) and (combo_name in le_combo.classes_):
                major_code = le_major.transform([major_name])[0]
                combo_code = le_combo.transform([combo_name])[0]
                
                # Input vector: [Major, Combo, S1, S2, S3]
                input_vec = np.array([[major_code, combo_code, s1, s2, s3]])
                pred = model.predict(input_vec)[0]
                return min(max(pred, 15.0), 29.9)
        
        # Fallback logic (nếu không có model hoặc ngành mới)
        trend = s3 - s2
        return s3 + (trend * 0.5)

    except Exception as e:
        return 0.0

# --- 3. HÀM TÍNH ĐIỂM TỔ HỢP TỪ KHO ĐIỂM ---
def calculate_combo_score(student_scores, combo_name):
    """
    Input: 
        - student_scores: {'Toán': 8, 'Văn': 7...}
        - combo_name: 'A00'
    Output: Tổng điểm (float) hoặc 0 nếu thiếu điểm
    """
    if combo_name not in KHOI_THI_MAPPING:
        return 0.0
    
    subjects = KHOI_THI_MAPPING[combo_name]
    total = 0.0
    for subj in subjects:
        score = student_scores.get(subj, 0.0)
        # Nếu điểm <= 0 hoặc None coi như không xét được khối này
        if score is None or score <= 0:
            return 0.0
        total += score
    return total

# --- DATA LOADER (Update để dùng hàm predict mới) ---
@st.cache_data
def load_data_with_prediction(file_path='data.csv'):
    # ... (Phần đọc file giữ nguyên như cũ) ...
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    df.columns = df.columns.str.strip()
    if 'Tổ hợp môn' in df.columns:
        df['Tổ hợp môn'] = df['Tổ hợp môn'].astype(str).apply(lambda x: re.split(r'[;,]\s*', x))
        df = df.explode('Tổ hợp môn').reset_index(drop=True)
        df['Tổ hợp môn'] = df['Tổ hợp môn'].str.strip()
        
    year_cols = [str(y) for y in range(2017, 2025)]
    for col in year_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # LOAD MODEL & PREDICT
    artifacts = load_artifacts()
    df['Dự báo 2025'] = df.apply(lambda row: predict_score_advanced(row, artifacts), axis=1)
    
    return df

# ... (Các hàm style, status giữ nguyên) ...
def get_status_text(diff):
    if diff >= 1.5: return "An toàn cao"
    elif diff >= 0.5: return "Khả quan"
    elif diff >= -0.5: return "Cân nhắc"
    else: return "Rủi ro"

def style_recommendation_table(df):
    def color_status(val):
        if val == "An toàn cao": color = '#d4edda' 
        elif val == "Khả quan": color = '#c3e6cb' 
        elif val == "Cân nhắc": color = '#fff3cd' 
        else: color = '#f8d7da' 
        return f'background-color: {color}; color: black; font-weight: bold;'
    
    return df.style.applymap(color_status, subset=['Đánh giá']).format("{:.2f}", subset=['Điểm của bạn', 'Dự báo 2025', 'Dư địa điểm'])