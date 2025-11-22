import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
}
MODEL_FILE = 'admission_model.pkl'
MODEL_FILE_V2 = 'admission_model_v2.pkl'

# --- 1. HÀM LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    """Load model và encoder đã train nếu có file .pkl"""
    if os.path.exists(MODEL_FILE_V2):
        try:
            return joblib.load(MODEL_FILE_V2)
        except:
            pass
    return None

# --- 2. HÀM DỰ BÁO (CORE LOGIC) ---
def predict_score_advanced(row, artifacts, custom_years=None):
    """
    Dự báo điểm chuẩn.
    custom_years: List [year1, year2, year3] để custom input (dùng cho backtest).
    Mặc định là ['2022', '2023', '2024'] để dự báo 2025.
    """
    try:
        if custom_years:
            y1, y2, y3 = custom_years
        else:
            y1, y2, y3 = '2022', '2023', '2024'

        s1 = float(row.get(y1, 0))
        s2 = float(row.get(y2, 0))
        s3 = float(row.get(y3, 0))
        
        if s3 == 0: return 0.0
        if s2 == 0: return s3 
        
        # ƯU TIÊN 1: Dùng Model AI
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
                return min(max(pred, 15.0), 29.9)
        
        # ƯU TIÊN 2: Fallback
        trend = s3 - s2
        return s3 + (trend * 0.5)

    except Exception:
        return 0.0

# --- 3. HÀM BACKTEST (MỚI) ---
def run_backtest(df, artifacts):
    """
    Chạy kiểm thử: Dùng dữ liệu 2021, 2022, 2023 để dự báo 2024.
    Sau đó so sánh với điểm thực tế 2024.
    """
    # Tạo bản sao để không ảnh hưởng df gốc
    df_test = df.copy()
    
    # Kiểm tra xem có đủ dữ liệu các năm để test không
    required_years = ['2021', '2022', '2023', '2024']
    for y in required_years:
        if y not in df_test.columns:
            return pd.DataFrame() # Không đủ dữ liệu để test

    # Dự báo 2024 giả định (Backtest Prediction)
    df_test['Backtest_2024'] = df_test.apply(
        lambda row: predict_score_advanced(row, artifacts, custom_years=['2021', '2022', '2023']), 
        axis=1
    )
    
    # Tính sai số
    # Chỉ lấy các dòng mà 2024 thực tế > 0 và dự báo > 0
    df_valid = df_test[(df_test['2024'] > 0) & (df_test['Backtest_2024'] > 0)].copy()
    
    if not df_valid.empty:
        df_valid['Sai_Số'] = df_valid['2024'] - df_valid['Backtest_2024']
        df_valid['Sai_Số_Tuyệt_Đối'] = df_valid['Sai_Số'].abs()
    
    return df_valid

# --- 4. HÀM TÍNH ĐIỂM TỔ HỢP ---
def calculate_combo_score(student_scores, combo_name):
    if combo_name not in KHOI_THI_MAPPING: return 0.0
    total = 0.0
    for subj in KHOI_THI_MAPPING[combo_name]:
        score = student_scores.get(subj, 0.0)
        if score is None or score <= 0: return 0.0
        total += score
    return total

# --- 5. DATA LOADER ---
@st.cache_data
def load_data_with_prediction(file_input='data.csv'):
    is_string_path = isinstance(file_input, str)
    if is_string_path and not os.path.exists(file_input):
        return pd.DataFrame()

    try:
        filename = file_input if is_string_path else file_input.name
        if filename.endswith('.csv'):
            df = pd.read_csv(file_input)
        else:
            df = pd.read_excel(file_input)
            
        df.columns = df.columns.str.strip()
        if 'Tổ hợp môn' in df.columns:
            df['Tổ hợp môn'] = df['Tổ hợp môn'].astype(str).apply(lambda x: re.split(r'[;,]\s*', x))
            df = df.explode('Tổ hợp môn').reset_index(drop=True)
            df['Tổ hợp môn'] = df['Tổ hợp môn'].str.strip()
        
        year_cols = [str(y) for y in range(2017, 2025)]
        for col in year_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        artifacts = load_artifacts()
        if 'Tên ngành' in df.columns and '2024' in df.columns:
            # Dự báo thật cho 2025
            df['Dự báo 2025'] = df.apply(lambda row: predict_score_advanced(row, artifacts), axis=1)
        else:
             df['Dự báo 2025'] = 0.0
        
        return df

    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return pd.DataFrame()

# --- 6. STYLING ---
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
    
    styler = df.style.applymap(color_status, subset=['Đánh giá'])
    cols_to_format = ['Điểm của bạn', 'Dự báo 2025', 'Dư địa điểm', '2023', '2024']
    valid_cols = [c for c in cols_to_format if c in df.columns]
    if valid_cols:
        styler = styler.format("{:.2f}", subset=valid_cols)
    return styler