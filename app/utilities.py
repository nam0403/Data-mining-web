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
}
MODEL_FILE = 'admission_model.pkl'
MODEL_FILE_V2 = 'admission_model_v2.pkl'

# --- 1. HÀM LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_FILE_V2):
        try:
            return joblib.load(MODEL_FILE_V2)
        except Exception as e:
            # st.warning(f"Không load được model V2: {e}")
            pass
    return None

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            return None
    return None

# --- 2. HÀM DỰ BÁO ---
def predict_score_advanced(row, artifacts):
    try:
        s1 = float(row.get('2022', 0))
        s2 = float(row.get('2023', 0))
        s3 = float(row.get('2024', 0))
        
        if s3 == 0: return 0.0
        if s2 == 0: return s3
        
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
        
        trend = s3 - s2
        return s3 + (trend * 0.5)

    except Exception:
        return 0.0

# --- 3. HÀM TÍNH ĐIỂM ---
def calculate_combo_score(student_scores, combo_name):
    if combo_name not in KHOI_THI_MAPPING:
        return 0.0
    
    subjects = KHOI_THI_MAPPING[combo_name]
    total = 0.0
    for subj in subjects:
        score = student_scores.get(subj, 0.0)
        if score is None or score <= 0:
            return 0.0
        total += score
    return total

# --- 4. DATA LOADER ---
@st.cache_data
def load_data_with_prediction(file_input='data.csv'):
    """
    Hàm load dữ liệu thông minh:
    - Nếu input là chuỗi (str): Kiểm tra xem file có trên ổ cứng không.
    - Nếu input là UploadedFile (từ nút upload): Đọc trực tiếp không cần kiểm tra path.
    """
    # Kiểm tra xem input là đường dẫn string hay object file
    is_string_path = isinstance(file_input, str)

    # Nếu là đường dẫn string thì mới kiểm tra tồn tại trên ổ cứng
    if is_string_path and not os.path.exists(file_input):
        return pd.DataFrame() # Trả về bảng rỗng nếu không thấy file

    try:
        # Lấy tên file để quyết định dùng read_csv hay read_excel
        # Nếu là string thì dùng chính nó, nếu là object thì dùng thuộc tính .name
        filename = file_input if is_string_path else file_input.name
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_input)
        else:
            df = pd.read_excel(file_input)
            
        # Chuẩn hóa tên cột (xóa khoảng trắng thừa)
        df.columns = df.columns.str.strip()
        
        # Xử lý tách tổ hợp môn (Ví dụ: "C00, D01" -> tách thành 2 dòng riêng)
        if 'Tổ hợp môn' in df.columns:
            df['Tổ hợp môn'] = df['Tổ hợp môn'].astype(str).apply(lambda x: re.split(r'[;,]\s*', x))
            df = df.explode('Tổ hợp môn').reset_index(drop=True)
            df['Tổ hợp môn'] = df['Tổ hợp môn'].str.strip()
        
        # Ép kiểu số cho các cột Năm (2017 -> 2024)
        year_cols = [str(y) for y in range(2017, 2025)]
        for col in year_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Load model và chạy dự báo
        artifacts = load_artifacts()
        
        # Chỉ chạy dự báo nếu file có đủ cột cần thiết
        if 'Tên ngành' in df.columns and '2024' in df.columns:
            df['Dự báo 2025'] = df.apply(lambda row: predict_score_advanced(row, artifacts), axis=1)
        else:
             df['Dự báo 2025'] = 0.0
        
        return df

    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return pd.DataFrame()


# --- 5. STYLING ---
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