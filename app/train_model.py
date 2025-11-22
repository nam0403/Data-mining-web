import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# --- CẤU HÌNH ---
DATA_PATH = 'C:/Users/phuon/OneDrive/Máy tính/Data mining web/Dataset/diem_chuan_ussh_wide_final (3).csv'
MODEL_PATH = 'admission_model_v2.pkl' # Model mới
ENCODER_PATH = 'encoders.pkl'         # File lưu bộ mã hóa
WINDOW_SIZE = 3                       # Dùng 3 năm cũ để đoán năm mới

def train_advanced_model():
    print("⏳ Đang xử lý dữ liệu huấn luyện...")
    
    # 1. Load Data
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH)
    
    # Clean Data
    df.columns = df.columns.str.strip()
    if 'Tổ hợp môn' in df.columns:
        df['Tổ hợp môn'] = df['Tổ hợp môn'].astype(str).apply(lambda x: re.split(r'[;,]\s*', x))
        df = df.explode('Tổ hợp môn').reset_index(drop=True)
        df['Tổ hợp môn'] = df['Tổ hợp môn'].str.strip()

    year_cols = sorted([c for c in df.columns if c.isdigit()])
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. MÃ HÓA (ENCODING) NGÀNH VÀ TỔ HỢP
    # Máy học không hiểu chữ "Báo chí", phải đổi thành số (ví dụ: 10)
    le_major = LabelEncoder()
    le_combo = LabelEncoder()
    
    df['Major_Encoded'] = le_major.fit_transform(df['Tên ngành'])
    df['Combo_Encoded'] = le_combo.fit_transform(df['Tổ hợp môn'])

    # 3. TẠO TRAINING SET
    # Input (X): [Mã_Ngành, Mã_TổHợp, Điểm_Năm_1, Điểm_Năm_2, Điểm_Năm_3]
    # Output (y): [Điểm_Năm_4]
    X = []
    y = []

    for idx, row in df.iterrows():
        major_code = row['Major_Encoded']
        combo_code = row['Combo_Encoded']
        scores = row[year_cols].values
        
        for i in range(len(scores) - WINDOW_SIZE):
            window = scores[i : i + WINDOW_SIZE]
            target = scores[i + WINDOW_SIZE]
            
            if np.all(window > 0) and target > 0:
                # Feature vector: [Major, Combo, Year1, Year2, Year3]
                feature_vector = np.concatenate(([major_code, combo_code], window))
                X.append(feature_vector)
                y.append(target)

    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 Kích thước dữ liệu train: {X.shape}")

    # 4. Huấn luyện (Dùng Random Forest để xử lý tốt cả Category và Số)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 5. Lưu Model và Encoders (Cần lưu cả 2 để lúc dự đoán còn dùng lại)
    artifacts = {
        'model': model,
        'le_major': le_major,
        'le_combo': le_combo
    }
    joblib.dump(artifacts, MODEL_PATH)
    print(f"✅ Đã lưu Model v2 và Encoders tại: {MODEL_PATH}")

if __name__ == "__main__":
    train_advanced_model()