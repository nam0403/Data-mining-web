import pandas as pd
import os
from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select

# ==========================================
# 1. TỪ ĐIỂN MAPPING (QUAN TRỌNG)
# ==========================================
# Hệ thống sẽ dựa vào đây để biết A00 gồm những môn gì
COMBINATION_MAPPING = {
    'A00': ['Toán', 'Vật lý', 'Hóa học'],
    'A01': ['Toán', 'Vật lý', 'Tiếng Anh'],
    'A02': ['Toán', 'Vật lý', 'Sinh học'],
    'B00': ['Toán', 'Hóa học', 'Sinh học'],
    'B08': ['Toán', 'Sinh học', 'Tiếng Anh'],
    'C00': ['Ngữ văn', 'Lịch sử', 'Địa lý'],
    'C01': ['Ngữ văn', 'Toán', 'Vật lý'],
    'C02': ['Ngữ văn', 'Toán', 'Hóa học'],
    'C03': ['Ngữ văn', 'Toán', 'Lịch sử'],
    'C04': ['Ngữ văn', 'Toán', 'Địa lý'],
    'C14': ['Ngữ văn', 'Toán', 'GDCD'],
    'C15': ['Ngữ văn', 'Toán', 'KHXH'],
    'C19': ['Ngữ văn', 'Lịch sử', 'GDCD'],
    'C20': ['Ngữ văn', 'Địa lý', 'GDCD'],
    'D01': ['Toán', 'Ngữ văn', 'Tiếng Anh'],
    'D02': ['Toán', 'Ngữ văn', 'Tiếng Nga'],
    'D03': ['Toán', 'Ngữ văn', 'Tiếng Pháp'],
    'D04': ['Toán', 'Ngữ văn', 'Tiếng Trung'],
    'D05': ['Toán', 'Ngữ văn', 'Tiếng Đức'],
    'D06': ['Toán', 'Ngữ văn', 'Tiếng Nhật'],
    'D07': ['Toán', 'Hóa học', 'Tiếng Anh'],
    'D08': ['Toán', 'Sinh học', 'Tiếng Anh'],
    'D09': ['Toán', 'Lịch sử', 'Tiếng Anh'],
    'D10': ['Toán', 'Địa lý', 'Tiếng Anh'],
    'D11': ['Ngữ văn', 'Vật lý', 'Tiếng Anh'],
    'D12': ['Ngữ văn', 'Hóa học', 'Tiếng Anh'],
    'D13': ['Ngữ văn', 'Sinh học', 'Tiếng Anh'],
    'D14': ['Ngữ văn', 'Lịch sử', 'Tiếng Anh'],
    'D15': ['Ngữ văn', 'Địa lý', 'Tiếng Anh'],
    'D66': ['Ngữ văn', 'GDCD', 'Tiếng Anh'],
    'D78': ['Ngữ văn', 'KHXH', 'Tiếng Anh'],
    'D84': ['Toán', 'GDCD', 'Tiếng Anh'],
    'D90': ['Toán', 'KHTN', 'Tiếng Anh'],
    'D96': ['Toán', 'KHXH', 'Tiếng Anh'],
    'H00': ['Ngữ văn', 'Năng khiếu 1', 'Năng khiếu 2'],
    'V00': ['Toán', 'Vật lý', 'Mỹ thuật'],
    'V01': ['Toán', 'Ngữ văn', 'Mỹ thuật'],
}

# ==========================================
# 2. DATABASE MODELS
# ==========================================
class Subject(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

class School(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True) 
    name: str 

class Major(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True) 
    name: str

class Combination(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True) 
    subject1_id: int = Field(foreign_key="subject.id")
    subject2_id: int = Field(foreign_key="subject.id")
    subject3_id: int = Field(foreign_key="subject.id")

class SchoolMajor(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    school_id: int = Field(foreign_key="school.id")
    major_id: int = Field(foreign_key="major.id")

# --- BẢNG MỚI: PredictionData ---
class PredictionData(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    
    school_major_id: int = Field(foreign_key="schoolmajor.id")
    combination_id: int = Field(foreign_key="combination.id")
    
    target_year: int
    
    # Các cột dữ liệu từ file CSV
    pred_ensemble: float      # Điểm dự báo
    pred_rf: Optional[float] = None
    pred_xgb: Optional[float] = None
    pred_cat: Optional[float] = None
    
    actual_score: Optional[float] = None # Điểm thực tế (nếu có)
    error_ensemble: Optional[float] = None

# ==========================================
# 3. LOGIC IMPORT
# ==========================================
def get_or_create_subject(session, name):
    """Tìm hoặc tạo môn học"""
    obj = session.exec(select(Subject).where(Subject.name == name)).first()
    if not obj:
        obj = Subject(name=name)
        session.add(obj)
        session.commit()
        session.refresh(obj)
    return obj.id

def ensure_combination_exists(session, code):
    """
    Tìm Combination theo code.
    Nếu chưa có -> Lấy danh sách môn từ MAPPING -> Tạo môn -> Tạo Combination
    """
    # 1. Kiểm tra xem tổ hợp đã có trong DB chưa
    combo = session.exec(select(Combination).where(Combination.code == code)).first()
    if combo:
        return combo.id
    
    # 2. Nếu chưa có, tra cứu từ điển mapping
    subjects = COMBINATION_MAPPING.get(code)
    
    if not subjects:
        print(f"⚠️ Cảnh báo: Không tìm thấy định nghĩa môn học cho tổ hợp '{code}'. Bỏ qua.")
        return None

    # 3. Tạo/Lấy ID cho 3 môn học
    try:
        s1_id = get_or_create_subject(session, subjects[0])
        s2_id = get_or_create_subject(session, subjects[1])
        s3_id = get_or_create_subject(session, subjects[2])

        # 4. Tạo tổ hợp mới
        new_combo = Combination(
            code=code,
            subject1_id=s1_id,
            subject2_id=s2_id,
            subject3_id=s3_id
        )
        session.add(new_combo)
        session.commit()
        session.refresh(new_combo)
        return new_combo.id
    except Exception as e:
        print(f"❌ Lỗi khi tạo tổ hợp {code}: {e}")
        return None

def import_predictions(csv_path: str, db_name: str = "admission_db.sqlite"):
    sqlite_url = f"sqlite:///{db_name}"
    engine = create_engine(sqlite_url)
    
    # Tạo bảng nếu chưa có
    SQLModel.metadata.create_all(engine)
    
    print(f"🔄 Đang đọc file CSV: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return

    # Cache ID để tăng tốc độ import (tránh query DB quá nhiều)
    school_cache = {}
    major_cache = {}
    school_major_cache = {}
    combo_cache = {}

    with Session(engine) as session:
        count = 0
        for idx, row in df.iterrows():
            try:
                # --- A. Xử lý Trường (School) ---
                s_code = str(row['university']).strip()
                # Nếu CSV không có cột tên trường, dùng tạm code làm tên
                s_name = s_code 
                
                if s_code not in school_cache:
                    school = session.exec(select(School).where(School.code == s_code)).first()
                    if not school:
                        school = School(code=s_code, name=s_name)
                        session.add(school)
                        session.commit()
                        session.refresh(school)
                    school_cache[s_code] = school.id
                
                # --- B. Xử lý Ngành (Major) ---
                m_code = str(row['major_code']).strip()
                m_name = str(row['major_name']).strip()
                
                if m_code not in major_cache:
                    major = session.exec(select(Major).where(Major.code == m_code)).first()
                    if not major:
                        major = Major(code=m_code, name=m_name)
                        session.add(major)
                        session.commit()
                        session.refresh(major)
                    major_cache[m_code] = major.id

                # --- C. Xử lý Liên kết Trường-Ngành (SchoolMajor) ---
                s_id = school_cache[s_code]
                m_id = major_cache[m_code]
                sm_key = (s_id, m_id)
                
                if sm_key not in school_major_cache:
                    sm = session.exec(select(SchoolMajor).where(SchoolMajor.school_id==s_id, SchoolMajor.major_id==m_id)).first()
                    if not sm:
                        sm = SchoolMajor(school_id=s_id, major_id=m_id)
                        session.add(sm)
                        session.commit()
                        session.refresh(sm)
                    school_major_cache[sm_key] = sm.id

                # --- D. Xử lý Tổ hợp (Combination) ---
                c_code = str(row['combination']).strip()
                
                if c_code not in combo_cache:
                    # Gọi hàm helper đã viết ở trên
                    c_id = ensure_combination_exists(session, c_code)
                    if c_id:
                        combo_cache[c_code] = c_id
                    else:
                        # Nếu không tạo được tổ hợp (do thiếu mapping), bỏ qua dòng này
                        continue
                
                # --- E. Lưu dữ liệu vào bảng PredictionData ---
                # Kiểm tra trùng lặp trước khi thêm (SchoolMajor + Combination + Year)
                sm_id = school_major_cache[sm_key]
                cmb_id = combo_cache[c_code]
                target_year = int(row['target_year'])
                
                # Logic này tùy chọn: Xóa cái cũ ghi đè cái mới, hoặc bỏ qua nếu đã tồn tại
                # Ở đây mình làm đơn giản là thêm mới luôn (nếu bạn muốn update hãy thêm logic check)
                
                pred = PredictionData(
                    school_major_id=sm_id,
                    combination_id=cmb_id,
                    target_year=target_year,
                    actual_score=float(row['diem_chuan']) if pd.notna(row['diem_chuan']) else 0.0,
                    pred_ensemble=float(row['pred_Ensemble']),
                    pred_rf=float(row['pred_RandomForest']),
                    pred_xgb=float(row['pred_XGBoost']),
                    pred_cat=float(row['pred_CatBoost']),
                    error_ensemble=float(row['error_Ensemble']) if 'error_Ensemble' in row else 0.0
                )
                session.add(pred)
                count += 1
                
            except Exception as e:
                print(f"❌ Lỗi dòng {idx}: {e}")
                continue
        
        session.commit()
        print(f"✅ Đã nhập thành công {count} dòng dự báo vào bảng PredictionData!")

# ==========================================
# CHẠY SCRIPT
# ==========================================
if __name__ == "__main__":
    # Đặt tên file CSV của bạn ở đây
    csv_filename = "../Dataset/predictions_2025.csv"
    
    if os.path.exists(csv_filename):
        import_predictions(csv_filename)
    else:
        print(f"Không tìm thấy file {csv_filename}. Vui lòng kiểm tra lại.")