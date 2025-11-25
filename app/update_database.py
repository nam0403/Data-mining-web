import pandas as pd
import numpy as np
from sqlmodel import Session, select, delete
from fast_api_service import engine, School, Major, Combination, SchoolMajor, AdmissionScore, create_db_and_tables

# Tên file CSV bạn vừa upload
CSV_FILE = 'diem-chuan-all-cleaned-merged-15006-keep-2024-2023.csv'

def clean_transaction_data(session: Session):
    """
    Xóa dữ liệu điểm và liên kết cũ để nạp mới sạch sẽ.
    Giữ lại bảng Subject, Combination vì đó là dữ liệu nền (Master Data).
    """
    print("🧹 Đang dọn dẹp dữ liệu cũ...")
    session.exec(delete(AdmissionScore))
    session.exec(delete(SchoolMajor))
    session.commit()
    print("✅ Đã xóa dữ liệu điểm và liên kết trường-ngành cũ.")

def get_or_create_school(session: Session, code: str, cache: dict) -> int:
    if code in cache:
        return cache[code]
    
    # Tìm trong DB
    school = session.exec(select(School).where(School.code == code)).first()
    if not school:
        # Nếu chưa có, tạo mới (Tạm dùng Code làm Name, sau này update sau)
        school = School(code=code, name=f"Trường {code}")
        session.add(school)
        session.commit()
        session.refresh(school)
    
    cache[code] = school.id
    return school.id

def get_or_create_major(session: Session, code: str, name: str, cache: dict) -> int:
    # Mã ngành là duy nhất (VD: 7480201)
    if code in cache:
        return cache[code]
    
    major = session.exec(select(Major).where(Major.code == code)).first()
    if not major:
        major = Major(code=code, name=name)
        session.add(major)
        session.commit()
        session.refresh(major)
    else:
        # Update tên ngành nếu có thay đổi (tùy chọn)
        if major.name != name:
            major.name = name
            session.add(major)
            session.commit()
            
    cache[code] = major.id
    return major.id

def get_combination_id(session: Session, code: str, cache: dict) -> int:
    if code in cache:
        return cache[code]
    
    combo = session.exec(select(Combination).where(Combination.code == code)).first()
    if combo:
        cache[code] = combo.id
        return combo.id
    return None

def import_data():
    print(f"🚀 Bắt đầu nạp dữ liệu từ {CSV_FILE}...")
    
    try:
        df = pd.read_csv(CSV_FILE, dtype={'Mã ngành': str, '2025': float}) # Ép kiểu mã ngành thành chuỗi
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {CSV_FILE}")
        return

    # Chuẩn hóa tên cột (Xóa khoảng trắng thừa nếu có)
    df.columns = df.columns.str.strip()
    
    # Các cột năm điểm chuẩn
    year_cols = [str(y) for y in range(2017, 2026)] # 2017 -> 2025
    
    with Session(engine) as session:
        # 1. Dọn dẹp dữ liệu cũ
        clean_transaction_data(session)
        
        # Cache để tăng tốc độ (tránh query DB liên tục)
        school_cache = {}
        major_cache = {}
        combo_cache = {}
        
        # Pre-load Combinations vào cache
        combos = session.exec(select(Combination)).all()
        for c in combos:
            combo_cache[c.code] = c.id
            
        print("⏳ Đang xử lý từng dòng dữ liệu (Điều này có thể mất vài phút)...")
        
        count_scores = 0
        count_links = 0
        
        for index, row in df.iterrows():
            # 1. Xử lý Trường
            school_code = str(row['university']).strip()
            school_id = get_or_create_school(session, school_code, school_cache)
            
            # 2. Xử lý Ngành
            major_code = str(row['Mã ngành']).strip()
            major_name = str(row['Tên ngành']).strip()
            major_id = get_or_create_major(session, major_code, major_name, major_cache)
            
            # 3. Tạo liên kết Trường - Ngành (Nếu chưa có trong phiên này)
            # Vì ta đã xóa SchoolMajor ở đầu, nên ta cần tạo lại.
            # Tuy nhiên trong vòng lặp này có thể lặp lại (1 ngành có nhiều tổ hợp -> nhiều dòng)
            # Nên cần check xem đã add trong session này chưa hoặc check DB.
            # Cách tối ưu: Kiểm tra DB
            link = session.exec(select(SchoolMajor).where(
                SchoolMajor.school_id == school_id,
                SchoolMajor.major_id == major_id
            )).first()
            
            if not link:
                link = SchoolMajor(school_id=school_id, major_id=major_id)
                session.add(link)
                session.commit()
                session.refresh(link)
                count_links += 1
            
            # 4. Xử lý Tổ hợp & Điểm
            combo_code = str(row['Tổ hợp môn']).strip()
            combo_id = get_combination_id(session, combo_code, combo_cache)
            
            if combo_id:
                # Duyệt qua các cột năm
                for year in year_cols:
                    if year in df.columns:
                        score_val = row[year]
                        # Kiểm tra điểm hợp lệ (không phải NaN và > 0)
                        if pd.notna(score_val) and score_val > 0:
                            score_record = AdmissionScore(
                                school_major_id=link.id,
                                combination_id=combo_id,
                                year=int(year),
                                score=float(score_val)
                            )
                            session.add(score_record)
                            count_scores += 1
            else:
                # Log warning nếu tổ hợp chưa có trong bảng Combination (Do chưa chạy seed_real_data.py hoặc file thiếu)
                # print(f"⚠️ Cảnh báo: Tổ hợp {combo_code} chưa có trong hệ thống. Bỏ qua điểm dòng {index}.")
                pass
                
            # Commit theo batch (ví dụ mỗi 100 dòng) hoặc commit cuối cùng
            if index % 100 == 0:
                session.commit()
                print(f"   -> Đã xử lý {index} dòng...")

        session.commit()
        print("------------------------------------------------")
        print(f"🎉 HOÀN TẤT!")
        print(f"📊 Thống kê:")
        print(f"   - Số liên kết Trường-Ngành mới: {count_links}")
        print(f"   - Số bản ghi điểm chuẩn đã nạp: {count_scores}")

if __name__ == "__main__":
    create_db_and_tables()
    import_data()