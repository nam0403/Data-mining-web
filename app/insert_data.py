import json
import pandas as pd
import numpy as np
import re
from sqlmodel import Session, select, delete
from fast_api_service import engine, School, Major, Combination, SchoolMajor, AdmissionScore, Subject, create_db_and_tables

# --- TÊN FILE DỮ LIỆU ---
FILE_JSON_MASTER = 'MASTER_SCHOOL_CODES.json'
FILE_CSV_COMBOS = 'danh-sach-to-hop-mon.csv'
FILE_CSV_SCORES = 'diem-chuan-all-cleaned-merged-15006-keep-2024-2023.csv'

# Mapping tên môn học sang tiếng Việt có dấu
SUBJECT_MAPPING = {
    "Toan": "Toán", "VatLy": "Vật lí", "HoaHoc": "Hóa học",
    "SinhHoc": "Sinh học", "NguVan": "Ngữ văn", "LichSu": "Lịch sử",
    "DiaLy": "Địa lí", "NgoaiNgu": "Ngoại ngữ", "GDCD": "GDCD"
}

def reset_database(session: Session):
    """Xóa toàn bộ dữ liệu cũ để nạp mới"""
    print("🧹 Đang dọn dẹp Database cũ...")
    session.exec(delete(AdmissionScore))
    session.exec(delete(SchoolMajor))
    session.exec(delete(Combination))
    session.exec(delete(Major))
    session.exec(delete(School))
    session.exec(delete(Subject))
    session.commit()
    print("✅ Đã xóa sạch dữ liệu.")

def get_or_create_subject(session: Session, raw_name: str, cache: dict) -> int:
    clean_name = raw_name.strip()
    final_name = SUBJECT_MAPPING.get(clean_name, clean_name)
    
    if final_name in cache: return cache[final_name]
    
    subj = session.exec(select(Subject).where(Subject.name == final_name)).first()
    if not subj:
        subj = Subject(name=final_name)
        session.add(subj)
        session.commit()
        session.refresh(subj)
    
    cache[final_name] = subj.id
    return subj.id

def get_active_school_codes():
    """Quét file CSV điểm chuẩn để lấy danh sách các trường CÓ dữ liệu"""
    try:
        df = pd.read_csv(FILE_CSV_SCORES, usecols=['university'])
        # Lấy danh sách mã trường duy nhất, loại bỏ khoảng trắng
        active_codes = set(df['university'].astype(str).str.strip().unique())
        print(f"🔍 Tìm thấy {len(active_codes)} trường có dữ liệu điểm chuẩn.")
        return active_codes
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {FILE_CSV_SCORES}")
        return set()

def import_master_data(session: Session, active_schools: set):
    """Nạp Môn học, Tổ hợp, Danh mục Trường/Ngành từ JSON (CÓ LỌC)"""
    print("🚀 Bắt đầu nạp Master Data...")
    
    # 1. NẠP TỔ HỢP MÔN
    subj_cache = {}
    combo_cache = {}
    
    try:
        df_combo = pd.read_csv(FILE_CSV_COMBOS)
        print(f"   - Đọc {len(df_combo)} dòng từ file tổ hợp.")
        
        for _, row in df_combo.iterrows():
            code = row['Group'].strip()
            subjects = [s.strip() for s in row['Subjects'].split(',')]
            
            if len(subjects) == 3:
                s1 = get_or_create_subject(session, subjects[0], subj_cache)
                s2 = get_or_create_subject(session, subjects[1], subj_cache)
                s3 = get_or_create_subject(session, subjects[2], subj_cache)
                
                combo = Combination(code=code, subject1_id=s1, subject2_id=s2, subject3_id=s3)
                session.add(combo)
                session.commit()
                session.refresh(combo)
                combo_cache[code] = combo.id
    except Exception as e:
        print(f"❌ Lỗi nạp tổ hợp: {e}")

    # 2. NẠP DANH MỤC TRƯỜNG & NGÀNH TỪ JSON (CÓ LỌC)
    school_cache = {}
    major_cache = {}
    
    try:
        with open(FILE_JSON_MASTER, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"   - Đọc danh mục gốc từ JSON: {len(data)} trường.")
        
        skipped_count = 0
        imported_count = 0
        
        for s_code, majors in data.items():
            # --- LOGIC LỌC: CHỈ NẠP NẾU CÓ TRONG DANH SÁCH ACTIVE ---
            if s_code not in active_schools:
                skipped_count += 1
                continue # Bỏ qua trường này
            
            # Tạo Trường
            school = School(code=s_code, name=f"Trường {s_code}")
            session.add(school)
            session.commit()
            session.refresh(school)
            school_cache[s_code] = school.id
            imported_count += 1
            
            # Tạo Ngành
            for m_code, m_name in majors.items():
                if m_code not in major_cache:
                    major = Major(code=m_code, name=m_name)
                    session.add(major)
                    session.commit()
                    session.refresh(major)
                    major_cache[m_code] = major.id
        
        print(f"   - Đã nạp: {imported_count} trường. Bỏ qua: {skipped_count} trường không có dữ liệu điểm.")
                    
    except Exception as e:
        print(f"❌ Lỗi nạp JSON Master: {e}")

    return school_cache, major_cache, combo_cache

def import_admission_scores(session: Session, school_cache, major_cache, combo_cache):
    """Nạp dữ liệu điểm chuẩn từ CSV lớn"""
    print("🚀 Bắt đầu nạp Dữ liệu Điểm chuẩn (Transaction Data)...")
    
    try:
        df = pd.read_csv(FILE_CSV_SCORES, dtype={'Mã ngành': str})
        print(f"   - Tìm thấy {len(df)} dòng dữ liệu điểm.")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {FILE_CSV_SCORES}")
        return

    # Chuẩn hóa cột
    df.columns = df.columns.str.strip()
    year_cols = [str(y) for y in range(2017, 2026)]
    
    count_links = 0
    count_scores = 0
    
    # Cache cục bộ cho SchoolMajor
    link_cache = {}

    for index, row in df.iterrows():
        try:
            # 1. Xử lý Trường
            s_code = str(row['university']).strip()
            
            # Nếu trường này chưa có trong cache (tức là chưa được tạo ở bước Master Data hoặc bị lọc)
            # Ta sẽ tạo mới nó ở đây để đảm bảo không mất dữ liệu điểm
            # (Trường hợp file CSV có mã trường mà JSON không có)
            if s_code not in school_cache:
                school = School(code=s_code, name=f"Trường {s_code}")
                session.add(school)
                session.commit()
                session.refresh(school)
                school_cache[s_code] = school.id
            
            s_id = school_cache[s_code]
            
            # 2. Xử lý Ngành
            m_code = str(row['Mã ngành']).strip()
            m_name = str(row['Tên ngành']).strip()
            
            if m_code not in major_cache:
                major = Major(code=m_code, name=m_name)
                session.add(major)
                session.commit()
                session.refresh(major)
                major_cache[m_code] = major.id
            m_id = major_cache[m_code]
            
            # 3. Tạo/Lấy liên kết Trường-Ngành
            link_key = f"{s_id}_{m_id}"
            if link_key in link_cache:
                sm_id = link_cache[link_key]
            else:
                link = session.exec(select(SchoolMajor).where(
                    SchoolMajor.school_id == s_id,
                    SchoolMajor.major_id == m_id
                )).first()
                
                if not link:
                    link = SchoolMajor(school_id=s_id, major_id=m_id)
                    session.add(link)
                    session.commit()
                    session.refresh(link)
                    count_links += 1
                
                sm_id = link.id
                link_cache[link_key] = sm_id
            
            # 4. Xử lý Điểm
            c_code = str(row['Tổ hợp môn']).strip()
            if c_code in combo_cache:
                c_id = combo_cache[c_code]
                
                for year in year_cols:
                    if year in df.columns:
                        val = row[year]
                        if pd.notna(val) and float(val) > 0:
                            score_rec = AdmissionScore(
                                school_major_id=sm_id,
                                combination_id=c_id,
                                year=int(year),
                                score=float(val)
                            )
                            session.add(score_rec)
                            count_scores += 1
            
            if index % 500 == 0:
                session.commit()
                print(f"   -> Đã xử lý {index} dòng...")
                
        except Exception as e:
            print(f"⚠️ Lỗi dòng {index}: {e}")
            continue

    session.commit()
    print("------------------------------------------------")
    print("🎉 NẠP DỮ LIỆU THÀNH CÔNG!")
    print(f"📊 Tổng kết:")
    print(f"   - Trường: {len(school_cache)}")
    print(f"   - Ngành: {len(major_cache)}")
    print(f"   - Liên kết đào tạo: {len(link_cache)}")
    print(f"   - Bản ghi điểm chuẩn: {count_scores}")

def main():
    create_db_and_tables()
    with Session(engine) as session:
        # 1. Xóa cũ
        reset_database(session)
        
        # 2. Lấy danh sách trường có điểm chuẩn để lọc
        active_schools = get_active_school_codes()
        
        # 3. Nạp Master Data (Có lọc)
        s_cache, m_cache, c_cache = import_master_data(session, active_schools)
        
        # 4. Nạp Transaction Data
        import_admission_scores(session, s_cache, m_cache, c_cache)

if __name__ == "__main__":
    main()