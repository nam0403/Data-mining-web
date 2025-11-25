import json
import pandas as pd
from sqlmodel import Session, select
from fast_api_service import engine, Subject, Combination, School, Major, SchoolMajor, create_db_and_tables

# ==========================================
# 1. HÀM CHUẨN HÓA TÊN MÔN HỌC
# ==========================================
# File CSV dùng tiếng Việt không dấu (Toan, VatLy...), ta map sang tiếng Việt có dấu
SUBJECT_MAPPING = {
    "Toan": "Toán",
    "VatLy": "Vật lí",
    "HoaHoc": "Hóa học",
    "SinhHoc": "Sinh học",
    "NguVan": "Ngữ văn",
    "LichSu": "Lịch sử",
    "DiaLy": "Địa lí",
    "NgoaiNgu": "Ngoại ngữ",
    "GDCD": "GDCD"
}

def get_or_create_subject(session: Session, raw_name: str) -> int:
    """Tìm ID môn học, nếu chưa có thì tạo mới"""
    clean_name = raw_name.strip()
    # Map sang tên đẹp nếu có, không thì giữ nguyên
    final_name = SUBJECT_MAPPING.get(clean_name, clean_name)
    
    # Kiểm tra DB
    statement = select(Subject).where(Subject.name == final_name)
    subject = session.exec(statement).first()
    
    if not subject:
        subject = Subject(name=final_name)
        session.add(subject)
        session.commit()
        session.refresh(subject)
        print(f"➕ Đã thêm môn: {final_name}")
        
    return subject.id

def seed_combinations(session: Session):
    print("⏳ Đang nạp dữ liệu Tổ hợp môn từ CSV...")
    
    # Đọc file CSV
    # Lưu ý: Cần đảm bảo file nằm cùng thư mục
    try:
        df = pd.read_csv('danh-sach-to-hop-mon.xlsx - danh-sach-to-hop-mon.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file danh-sach-to-hop-mon.csv")
        return

    count = 0
    for index, row in df.iterrows():
        code = row['Group'].strip()
        subjects_str = row['Subjects']
        
        # Tách chuỗi môn học: "Toan, VatLy, HoaHoc" -> ["Toan", "VatLy", "HoaHoc"]
        subj_list = [s.strip() for s in subjects_str.split(',')]
        
        # Chỉ xử lý nếu đủ 3 môn (theo thiết kế DB hiện tại)
        if len(subj_list) == 3:
            # Lấy ID của 3 môn
            s1_id = get_or_create_subject(session, subj_list[0])
            s2_id = get_or_create_subject(session, subj_list[1])
            s3_id = get_or_create_subject(session, subj_list[2])
            
            # Kiểm tra xem tổ hợp tồn tại chưa
            existing = session.exec(select(Combination).where(Combination.code == code)).first()
            if not existing:
                combo = Combination(
                    code=code,
                    subject1_id=s1_id,
                    subject2_id=s2_id,
                    subject3_id=s3_id
                )
                session.add(combo)
                count += 1
    
    session.commit()
    print(f"✅ Đã nạp xong {count} tổ hợp môn.")

def seed_schools_and_majors(session: Session):
    print("⏳ Đang nạp dữ liệu Trường & Ngành từ JSON...")
    
    try:
        with open('MASTER_SCHOOL_CODES.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file MASTER_SCHOOL_CODES.json")
        return

    # Data structure: {"Mã Trường": {"Mã Ngành": "Tên Ngành", ...}}
    
    school_count = 0
    major_count = 0
    link_count = 0

    for school_code, majors_dict in data.items():
        # 1. Tạo/Lấy Trường
        # JSON không có tên trường đầy đủ, tạm thời dùng Mã trường làm Tên trường
        # Sau này có thể update lại tên trường nếu có file mapping khác
        school = session.exec(select(School).where(School.code == school_code)).first()
        if not school:
            school = School(code=school_code, name=f"Trường {school_code}") 
            session.add(school)
            session.commit()
            session.refresh(school)
            school_count += 1
        
        # 2. Duyệt qua các ngành trong trường đó
        for major_code, major_name in majors_dict.items():
            # Tạo/Lấy Ngành (Dựa trên Mã ngành)
            # Lưu ý: Mã ngành có thể trùng tên ở các trường khác nhau, nhưng trong bảng Major
            # ta lưu danh mục duy nhất.
            
            major = session.exec(select(Major).where(Major.code == major_code)).first()
            if not major:
                major = Major(code=major_code, name=major_name)
                session.add(major)
                session.commit()
                session.refresh(major)
                major_count += 1
            
            # 3. Tạo liên kết School - Major
            existing_link = session.exec(select(SchoolMajor).where(
                SchoolMajor.school_id == school.id,
                SchoolMajor.major_id == major.id
            )).first()
            
            if not existing_link:
                link = SchoolMajor(school_id=school.id, major_id=major.id)
                session.add(link)
                link_count += 1
    
    session.commit()
    print(f"✅ Đã xử lý: {school_count} trường mới, {major_count} ngành mới.")
    print(f"🔗 Đã tạo {link_count} liên kết đào tạo (Trường-Ngành).")

def main():
    # 1. Tạo bảng nếu chưa có
    create_db_and_tables()
    
    # 2. Mở session và chạy nạp dữ liệu
    with Session(engine) as session:
        seed_combinations(session)
        seed_schools_and_majors(session)
        print("\n🎉 HOÀN TẤT QUÁ TRÌNH NẠP DỮ LIỆU CẤU TRÚC!")
        print("Lưu ý: Bảng điểm chuẩn (AdmissionScore) hiện tại vẫn trống vì cần file dữ liệu điểm.")

if __name__ == "__main__":
    main()