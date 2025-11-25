import pandas as pd
from sqlmodel import Session, select
from fast_api_service import engine, School

# Tên file CSV chứa danh sách tên file
LIST_FILE = 'DANH_SACH_FILE_DA_XU_LY.csv'

def parse_filename(filename):
    """
    Phân tích tên file để lấy Mã trường và Tên trường.
    Format: "MÃ - Tên Trường_filtered.csv"
    """
    try:
        # 1. Loại bỏ đuôi file
        clean_name = filename.replace('_filtered.csv', '').replace('.csv', '')
        
        # 2. Tách Mã và Tên bằng dấu gạch ngang " - "
        # Dùng split(..., 1) để chỉ tách ở dấu gạch ngang đầu tiên
        parts = clean_name.split(' - ', 1)
        
        if len(parts) == 2:
            code = parts[0].strip()
            name = parts[1].strip()
            return code, name
        return None, None
    except Exception:
        return None, None

def update_names():
    print(f"🚀 Bắt đầu cập nhật tên trường từ file {LIST_FILE}...")
    
    try:
        # Đọc file CSV (giả sử cột đầu tiên chứa tên file)
        df = pd.read_csv(LIST_FILE, header=0)
        # Lấy dữ liệu cột đầu tiên bất kể tên cột là gì
        file_list = df.iloc[:, 0].dropna().tolist()
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {LIST_FILE}")
        return

    with Session(engine) as session:
        updated_count = 0
        not_found_count = 0
        
        for fname in file_list:
            code, new_name = parse_filename(fname)
            
            if code and new_name:
                # Tìm trường trong DB theo mã
                statement = select(School).where(School.code == code)
                school = session.exec(statement).first()
                
                if school:
                    # Cập nhật tên mới nếu khác tên cũ
                    if school.name != new_name:
                        print(f"🔄 Cập nhật {code}: {school.name} -> {new_name}")
                        school.name = new_name
                        session.add(school)
                        updated_count += 1
                else:
                    # Trường hợp mã có trong danh sách file nhưng chưa có trong DB
                    # (Có thể bỏ qua hoặc log lại)
                    # print(f"⚠️ Không tìm thấy trường mã {code} trong DB.")
                    not_found_count += 1
        
        # Commit thay đổi vào DB
        session.commit()
        
        print("-" * 40)
        print(f"✅ HOÀN TẤT CẬP NHẬT!")
        print(f"📊 Thống kê:")
        print(f"   - Số trường được cập nhật tên: {updated_count}")
        print(f"   - Số mã trong file không khớp DB: {not_found_count}")

if __name__ == "__main__":
    update_names()