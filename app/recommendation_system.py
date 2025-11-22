import streamlit as st
import ultilities as utils
import pandas as pd

# Danh sách môn phổ biến để tạo form nhập
ALL_SUBJECT_INPUTS = ['Toán', 'Văn', 'Ngoại ngữ (Anh)', 'Lý', 'Hóa', 'Sinh', 'Sử', 'Địa', 'GDCD']

# Mapping tên môn từ input về tên chuẩn trong KHOI_THI_MAPPING
# (Vì form nhập có thể ghi "Ngoại ngữ (Anh)" cho rõ nghĩa)
SUBJECT_MAP_STANDARD = {
    'Ngoại ngữ (Anh)': 'Anh',
    'Ngoại ngữ (Trung)': 'Tiếng Trung',
    # Các môn khác tên giống nhau thì không cần map
}

def standardize_score_dict(raw_scores):
    """Chuẩn hóa tên môn để khớp với KHOI_THI_MAPPING"""
    std_scores = {}
    for k, v in raw_scores.items():
        std_name = SUBJECT_MAP_STANDARD.get(k, k) # Lấy tên chuẩn
        std_scores[std_name] = v
    return std_scores

def show_prediction_system():
    st.title("🤖 Hệ thống Dự báo Điểm chuẩn AI")
    st.markdown("Nhập điểm thi của bạn, hệ thống sẽ tự động ghép tổ hợp và dự báo khả năng đỗ.")

    # Load dữ liệu (đã có cột Dự báo 2025 từ Model)
    df_data = utils.load_data_with_prediction('Data-mining-web/Dataset/diem_chuan_ussh_wide_final (3).csv')
    if df_data.empty:
        st.error("Thiếu file dữ liệu.")
        st.stop()

    ALL_MAJORS = sorted(df_data['Tên ngành'].unique())

    # --- PHẦN 1: NHẬP ĐIỂM (TOÀN BỘ MÔN) ---
    with st.container(border=True):
        st.subheader("1. Nhập điểm thi THPT")
        st.caption("Nhập điểm các môn bạn đã thi (thang điểm 10).")
        
        cols = st.columns(3)
        raw_scores = {}
        for i, subj in enumerate(ALL_SUBJECT_INPUTS):
            with cols[i % 3]:
                raw_scores[subj] = st.number_input(f"Điểm {subj}", 0.0, 10.0, step=0.25, key=f"in_{i}")
        
        # Chuẩn hóa tên môn
        std_scores = standardize_score_dict(raw_scores)

    # --- PHẦN 2: CHỌN NGÀNH MỤC TIÊU ---
    st.markdown("---")
    st.subheader("2. Chọn Ngành Xét Tuyển")
    
    col_sel, col_act = st.columns([3, 1])
    with col_sel:
        target_major = st.selectbox("Tìm kiếm ngành bạn muốn vào:", ALL_MAJORS, index=None, placeholder="Ví dụ: Tâm lý học...")
    
    # Nút Action
    analyze_clicked = False
    with col_act:
        st.write("") # Spacer
        st.write("")
        if st.button("🔮 DỰ BÁO NGAY", type="primary", use_container_width=True):
            analyze_clicked = True

    # --- PHẦN 3: XỬ LÝ VÀ HIỂN THỊ ---
    if analyze_clicked and target_major:
        # 1. Lọc dữ liệu ngành mục tiêu
        df_target = df_data[df_data['Tên ngành'] == target_major].copy()
        
        if df_target.empty:
            st.warning("Không có dữ liệu cho ngành này.")
        else:
            results = []
            # 2. Duyệt qua các tổ hợp của ngành này
            for idx, row in df_target.iterrows():
                combo = row['Tổ hợp môn']
                predicted_score = row['Dự báo 2025']
                
                # 3. Tự động tính điểm của thí sinh cho tổ hợp này
                my_score = utils.calculate_combo_score(std_scores, combo)
                
                if my_score > 0: # Chỉ hiện nếu thí sinh có điểm > 0 cho tổ hợp này
                    diff = my_score - predicted_score
                    results.append({
                        'Tổ hợp môn': combo,
                        'Điểm của bạn': my_score,
                        'Dự báo 2025': predicted_score,
                        'Dư địa điểm': diff,
                        'Đánh giá': utils.get_status_text(diff)
                    })
            
            st.divider()
            if not results:
                st.error(f"Bạn chưa nhập đủ điểm cho các tổ hợp xét tuyển của ngành {target_major}.")
                st.write(f"Ngành này xét các khối: {', '.join(df_target['Tổ hợp môn'].unique())}")
            else:
                # Tạo DataFrame kết quả
                df_res = pd.DataFrame(results).sort_values(by='Dư địa điểm', ascending=False)
                best_option = df_res.iloc[0] # Tổ hợp tốt nhất
                
                # Hiển thị kết quả tốt nhất
                st.success(f"✅ Tổ hợp tối ưu nhất cho bạn: **{best_option['Tổ hợp môn']}**")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Điểm xét tuyển của bạn", f"{best_option['Điểm của bạn']:.2f}")
                c2.metric(f"Điểm chuẩn dự báo (AI)", f"{best_option['Dự báo 2025']:.2f}")
                c3.metric("Khả năng đậu", best_option['Đánh giá'], delta=f"{best_option['Dư địa điểm']:.2f}")
                
                # Hiển thị bảng chi tiết
                st.markdown("#### Chi tiết các tổ hợp khác:")
                st.dataframe(
                    utils.style_recommendation_table(df_res),
                    use_container_width=True,
                    hide_index=True
                )

    # --- PHẦN 4: GỢI Ý NGÀNH KHÁC (RECOMMENDATION) ---
    # Logic: Quét toàn bộ database xem điểm này đậu được ngành nào
    if analyze_clicked:
        st.markdown("---")
        st.subheader("💡 Gợi ý: Các ngành khác bạn có cơ hội trúng tuyển")
        
        rec_results = []
        # Duyệt toàn bộ data
        for idx, row in df_data.iterrows():
            # Bỏ qua ngành đang chọn ở trên
            if row['Tên ngành'] == target_major: continue
            
            combo = row['Tổ hợp môn']
            my_score = utils.calculate_combo_score(std_scores, combo)
            
            if my_score > 0:
                pred = row['Dự báo 2025']
                diff = my_score - pred
                # Chỉ lấy ngành an toàn hoặc rủi ro thấp
                if diff >= -0.5: 
                    rec_results.append({
                        'Tên ngành': row['Tên ngành'],
                        'Tổ hợp môn': combo,
                        'Điểm của bạn': my_score,
                        'Dự báo 2025': pred,
                        'Dư địa điểm': diff,
                        'Đánh giá': utils.get_status_text(diff)
                    })
        
        if rec_results:
            df_rec = pd.DataFrame(rec_results)
            # Mỗi ngành chỉ lấy tổ hợp tốt nhất
            df_rec = df_rec.sort_values(by='Dư địa điểm', ascending=False).drop_duplicates(subset=['Tên ngành'])
            
            st.dataframe(
                utils.style_recommendation_table(df_rec.head(10)), # Top 10 ngành
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Với mức điểm hiện tại, hệ thống chưa tìm thấy ngành gợi ý phù hợp.")