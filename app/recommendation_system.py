import streamlit as st
import utilities as utils
import pandas as pd
import plotly.express as px # Thêm thư viện vẽ biểu đồ
import os

# Danh sách môn phổ biến
ALL_SUBJECT_INPUTS = ['Toán', 'Văn', 'Ngoại ngữ (Anh)', 'Lý', 'Hóa', 'Sinh', 'Sử', 'Địa', 'GDCD']

SUBJECT_MAP_STANDARD = {
    'Ngoại ngữ (Anh)': 'Anh',
    'Ngoại ngữ (Trung)': 'Tiếng Trung',
}

def standardize_score_dict(raw_scores):
    std_scores = {}
    for k, v in raw_scores.items():
        std_name = SUBJECT_MAP_STANDARD.get(k, k)
        std_scores[std_name] = v
    return std_scores

# Hàm tạo tên hiển thị đẹp cho biểu đồ (Giống Dashboard)
def create_display_name(row):
    major_name = row['Tên ngành']
    combo = row['Tổ hợp môn']
    is_clc = 'CLC' in major_name.upper() or 'CHẤT LƯỢNG CAO' in major_name.upper()
    type_label = " (CLC)" if is_clc else ""
    return f"{major_name}{type_label} - {combo}"

def show_prediction_system():
    st.title("🤖 Hệ thống Dự báo & Tư vấn Tuyển sinh")
    st.markdown("Nhập điểm thi để nhận dự báo chi tiết và biểu đồ xu hướng điểm chuẩn.")
    
    # --- LOAD DATA ---
    default_file = 'data.csv'
    if os.path.exists(default_file):
        df_data = utils.load_data_with_prediction(default_file)
    else:
        st.warning(f"⚠️ Không tìm thấy file '{default_file}'.")
        uploaded_file = st.file_uploader("Tải lên file dữ liệu (data.csv):", type=['csv', 'xlsx'])
        if uploaded_file:
            df_data = utils.load_data_with_prediction(uploaded_file)
        else:
            st.stop()

    if df_data.empty:
        st.error("Dữ liệu rỗng.")
        st.stop()

    ALL_MAJORS = sorted(df_data['Tên ngành'].unique())

    # --- PHẦN 1: NHẬP ĐIỂM ---
    with st.container(border=True):
        st.subheader("1. Nhập điểm thi THPT")
        cols = st.columns(3)
        raw_scores = {}
        for i, subj in enumerate(ALL_SUBJECT_INPUTS):
            with cols[i % 3]:
                raw_scores[subj] = st.number_input(f"Điểm {subj}", 0.0, 10.0, step=0.25, key=f"in_{i}")
        
        std_scores = standardize_score_dict(raw_scores)

    # --- PHẦN 2: CHỌN NGÀNH ---
    st.markdown("---")
    st.subheader("2. Phân tích Ngành Mục tiêu")
    
    col_sel, col_act = st.columns([3, 1])
    with col_sel:
        target_major = st.selectbox("Chọn ngành bạn quan tâm:", ALL_MAJORS, index=None, placeholder="Ví dụ: Báo chí...")
    
    analyze_clicked = False
    with col_act:
        st.write("") 
        st.write("")
        if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
            analyze_clicked = True

    # --- PHẦN 3: KẾT QUẢ PHÂN TÍCH ---
    if analyze_clicked and target_major:
        # Lọc dữ liệu ngành mục tiêu
        df_target = df_data[df_data['Tên ngành'] == target_major].copy()
        
        if df_target.empty:
            st.warning("Không có dữ liệu cho ngành này.")
        else:
            # 3.1 Tính toán điểm thí sinh
            results = []
            for idx, row in df_target.iterrows():
                combo = row['Tổ hợp môn']
                predicted_score = row['Dự báo 2025']
                my_score = utils.calculate_combo_score(std_scores, combo)
                
                if my_score > 0:
                    diff = my_score - predicted_score
                    results.append({
                        'Tổ hợp môn': combo,
                        'Điểm của bạn': my_score,
                        'Dự báo 2025': predicted_score,
                        'Dư địa điểm': diff,
                        'Đánh giá': utils.get_status_text(diff)
                    })
            
            st.divider()
            
            # Nếu chưa nhập điểm
            if not results:
                st.error(f"⚠️ Bạn chưa nhập đủ điểm cho các khối xét tuyển của ngành này: {', '.join(df_target['Tổ hợp môn'].unique())}")
            else:
                df_res = pd.DataFrame(results).sort_values(by='Dư địa điểm', ascending=False)
                best_option = df_res.iloc[0]
                
                # --- HIỂN THỊ KẾT QUẢ TỐI ƯU ---
                st.success(f"✅ Tổ hợp tối ưu nhất: **{best_option['Tổ hợp môn']}**")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Điểm của bạn", f"{best_option['Điểm của bạn']:.2f}")
                c2.metric("Dự báo Điểm chuẩn 2025", f"{best_option['Dự báo 2025']:.2f}")
                c3.metric("Đánh giá cơ hội", best_option['Đánh giá'], delta=f"{best_option['Dư địa điểm']:.2f}")

                # --- VISUALIZATION: BIỂU ĐỒ XU HƯỚNG (LINE CHART) ---
                st.markdown("#### 📈 Xu hướng điểm chuẩn qua các năm")
                
                # Chuẩn bị dữ liệu vẽ chart
                year_cols = [c for c in df_target.columns if c.isdigit()]
                year_cols = sorted(year_cols)
                
                if year_cols:
                    # Tạo tên hiển thị đẹp (Tách CLC)
                    df_target['Tên hiển thị'] = df_target.apply(create_display_name, axis=1)
                    
                    df_melt = df_target.melt(
                        id_vars=['Tên hiển thị', 'Tổ hợp môn'], 
                        value_vars=year_cols, 
                        var_name='Năm', 
                        value_name='Điểm chuẩn'
                    )
                    df_melt = df_melt[df_melt['Điểm chuẩn'] > 0]
                    
                    fig = px.line(
                        df_melt, 
                        x='Năm', y='Điểm chuẩn', 
                        color='Tên hiển thị', 
                        markers=True,
                        title=f"Lịch sử điểm chuẩn ngành {target_major}",
                        labels={'Tên hiển thị': 'Tổ hợp & Hệ'}
                    )
                    # Thêm điểm dự báo 2025 vào biểu đồ (nếu muốn)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- BẢNG CHI TIẾT ---
                st.markdown("#### 📋 Chi tiết các tổ hợp")
                st.dataframe(utils.style_recommendation_table(df_res), use_container_width=True, hide_index=True)

    # --- PHẦN 4: GỢI Ý NGÀNH KHÁC (RECOMMENDATION) ---
    if analyze_clicked:
        st.markdown("---")
        st.subheader("💡 Gợi ý: Các ngành phù hợp với điểm của bạn")
        
        rec_results = []
        for idx, row in df_data.iterrows():
            if row['Tên ngành'] == target_major: continue
            
            combo = row['Tổ hợp môn']
            my_score = utils.calculate_combo_score(std_scores, combo)
            
            if my_score > 0:
                pred = row['Dự báo 2025']
                diff = my_score - pred
                # Lấy ngành an toàn hoặc rủi ro thấp (lệch không quá -1 điểm)
                if diff >= -1.0: 
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
            # Lấy top 10 ngành tốt nhất, mỗi ngành chỉ lấy 1 tổ hợp cao điểm nhất
            df_rec = df_rec.sort_values(by='Dư địa điểm', ascending=False).drop_duplicates(subset=['Tên ngành']).head(10)
            
            # --- VISUALIZATION: SO SÁNH ĐIỂM (BAR CHART) ---
            st.caption("Biểu đồ so sánh: Điểm của bạn vs Điểm Dự báo cho Top 5 ngành phù hợp nhất")
            
            top_5_chart = df_rec.head(5).copy()
            # Tạo tên hiển thị ngắn gọn cho chart
            top_5_chart['Label'] = top_5_chart['Tên ngành'] + " (" + top_5_chart['Tổ hợp môn'] + ")"
            
            # Melt dữ liệu để vẽ Grouped Bar Chart
            df_bar = top_5_chart.melt(
                id_vars=['Label'], 
                value_vars=['Điểm của bạn', 'Dự báo 2025'], 
                var_name='Loại điểm', 
                value_name='Điểm số'
            )
            
            fig_bar = px.bar(
                df_bar, 
                x='Điểm số', 
                y='Label', 
                color='Loại điểm', 
                barmode='group',
                orientation='h', # Biểu đồ ngang cho dễ đọc tên ngành dài
                title="Top 5 Ngành Tiềm năng nhất",
                color_discrete_map={'Điểm của bạn': '#2ECC71', 'Dự báo 2025': '#3498DB'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- BẢNG DỮ LIỆU GỢI Ý ---
            st.dataframe(utils.style_recommendation_table(df_rec), use_container_width=True, hide_index=True)
        else:
            st.info("Với mức điểm hiện tại, chưa tìm thấy ngành gợi ý phù hợp trong vùng an toàn.")