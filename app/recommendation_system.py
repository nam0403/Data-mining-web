import streamlit as st
import utilities as utils
import pandas as pd
import plotly.express as px

def show_prediction_system():
    # --- CONFIG UI & CSS ---
    st.markdown("""
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        
        /* Custom Card Style */
        .metric-card {
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        /* Light Mode Colors */
        [data-theme="light"] .metric-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
        }
        
        /* Status Colors */
        .status-pass { color: #059669 !important; } /* Green */
        .status-fail { color: #dc2626 !important; } /* Red */
        .bg-pass { background-color: #ecfdf5; border: 1px solid #a7f3d0; }
        .bg-fail { background-color: #fef2f2; border: 1px solid #fecaca; }

        /* Button Style */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
            font-size: 16px;
            margin-top: 10px;
        }
        
        /* Remove extra padding for dataframe to fit content */
        .stDataFrame {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Hệ Thống Tư Vấn Tuyển Sinh AI")
    st.markdown("Nhập thông tin hồ sơ để nhận dự báo và gợi ý nguyện vọng tối ưu.")

    # --- INITIALIZE PREDICTOR ---
    # Sử dụng Session State để giữ instance của predictor (tránh reload model nhiều lần)
    if 'predictor' not in st.session_state:
        st.session_state.predictor = utils.AdmissionPredictor()
    
    predictor = st.session_state.predictor
    schools = utils.get_all_schools()
    
    if not schools:
        st.error("⚠️ Không tìm thấy dữ liệu trường học. Vui lòng kiểm tra Database.")
        st.stop()

    # =========================================================
    # PHẦN 1: NHẬP LIỆU (LAYOUT NGANG - GIỮ NGUYÊN BAN ĐẦU)
    # =========================================================
    with st.container(border=True):
        st.subheader("📝 Hồ Sơ Thí Sinh")
        
        # Hàng 1: Trường & Ngành
        c1, c2 = st.columns(2)
        
        # 1. Chọn Trường
        # Mapping: "Mã - Tên" -> ID
        # Cần giữ lại Mã trường để tạo StudentInput sau này
        # Fix: Lưu object School thay vì ID để truy cập được code và name sau này
        school_options = {f"{s.code} - {s.name}": s for s in schools}
        
        selected_school_obj = None
        school_id = None

        with c1:
            sel_school_label = st.selectbox(
                "1. Chọn Trường Đại học:",
                options=list(school_options.keys()),
                index=None,
                placeholder="Tìm kiếm trường..."
            )
            if sel_school_label:
                selected_school_obj = school_options.get(sel_school_label)
                school_id = selected_school_obj.id

        # 2. Chọn Ngành (Lọc động)
        major_id = None
        sel_major_label = None
        selected_major_obj = None
        
        with c2:
            if school_id:
                majors = utils.get_majors_by_school(school_id)
                if majors:
                    # Store major object as value: "Mã - Tên" -> Major Object
                    major_options = {f"{m.code} - {m.name}": m for m in majors}
                    sel_major_label = st.selectbox(
                        "2. Chọn Ngành học:",
                        options=list(major_options.keys()),
                        index=None,
                        placeholder="Chọn ngành..."
                    )
                    if sel_major_label:
                        selected_major_obj = major_options.get(sel_major_label)
                        major_id = selected_major_obj.id
                else:
                    st.warning("Trường này chưa có dữ liệu ngành.")
            else:
                st.selectbox("2. Chọn Ngành học:", [], disabled=True, placeholder="Vui lòng chọn trường trước")

        # Hàng 2: Tổ hợp & Điểm Vùng
        c3, c4 = st.columns(2)
        
        combo_id = None
        combo_code = None
        region_bonus = 0.0
        
        with c3:
            if major_id:
                combos = utils.get_combinations_by_school_major(school_id, major_id)
                if combos:
                    # Combo object: (id, code)
                    combo_options = {c.code: c.id for c in combos}
                    combo_code = st.selectbox(
                        "3. Chọn Tổ hợp xét tuyển:",
                        options=list(combo_options.keys()),
                        index=None,
                        placeholder="Chọn khối..."
                    )
                    if combo_code:
                        combo_id = combo_options.get(combo_code)
                else:
                    st.warning("Chưa có dữ liệu tổ hợp.")
            else:
                st.selectbox("3. Chọn Tổ hợp xét tuyển:", [], disabled=True, placeholder="Vui lòng chọn ngành trước")
                
        with c4:
            region_bonus = st.selectbox(
                "4. Điểm ưu tiên (Khu vực/Đối tượng):", 
                options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.75],
                format_func=lambda x: f"+{x} điểm" if x > 0 else "Không có"
            )

        # Hàng 3: Nhập Điểm Thi (Chỉ hiện khi chọn xong tổ hợp)
        scores = {}
        analyze = False
        
        if combo_id:
            st.divider()
            st.markdown(f"**🎯 Nhập điểm thi tổ hợp {combo_code}:**")
            
            subjects = utils.get_subjects_of_combination(combo_id)
            if subjects:
                # Dàn hàng ngang các môn
                cols = st.columns(len(subjects))
                for i, subj in enumerate(subjects):
                    with cols[i]:
                        scores[subj] = st.number_input(
                            f"Điểm {subj}",
                            min_value=0.0, max_value=10.0, step=0.25,
                            key=f"score_{subj}"
                        )
                
                st.write("")
                analyze = st.button("🚀 PHÂN TÍCH KẾT QUẢ", type="primary")
            else:
                st.error("Lỗi dữ liệu môn học.")

    # =========================================================
    # PHẦN 2: KẾT QUẢ DỰ BÁO (HIỆN ĐẠI HÓA)
    # =========================================================
    
    # Biến lưu trữ kết quả để dùng cho phần Recommend ở dưới
    is_analyzed = False
    prediction_result = None
    
    if analyze:
        is_analyzed = True
        
        # 1. Tạo đối tượng StudentInput
        # Đảm bảo selected_school_obj và selected_major_obj không None trước khi truy cập
        if selected_school_obj and selected_major_obj:
            student_input = utils.StudentInput(
                university=selected_school_obj.code, 
                major_name=selected_major_obj.name,  
                major_code=selected_major_obj.code, 
                combination=combo_code,              
                subject_scores=scores,               
                priority_score=region_bonus
            )
            
            # 2. Gọi hàm dự báo từ class Predictor
            prediction_result = predictor.predict(student_input)
            
            # Lấy dữ liệu từ kết quả để hiển thị
            final_pred = prediction_result.best_prediction
            user_total_score = prediction_result.student_score
            diff = prediction_result.margin
            pass_status = prediction_result.is_passed
            
            # Status text logic
            status_label = "ĐẬU AN TOÀN" if pass_status else "NGUY CƠ TRƯỢT"
            if diff >= 0 and diff < 1.0: status_label = "CƠ HỘI CAO"
            if diff < 0 and diff >= -0.5: status_label = "CÂN NHẮC"

            st.markdown("---")
            
            # --- A. CARDS KẾT QUẢ ---
            st.subheader(f"📊 Kết Quả Dự Báo: {prediction_result.major_name}")
            
            bg_class = "bg-pass" if pass_status else "bg-fail"
            text_class = "status-pass" if pass_status else "status-fail"

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase;">Điểm Của Bạn</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #1E40AF;">{user_total_score:.2f}</div>
                    <div style="font-size: 0.8rem; color: #9CA3AF;">(Thi: {student_input.raw_score:.2f} + Vùng: {region_bonus})</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase;">Dự Báo AI 2025</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #4B5563;">{final_pred:.2f}</div>
                    <div style="font-size: 0.8rem; color: #9CA3AF;">(Random Forest Model)</div>
                </div>
                <div class="metric-card {bg_class}">
                    <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase;">Khả Năng Đậu</div>
                    <div class="{text_class}" style="font-size: 1.6rem; font-weight: 800; padding: 5px 0;">{status_label}</div>
                    <div style="font-size: 1rem; font-weight: 600;">{diff:+.2f} điểm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- B. BIỂU ĐỒ & BẢNG CHI TIẾT (2 CỘT) ---
            col_chart, col_data = st.columns([1.5, 1])
            
            # Lấy lại lịch sử để vẽ biểu đồ
            history = predictor.get_historical_cutoffs(
                student_input.university, 
                student_input.major_code, 
                student_input.combination
            )
            
            with col_chart:
                st.markdown("**📉 Xu hướng điểm chuẩn**")
                # Chuẩn bị data
                clean_history = {}
                if history:
                    for k, v in history.items():
                        clean_history[k] = v
                
                df_chart = pd.DataFrame(list(clean_history.items()), columns=['Năm', 'Điểm'])
                if not df_chart.empty:
                    df_chart = df_chart.sort_values('Năm')
                    
                    fig = px.line(df_chart, x='Năm', y='Điểm', markers=True)
                    fig.add_scatter(x=[2025], y=[final_pred], mode='markers+text', 
                                    name='Dự báo', text=[f"{final_pred:.2f}"], textposition="top center",
                                    marker=dict(color='red', size=12, symbol='star'))
                    fig.add_scatter(x=[2025], y=[user_total_score], mode='markers', 
                                    name='Điểm của bạn', 
                                    marker=dict(color='green', size=10, symbol='circle'))
                    
                    fig.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=20, r=20, t=10, b=20), height=300, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Chưa có dữ liệu biểu đồ.")
                
            with col_data:
                st.markdown("**📋 Lịch sử & Dự báo**")
                if history:
                    # Chuyển history thành bảng ngang
                    clean_history['2025 (Dự báo)'] = final_pred
                    
                    # Tạo bảng ngang
                    display_row = {str(k): v for k, v in clean_history.items()}
                    
                    df_hist = pd.DataFrame([display_row])
                    
                    # Sắp xếp cột
                    cols = sorted([c for c in df_hist.columns if c != '2025 (Dự báo)']) + ['2025 (Dự báo)']
                    df_hist = df_hist.reindex(columns=cols)
                    
                    st.dataframe(
                        df_hist.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.info("Không có dữ liệu lịch sử.")

    # =========================================================
    # PHẦN 3: RECOMMENDATION (NẰM DƯỚI CÙNG - FULL WIDTH)
    # =========================================================
    if is_analyzed and prediction_result:
        st.markdown("---")
        st.markdown("### 💡 Gợi Ý Nguyện Vọng Tối Ưu")
        
        rec_df = pd.DataFrame()
        msg_type = "success"
        msg_title = ""
        msg_content = ""
        
        # Flag hiển thị cột trường
        show_school_col = False

        # --- LOGIC KỊCH BẢN ---
        if prediction_result.is_passed:
            # KỊCH BẢN 1: ĐẬU -> Gợi ý cùng trường
            msg_type = "success"
            msg_title = "Chúc mừng! Kết quả rất khả quan."
            msg_content = f"Bạn có khả năng cao trúng tuyển ngành **{prediction_result.major_name}**. Dưới đây là các ngành khác tại **{selected_school_obj.name}** cùng khối **{prediction_result.combination}** mà bạn cũng có thể đậu:"
            
            rec_df = utils.get_recommendations_by_specific_combo(
                prediction_result.combination, 
                prediction_result.student_score, 
                current_school_id=school_id, 
                limit=5
            )
            show_school_col = False
            
        else:
            msg_type = "warning"
            msg_title = "Cần cân nhắc kỹ!"
            msg_content = f"Điểm của bạn đang thấp hơn dự báo ngành **{prediction_result.major_name}**. Hệ thống đề xuất các lựa chọn **An Toàn Hơn**:"
            
            # CASE 2: TRƯỢT -> Tìm trong trường (Ẩn tên trường)
            rec_df = utils.get_recommendations_by_specific_combo(
                prediction_result.combination, 
                prediction_result.student_score, 
                current_school_id=school_id, 
                limit=5
            )
            
            if not rec_df.empty:
                show_school_col = False
            else:
                # CASE 3 (Fallback của 2): Tìm toàn hệ thống (Hiện tên trường)
                msg_content += " (Đã mở rộng tìm kiếm sang các trường khác do không tìm thấy ngành phù hợp tại trường này)"
                rec_df = utils.get_recommendations_by_specific_combo(
                    prediction_result.combination, 
                    prediction_result.student_score, 
                    current_school_id=None, 
                    limit=10
                )
                show_school_col = True

        # CASE 3 (Thuần túy): Điểm quá thấp, trượt hết -> Gợi ý ngành thấp nhất (Hiện tên trường)
        if rec_df.empty:
            st.warning("😔 Với mức điểm hiện tại, rất khó tìm được ngành phù hợp xét tuyển khối này.")
            st.info("💪 **Lời khuyên:** Bạn có thể tham khảo các ngành có điểm chuẩn thấp nhất trên toàn hệ thống dưới đây:")
            rec_df = utils.get_lowest_score_majors(limit=10)
            show_school_col = True
        else:
            if msg_type == "success":
                st.success(f"**{msg_title}** {msg_content}")
            else:
                st.warning(f"**{msg_title}** {msg_content}")

        # --- HIỂN THỊ BẢNG RECOMMENDATION ---
        if not rec_df.empty:
            # Chọn và đổi tên cột
            display_cols = {
                'Trường': 'Trường Đại Học',
                'Ngành': 'Ngành Học',
                'Tổ hợp': 'Khối',
                '2023': 'Điểm 2023',
                '2024': 'Điểm 2024',
                'Dự báo 2025': 'Dự Báo 2025',
                'Chênh lệch': '+/- Điểm'
            }
            
            # Logic ẩn/hiện cột trường dựa trên cờ show_school_col
            if not show_school_col and 'Trường' in display_cols:
                del display_cols['Trường']

            valid_cols = [c for c in display_cols.keys() if c in rec_df.columns]
            rec_display = rec_df[valid_cols].rename(columns=display_cols)
            
            # Style tô màu chênh lệch
            def style_diff_color(val):
                try:
                    val_float = float(val)
                    return 'color: #059669; font-weight: bold;' if val_float >= 0 else 'color: #dc2626; font-weight: bold;'
                except: return ''

            # Tính toán chiều cao động
            num_rows = len(rec_display)
            row_height = 35
            dynamic_height = (num_rows + 1) * row_height + 5 

            # Cấu hình hiển thị
            column_config = {
                "Ngành Học": st.column_config.TextColumn("Ngành Học", width="medium"),
                "Dự Báo 2025": st.column_config.NumberColumn("Dự Báo 2025", format="%.2f ⭐"),
                "+/- Điểm": st.column_config.NumberColumn("Dư Điểm", format="%+.2f"),
            }
            
            # Chỉ thêm config cho cột Trường nếu nó tồn tại
            if show_school_col:
                column_config["Trường Đại Học"] = st.column_config.TextColumn("Trường Đại Học", width="medium")

            st.dataframe(
                rec_display.style.map(style_diff_color, subset=['+/- Điểm'])
                .format("{:.2f}", subset=['Điểm 2023', 'Điểm 2024', 'Dự Báo 2025', '+/- Điểm']),
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                height=dynamic_height
            )