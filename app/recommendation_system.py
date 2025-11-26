import streamlit as st
import utilities as utils
import pandas as pd
import plotly.express as px

def show_prediction_system():
    # --- 1. CONFIG UI & CSS (Giao diện đẹp) ---
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
            background-color: white;
            border: 1px solid #e5e7eb;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        /* Status Colors */
        .status-pass { color: #059669 !important; font-weight: 800; } /* Green */
        .status-fail { color: #dc2626 !important; font-weight: 800; } /* Red */
        .bg-pass { background-color: #ecfdf5 !important; border: 1px solid #a7f3d0 !important; }
        .bg-fail { background-color: #fef2f2 !important; border: 1px solid #fecaca !important; }

        /* Button Style */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
            font-size: 16px;
            margin-top: 10px;
        }
        
        /* Table Headers */
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Hệ Thống Tư Vấn Tuyển Sinh 2025")
    st.markdown("Nhập thông tin hồ sơ để nhận dự báo điểm chuẩn và gợi ý nguyện vọng tối ưu.")

    # --- 2. INITIALIZE (Khởi tạo Predictor) ---
    if 'predictor' not in st.session_state:
        # Gọi class mới từ utilities (không cần load model file nữa)
        st.session_state.predictor = utils.AdmissionPredictor()
    
    predictor = st.session_state.predictor
    
    # Lấy danh sách trường để hiển thị
    schools = utils.get_all_schools()
    
    if not schools:
        st.error("⚠️ Không tìm thấy dữ liệu trường học. Vui lòng kiểm tra Database.")
        st.stop()

    # =========================================================
    # PHẦN 3: NHẬP LIỆU (INPUT FORM)
    # =========================================================
    with st.container(border=True):
        st.subheader("📝 Hồ Sơ Thí Sinh")
        
        # Hàng 1: Trường & Ngành
        c1, c2 = st.columns(2)
        
        # A. Chọn Trường
        # Mapping: "Mã - Tên" -> Object School
        school_options = {f"{s.code} - {s.name}": s for s in schools}
        
        selected_school_obj = None
        school_id = None

        with c1:
            sel_school_label = st.selectbox(
                "1. Chọn Trường Đại học:",
                options=list(school_options.keys()),
                index=None,
                placeholder="Tìm kiếm tên hoặc mã trường..."
            )
            if sel_school_label:
                selected_school_obj = school_options.get(sel_school_label)
                school_id = selected_school_obj.id

        # B. Chọn Ngành (Lọc theo trường đã chọn)
        major_id = None
        selected_major_obj = None
        
        with c2:
            if school_id:
                # Gọi hàm mới nhận ID
                majors = utils.get_majors_by_school(school_id)
                if majors:
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
                # Gọi hàm mới nhận ID
                combos = utils.get_combinations_by_school_major(school_id, major_id)
                if combos:
                    # Combo object: (id, code)
                    combo_options = {c.code: c.id for c in combos}
                    combo_code = st.selectbox(
                        "3. Chọn Tổ hợp xét tuyển:",
                        options=list(combo_options.keys()),
                        index=None,
                        placeholder="Chọn khối (A00, A01...)"
                    )
                    if combo_code:
                        combo_id = combo_options.get(combo_code)
                else:
                    st.warning("Chưa có dữ liệu tổ hợp cho ngành này.")
            else:
                st.selectbox("3. Chọn Tổ hợp xét tuyển:", [], disabled=True, placeholder="Vui lòng chọn ngành trước")
                
        with c4:
            region_bonus = st.selectbox(
                "4. Điểm ưu tiên (Khu vực/Đối tượng):", 
                options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.75],
                format_func=lambda x: f"+{x} điểm" if x > 0 else "Không có"
            )

        # Hàng 3: Nhập Điểm Thi (Hiện động theo tổ hợp)
        scores = {}
        analyze = False
        
        if combo_id:
            st.divider()
            st.markdown(f"**🎯 Nhập điểm thi tổ hợp {combo_code}:**")
            
            # Gọi hàm mới nhận ID
            subjects = utils.get_subjects_of_combination(combo_id)
            if subjects:
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
                st.error("Không tải được môn học của tổ hợp này.")

    # =========================================================
    # PHẦN 4: HIỂN THỊ KẾT QUẢ (ANALYSIS)
    # =========================================================
    
    if analyze and selected_school_obj and selected_major_obj:
        # 1. Chuẩn bị dữ liệu input
        student_input = utils.StudentInput(
            university=selected_school_obj.code, 
            major_name=selected_major_obj.name,  
            major_code=selected_major_obj.code, 
            combination=combo_code,              
            subject_scores=scores,               
            priority_score=region_bonus
        )
        
        # 2. Gọi hàm dự báo (Query DB)
        prediction_result = predictor.predict(student_input)
        
        if not prediction_result:
            st.error("Không tìm thấy dữ liệu dự báo cho ngành này.")
            st.stop()
            
        # Lấy dữ liệu hiển thị
        final_pred = prediction_result.best_prediction
        user_total_score = prediction_result.student_score
        diff = prediction_result.margin
        pass_status = prediction_result.is_passed
        
        # Logic nhãn trạng thái
        status_label = "ĐẬU AN TOÀN" if pass_status else "NGUY CƠ TRƯỢT"
        if diff >= 0 and diff < 1.0: status_label = "CƠ HỘI CAO" # Đậu nhưng sát nút
        if diff < 0 and diff >= -0.5: status_label = "CÂN NHẮC"  # Trượt nhưng sát nút

        st.markdown("---")
        
        # --- A. CARDS KẾT QUẢ ---
        st.subheader(f"📊 Kết Quả Phân Tích: {prediction_result.major_name}")
        
        bg_class = "bg-pass" if pass_status else "bg-fail"
        text_class = "status-pass" if pass_status else "status-fail"

        # HTML Custom Card
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase;">Tổng Điểm Của Bạn</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1E40AF;">{user_total_score:.2f}</div>
                <div style="font-size: 0.8rem; color: #9CA3AF;">(Thi: {student_input.raw_score:.2f} + Ưu tiên: {region_bonus})</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase;">Dự Báo Điểm Chuẩn 2025</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #4B5563;">{final_pred:.2f}</div>
                <div style="font-size: 0.8rem; color: #9CA3AF;">(Dữ liệu AI Ensemble)</div>
            </div>
            <div class="metric-card {bg_class}">
                <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase;">Đánh Giá Cơ Hội</div>
                <div class="{text_class}" style="font-size: 1.6rem; padding: 5px 0;">{status_label}</div>
                <div style="font-size: 1rem; font-weight: 600;">{diff:+.2f} điểm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- B. BIỂU ĐỒ & LỊCH SỬ ---
        col_chart, col_data = st.columns([1.5, 1])
        
        # Gọi hàm lấy lịch sử từ class Predictor (đã cập nhật)
        history = predictor.get_historical_cutoffs(
            student_input.university, 
            student_input.major_code, 
            student_input.combination
        )
        
        # Vẽ biểu đồ
        with col_chart:
            st.markdown("**📉 Xu hướng điểm chuẩn qua các năm**")
            clean_history = {k: v for k, v in history.items()}
            
            df_chart = pd.DataFrame(list(clean_history.items()), columns=['Năm', 'Điểm'])
            
            if not df_chart.empty:
                df_chart = df_chart.sort_values('Năm')
                
                # Biểu đồ đường
                fig = px.line(df_chart, x='Năm', y='Điểm', markers=True)
                
                # Thêm điểm Dự báo 2025
                fig.add_scatter(x=[2025], y=[final_pred], mode='markers+text', 
                                name='Dự báo 2025', text=[f"{final_pred:.2f}"], textposition="top center",
                                marker=dict(color='red', size=12, symbol='star'))
                
                # Thêm điểm của User
                fig.add_scatter(x=[2025], y=[user_total_score], mode='markers', 
                                name='Điểm của bạn', 
                                marker=dict(color='green', size=10, symbol='circle'))
                
                fig.update_layout(
                    xaxis_title=None, yaxis_title=None, 
                    margin=dict(l=20, r=20, t=10, b=20), 
                    height=300, 
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chưa có đủ dữ liệu lịch sử để vẽ biểu đồ.")
        
        # Hiển thị bảng số liệu
        with col_data:
            st.markdown("**📋 Chi tiết điểm chuẩn**")
            if history:
                # Thêm cột dự báo vào bảng
                clean_history['2025 (Dự báo)'] = final_pred
                
                # Xoay bảng ngang cho dễ nhìn
                display_row = {str(k): v for k, v in clean_history.items()}
                df_hist = pd.DataFrame([display_row])
                
                # Sắp xếp cột theo năm
                cols = sorted([c for c in df_hist.columns if c != '2025 (Dự báo)']) + ['2025 (Dự báo)']
                df_hist = df_hist.reindex(columns=cols)
                
                st.dataframe(
                    df_hist.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                    use_container_width=True,
                    height=200
                )
            else:
                st.info("Không có dữ liệu lịch sử.")

        # --- C. GỢI Ý NGUYỆN VỌNG (RECOMMENDATION) ---
        st.markdown("---")
        st.markdown("### 💡 Gợi Ý Nguyện Vọng Tối Ưu")
        
        rec_df = pd.DataFrame()
        msg_content = ""
        msg_type = "info"
        show_school_col = False

        # --- Logic gợi ý ---
        if pass_status:
            # Case 1: ĐẬU -> Gợi ý các ngành KHÁC cùng trường
            msg_type = "success"
            msg_content = f"Bạn có khả năng cao trúng tuyển ngành **{prediction_result.major_name}**. Dưới đây là các ngành khác tại **{selected_school_obj.name}** cùng khối xét tuyển mà bạn cũng có thể quan tâm:"
            
            rec_df = utils.get_recommendations_by_specific_combo(
                prediction_result.combination, 
                prediction_result.student_score, 
                current_school_id=school_id, # Chỉ tìm trong trường này
                limit=5
            )
            show_school_col = False
            
        else:
            # Case 2: RỚT -> Gợi ý ngành an toàn hơn
            msg_type = "warning"
            msg_content = f"Điểm của bạn thấp hơn dự báo ngành **{prediction_result.major_name}**. Hệ thống đề xuất các lựa chọn **An Toàn Hơn**:"
            
            # Ưu tiên tìm trong trường hiện tại
            rec_df = utils.get_recommendations_by_specific_combo(
                prediction_result.combination, 
                prediction_result.student_score, 
                current_school_id=school_id, 
                limit=5
            )
            
            if rec_df.empty:
                # Nếu trường này không còn ngành nào đậu -> Tìm toàn hệ thống
                msg_content += " (Đã mở rộng tìm kiếm sang các trường khác do không tìm thấy ngành phù hợp tại trường này)"
                rec_df = utils.get_recommendations_by_specific_combo(
                    prediction_result.combination, 
                    prediction_result.student_score, 
                    current_school_id=None, # Tìm tất cả
                    limit=10
                )
                show_school_col = True
            else:
                show_school_col = False

        # Case 3: Điểm quá thấp, không đậu ngành nào -> Gợi ý ngành thấp nhất hệ thống
        if rec_df.empty:
            st.warning("😔 Với mức điểm hiện tại, rất khó tìm được ngành phù hợp xét tuyển khối này.")
            st.info("💪 **Lời khuyên:** Tham khảo các ngành có điểm chuẩn thấp nhất hệ thống:")
            rec_df = utils.get_lowest_score_majors(limit=10)
            show_school_col = True
        else:
            if msg_type == "success":
                st.success(msg_content)
            else:
                st.warning(msg_content)

        # --- Hiển thị bảng Recommendation ---
        if not rec_df.empty:
            # Cấu hình cột hiển thị
            display_cols = {
                'Trường': 'Trường',
                'Ngành': 'Ngành Học',
                'Tổ hợp': 'Khối',
                'Dự báo 2025': 'Dự Báo 2025',
                'Chênh lệch': 'Dư Điểm (+/-)'
            }
            
            # Ẩn cột trường nếu đang xem trong cùng 1 trường
            if not show_school_col and 'Trường' in display_cols:
                del display_cols['Trường']

            # Lọc cột tồn tại
            valid_cols = [c for c in display_cols.keys() if c in rec_df.columns]
            rec_display = rec_df[valid_cols].rename(columns=display_cols)
            
            # Tô màu cột Chênh lệch
            def style_diff(val):
                try:
                    v = float(val)
                    return 'color: #059669; font-weight: bold;' if v >= 0 else 'color: #dc2626; font-weight: bold;'
                except: return ''

            # Cấu hình hiển thị dataframe
            column_config = {
                "Ngành Học": st.column_config.TextColumn("Ngành Học", width="medium"),
                "Dự Báo 2025": st.column_config.NumberColumn("Dự Báo 2025", format="%.2f"),
                "Dư Điểm (+/-)": st.column_config.NumberColumn("Dư Điểm (+/-)", format="%+.2f"),
            }
            if show_school_col:
                column_config["Trường"] = st.column_config.TextColumn("Trường", width="medium")

            st.dataframe(
                rec_display.style.map(style_diff, subset=['Dư Điểm (+/-)'])
                .format("{:.2f}", subset=['Dự Báo 2025', 'Dư Điểm (+/-)']),
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )