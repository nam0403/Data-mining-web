import streamlit as st
import utilities as utils
import pandas as pd
import plotly.express as px

def show_prediction_system():
    st.title("🎯 Hệ thống Tư vấn Tuyển sinh")
    st.markdown("Nhập thông tin nguyện vọng và điểm thi để nhận dự báo chi tiết.")

    # --- LOAD MODELS ---
    models = utils.load_all_models()

    # =========================================================
    # KHUNG NHẬP LIỆU (INPUT)
    # =========================================================
    with st.container(border=True):
        st.subheader("📝 Thông tin Hồ sơ")
        
        # --- BƯỚC 1: CHỌN TRƯỜNG ---
        # Lấy danh sách trường từ Database
        schools = utils.get_all_schools()
        if not schools:
            st.error("Database trống. Vui lòng nạp dữ liệu.")
            st.stop()
            
        # Mapping: "Mã - Tên" -> ID
        school_map = {f"{s.code} - {s.name}": s.id for s in schools}
        sel_school_label = st.selectbox(
            "1. Chọn Trường Đại học:", 
            list(school_map.keys()), 
            index=None, 
            placeholder="Gõ mã hoặc tên trường..."
        )
        school_id = school_map.get(sel_school_label)

        # --- BƯỚC 2: CHỌN NGÀNH ---
        major_id = None
        sel_major_label = None
        if school_id:
            # Lấy danh sách ngành của trường đã chọn
            majors = utils.get_majors_by_school(school_id)
            major_map = {f"{m.code} - {m.name}": m.id for m in majors}
            
            sel_major_label = st.selectbox(
                "2. Chọn Ngành học:", 
                list(major_map.keys()), 
                index=None, 
                placeholder="Chọn ngành..."
            )
            major_id = major_map.get(sel_major_label)

        # --- BƯỚC 3: CHỌN TỔ HỢP & ĐIỂM VÙNG ---
        combo_id = None
        combo_code = None
        if major_id:
            # Lấy danh sách tổ hợp của ngành đã chọn
            combos = utils.get_combinations_by_school_major(school_id, major_id)
            combo_map = {c.code: c.id for c in combos}
            
            c1, c2 = st.columns(2)
            with c1:
                combo_code = st.selectbox(
                    "3. Chọn Tổ hợp xét tuyển:", 
                    list(combo_map.keys()), 
                    index=None, 
                    placeholder="Chọn khối..."
                )
                combo_id = combo_map.get(combo_code)
            
            with c2:
                region_bonus = st.selectbox(
                    "4. Điểm ưu tiên (Khu vực/Đối tượng):", 
                    options=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.75],
                    format_func=lambda x: f"+{x} điểm" if x > 0 else "Không có"
                )

        # --- BƯỚC 5: NHẬP ĐIỂM THI (DYNAMIC INPUTS) ---
        scores = {}
        analyze = False
        if combo_id:
            st.divider()
            st.markdown(f"**5. Nhập điểm thi tổ hợp {combo_code}:**")
            
            # Lấy chính xác tên môn học của tổ hợp từ DB
            subjects = utils.get_subjects_of_combination(combo_id)
            
            if not subjects:
                st.error(f"Không tìm thấy môn thi cho tổ hợp {combo_code}.")
            else:
                # Tạo các ô nhập điểm tương ứng
                cols = st.columns(len(subjects))
                for i, subj in enumerate(subjects):
                    with cols[i]:
                        scores[subj] = st.number_input(f"Điểm {subj}", 0.0, 10.0, step=0.25, key=f"s_{i}")
                
                st.write("")
                analyze = st.button("🚀 DỰ BÁO & TƯ VẤN", type="primary", use_container_width=True)

    # =========================================================
    # XỬ LÝ KẾT QUẢ (OUTPUT)
    # =========================================================
    if analyze:
        # 1. Tính tổng điểm
        raw_sum = sum(scores.values())
        total_score = raw_sum + region_bonus
        
        # 2. Lấy dữ liệu lịch sử ngành mục tiêu
        history = utils.get_history_dict(school_id, major_id, combo_id)
        if not history:
            st.warning("Không có dữ liệu lịch sử cho ngành này.")
            st.stop()
            
        # Tách tên ngành để hiển thị đẹp hơn
        if sel_major_label:
            parts = sel_major_label.split(" - ", 1)
            major_name = parts[1] if len(parts) > 1 else parts[0]
        else:
            major_name = ""

        # Chuẩn bị dữ liệu chạy mô hình dự báo
        row_data = history.copy()
        row_data.update({'Tên ngành': major_name, 'Tổ hợp môn': combo_code})
        
        # Chạy dự báo đa mô hình
        predictions = utils.predict_multimodel(row_data, models)
        final_pred = predictions['RandomForest'] # Lấy RF làm kết quả chính
        diff = total_score - final_pred
        status = utils.get_status_text(diff)
        is_pass = diff >= 0

        # --- A. GIAO DIỆN KẾT QUẢ CHÍNH ---
        st.markdown("---")
        st.header("📊 Kết quả Phân tích Ngành Mục tiêu")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng điểm xét tuyển", f"{total_score:.2f}", help=f"Điểm thi: {raw_sum} + Ưu tiên: {region_bonus}")
        m2.metric("Dự báo (Random Forest)", f"{final_pred:.2f}")
        m3.metric("Baseline (Trend)", f"{predictions['Baseline']:.2f}")
        m4.metric("Kết luận", status, delta=f"{diff:+.2f}", delta_color="normal" if is_pass else "inverse")

        # Biểu đồ & Bảng
        col_chart, col_table = st.columns([1, 1])
        
        with col_chart:
            st.subheader("📈 So sánh các Mô hình")
            model_df = pd.DataFrame(list(predictions.items()), columns=['Mô hình', 'Điểm dự báo'])
            fig = px.bar(model_df, x='Điểm dự báo', y='Mô hình', orientation='h', 
                         text_auto='.2f', color='Mô hình', title=f"Dự báo điểm chuẩn {major_name}")
            fig.add_vline(x=total_score, line_dash="dash", line_color="red", annotation_text="Điểm của bạn")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            st.subheader("📋 Lịch sử & Dự báo")
            # Xử lý bảng ngang (Pivot)
            clean_history = {}
            for k, v in history.items():
                try:
                    clean_year = str(int(float(k))) 
                    clean_history[clean_year] = v
                except:
                    clean_history[k] = v
            
            clean_history['2025 (Dự báo)'] = final_pred
            
            df_hist = pd.DataFrame([clean_history])
            # Sắp xếp cột năm tăng dần, đưa 2025 về cuối
            cols = sorted([c for c in df_hist.columns if c != '2025 (Dự báo)']) + ['2025 (Dự báo)']
            df_hist = df_hist.reindex(columns=cols)
            
            st.dataframe(
                df_hist.style.format("{:.2f}")
                .background_gradient(cmap='Blues', axis=1),
                use_container_width=True,
                hide_index=True
            )

        # --- B. RECOMMENDATION SYSTEM (AI DRIVEN) ---
        st.markdown("---")
        st.header(f"💡 Gợi ý Nguyện vọng (Khối {combo_code})")
        
        rec_df = pd.DataFrame()
        scenario_msg = ""

        # KỊCH BẢN 1: ĐẬU -> Gợi ý các ngành khác trong trường CÙNG TỔ HỢP
        if is_pass:
            current_school_name = sel_school_label.split(' - ')[1] if ' - ' in sel_school_label else sel_school_label
            scenario_msg = f"🎉 **Chúc mừng!** Bạn có khả năng cao trúng tuyển ngành **{major_name}**. Dưới đây là các ngành khác tại **{current_school_name}** xét tuyển khối **{combo_code}** phù hợp với điểm của bạn:"
            
            # Gọi hàm tìm kiếm theo tổ hợp cụ thể
            rec_df = utils.get_recommendations_by_specific_combo(
                combo_code, total_score, current_school_id=school_id, limit=5
            )
        
        # KỊCH BẢN 2: TRƯỢT -> Tìm ngành khác trong trường -> Nếu ko có thì tìm trường khác
        else:
            scenario_msg = f"⚠️ Điểm của bạn hơi thấp so với dự báo. Hệ thống đề xuất các lựa chọn an toàn hơn với khối **{combo_code}**:"
            
            # 2a. Tìm trong trường hiện tại
            rec_df = utils.get_recommendations_by_specific_combo(
                combo_code, total_score, current_school_id=school_id, limit=5
            )
            
            # 2b. Nếu trường này ko có -> Tìm toàn hệ thống
            if rec_df.empty:
                scenario_msg += "\n\n*(Không tìm thấy ngành phù hợp tại trường này, hệ thống đã mở rộng tìm kiếm sang các trường khác trên toàn quốc...)*"
                rec_df = utils.get_recommendations_by_specific_combo(
                    combo_code, total_score, current_school_id=None, limit=10
                )

        # KỊCH BẢN 3: TRƯỢT HẾT (Điểm quá thấp) -> Show ngành điểm thấp nhất
        if rec_df.empty:
            st.warning(f"😔 Với mức điểm hiện tại, hệ thống chưa tìm thấy ngành nào xét tuyển khối **{combo_code}** phù hợp.")
            st.info("💪 **Đừng lo lắng!** Dưới đây là một số ngành có điểm chuẩn thấp nhất năm ngoái trên toàn hệ thống để bạn tham khảo:")
            rec_df = utils.get_lowest_score_majors(limit=5)
        else:
            st.success(scenario_msg)

        # HIỂN THỊ BẢNG GỢI Ý
        if not rec_df.empty:
            # Chọn các cột quan trọng để hiển thị
            display_cols = ['Trường', 'Ngành', 'Tổ hợp', '2023', '2024', 'Dự báo 2025', 'Chênh lệch']
            # Lọc cột chỉ lấy những cột có trong DataFrame kết quả (để tránh lỗi nếu thiếu cột)
            valid_cols = [c for c in display_cols if c in rec_df.columns]
            
            st.dataframe(
                rec_df[valid_cols].style
                .format("{:.2f}", subset=[c for c in ['2023', '2024', 'Dự báo 2025', 'Chênh lệch'] if c in rec_df.columns])
                .background_gradient(cmap='Greens', subset=['Dự báo 2025'] if 'Dự báo 2025' in rec_df.columns else None),
                use_container_width=True,
                hide_index=True
            )