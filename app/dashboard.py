import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import utilities as utils
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score

def show_model_analysis():
    st.header("📊 Dashboard Quản trị & Phân tích Tuyển sinh")
    
    # --- 1. LOAD DATA ---
    default_file = 'data.csv'
    if os.path.exists(default_file):
        df = utils.load_data_with_prediction(default_file)
    else:
        st.warning(f"⚠️ Không tìm thấy file '{default_file}'. Vui lòng tải file dữ liệu lên hệ thống.")
        return

    if df.empty:
        st.error("File dữ liệu rỗng hoặc sai định dạng.")
        return

    # --- 2. XỬ LÝ DỮ LIỆU CHO VISUALIZATION ---
    # Tạo cột 'Tên hiển thị' để tách biệt ngành CLC và ngành thường trên biểu đồ
    # Logic: Nếu tên ngành chứa "CLC" hoặc "Chất lượng cao" -> Gán nhãn CLC
    def create_display_name(row):
        major_name = row['Tên ngành']
        combo = row['Tổ hợp môn']
        is_clc = 'CLC' in major_name.upper() or 'CHẤT LƯỢNG CAO' in major_name.upper()
        type_label = " (CLC)" if is_clc else ""
        # Tên hiển thị = Tên ngành + (CLC nếu có) + Tổ hợp
        return f"{major_name}{type_label} - {combo}"

    df['Tên hiển thị'] = df.apply(create_display_name, axis=1)

    # --- 3. METRICS TỔNG QUAN ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số ngành đào tạo", df['Tên ngành'].nunique())
    c2.metric("Số lượng tổ hợp xét tuyển", len(df))
    
    # Tính trung bình điểm chuẩn nếu có cột 2024
    if '2024' in df.columns:
        avg_24 = df[df['2024'] > 0]['2024'].mean()
        c3.metric("Điểm chuẩn TB 2024", f"{avg_24:.2f}")
    
    # Tính trung bình dự báo 2025
    if 'Dự báo 2025' in df.columns:
        avg_25 = df[df['Dự báo 2025'] > 0]['Dự báo 2025'].mean()
        delta = avg_25 - avg_24 if '2024' in df.columns else 0
        c4.metric("Dự báo TB 2025", f"{avg_25:.2f}", delta=f"{delta:+.2f}")

    st.markdown("---")

    # --- 4. TABS GIAO DIỆN CHÍNH ---
    tab1, tab2, tab3 = st.tabs(["📈 Xu hướng & So sánh", "🔮 Bảng Dự báo 2025", "🎯 Độ chính xác Mô hình"])

    # === TAB 1: BIỂU ĐỒ XU HƯỚNG (LINE CHART) ===
    with tab1:
        st.subheader("So sánh biến động điểm chuẩn qua các năm")
        st.caption("Hỗ trợ so sánh đa chiều: Giữa các ngành, giữa hệ Chuẩn và CLC, giữa các tổ hợp môn.")
        
        # Multiselect cho phép chọn nhiều ngành cùng lúc
        all_majors = sorted(df['Tên ngành'].unique())
        selected_majors = st.multiselect(
            "Chọn các ngành để so sánh:", 
            all_majors,
            default=[all_majors[0]] if len(all_majors) > 0 else None
        )
        
        if selected_majors:
            # Lọc dữ liệu theo ngành đã chọn
            df_chart = df[df['Tên ngành'].isin(selected_majors)].copy()
            
            # Lấy danh sách các cột năm (chỉ lấy cột số)
            year_cols = [c for c in df.columns if c.isdigit()]
            year_cols = sorted(year_cols) # Sắp xếp năm tăng dần (2017 -> 2024)

            if year_cols:
                # Melt dữ liệu: Chuyển cột Năm thành hàng để vẽ biểu đồ
                df_melt = df_chart.melt(
                    id_vars=['Tên hiển thị', 'Tên ngành', 'Tổ hợp môn'], 
                    value_vars=year_cols, 
                    var_name='Năm', 
                    value_name='Điểm chuẩn'
                )
                # Loại bỏ các điểm dữ liệu bằng 0 (năm không tuyển sinh)
                df_melt = df_melt[df_melt['Điểm chuẩn'] > 0]
                
                # Vẽ biểu đồ Line Chart bằng Plotly
                # color='Tên hiển thị' -> Tự động tách màu cho Ngành Thường/CLC và Tổ hợp
                fig = px.line(
                    df_melt, 
                    x='Năm', 
                    y='Điểm chuẩn', 
                    color='Tên hiển thị', 
                    markers=True,
                    symbol='Tên hiển thị', # Thêm ký hiệu khác nhau cho mỗi đường để dễ phân biệt
                    title="Biểu đồ xu hướng điểm chuẩn",
                    hover_data=['Tên ngành', 'Tổ hợp môn']
                )
                
                # Tinh chỉnh layout
                fig.update_layout(
                    xaxis_title="Năm tuyển sinh", 
                    yaxis_title="Điểm chuẩn", 
                    legend_title="Chi tiết Ngành - Tổ hợp",
                    hovermode="x unified" # Hiển thị tooltip gộp khi di chuột
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dữ liệu không có cột năm (2017-2024) để vẽ biểu đồ.")
        else:
            st.info("👉 Vui lòng chọn ít nhất một ngành từ danh sách trên để xem biểu đồ.")

    # === TAB 2: BẢNG DỰ BÁO CHI TIẾT ===
    with tab2:
        st.subheader("Dữ liệu Dự báo Tuyển sinh 2025")
        
        # Chọn các cột cần hiển thị
        display_cols = ['Tên ngành', 'Mã ngành', 'Tổ hợp môn', '2023', '2024', 'Dự báo 2025']
        # Chỉ lấy cột nào thực sự có trong file
        valid_cols = [c for c in display_cols if c in df.columns]
        
        df_display = df[valid_cols].copy()
        
        # Hiển thị DataFrame với Style (Tô màu cột Dự báo)
        st.dataframe(
            df_display.style
            .format("{:.2f}", subset=[c for c in ['2023', '2024', 'Dự báo 2025'] if c in df_display.columns])
            .background_gradient(cmap='Blues', subset=['Dự báo 2025']), # Tô màu xanh đậm nhạt theo điểm
            use_container_width=True,
            height=600
        )

    # === TAB 3: KIỂM ĐỊNH ĐỘ CHÍNH XÁC (BACKTESTING) ===
    with tab3:
        st.subheader("Kiểm định độ tin cậy của Mô hình (Backtesting)")
        st.markdown("""
        **Phương pháp kiểm thử:** Hệ thống sẽ ẩn đi dữ liệu năm **2024**, dùng dữ liệu **2021-2023** để chạy mô hình dự báo lại năm 2024.
        Sau đó so sánh **Kết quả dự báo giả định** này với **Điểm chuẩn thực tế 2024**.
        """)
        
        # Load model artifacts
        artifacts = utils.load_artifacts()
        
        # Chạy Backtest (Hàm này nằm trong utilities.py)
        df_backtest = utils.run_backtest(df, artifacts)
        
        if not df_backtest.empty:
            # Tính toán các chỉ số đánh giá (Metrics)
            mae = mean_absolute_error(df_backtest['2024'], df_backtest['Backtest_2024'])
            r2 = r2_score(df_backtest['2024'], df_backtest['Backtest_2024'])
            
            # Hiển thị Metrics
            m1, m2 = st.columns(2)
            m1.metric("Sai số tuyệt đối trung bình (MAE)", f"{mae:.2f} điểm", 
                      help="Trung bình mô hình đoán lệch bao nhiêu điểm so với thực tế.")
            m2.metric("Độ phù hợp (R² Score)", f"{r2:.2%}", 
                      help="Mô hình giải thích được bao nhiêu % sự biến động của dữ liệu. Càng gần 100% càng tốt.")
            
            st.divider()
            
            col_chart1, col_chart2 = st.columns(2)
            
            # Chart 1: Scatter Plot (Tương quan Thực tế - Dự báo)
            with col_chart1:
                st.markdown("**1. Tương quan: Điểm Thực tế vs Dự báo (2024)**")
                fig_scatter = px.scatter(
                    df_backtest,
                    x='2024',
                    y='Backtest_2024',
                    color='Tổ hợp môn',
                    hover_data=['Tên ngành'],
                    labels={'2024': 'Điểm Thực tế 2024', 'Backtest_2024': 'Model Dự báo 2024'}
                )
                # Vẽ đường chéo y=x (Đường hoàn hảo)
                fig_scatter.add_shape(
                    type="line", x0=15, y0=15, x1=30, y1=30, 
                    line=dict(color="Red", dash="dash"),
                    name="Dự báo chuẩn xác"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            # Chart 2: Histogram Sai số (Residuals)
            with col_chart2:
                st.markdown("**2. Phân bố độ lệch (Sai số)**")
                fig_hist = px.histogram(
                    df_backtest, 
                    x='Sai_Số', 
                    nbins=15, 
                    labels={'Sai_Số': 'Độ lệch (Thực tế - Dự báo)'},
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            # Bảng dữ liệu chi tiết Backtest
            with st.expander("🔎 Xem bảng dữ liệu chi tiết kiểm định"):
                st.dataframe(
                    df_backtest[['Tên ngành', 'Tổ hợp môn', '2024', 'Backtest_2024', 'Sai_Số']]
                    .style.format("{:.2f}", subset=['2024', 'Backtest_2024', 'Sai_Số'])
                    .applymap(lambda x: 'color: red' if abs(x) > 1.0 else 'color: green', subset=['Sai_Số']),
                    use_container_width=True
                )
                
        else:
            st.warning("""
            ⚠️ Không thể chạy kiểm định do thiếu dữ liệu lịch sử. 
            Để chạy Backtest, file data.csv cần có đủ các cột: 2021, 2022, 2023 và 2024.
            """)