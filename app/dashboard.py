import streamlit as st
import plotly.express as px
import utilities as utils # Đã sửa import
import os

def show_model_analysis():
    st.header("📊 Phân tích Dữ liệu & Trạng thái Model")
    
    # --- CHECK MODEL ---
    model_path = 'admission_model_v2.pkl'
    if os.path.exists(model_path):
        st.success(f"✅ **AI Model đang hoạt động**: `{model_path}`")
    else:
        st.warning("⚠️ **Chế độ cơ bản**: Không tìm thấy file model. Đang dùng công thức toán học.")
    
    st.markdown("---")
    
    # --- LOAD DATA ---
    default_file = 'data.csv'
    if os.path.exists(default_file):
        df = utils.load_data_with_prediction(default_file)
    else:
        st.warning(f"Không tìm thấy '{default_file}'. Vui lòng upload file.")
        uploaded_file = st.file_uploader("Upload data.csv", type=['csv', 'xlsx'])
        if uploaded_file:
            df = utils.load_data_with_prediction(uploaded_file)
        else:
            return

    if df.empty:
        st.error("Dữ liệu rỗng.")
        return

    # Thống kê
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số Ngành", df['Tên ngành'].nunique())
    if 'Mã ngành' in df.columns:
        c2.metric("Mã tuyển sinh", df['Mã ngành'].nunique())
    if '2024' in df.columns:
        c3.metric("Điểm chuẩn TB 2024", f"{df['2024'].mean():.2f}")
    
    st.subheader("📈 Xu hướng điểm chuẩn")
    selected_major = st.selectbox("Chọn Ngành:", df['Tên ngành'].unique())
    df_major = df[df['Tên ngành'] == selected_major]
    
    year_cols = [c for c in df.columns if c.isdigit()]
    if year_cols:
        df_melt = df_major.melt(id_vars=['Tên ngành', 'Tổ hợp môn'], value_vars=year_cols, var_name='Năm', value_name='Điểm chuẩn')
        df_melt = df_melt[df_melt['Điểm chuẩn'] > 0].sort_values('Năm')
        fig = px.line(df_melt, x='Năm', y='Điểm chuẩn', color='Tổ hợp môn', markers=True)
        st.plotly_chart(fig, use_container_width=True)