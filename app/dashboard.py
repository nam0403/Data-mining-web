import streamlit as st
import plotly.express as px
import ultilities as utils

def show_model_analysis():
    st.header("📊 Phân tích Dữ liệu Tuyển sinh (2017 - 2024)")
    
    # Load dữ liệu
    df = utils.load_data_from_file('C:/Users/phuon/OneDrive/Máy tính/Data mining web/Dataset/diem_chuan_ussh_wide_final (3).csv')
    
    if df.empty:
        st.warning("Chưa có dữ liệu. Vui lòng tải file 'data.csv' vào thư mục gốc.")
        return

    # Thống kê cơ bản
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số Ngành đào tạo", df['Tên ngành'].nunique())
    c2.metric("Số lượng Mã tuyển sinh", df['Mã ngành'].nunique())
    c3.metric("Điểm chuẩn TB 2024", f"{df['2024'].mean():.2f}")

    st.markdown("---")
    
    # 1. Biểu đồ xu hướng điểm chuẩn qua các năm
    st.subheader("📈 Xu hướng biến động điểm chuẩn")
    
    # Chọn ngành để xem
    selected_major = st.selectbox("Chọn Ngành để xem biểu đồ xu hướng:", df['Tên ngành'].unique())
    
    # Lấy dữ liệu ngành đó
    df_major = df[df['Tên ngành'] == selected_major]
    
    # Melt dữ liệu (Chuyển cột Năm thành dòng để vẽ biểu đồ)
    # Chỉ lấy các cột số (Năm)
    year_cols = [c for c in df.columns if c.isdigit()]
    df_melt = df_major.melt(id_vars=['Tên ngành', 'Tổ hợp môn'], value_vars=year_cols, var_name='Năm', value_name='Điểm chuẩn')
    df_melt = df_melt[df_melt['Điểm chuẩn'] > 0] # Bỏ các năm không tuyển sinh (điểm = 0)
    df_melt = df_melt.sort_values('Năm')

    fig = px.line(df_melt, x='Năm', y='Điểm chuẩn', color='Tổ hợp môn', markers=True,
                  title=f"Biểu đồ điểm chuẩn ngành {selected_major} qua các năm")
    st.plotly_chart(fig, use_container_width=True)

    # 2. Phân bố điểm năm 2024
    st.subheader("📊 Phân bố điểm chuẩn năm 2024 (Tất cả các ngành)")
    fig_hist = px.histogram(df, x="2024", nbins=20, title="Phổ điểm chuẩn 2024", text_auto=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("📂 Xem dữ liệu thô (Raw Data)"):
        st.dataframe(df)