import streamlit as st
import dashboard as admin_dashboard
import recommendation_system

# --- PAGE CONFIG (Phải đặt đầu tiên ở file chạy chính) ---
st.set_page_config(
    layout="wide", 
    page_title="AI Admission System v3", 
    page_icon="🏫"
)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2231/2231649.png", width=100) # Placeholder Logo
    st.title("Điều khiển")
    page = st.radio(
        "Chức năng:", 
        ["🎓 Hệ thống Dự đoán"],
        index=0
    )
# --- ROUTING ---
if page == "🎓 Hệ thống Dự đoán":
    recommendation_system.show_prediction_system()