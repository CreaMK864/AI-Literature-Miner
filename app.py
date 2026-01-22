import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import numpy as np
import altair as alt
from datetime import date
import inference

# ================= 1. PAGE CONFIG (Research Mode) =================
st.set_page_config(layout="wide", page_title="Microcirculation Research Platform")

# CSS: Academic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;700&display=swap');
    
    .main-header {font-family: 'Roboto', sans-serif; font-size: 36px; font-weight: 700; color: #0e2a47;}
    .sub-text {font-family: 'Roboto', sans-serif; font-size: 16px; color: #555; font-style: italic;}
    .paper-container {
        background-color: #ffffff; padding: 50px; border: 1px solid #dcdcdc; 
        box-shadow: 0 0 15px rgba(0,0,0,0.1); margin-top: 20px; font-family: 'Times New Roman', serif;
    }
    h3 {color: #0e2a47; border-bottom: 2px solid #0e2a47; padding-bottom: 5px; margin-top: 30px;}
    .stButton>button {background-color: #0e2a47; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ================= STATE =================
if 'user_data' not in st.session_state: st.session_state.user_data = None
if 'run_analysis' not in st.session_state: st.session_state.run_analysis = False

# ================= 2. SUBJECT DATA INPUT =================
@st.dialog("🧪 Subject & Clinical Metadata")
def get_patient_info():
    st.write("Enter clinical parameters for quantitative standardization.")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Subject ID", value="Subj-001")
            age = st.number_input("Age", 18, 100, 30)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            # 重要：讓使用者確認視野寬度，以便計算 Density
            fov = st.number_input("Field of View Width (mm)", 1.0, 5.0, 3.0)
            
        st.markdown("---")
        st.caption("Reference Standard: Etehad Tavakol et al. (2015)")
        
        if st.form_submit_button("✅ Initialize Research Pipeline"):
            st.session_state.user_data = {"name": name, "age": age, "gender": gender, "fov": fov}
            st.session_state.run_analysis = True
            st.rerun()

# ================= 3. GAUSSIAN PLOT (Based on Etehad Tavakol 2015) =================
def plot_gaussian_comparison(density_val):
    # 文獻數據: Etehad Tavakol et al. (2015)
    # Healthy Mean = 9 loops/mm, Std Dev approx 2
    mu, sigma = 9, 2
    x = np.linspace(2, 16, 200)
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (x - mu))**2)
    
    source = pd.DataFrame({'Density': x, 'Probability': y})
    
    # 1. 正常人群分佈 (綠色區域)
    base = alt.Chart(source).mark_area(
        opacity=0.3, color='#4c78a8'
    ).encode(
        x=alt.X('Density', title='Capillary Density (loops/mm)'),
        y=alt.Y('Probability', axis=None)
    )
    
    # 2. 受試者位置 (紅線)
    rule = alt.Chart(pd.DataFrame({'x': [density_val]})).mark_rule(color='red', size=4).encode(x='x')
    
    # 3. 標籤
    text = alt.Chart(pd.DataFrame({'x': [density_val], 'label': [f'Subject: {density_val:.1f}']})).mark_text(
        align='left', dx=5, dy=-10, color='red', fontWeight='bold'
    ).encode(x='x', text='label')
    
    return (base + rule + text).properties(
        title="Comparison vs. Healthy Norm (Etehad Tavakol et al., 2015)",
        height=250
    )

# ================= 4. SOTA BENCHMARK PLOT =================
def plot_model_performance():
    # 這裡預留給第二階段的論文數據
    data = pd.DataFrame({
        'Model': ['U-Net (Baseline)', 'ResNet-50', 'MiT-B5 (Ours)'],
        'mIoU': [0.71, 0.75, 0.44], # 暫時數據，之後找到新論文再更新
        'Color': ['#ccc', '#ccc', '#0e2a47']
    })
    
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('mIoU', scale=alt.Scale(domain=[0, 1.0])),
        y=alt.Y('Model', sort='-x'),
        color=alt.Color('Color', scale=None),
        tooltip=['Model', 'mIoU']
    ).properties(title="Model Performance Benchmark", height=150)
    return chart

# ================= MAIN UI =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083206.png", width=60)
    st.title("Research Config")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("### 📚 Core References")
    st.info("""
    **1. Density Norms:**
    *Etehad Tavakol et al. (2015)*
    BioMed Research International
    
    **2. Pattern Classification:**
    *Smith et al. (2023)* / *Cutolo*
    EULAR Study Group
    """)

st.markdown('<div class="main-header">🧬 Quantitative Nailfold Capillaroscopy System</div>', unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload Microscopy Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(uploaded_file, caption="Raw Input", use_container_width=True)
        if st.button("🚀 Execute Analysis"):
            if not api_key: st.error("Error: API Key Missing.")
            else: get_patient_info()
        
        with st.expander("Show Technical Benchmarks", expanded=True):
            st.altair_chart(plot_model_performance(), use_container_width=True)

    if st.session_state.run_analysis and st.session_state.user_data:
        user = st.session_state.user_data
        
        try:
            # 1. Inference
            with st.spinner("Step 1/3: Segmentation & Feature Extraction..."):
                uploaded_file.seek(0)
                original, overlay, stats = inference.process_image(uploaded_file)
            
            # 計算密度 (Loops per mm)
            total_loops = stats.get("Normal", 0) + stats.get("Abnormal", 0) + stats.get("Aggregation", 0)
            density = total_loops / user['fov'] 
            
            with col2:
                st.image(overlay, caption="Semantic Segmentation Overlay", use_container_width=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Count", f"{total_loops}")
                m2.metric("Linear Density", f"{density:.2f} /mm")
                m3.metric("Abnormalities", f"{stats.get('Hemo', 0) + stats.get('Abnormal', 0)}")
                
                # 顯示引用了 Etehad Tavakol 的圖表
                st.altair_chart(plot_gaussian_comparison(density), use_container_width=True)

            # 2. Gemini Report
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            today_str = date.today().strftime("%Y-%m-%d")

            # Prompt 引用論文
            prompt = f"""
            Act as a Medical Research Assistant. Write a quantitative analysis report.
            
            **Subject Data:**
            - ID: {user['name']} ({user['age']}y, {user['gender']})
            - Date: {today_str}
            - Field of View: {user['fov']} mm
            
            **Computed Metrics:**
            - Linear Density: {density:.2f} loops/mm
            - Morphological Counts: {stats}
            
            **Reference Standards (Cite in report):**
            - Normal Density: 9 ± 2 loops/mm (Source: Etehad Tavakol et al., 2015 )
            - Abnormal Parameters: Smith et al., 2023
            
            **Instructions:**
            1. **Quantitative Analysis**: Compare patient's density ({density:.2f}) with the Etehad Tavakol norm (9±2).
            2. **Morphological Assessment**: Identify any anomalies.
            3. **Conclusion**: Standard medical summary.
            
            **Tone**: Academic, objective.
            """
            
            st.markdown("---")
            st.markdown('<div class="main-header">📑 Computer-Aided Diagnosis Report</div>', unsafe_allow_html=True)
            
            with st.spinner("Synthesizing Research Data..."):
                response = model.generate_content(prompt)
                
            st.markdown(f'<div class="paper-container">{response.text}</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Analyze Next Subject"):
                st.session_state.run_analysis = False
                st.rerun()

        except Exception as e:
            st.error(f"Execution Error: {e}")