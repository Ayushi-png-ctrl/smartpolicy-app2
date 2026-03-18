import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SmartPolicy - Insurance Premium Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3D58;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5A7F;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-amount {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the correct directory.")
        return None, None

# Load model
model, scaler = load_model()

# Header section
st.markdown('<h1 class="main-header">🏥 SmartPolicy</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Insurance Premium Predictor</p>', unsafe_allow_html=True)

# Input form — full width now (col2 Quick Insights moved to bottom)
st.markdown("### 📋 Enter Customer Details")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

    sex = st.selectbox("Gender", options=["male", "female"])

    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        help="BMI between 18.5-24.9 is considered healthy"
    )

    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)

    smoker = st.selectbox("Smoker", options=["yes", "no"])

    submitted = st.form_submit_button("💰 Predict Premium", use_container_width=True)

if submitted and model is not None:
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'bmi_risk': [bmi / age]
    })

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display prediction
    st.markdown("---")
    col3, col4, col5 = st.columns(3)

    with col4:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Estimated Premium</h2>
            <div class="prediction-amount">${prediction:,.2f}</div>
            <p>per year</p>
        </div>
        """, unsafe_allow_html=True)

    # Comparison with average
    if smoker == 'yes':
        avg_smoker = 32050.23
        st.metric("vs Average Smoker", f"${prediction - avg_smoker:,.2f}")
    else:
        avg_non_smoker = 8440.66
        st.metric("vs Average Non-Smoker", f"${prediction - avg_non_smoker:,.2f}")

    # Feature importance visualization
    st.markdown("### 🔍 Factors Influencing This Prediction")

    risk_factors = []
    risk_values = []

    if smoker == 'yes':
        risk_factors.append("Smoking")
        risk_values.append(90)

    if age > 50:
        risk_factors.append("Age > 50")
        risk_values.append(min((age-50)*3, 80))

    if bmi > 30:
        risk_factors.append("High BMI")
        risk_values.append(min((bmi-30)*5, 70))

    if risk_factors:
        fig = go.Figure(data=[
            go.Bar(name='Risk Level', x=risk_factors, y=risk_values, marker_color=['red' if v>70 else 'orange' for v in risk_values])
        ])
        fig.update_layout(
            title="Risk Factor Impact",
            xaxis_title="Factors",
            yaxis_title="Risk Score",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Educational note
    with st.expander("📚 Understanding Your Premium"):
        st.markdown("""
        Insurance premiums are calculated based on:
        - **Age**: Older individuals typically have higher health risks
        - **Smoking**: Significantly increases health risks and costs
        - **BMI**: Higher BMI correlates with more health issues
        - **Children**: Family coverage affects premium calculations
        - **Region**: Local healthcare costs vary by location

        This prediction is based on historical data and machine learning analysis.
        """)

# ── Quick Insights — moved to bottom ──────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Quick Insights")

# BMI Category display
bmi_val = bmi if 'bmi' in locals() else 25.0
if bmi_val < 18.5:
    bmi_category, bmi_color = "Underweight", "#FFA500"
elif bmi_val < 25:
    bmi_category, bmi_color = "Healthy Weight", "#00CC00"
elif bmi_val < 30:
    bmi_category, bmi_color = "Overweight", "#FFA500"
else:
    bmi_category, bmi_color = "Obese", "#FF0000"

st.markdown(f"**BMI Category:** <span style='color:{bmi_color}'>{bmi_category}</span>", unsafe_allow_html=True)

st.info(
    """
    **💡 Did you know?**
    - Smokers pay up to 3-4x higher premiums
    - Premiums increase with age
    - Higher BMI may increase costs
    - Region has minimal impact
    """
)

# Smoker vs Non-Smoker cards
insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.markdown("""
    <div style="background:#fff5f5; padding:1.5rem; border-radius:12px; border-left:5px solid #e74c3c;">
        <h4>🚬 Average Smoker Premium</h4>
        <h2 style="color:#e74c3c;">$32,050</h2>
        <p style="color:gray;">per year (dataset average)</p>
    </div>
    """, unsafe_allow_html=True)

with insight_col2:
    st.markdown("""
    <div style="background:#f0fff4; padding:1.5rem; border-radius:12px; border-left:5px solid #27ae60;">
        <h4>🚭 Average Non-Smoker Premium</h4>
        <h2 style="color:#27ae60;">$8,434</h2>
        <p style="color:gray;">per year (dataset average)</p>
    </div>
    """, unsafe_allow_html=True)

with insight_col3:
    st.markdown("""
    <div style="background:#fffbf0; padding:1.5rem; border-radius:12px; border-left:5px solid #e67e22;">
        <h4>📈 Smoking Cost Multiplier</h4>
        <h2 style="color:#e67e22;">~3.8x</h2>
        <p style="color:gray;">Smokers pay ~3.8x more on average</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>SmartPolicy | AI-Powered Insurance Prediction | v1.0</p>",
    unsafe_allow_html=True
)
