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
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model only (NO scaler — Random Forest doesn't need scaling)
# ✅ CHANGE 1: Removed scaler loading; RF was trained on unscaled data
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'model.pkl' is in the correct directory.")
        return None

# Load model
model = load_model()

# Header section
st.markdown('<h1 class="main-header">🏥 SmartPolicy</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Insurance Premium Predictor</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Enter Customer Details")
    
    # Input form
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
        
        # ✅ CHANGE 2: Removed region selectbox — region removed from model input
        
        submitted = st.form_submit_button("💰 Predict Premium", use_container_width=True)

with col2:
    st.markdown("### 📊 Quick Insights")
    
    # Display key factors affecting insurance costs
    st.info(
        """
        **💡 Did you know?**
        - Smokers pay up to 3-4x higher premiums
        - Premiums increase with age
        - Higher BMI may increase costs
        """
    )
    # ✅ CHANGE 3: Removed "Region has minimal impact" from info box since region is removed
    
    # BMI Categories
    if 'bmi' in locals():
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "#FFA500"
        elif 18.5 <= bmi < 25:
            bmi_category = "Healthy Weight"
            bmi_color = "#00FF00"
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "#FFA500"
        else:
            bmi_category = "Obese"
            bmi_color = "#FF0000"
        
        st.markdown(f"**BMI Category:** <span style='color:{bmi_color}'>{bmi_category}</span>", unsafe_allow_html=True)

if submitted and model is not None:
    # ✅ CHANGE 4: Removed 'region' from input_data
    # ✅ CHANGE 1 (cont): No scaler.transform — passing raw data directly to RF model
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'bmi_risk': [bmi / age]
    })
    
    # Make prediction (no scaling needed for Random Forest)
    prediction = model.predict(input_data)[0]
    
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
    
    # ✅ CHANGE 5: Fixed avg_non_smoker value (was 8440.66, corrected to 8434.27 from actual data)
    if smoker == 'yes':
        avg_smoker = 32050.23
        st.metric("vs Average Smoker", f"${prediction - avg_smoker:,.2f}")
    else:
        avg_non_smoker = 8434.27
        st.metric("vs Average Non-Smoker", f"${prediction - avg_non_smoker:,.2f}")
    
    # Feature importance visualization
    st.markdown("### 🔍 Factors Influencing This Prediction")
    
    # Create a simple gauge for risk factors
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
        
        This prediction is based on historical data and machine learning analysis.
        """)
        # ✅ CHANGE 6: Removed Region bullet point from expander since it's no longer used


# Footer
st.markdown("---")
# ─────────────────────────────────────────────
# ✅ CHANGE 7: NEW — Quick Insight section at the very bottom
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Quick Insight: Smoker vs Non-Smoker")
 
insight_col1, insight_col2, insight_col3 = st.columns(3)
 
with insight_col1:
    st.markdown("""
    <div class="insight-box">
        <h4>🚬 Average Smoker Premium</h4>
        <h2 style="color:#e74c3c;">$32,050</h2>
        <p style="color:gray;">per year (dataset average)</p>
    </div>
    """, unsafe_allow_html=True)
 
with insight_col2:
    st.markdown("""
    <div class="insight-box">
        <h4>🚭 Average Non-Smoker Premium</h4>
        <h2 style="color:#27ae60;">$8,434</h2>
        <p style="color:gray;">per year (dataset average)</p>
    </div>
    """, unsafe_allow_html=True)
 
with insight_col3:
    st.markdown("""
    <div class="insight-box">
        <h4>📈 Smoking Cost Multiplier</h4>
        <h2 style="color:#e67e22;">~3.8x</h2>
        <p style="color:gray;">smokers pay ~3.8x more on average</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>SmartPolicy | AI-Powered Insurance Prediction | v1.0</p>",
    unsafe_allow_html=True
)
