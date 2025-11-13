import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AirQuality AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .prediction-result {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .good { background: #d1fae5; color: #065f46; }
    .moderate { background: #fef3c7; color: #92400e; }
    .unhealthy-sensitive { background: #fed7aa; color: #9a3412; }
    .unhealthy { background: #fecaca; color: #991b1b; }
    .very-unhealthy { background: #e9d5ff; color: #6b21a8; }
    .hazardous { background: #fca5a5; color: #7f1d1d; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        st.error("❌ Model files not found! Please upload model.pkl and scaler.pkl")
        return None, None

model, scaler = load_model()

# AQI Category function
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "good", "😊 Air quality is satisfactory"
    elif aqi_value <= 100:
        return "Moderate", "moderate", "😐 Air quality is acceptable"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "unhealthy-sensitive", "😷 Sensitive groups may be affected"
    elif aqi_value <= 200:
        return "Unhealthy", "unhealthy", "🤒 Everyone may be affected"
    elif aqi_value <= 300:
        return "Very Unhealthy", "very-unhealthy", "🚨 Health alert"
    else:
        return "Hazardous", "hazardous", "💀 Health warning"

# Health recommendations
def get_recommendations(category):
    recommendations = {
        "Good": [
            "✅ Perfect for outdoor activities and exercise",
            "✅ No restrictions needed",
            "✅ Great day for children to play outside"
        ],
        "Moderate": [
            "⚠️ Generally acceptable for most people",
            "⚠️ Sensitive individuals should reduce prolonged exertion",
            "💧 Stay hydrated if exercising outdoors"
        ],
        "Unhealthy for Sensitive Groups": [
            "🚫 Sensitive groups should reduce outdoor activities",
            "👵 Older adults and children should limit exertion",
            "🏠 Consider moving activities indoors"
        ],
        "Unhealthy": [
            "🚨 Everyone should reduce outdoor activities",
            "💪 Avoid strenuous exercise outdoors",
            "😷 Wear masks if going outside"
        ],
        "Very Unhealthy": [
            "🆘 Avoid all outdoor activities",
            "🏠 Stay indoors with windows closed",
            "💨 Use air purifiers"
        ],
        "Hazardous": [
            "🔥 HEALTH EMERGENCY - Stay indoors",
            "🆘 Cancel all outdoor activities",
            "🏥 Seek medical help if symptoms appear"
        ]
    }
    return recommendations.get(category, [])

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌤️ AirQuality AI Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
        st.title("Quick Guide")
        st.info("""
        **Safe Levels (WHO):**
        - PM2.5: ≤ 12 µg/m³
        - PM10: ≤ 45 µg/m³  
        - NO₂: ≤ 25 µg/m³
        - CO: ≤ 4 mg/m³
        - O₃: ≤ 100 µg/m³
        """)
        
        st.warning("💡 **Tip:** Enter current pollutant levels from your local air quality monitor")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Enter Pollutant Levels")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=25.0, step=1.0)
                pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=50.0, step=1.0)
                
            with col2:
                no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=20.0, step=1.0)
                co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0, step=0.1)
                
            with col3:
                o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, value=60.0, step=1.0)
            
            submitted = st.form_submit_button("🚀 Predict AQI", use_container_width=True)
    
    with col2:
        st.subheader("🎯 Real-time Analysis")
        
        # Show current input analysis
        if pm25 > 35:
            st.error(f"PM2.5: {pm25} µg/m³ (High)")
        elif pm25 > 12:
            st.warning(f"PM2.5: {pm25} µg/m³ (Moderate)")
        else:
            st.success(f"PM2.5: {pm25} µg/m³ (Good)")
            
        if pm10 > 100:
            st.error(f"PM10: {pm10} µg/m³ (High)")
        elif pm10 > 45:
            st.warning(f"PM10: {pm10} µg/m³ (Moderate)")
        else:
            st.success(f"PM10: {pm10} µg/m³ (Good)")
    
    # Prediction
    if submitted and model is not None:
        try:
            # Make prediction
            input_data = np.array([[pm25, pm10, no2, co, o3]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Get category
            category, css_class, description = get_aqi_category(prediction)
            recommendations = get_recommendations(category)
            
            # Display result
            st.markdown(f"""
            <div class="prediction-result {css_class}">
                <h2>Predicted AQI: {prediction:.1f}</h2>
                <h3>{category}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show recommendations
            st.subheader("💡 Health Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("PM2.5", f"{pm25} µg/m³")
                with col2: st.metric("PM10", f"{pm10} µg/m³")
                with col3: st.metric("NO₂", f"{no2} µg/m³")
                with col4: st.metric("CO", f"{co} mg/m³")
                with col5: st.metric("O₃", f"{o3} µg/m³")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    elif submitted and model is None:
        st.error("Model not loaded. Please check if model files are uploaded.")
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **AirQuality AI** uses machine learning to predict Air Quality Index (AQI) based on pollutant levels.
        
        **Features:**
        - 🌿 Real-time AQI prediction
        - 💡 Personalized health recommendations  
        - 📱 Mobile-friendly interface
        - 🎯 Accurate Gradient Boosting model
        
        **Pollutants Monitored:**
        - PM2.5 (Fine particles)
        - PM10 (Coarse particles) 
        - NO₂ (Nitrogen dioxide)
        - CO (Carbon monoxide)
        - O₃ (Ozone)
        
        *Built with ❤️ using Streamlit and Scikit-learn*
        """)

if __name__ == "__main__":
    main()