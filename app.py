import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Wind Power Prediction",
    page_icon="🌬️",
    layout="wide"
)

# Define the ANN Model (same architecture as training)
class WindPowerANN(nn.Module):
    def __init__(self, input_dim):
        super(WindPowerANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Load the trained model
        device = torch.device('cpu')  # Use CPU for deployment
        model = WindPowerANN(input_dim=8)  # 8 features
        model.load_state_dict(torch.load('wind_power_model.pth', map_location=device))
        model.eval()
        
        # Load the scaler
        scaler = joblib.load('scaler.pkl')
        
        return model, scaler, device
    except FileNotFoundError:
        st.error("⚠️ Model or scaler files not found. Please ensure 'wind_power_model.pth' and 'scaler.pkl' are in the same directory.")
        return None, None, None

# Main app
def main():
    # Header
    st.title("🌬️ Wind Power Generation Predictor")
    st.markdown("### Predict power output based on weather conditions")
    st.markdown("---")
    
    # Load model
    model, scaler, device = load_model_and_scaler()
    
    if model is None:
        st.info("📋 **To use this app, you need to save your model and scaler:**")
        st.code("""
    # After training, add these lines to save your model:
      torch.save(model.state_dict(), 'wind_power_model.pth')
      joblib.dump(scaler, 'scaler.pkl')
        """, language='python')
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Input Weather Parameters")
        
        # Input fields with reasonable defaults
        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-20.0,
            max_value=50.0,
            value=19.0,
            step=0.1,
            help="Ambient temperature in Celsius"
        )
        
        humidity = st.slider(
            "💧 Humidity (%)",
            min_value=0,
            max_value=100,
            value=90,
            help="Relative humidity percentage"
        )
        
        wind_speed = st.number_input(
            "💨 Wind Speed (m/s)",
            min_value=0.0,
            max_value=50.0,
            value=12.6,
            step=0.1,
            help="Wind speed in meters per second"
        )
        
        wind_direction = st.number_input(
            "🧭 Wind Direction (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=5.9,
            step=0.1,
            help="Wind direction in degrees"
        )
        
    with col2:
        st.subheader("📈 Additional Parameters")
        
        pressure = st.number_input(
            "🔽 Atmospheric Pressure",
            min_value=0.0,
            max_value=200.0,
            value=9.3,
            step=0.1,
            help="Atmospheric pressure parameter"
        )
        
        param1 = st.number_input(
            "📊 Parameter 6",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.1,
            help="Additional weather parameter"
        )
        
        param2 = st.number_input(
            "📊 Parameter 7",
            min_value=0.0,
            max_value=200.0,
            value=65.0,
            step=0.1,
            help="Additional weather parameter"
        )
        
        param3 = st.number_input(
            "📊 Parameter 8",
            min_value=0.0,
            max_value=200.0,
            value=10.0,
            step=0.1,
            help="Additional weather parameter"
        )
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("⚡ Predict Power Output", type="primary", use_container_width=True)
    
    if predict_button:
        if model is None or scaler is None:
            st.error("⚠️ Model or scaler not loaded. Please check the files.")
            return
            
        # Prepare input data
        input_data = np.array([[
            temperature, humidity, wind_speed, wind_direction,
            pressure, param1, param2, param3
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Convert to tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0][0]
        
        # Display result
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Predicted Power Output",
                value=f"{prediction:.2f} kW",
                delta=None
            )
        
        with col2:
            if prediction > 0:
                efficiency = min((prediction / 100) * 100, 100)
                st.metric(
                    label="Efficiency Estimate",
                    value=f"{efficiency:.1f}%"
                )
        
        with col3:
            if prediction > 0:
                daily_output = prediction * 24
                st.metric(
                    label="Daily Output (Est.)",
                    value=f"{daily_output:.1f} kWh"
                )

        # Visualization
        st.markdown("---")
        st.subheader("📊 Input Parameters Visualization")
        
        # Create radar chart for input parameters
        categories = ['Temperature', 'Humidity', 'Wind Speed', 'Wind Dir', 
                     'Pressure', 'Param 6', 'Param 7', 'Param 8']
        
        # Normalize values to 0-1 scale for visualization
        normalized_values = [
            temperature / 50,
            humidity / 100,
            wind_speed / 50,
            wind_direction / 360,
            pressure / 100,
            param1 / 200,
            param2 / 200,
            param3 / 200
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='Current Input',
            line_color='#1f77b4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Normalized Input Parameters"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Power output gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Power Output (kW)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "gray"},
                    {'range': [60, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app predicts wind power generation using a trained 
        Artificial Neural Network (ANN) model.
        
        **Model Architecture:**
        - Input Layer: 8 features
        - Hidden Layers: 64 → 32 → 16 neurons
        - Output: Power (kW)
        - Activation: ReLU
        
        **Features:**
        - Real-time prediction
        - Interactive visualizations
        - Parameter analysis
        
        ---
        
        **Instructions:**
        1. Enter weather parameters
        2. Click "Predict Power Output"
        3. View results and analysis
        """)
        
        st.markdown("---")
        st.markdown("**🔋 Power Generation Guide**")
        st.markdown("""
        - **Low**: < 30 kW
        - **Medium**: 30-60 kW
        - **High**: > 60 kW
        """)

if __name__ == "__main__":
    main()