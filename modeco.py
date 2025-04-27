import streamlit as st
import numpy as np
import torch
import joblib

# === Function to Load Model and Scaler ===
def load_model_and_scaler(profile_number):
    if profile_number == 1:  # Sinusoidal
        scaler = joblib.load('scaler1.pkl')
        model = torch.load('trained_model1.pth')
    elif profile_number == 2:  # Hyperbolic Tangent
        scaler = joblib.load('scaler2.pkl')
        model = torch.load('trained_model2.pth')
    model.eval()
    return model, scaler

st.title("MoDECO - Motor Driven Energy Consumption and Optimization")
st.subheader("Forecast Energy Savings for Motor-Driven Systems")
st.markdown("""
This tool estimates energy savings (%) based on measurable electrical parameters.
""")

# === USER INPUTS ===

# Electrical Measurements
st.header("Input Electrical Measurements")
input_option = st.radio(
    "Select Measurement Type:",
    ('Measured Current (I)', 'Measured Real Power (P)')
)

if input_option == 'Measured Current (I)':
    I_measured = st.number_input('Enter Measured Motor Current, I(t) [Amps]', min_value=0.0, value=100.0, step=1.0)
    I_rated = st.number_input('Enter Rated Motor Current, I_r [Amps]', min_value=0.0, value=120.0, step=1.0)
    epsilon = 1.0 - np.sqrt(I_measured / I_rated)
else:
    P_measured = st.number_input('Enter Measured Motor Power, P(t) [kW]', min_value=0.0, value=75.0, step=1.0)
    P_rated = st.number_input('Enter Rated Motor Power, P_rated [kW]', min_value=0.0, value=100.0, step=1.0)
    epsilon = 1.0 - (P_measured / P_rated)

epsilon = np.clip(epsilon, 0, 1)
st.write(f"**Calculated Dip Depth (ε):** {epsilon:.4f}")

# Rated Power Factor
cos_phi_r = st.number_input('Enter Rated Power Factor (cos(φᵣ))', min_value=0.0, max_value=1.0, value=0.85, step=0.01)

# Load Type
st.header("Select Load Type")
load_type = st.selectbox(
    "Select the Mechanical Load Type Connected to the Motor:",
    ('Variable Torque Load (Pump/Fan)', 'Constant Torque Load (Conveyor/Hoist)', 'Custom δ')
)

if load_type == 'Variable Torque Load (Pump/Fan)':
    delta = 0.12
elif load_type == 'Constant Torque Load (Conveyor/Hoist)':
    delta = 0.06
else:
    delta = st.number_input('Enter Custom Torque Deviation (δ)', min_value=0.0, max_value=1.0, value=0.08, step=0.01)

st.write(f"**Assigned Torque Deviation (δ):** {delta:.3f}")

# === PREDICTION ===
if st.button('Predict Energy Savings'):

    # Input array
    X = np.array([[epsilon, cos_phi_r, delta]])

    results = {}
    for profile_number, profile_name in zip([1,2], ['Sinusoidal Profile', 'Hyperbolic Tangent Profile']):
        
        # Load model and scaler
        model, scaler = load_model_and_scaler(profile_number)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_pred = model(X_tensor).detach().numpy().flatten()  # 4 outputs: energy ratios

        # Calculate Energy Savings (%)
        savings = (1 - y_pred) * 100

        results[profile_name] = {
            'Savings (VFD Only)': savings[0],     # (1 - E/E0) * 100
            'Savings (VFD + SS)': savings[1],      # (1 - Es/E0) * 100
            'Savings (SS Gain after VFD)': savings[2],  # (1 - Es/E) * 100
            'Savings (SS Only)': savings[3]        # (1 - Es0/E0) * 100
        }

    # === Display Results ===
    st.header("Predicted Energy Savings (%)")

    for profile_name, values in results.items():
        st.subheader(profile_name)
        st.metric(label="Savings with VFD Only (compared to baseline)", value=f"{values['Savings (VFD Only)']:.2f}%")
		st.metric(label="Savings with VFD and Soft Starter (compared to baseline)", value=f"{values['Savings (VFD + SS)']:.2f}%")
		st.metric(label="Additional Savings by Adding Soft Starter to VFD", value=f"{values['Savings (SS Gain after VFD)']:.2f}%")
		st.metric(label="Savings with Soft Starter Only (no VFD)", value=f"{values['Savings (SS Only)']:.2f}%")