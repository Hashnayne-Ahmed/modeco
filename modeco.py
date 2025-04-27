import streamlit as st
import numpy as np
import torch
import joblib

class EnergyModel(torch.nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 48),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(48, 48),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(48, 4)
        )
    def forward(self, x):
        return self.net(x)

def load_model_and_scaler(profile_number):
    if profile_number == 1:
        scaler = joblib.load('scaler1.pkl')
        model = EnergyModel()
        model.load_state_dict(torch.load('trained_model1.pth', map_location=torch.device('cpu')))
    elif profile_number == 2:
        scaler = joblib.load('scaler2.pkl')
        model = EnergyModel()
        model.load_state_dict(torch.load('trained_model2.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

st.title("MoDECO - Motor Driven Energy Consumption and Optimization")
st.subheader("Forecast Energy Savings for Motor-Driven Systems")
st.markdown("This tool estimates energy savings (%) based on measurable electrical parameters.")

st.header("Input Electrical Measurements")
input_option = st.radio("Select Measurement Type:", ('Measured Current (I)', 'Measured Real Power (P)'))

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

cos_phi_r = st.number_input('Enter Rated Power Factor (cos(φᵣ))', min_value=0.0, max_value=1.0, value=0.85, step=0.01)

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

if st.button('Predict Energy Savings'):
    X = np.array([[epsilon, cos_phi_r, delta]])

    results = {}
    for profile_number, profile_name in zip([1, 2], ['Sinusoidal Profile', 'Hyperbolic Tangent Profile']):
        model, scaler = load_model_and_scaler(profile_number)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_pred = model(X_tensor).detach().numpy().flatten()
        savings = (1 - y_pred) * 100
        results[profile_name] = {
            'Savings (VFD Only)': savings[0],
            'Savings (VFD + SS)': savings[1],
            'Savings (SS Gain after VFD)': savings[2],
            'Savings (SS Only)': savings[3]
        }

    st.header("Predicted Energy Savings (%)")

metrics_mapping = {
    'Savings (VFD Only)': "Savings with VFD Only (compared to baseline)",
    'Savings (VFD + SS)': "Savings with VFD and Soft Starter (compared to baseline)",
    'Savings (SS Gain after VFD)': "Additional Savings by Adding Soft Starter to VFD",
    'Savings (SS Only)': "Savings with Soft Starter Only (no VFD)"
}

combined_results = {}

for metric in metrics_mapping.keys():
    vals = [results[profile][metric] for profile in results]
    min_val = min(vals)
    max_val = max(vals)
    combined_results[metric] = (min_val, max_val)

for metric, (min_val, max_val) in combined_results.items():
    # Bigger font for label
    label = f"<div style='font-size:26px; font-weight:600; color:#333; margin-top:20px;'>{metrics_mapping[metric]}</div>"
    # Big font for value
    if abs(min_val - max_val) < 0.05:
        value = f"<div style='font-size:38px; font-weight:800; color:#000;'>Approximately {((min_val + max_val)/2):.1f}%</div>"
    else:
        value = f"<div style='font-size:38px; font-weight:800; color:#000;'>Approximately {min_val:.1f}% – {max_val:.1f}%</div>"
    st.markdown(label, unsafe_allow_html=True)
    st.markdown(value, unsafe_allow_html=True)

st.markdown("<div style='font-style: italic; font-size:18px; color: #555;'>Baseline corresponds to operation without VFD and without Soft Starter.</div>", unsafe_allow_html=True)