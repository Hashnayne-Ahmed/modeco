# **MoDECO** — **Mo**tor **D**riven **E**nergy **C**onsumption and **O**ptimization

**MoDECO** - **Mo**tor **D**riven **E**nergy **C**onsumption and **O**ptimization is an open-source tool developed to forecast and optimize energy consumption in motor-driven systems based on a combined theoretical and data-driven framework.

<p align="justify">
The theoretical foundation of MoDECO relies on physics-based formulations that describe motor energy behavior under variable frequency drive (VFD) control, soft starter (SS) operation, and baseline constant-speed scenarios. Key electrical parameters—such as dip depth (ε), rated power factor (cos(φᵣ)), and torque deviation (δ)—are used to quantify normalized energy consumption under different operational modes.
</p>

<p align="justify">
To enable real-time and user-friendly predictions, MoDECO integrates these physics-derived relations into a lightweight forecasting model built using a physics-informed neural network (PINN). The trained model accurately maps observable electrical inputs to four normalized energy ratios, allowing facility managers, engineers, and energy analysts to estimate energy savings without requiring detailed speed profile measurements or extensive physical simulations.
</p>
---

## 🚀 Interactive Web Application

Access the live interactive MoDECO app here:

👉 [Launch MoDECO Web App](https://modeco-o8dt5odosc9lxzyecclwpu.streamlit.app/)

The app allows users to:
- Input basic motor electrical parameters.
- Estimate energy savings for different operational scenarios (VFD only, VFD + SS, SS only).
- Compare results for different motor speed profiles (sinusoidal and hyperbolic tangent).

---

## 📦 Repository Contents

- `modeco.py` — Source code for the interactive Streamlit web application.
- `scaler1.pkl`, `scaler2.pkl` — Pre-trained scalers for input normalization.
- `trained_model1.pth`, `trained_model2.pth` — Pre-trained PINN models for energy forecasting.
- `requirements.txt` — Python dependencies needed to run the app locally.

---

## ⚙️ Local Installation

To run MoDECO locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/Hashnayne-Ahmed/modeco.git
    cd modeco
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## ✨ Features

- Fast and accurate energy savings predictions.
- No need for detailed motor speed measurements.
- Lightweight deployment via Streamlit.
- Suitable for facility managers, energy analysts, and engineers.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

✅ Baseline corresponds to motor operation without VFD and without soft starter.

---

## 🎓 Acknowledgment
<p align="justify">
This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Advanced Manufacturing Office Award No. DE-EE0009729 ([Funder ID: 10.13039/100000015](https://doi.org/10.13039/100000015)). Support from the [University of Florida Industrial Training and Assessment Center (UF-ITAC)](https://iac.mae.ufl.edu/) ([IAC Network Link](https://iac.university/center/UF)) and the [Department of Mechanical and Aerospace Engineering at the University of Florida](https://mae.ufl.edu/) is also gratefully acknowledged.

This project was developed by [Hashnayne Ahmed](https://www.hashnayneahmed.com/) as part of ongoing research efforts in energy optimization and motor-driven systems.
</p>
