# **MoDECO** — **Mo**tor **D**riven **E**nergy **C**onsumption and **O**ptimization

MoDECO is an open-source tool developed to forecast and optimize energy consumption in motor-driven systems based on a combined theoretical and data-driven framework. 

The theoretical foundation of MoDECO relies on physics-based formulations that describe motor energy behavior under variable frequency drive (VFD) control, soft starter (SS) operation, and baseline constant-speed scenarios. Key electrical parameters—such as dip depth (ε), rated power factor (cos(φᵣ)), and torque deviation (δ)—are used to quantify normalized energy consumption under different operational modes.

To enable real-time and user-friendly predictions, MoDECO integrates these physics-derived relations into a lightweight forecasting model built using a physics-informed neural network (PINN). The trained model accurately maps observable electrical inputs to four normalized energy ratios, allowing facility managers, engineers, and energy analysts to estimate energy savings without requiring detailed speed profile measurements or extensive physical simulations.

MoDECO supports:
- Fast estimation of energy savings achievable through VFDs, soft starters, or both.
- Optimization-based decision-making for energy-efficient motor operation.
- Deployment via an interactive web interface or local scripts.

This repository contains the trained models, datasets, and an interactive app to help users explore and apply MoDECO to real-world motor-driven systems.
