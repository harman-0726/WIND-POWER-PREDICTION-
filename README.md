🌬️ Wind Power Prediction System (AI + Deep Learning)

A complete Wind Power Prediction System using Artificial Neural Networks (ANN) built with Python, PyTorch, and Streamlit. This project provides accurate wind energy forecasting based on meteorological parameters, enabling efficient grid management and renewable energy optimization.

🚀 Project Overview

This system predicts wind power output (in watts) using real-time weather data.
It solves the challenge of intermittent wind energy generation by using a trained deep learning model that delivers accurate, fast predictions suitable for industrial use.

🧠 Key Features
✔ Artificial Neural Network (ANN)

4-layer deep neural network

Hidden layers: 64 → 32 → 16 neurons

Activation: ReLU

Optimizer: Adam

Loss: MSE (Mean Squared Error)

Trained for 100 epochs

R² score above 0.85 (High accuracy)

✔ Input Features (8 parameters)

Temperature

Humidity

Wind speed

Wind direction

Atmospheric pressure

Additional meteorological parameters

✔ Output

Predicted power output (Watts)

Converts kW → W automatically for better readability

Example: 0.03 kW → 30 W

🖥 Frontend: Streamlit Web App

The system includes a user-friendly interface built with Streamlit:

Enter weather parameters in real time

Get instant predictions (under 100 ms)

Interactive charts:

Radar chart

Gauge meter

Metrics display

No technical expertise needed

🗄 Backend & Data Handling

Uses StandardScaler for normalization

Model saved using:

PyTorch → .pth

Joblib → .pkl (for scaler)

Accepts CSV datasets

Fully automated preprocessing pipeline

📂 Project Structure
wind_power_project/
│
├── backend/
│   ├── model.py
│   ├── train.py
│   ├── saved_model.pth
│   ├── saved_scaler.pkl
│   └── dataset.csv
│
├── frontend/
│   └── app.py
│
├── README.md
└── requirements.txt

🛠 Technologies Used

Python

PyTorch

NumPy / Pandas

Matplotlib / Seaborn

Scikit-Learn

Streamlit

📊 Model Performance

R² Score: > 0.85

Low MSE and MAE

Fast inference time: < 100 ms

Suitable for real-time deployment in wind farms.

🏗 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Model (Optional)
python backend/train.py

3️⃣ Run Streamlit Frontend
cd frontend
streamlit run app.py

🌱 Contribution to Clean Energy

This project represents an efficient, cost-effective, and open-source solution for wind energy forecasting.
It demonstrates how deep learning can support sustainable power management and help transition toward global renewable energy.

👨‍💻 Developer

Harmandeep Singh
Computer Science Student | AI & ML Enthusiast
Working on real-world AI applications.

⭐ Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!
