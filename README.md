# Predictive Health Monitoring (PHM) for Mechanical Systems

An end-to-end industrial analytics platform designed to forecast the Remaining Useful Life (RUL) of mechanical components using multivariate sensor telemetry.

This project demonstrates the application of Machine Learning to enable zero-failure reliability in mission-critical systems such as liquid propulsion and industrial machinery.

---

## 1. Problem Statement

Mechanical systems degrade over time due to operational stress. Unexpected failures can result in:

- Operational downtime
- Financial losses
- Safety risks

This system predicts component failure in advance by estimating Remaining Useful Life (RUL).

---

## 2. Solution Overview

The platform integrates:

- Machine Learning-based RUL prediction
- Real-time health monitoring dashboard
- REST API backend for prediction serving
- Standardized industrial dataset (NASA CMAPSS)

---

## 3. Key Features

### 3.1 Predictive Engine
- Implements Random Forest Regression
- Captures non-linear degradation patterns
- Trained on time-series sensor data

### 3.2 Mission Control Dashboard
- Built using React.js and Tailwind CSS
- Displays system health categorized as:
  - Healthy
  - Warning
  - Critical

### 3.3 Backend API
- Developed using Django REST Framework
- Modular and scalable architecture
- Secure data handling and prediction endpoints

### 3.4 Analytics Logic
- Processes 21 sensor channels including:
  - Temperature
  - Pressure
  - Vibration
- Maps sensor trends against historical failure patterns
- Estimates Remaining Useful Life dynamically

---

## 4. Project Structure

RUL_Prediction_Project/
│
├── backend/          # Django REST API and business logic
├── dashboard/        # React frontend application
├── ml/               # Model training and prediction scripts
├── data/             # NASA CMAPSS telemetry dataset
└── README.md         # Project documentation

---

## 5. Tech Stack

### Frontend
- React.js
- Tailwind CSS

### Backend
- Django
- Django REST Framework

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy

### Environment
- Python Virtual Environment (venv)

---

## 6. Dataset

The model is trained and validated using the NASA CMAPSS 
(Commercial Modular Aero-Propulsion System Simulation) dataset.

The dataset contains simulated engine degradation data used for predictive maintenance research.

---

## 7. System Workflow

1. Load and preprocess sensor data
2. Perform feature engineering
3. Train Random Forest regression model
4. Predict Remaining Useful Life
5. Serve predictions via REST API
6. Visualize system health in dashboard

---

## 8. Future Enhancements

- Integrate LSTM-based deep learning models
- Add real-time streaming support
- Containerize using Docker
- Deploy on cloud infrastructure (AWS/Azure)
- Implement model monitoring and performance tracking

---

## 9. Use Cases

- Aerospace engine monitoring
- Industrial equipment predictive maintenance
- Reliability engineering systems
- Mission-critical hardware diagnostics

---

## 10. Getting Started

### Clone Repository

git clone https://github.com/your-username/your-repo-name.git
cd RUL_Prediction_Project

### Backend Setup

cd backend
python -m venv venv
venv\Scripts\activate      # For Windows
pip install -r requirements.txt
python manage.py runserver

### Frontend Setup

cd dashboard
npm install
npm start
