# Predictive Health Monitoring System  
## Remaining Useful Life (RUL) Estimation for Industrial Engines

---

## Overview

This project implements a machine learning–based predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of turbofan engines using multivariate time-series sensor data from the NASA CMAPSS dataset.

The system learns degradation patterns from historical run-to-failure data and predicts how many operational cycles remain before an engine is expected to fail.

In addition to model training and evaluation, the project simulates a real-time monitoring system that dynamically evaluates engine health and triggers alerts when failure risk increases.

---

## Problem Statement

Industrial and aerospace engines degrade gradually over time under varying operating conditions.

Traditional maintenance strategies include:

- Fixed interval servicing  
- Reactive maintenance after failure  

These approaches may lead to:

- Increased downtime  
- Unexpected breakdowns  
- Higher operational cost  

This project demonstrates a **predictive maintenance approach**, where sensor data is used to forecast failure before it occurs.

---

## Dataset

NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)

- 100 simulated engines  
- 21 sensor measurements  
- 3 operational settings  
- Run-to-failure lifecycle data  
- 20,631 time-series records  

Each engine operates until failure, enabling supervised learning of degradation behavior.

---

## Methodology

### 1. Data Preprocessing

- Removed redundant columns  
- Assigned meaningful feature names  
- Computed Remaining Useful Life (RUL):

  ```
  RUL = Maximum Cycle per Engine − Current Cycle
  ```

- Removed constant (zero-variance) sensor features  
- Performed engine-wise train/test split to prevent data leakage  

---

### 2. Model Development

Two ensemble regression models were implemented:

- RandomForest Regressor  
- XGBoost Regressor  

RandomForest configuration:

- 200 trees  
- Controlled maximum depth  
- Regularization through minimum samples  

The model captures non-linear degradation patterns in mechanical systems.

---

## Model Evaluation

Performance was evaluated using:

- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  

Example Results:

- RandomForest RMSE ≈ 48 cycles  
- XGBoost RMSE (comparison model)

---

## Visualizations

The system includes:

- True vs Predicted RUL scatter plot  
- Feature importance ranking  
- Engine degradation trend (Cycle vs Predicted RUL)  

These visualizations validate model behavior and highlight key degradation indicators.

---

## Simulated Real-Time Monitoring

The project extends beyond offline prediction by simulating real-time monitoring:

- Cycle-by-cycle prediction  
- Dynamic RUL updates  
- Health classification system:

  - RUL > 100 → Healthy  
  - 50 < RUL ≤ 100 → Warning  
  - RUL ≤ 50 → Critical  

- Automatic maintenance alert when critical threshold is reached  

This simulates how predictive maintenance systems operate in industrial environments.

---

## Project Structure

```
RUL_Prediction_Project/
│
├── data/
│   └── train_FD001.txt
│
├── model/
│   └── rul_model.pkl
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Joblib  

---

## Key Learnings

- Preventing data leakage in time-series systems  
- Regression modeling for reliability engineering  
- Ensemble methods comparison  
- Feature importance interpretation  
- Simulating real-time predictive maintenance systems  

---

## Future Improvements

- Hyperparameter tuning with GridSearchCV  
- LSTM-based deep learning model for sequential prediction  
- REST API deployment using Django or FastAPI  
- Web-based real-time monitoring dashboard  

---

## Conclusion

This project demonstrates an applied machine learning solution for predictive maintenance in high-reliability mechanical systems. By modeling degradation behavior from historical sensor data, the system enables condition-based maintenance and early failure detection.