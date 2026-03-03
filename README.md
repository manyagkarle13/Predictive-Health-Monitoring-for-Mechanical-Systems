Predictive Health Monitoring (PHM) for Industrial Systems
An end-to-end industrial analytics platform designed to forecast the Remaining Useful Life (RUL) of mechanical components using multivariate sensor telemetry. This project demonstrates the application of Machine Learning in ensuring "Zero-Failure" reliability for mission-critical hardware, such as liquid propulsion systems.

Key Features
Predictive Engine: Utilizes Random Forest Regression to analyze non-linear degradation patterns in time-series data.

Mission Control Dashboard: A React-based interface providing real-time health status categorized as Healthy, Warning, or Critical.

Robust Backend: Powered by the Django REST Framework for secure and scalable data handling.

Standardized Dataset: Trained and validated on the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

Tech Stack
Frontend: React.js, Tailwind CSS

Backend: Django, Django REST Framework

Machine Learning: Python, Scikit-learn, Pandas, NumPy

Environment: Virtual Environment (venv) for dependency isolation

Analytics Logic
The system processes 21 sensor channels (including temperature, pressure, and vibration) to estimate the RUL. By mapping sensor trends against historical failure profiles, the model identifies precursors to mechanical failure before they occur.

Project Structure
Plaintext
RUL_Prediction_Project/
├── backend/            # Django API and logic
├── dashboard/          # React frontend interface
├── data/               # NASA CMAPSS telemetry files
├── ml/                 # Model training and prediction scripts
└── README.md           # Project documentation