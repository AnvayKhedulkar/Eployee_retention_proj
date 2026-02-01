Employee Retention Prediction Project

Project Overview
This project focuses on predicting employee retention using machine learning models such as LightGBM and XGBoost. The project also includes an API layer built using FastAPI to serve predictions. The following steps describe the complete project configuration that must be performed before uploading the project to GitHub.

Environment and Dependency Setup

The following libraries must be installed before running the project:

!pip install xgboost lightgbm imbalanced-learn

The imbalanced-learn library is used to handle class imbalance in the dataset. Oversampling is performed using SMOTE (Synthetic Minority Oversampling Technique).

imb learn over sampling for smote

Additional dependencies required for API development and deployment:

!pip install fastapi uvicorn pyngrok

API Configuration

The API part of the code must be configured before execution. The FastAPI application is defined in app.py and is responsible for loading the trained model and preprocessing artifacts and serving predictions through REST endpoints.

configure the api part of the code

Running the FastAPI Server

Start the FastAPI server using Uvicorn in background mode:

!nohup uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

Verify that the Uvicorn process is running:

!ps aux | grep uvicorn

If required, stop the running Uvicorn server:

!pkill -f uvicorn

Restart the FastAPI server using the same command:

!nohup uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

Ngrok Configuration

Install pyngrok to expose the local FastAPI server to a public URL, especially when running the project in a cloud or Google Colab environment:

install pyngrok

Check the Uvicorn server logs for debugging and verification:

!cat uvicorn.log

Google Colab Setup

This project is designed to run in a Google Colab environment. Ensure that Google Colab is properly installed and configured before execution:

install google collab

Project Packaging and File Management

Create the project directory where all required files will be stored:

!mkdir -p "$project_path"

Copy all necessary model files, preprocessing artifacts, API code, and dataset into the project directory:

!cp final_lightgbm_model.pkl "$project_path/"
!cp scaler.pkl "$project_path/"
!cp outlier_bounds.pkl "$project_path/"
!cp app.py "$project_path/"
!cp employee_retention_powerbi.csv "$project_path/"

Verify that all files have been copied successfully:

!ls "$project_path"

Project Files Included

final_lightgbm_model.pkl – Trained LightGBM model
scaler.pkl – Feature scaling object
outlier_bounds.pkl – Outlier detection and handling configuration
app.py – FastAPI application file
employee_retention_powerbi.csv – Dataset used for analysis and Power BI visualization

Notes

All commands prefixed with "!" are intended to be executed in a Jupyter or Google Colab environment.
Ensure all dependencies are installed before starting the API server.
Restart the Uvicorn server after making any changes to app.py.
Check uvicorn.log for troubleshooting if the server fails to start.
