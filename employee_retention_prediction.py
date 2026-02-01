import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

df = pd.read_csv('/content/aug_train (1).csv')

df.head()

df = df.dropna()

df.shape

df.info()

df.describe()

#target distribution bar graph

plt.figure(figsize=(5,4))
sns.histplot(df['target'], bins=2, discrete=True)
plt.xlabel("Target (0 = No Job Change, 1 = Job Change)")
plt.ylabel("Count")
plt.title("Target Distribution")
plt.show()

#feature Engineering
education_mapping = {
    'Primary School': 0,
    'High School': 1,
    'Graduate': 2,
    'Masters': 3,
    'Phd': 4
}
experience_mapping = {
    'No relevent experience': 0,
    'Has relevent experience': 1
}
course_mapping = {
    'Full time course': 2,
    'Part time course': 1,
    'no_enrollment': 0,
}
df['experience'] = df['experience'].replace({
    '>20': 21,
    '<1': 0
})

df['gender'] = df['gender'].replace({
    'Male': 1,
    'Female': 0,
    'Other': 2
})

df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
df['experience'] = pd.to_numeric(df['experience'], errors='coerce')

df['last_new_job'] = df['last_new_job'].replace({
    '>4': 5,
    'never': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4
})

company_size_mapping = {
    '<10': 0,
    '10/49': 1,
    '50-99': 2,
    '100-500': 3,
    '500-999': 4,
    '1000-4999': 5,
    '5000-9999': 6,
    '10000+': 7
}

df['company_size'] = df['company_size'].map(company_size_mapping)
df['company_size'] = pd.to_numeric(df['company_size'], errors='coerce')
df['last_new_job'] = pd.to_numeric(df['last_new_job'], errors='coerce')

df['enrolled_university'] = df['enrolled_university'].map(course_mapping)
df['relevent_experience'] = df['relevent_experience'].map(experience_mapping)
df['education_level'] = df['education_level'].map(education_mapping)

#outliers handling
df['training_hours'] = df['training_hours'].clip(upper=170)

#converted numerical columns
numeric_cols = [
    'experience',
    'last_new_job',
    'training_hours',
    'education_level',
    'relevent_experience',
    'enrolled_university',
    'gender'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

df.head()

n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(15, 4 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=df['target'], y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Select only numeric columns for correlation calculation
corr = df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(12,8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

#Feature selection
X = df.drop(['target','enrollee_id','city','gender','company_type','major_discipline','enrolled_university'], axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#handling imbalance data (50-50)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(y.value_counts())
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

accuracy_score(y_test, y_pred_lr)
classification_report(y_test, y_pred_lr)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]

accuracy_score(y_test, y_pred_rf)
roc_auc_score(y_test, y_prob_rf)
classification_report(y_test, y_pred_rf)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

accuracy_score(y_test, y_pred_xgb)
roc_auc_score(y_test, y_prob_xgb)
classification_report(y_test, y_pred_xgb)

lgb_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=12,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

accuracy_score(y_test, y_pred_lgb)
roc_auc_score(y_test, y_prob_lgb)
classification_report(y_test, y_pred_lgb)

pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_lgb)
    ],
    'ROC_AUC': [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb),
        roc_auc_score(y_test, y_prob_lgb)
    ]
})

# plot fig of ROC-AUC
from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(7,6))

models = {
    "Logistic Regression": lr,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model
}

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Curve Comparison")
plt.legend(loc="lower right")
plt.show()

# model Evaluation

accuracy_score(y_test, y_pred_lgb)
roc_auc_score(y_test, y_prob_lgb)

pd.DataFrame(
    classification_report(y_test, y_pred_lgb, output_dict=True)
).transpose().round(3)

cm = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – LightGBM")
plt.show()

#Feature Importance (This IS Feature Selection Output)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

feature_importance

# Predict probability
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = lgb_model.predict(X_test)

# Create risk levels
risk_level = pd.cut(
    y_prob_lgb,
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Power BI dataset
powerbi_df = X_test.copy()

powerbi_df['Actual_Target'] = y_test.values
powerbi_df['Predicted_Target'] = y_pred_lgb
powerbi_df['Prediction_Probability'] = y_prob_lgb
powerbi_df['Risk_Level'] = risk_level

powerbi_df.to_csv("employee_retention_powerbi.csv", index=False)

import joblib

joblib.dump(lgb_model, "final_lightgbm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

"""Model Deployment

The trained LightGBM model was deployed as a REST API using FastAPI. Since the project was developed in Google Colab, ngrok was used to expose the API endpoint for live testing and demonstration.
"""


# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# from fastapi import FastAPI
# import pandas as pd
# import joblib
# 
# model = joblib.load("final_lightgbm_model.pkl")
# scaler = joblib.load("scaler.pkl")
# 
# app = FastAPI(title="Employee Retention API")
# 
# @app.get("/")
# def home():
#     return {"status": "LightGBM Employee Retention API is running"}
# 
# @app.post("/predict")
# def predict(data: dict):
#     # Define the exact order of features that the scaler was fitted on
#     feature_order = ['city_development_index','relevent_experience', 'education_level', 'experience', 'company_size', 'last_new_job', 'training_hours']
#     df = pd.DataFrame([data], columns=feature_order)
#     df_scaled = scaler.transform(df)
#     prob = model.predict_proba(df)[0][1]
# 
#     if prob < 0.33:
#         risk = "High Risk"
#     elif prob > 0.33 and prob < 0.66:
#         risk = "Medium Risk"
#     elif prob > 0.66 :
#         risk = "Low Risk"
# 
#     return {
#         "prediction_probability": round(float(prob), 4),
#         "risk_level": risk
#     }

# !nohup uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

# !ps aux | grep uvicorn

# !pkill -f uvicorn

# !nohup uvicorn app:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

from pyngrok import ngrok

# Kill any existing ngrok processes to free up tunnels
ngrok.kill()

ngrok.set_auth_token("389RxZNAxbKci5ZNvf6JlrPN9gB_xEx71QmbrW5zsmcDHKrb")
public_url = ngrok.connect(8000)
public_url

# !cat uvicorn.log

import requests

url = "https://debbi-courageous-tunelessly.ngrok-free.dev/predict"

sample_input = {
    "city_development_index": 0.300,
    "relevent_experience": 0,
    "education_level": 2,
    "experience": 7,
    "company_size": 2,
    "last_new_job": 1,
    "training_hours": 45

}

response = requests.post(url, json=sample_input)
response.json()

"""Model and files saved in google drive

"""

from google.colab import drive
drive.mount('/content/drive')

project_path = "/content/drive/MyDrive/Employee_Retention_Project"
# !mkdir -p "$project_path"

# !cp final_lightgbm_model.pkl "$project_path/"
# !cp scaler.pkl "$project_path/"

# !cp outlier_bounds.pkl "$project_path/"

# !cp app.py "$project_path/"

# !cp employee_retention_powerbi.csv "$project_path/"

# !ls "$project_path"