import streamlit as st # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pickle
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score  # type: ignore
from imblearn.over_sampling import SMOTE  # type: ignore

st.title("💖 Heart Disease Risk Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart (1).csv")

heart_data = load_data()



# Check for missing values
st.subheader("Missing Values Check")
st.write(heart_data.isnull().sum())



# Class Distribution
st.subheader("Target Variable Distribution")
fig, ax = plt.subplots()
sns.countplot(x="target", data=heart_data, ax=ax)
st.pyplot(fig)

# Handling Imbalance using SMOTE
X = heart_data.drop(columns=["target"])
y = heart_data["target"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

st.subheader("Target Variable Distribution After SMOTE")
fig, ax = plt.subplots()
sns.countplot(x=y_resampled, ax=ax)
st.pyplot(fig)

# Correlation Analysis
st.subheader("Correlation Matrix")
correlation_matrix = heart_data.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=2)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Model Selection
model_choice = st.sidebar.selectbox("Select a Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(C=0.1, penalty="l2")
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)

# Train Model
model.fit(X_train, y_train)

# Model Evaluation
st.subheader("Model Evaluation")
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

st.write(f"✅ **Training Accuracy:** {train_accuracy:.2f}")
st.write(f"✅ **Testing Accuracy:** {test_accuracy:.2f}")

st.write("📊 **Classification Report:**")
st.text(classification_report(y_test, model.predict(X_test)))

st.write(f"🏆 **ROC AUC Score:** {roc_auc_score(y_test, model.predict(X_test)):.2f}")

# Save the trained model
model_filename = "logistic_regression_model.pkl" if model_choice == "Logistic Regression" else "random_forest_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

st.write(f"🎯 Model saved as {model_filename}")

# User Input for Prediction

st.sidebar.header("  💖  SmartCardia")

def user_input():
    age = st.sidebar.number_input("Age", 20, 100, 50, help="Age of the person in years.")
    sex = st.sidebar.radio("Sex", [0, 1], help="0: Female, 1: Male")
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120, help="Resting blood pressure (in mm Hg).")
    chol = st.sidebar.number_input("Serum Cholesterol", 100, 500, 200, help="Serum cholesterol level in mg/dL.")
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="0: False, 1: True")
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", 70, 220, 150, help="Maximum heart rate achieved during exercise.")
    exang = st.sidebar.radio("Exercise-Induced Angina", [0, 1], help="0: No, 1: Yes")
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
    slope = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], help="Number of major vessels (0-3) colored by fluoroscopy.")
    thal = st.sidebar.selectbox("Thalassemia Type (0-3)", [0, 1, 2, 3], help="0: Normal, 1: Fixed defect, 2: Reversible defect")
    
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

user_data = user_input()

# Load the saved scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Scale user input
user_data_scaled = scaler.transform(user_data)

# Predict
if st.sidebar.button("Predict"):
    prediction = model.predict(user_data_scaled)[0]
    if prediction == 1:
        st.sidebar.success("✅ The patient is unlikely to have heart disease.")
    else:
        st.sidebar.error("🚨 The patient is likely to have heart disease.")
