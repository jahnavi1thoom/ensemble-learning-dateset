import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System – Stacking Model",
    layout="wide"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict whether a loan "
    "will be approved by combining multiple ML models for better decision making."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Drop ID column safely
df = df.drop(["Loan_ID", "Lon_Amount_Term"], axis=1, errors="ignore")

# --------------------------------------------------
# Target & Features
# --------------------------------------------------
TARGET = "Loan_Status"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# --------------------------------------------------
# Stacking Model
# --------------------------------------------------
base_models = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("dt", DecisionTreeClassifier(max_depth=6, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
]

meta_model = LogisticRegression(max_iter=1000)

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Full pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", stack_model)
])

# --------------------------------------------------
# Train Model
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Sidebar – User Input
# --------------------------------------------------
st.sidebar.header("📝 Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Build user input (MUST match training columns)
user_input = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": employment,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amt,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1.0 if credit_history == "Yes" else 0.0,
    "Property_Area": property_area
}])

# --------------------------------------------------
# Model Architecture Display
# --------------------------------------------------
st.markdown("---")
st.subheader("🧩 Stacking Model Architecture")

st.markdown("""
**Base Models Used**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used**
- Logistic Regression  

📌 The meta-model learns how to combine predictions from base models.
""")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("---")
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    

    proba = model_pipeline.predict_proba(user_input)[0]
    approval_prob = proba[1]
    final_pred = 1 if approval_prob >= 0.4 else 0
    confidence = approval_prob * 100 if final_pred == 1 else (1-approval_prob) * 100

    st.subheader("📊 Loan Decision Result")

    if final_pred == 1:
        st.markdown("<h2 style='color:green;'>✅ Loan Approved</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:red;'>❌ Loan Rejected</h2>", unsafe_allow_html=True)

    # Base model predictions
    st.write("### 📊 Base Model Predictions")

    for name, model in base_models:
        temp_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        temp_pipe.fit(X_train, y_train)
        pred = temp_pipe.predict(user_input)[0]
        st.write(f"• {name.upper()} → {'Approved' if pred == 1 else 'Rejected'}")

    st.write("### 🧠 Final Stacking Decision")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    # --------------------------------------------------
    # Business Explanation
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("💼 Business Explanation")

    st.write(
        "Based on the applicant’s income, credit history, employment status, education, and property area, "
        "multiple machine learning models evaluated loan repayment risk. "
        "Their predictions were combined using a stacking ensemble approach, "
        f"leading to a final decision of **{'loan approval' if final_pred == 1 else 'loan rejection'}**."
    )
