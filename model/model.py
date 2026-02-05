import pandas as pd
import joblib
import os
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------
# Project root directory (auto detect)
# -----------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ✅ Correct dataset path (as per your project)
CSV_PATH = os.path.join(ROOT_DIR, "Dataset", "student_data.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

# -----------------------------------------
# Load dataset
# -----------------------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------------------
# Features & target
# -----------------------------------------
X = df[["StudyHours", "Attendance"]]
y = df["ResultNumeric"]   # 0 / 1 (Fail / Pass)

# -----------------------------------------
# Scaling
# -----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================================================
# 🔹 STEP 1: Train Linear Regression
# =====================================================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict scores
train_linear_pred = linear_model.predict(X_train)
test_linear_pred = linear_model.predict(X_test)

# =====================================================
# 🔹 STEP 2: Create Hybrid Features
# =====================================================
X_train_hybrid = np.column_stack((X_train, train_linear_pred))
X_test_hybrid = np.column_stack((X_test, test_linear_pred))

# =====================================================
# 🔹 STEP 3: Train Logistic Regression (Final Model)
# =====================================================
logistic_model = LogisticRegression()
logistic_model.fit(X_train_hybrid, y_train)

# -----------------------------------------
# Save models
# -----------------------------------------
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(linear_model, os.path.join(MODEL_DIR, "linear_model.pkl"))
joblib.dump(logistic_model, os.path.join(MODEL_DIR, "hybrid_logistic_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Hybrid model created successfully")
