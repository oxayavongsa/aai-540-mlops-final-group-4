import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SageMaker input directory
INPUT_PATH = "/opt/ml/input/data/train/cardio_engineered.csv"
OUTPUT_PATH = "/opt/ml/model"

try:
    print("✅ Reading input CSV from:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    # Encode categories
    df["age_group"] = df["age_group"].map({'30s':1, '40s':2, '50s':3, '60s':4, '70s':5})
    df["cholesterol_label"] = df["cholesterol_label"].map({'Normal':1, 'Above Normal':2, 'Well Above Normal':3})
    df["bp_category"] = df["bp_category"].map({'normal':1, 'stage1':2, 'stage2':3})
    df["bmi_category"] = df["bmi_category"].map({'normal':1, 'overweight':2, 'obese':3})
    df.dropna(inplace=True)

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_PATH, "model.joblib"))

    print("✅ Training completed and model saved.")
except Exception as e:
    print("❌ ERROR during training:", str(e))
    raise