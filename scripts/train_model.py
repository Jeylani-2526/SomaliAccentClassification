import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Load datasets
somali_df = pd.read_csv("extracted_features.csv")
nonsomali_df = pd.read_csv("extracted_features_nonSomA.csv")

# Combine datasets
df = pd.concat([somali_df, nonsomali_df], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop non-feature columns if any (like filenames)
if 'Filename' in df.columns:
    df = df.drop(columns=['Filename'])

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=["non-Somali (-1)", "Somali (+1)"]))

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("✅ Model and scaler saved in 'models/' directory.")
