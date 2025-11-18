# =====================================================================
# Task 2: Supervised Learning - Predict Test Results
# Random Forest Classifier for Healthcare Test Result Prediction
# =====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Load and Clean Dataset
# ---------------------------------------------------------------------
print("=" * 60)
print("TASK 2: SUPERVISED LEARNING - TEST RESULTS PREDICTION")
print("=" * 60)

df = pd.read_csv("healthcare_dataset.csv")

# Remove missing values related to prediction
df = df.dropna(subset=["Test Results", "Billing Amount"])

# Feature Selection
features = [
    "Age", "Gender", "Blood Type", "Medical Condition",
    "Billing Amount", "Room Number", "Admission Type", "Medication"
]

X = df[features].copy()
y = df["Test Results"]

# ---------------------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------------------
label_encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ---------------------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------------------------------------------------
# Model Training - Random Forest Classifier
# ---------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# Model Prediction
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le_target.classes_
))

# ---------------------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le_target.classes_,
    yticklabels=le_target.classes_
)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# Feature Importance Plot
# ---------------------------------------------------------------------
feature_importances = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importances["Feature"],
    feature_importances["Importance"],
    color="teal"
)
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

print("\nFeature Importance:")
print(feature_importances.to_string(index=False))

# ---------------------------------------------------------------------
# Actual vs Predicted Results
# ---------------------------------------------------------------------
results_df = pd.DataFrame({
    "Actual": le_target.inverse_transform(y_test),
    "Predicted": le_target.inverse_transform(y_pred)
})

print("\n" + "=" * 60)
print("PREDICTED VS ACTUAL (First 20 Samples)")
print("=" * 60)
print(results_df.head(20).to_string(index=False))

# Save predictions to file
results_df.to_csv("predictions.csv", index=False)
print("\n✓ Predictions saved to 'predictions.csv'")
