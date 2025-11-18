
# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Visual style settings
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------
df = pd.read_csv("healthcare_dataset.csv")

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}\n")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFO:")
df.info()

# ---------------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------------
print("\nMISSING VALUES:")
print(df.isnull().sum())

# ---------------------------------------------------------------------
# Basic Statistics
# ---------------------------------------------------------------------
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# =====================================================================
# 1.1 DISTRIBUTION ANALYSIS
# Age, Billing Amount, Room Number
# =====================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# --- AGE DISTRIBUTION ---
axes[0, 0].hist(df["Age"], bins=30, color="skyblue", edgecolor="black")
axes[0, 0].set_title("Age Distribution", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Frequency")

axes[1, 0].boxplot(df["Age"], vert=False)
axes[1, 0].set_title("Age Boxplot", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Age")

# --- BILLING AMOUNT ---
axes[0, 1].hist(df["Billing Amount"].dropna(), bins=30, color="lightcoral", edgecolor="black")
axes[0, 1].set_title("Billing Amount Distribution", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Billing Amount ($)")

axes[1, 1].boxplot(df["Billing Amount"].dropna(), vert=False)
axes[1, 1].set_title("Billing Amount Boxplot", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Billing Amount ($)")

# --- ROOM NUMBER ---
axes[0, 2].hist(df["Room Number"], bins=30, color="lightgreen", edgecolor="black")
axes[0, 2].set_title("Room Number Distribution", fontsize=12, fontweight="bold")
axes[0, 2].set_xlabel("Room Number")

axes[1, 2].boxplot(df["Room Number"], vert=False)
axes[1, 2].set_title("Room Number Boxplot", fontsize=12, fontweight="bold")
axes[1, 2].set_xlabel("Room Number")

plt.tight_layout()
plt.show()

# Simple numerical insight summary
print("\nSUMMARY STATISTICS:")
print(f"Age: Mean={df['Age'].mean():.1f}, Median={df['Age'].median():.1f}, Std={df['Age'].std():.1f}")
print(f"Billing Amount: Mean=${df['Billing Amount'].mean():.2f}, Median=${df['Billing Amount'].median():.2f}")
print(f"Room Number: Mean={df['Room Number'].mean():.1f}, Median={df['Room Number'].median():.1f}")

# =====================================================================
# 1.2 FREQUENCY ANALYSIS
# Medical Condition, Admission Type, Medication
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- MEDICAL CONDITION ---
med_cond = df["Medical Condition"].value_counts()
axes[0].bar(med_cond.index, med_cond.values, color="steelblue", edgecolor="black")
axes[0].set_title("Medical Condition Frequency", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Medical Condition")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

# --- ADMISSION TYPE ---
adm_type = df["Admission Type"].value_counts()
axes[1].bar(adm_type.index, adm_type.values, color="coral", edgecolor="black")
axes[1].set_title("Admission Type Frequency", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Admission Type")
axes[1].set_ylabel("Count")

# --- MEDICATION ---
medication = df["Medication"].value_counts()
axes[2].bar(medication.index, medication.values, color="mediumseagreen", edgecolor="black")
axes[2].set_title("Medication Frequency", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Medication")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print("\nFREQUENCY COUNTS:")
print("\nMedical Condition:\n", med_cond)
print("\nAdmission Type:\n", adm_type)
print("\nMedication:\n", medication)

# =====================================================================
# Additional Insights
# Test Results & Gender Distribution
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- TEST RESULTS PIE ---
test_results = df["Test Results"].value_counts()
axes[0].pie(
    test_results.values,
    labels=test_results.index,
    autopct="%1.1f%%",
    startangle=90
)
axes[0].set_title("Test Results Distribution", fontsize=12, fontweight="bold")

# --- GENDER PIE ---
gender = df["Gender"].value_counts()
axes[1].pie(
    gender.values,
    labels=gender.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["lightblue", "pink"]
)
axes[1].set_title("Gender Distribution", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()
