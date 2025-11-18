"""
TASK 3 - Healthcare Billing Anomaly Detection
---------------------------------------------
Uses Isolation Forest to detect unusual billing patterns.
Outputs:
    - anomaly_detection.png
    - detected_anomalies.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    df = pd.read_csv(path)
    return df.dropna(subset=['Billing Amount'])


def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(df[['Billing Amount']])
    df['Anomaly_Label'] = df['Anomaly'].map({1: "Normal", -1: "Anomaly"})
    return df


def generate_plots(df, normal, anomalies):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(normal['Billing Amount'], bins=30, alpha=0.7, label="Normal", color="green")
    axes[0, 0].hist(anomalies['Billing Amount'], bins=30, alpha=0.7, label="Anomaly", color="red")
    axes[0, 0].set_title("Billing Amount Distribution", fontweight="bold")
    axes[0, 0].legend()

    # Scatter Plot
    axes[0, 1].scatter(normal.index, normal['Billing Amount'], c='green', s=20, alpha=0.5)
    axes[0, 1].scatter(anomalies.index, anomalies['Billing Amount'], c='red', s=50, alpha=0.8)
    axes[0, 1].set_title("Anomaly Detection Scatter Plot", fontweight="bold")

    # Box Plot
    axes[1, 0].boxplot(
        [normal['Billing Amount'], anomalies['Billing Amount']],
        labels=['Normal', 'Anomaly']
    )
    axes[1, 0].set_title("Box Plot Comparison", fontweight="bold")

    # Medical Condition Breakdown
    if 'Medical Condition' in df.columns:
        condition_counts = anomalies['Medical Condition'].value_counts()
        axes[1, 1].bar(condition_counts.index, condition_counts.values, color='crimson')
        axes[1, 1].set_title("Anomalies by Medical Condition")
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("anomaly_detection.png", dpi=300)
    plt.show()


def print_summary(df, normal, anomalies):
    print("=" * 60)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Normal Cases: {len(normal)}")
    print(f"Anomalies: {len(anomalies)}")

    print("\nBilling Statistics")
    print("-" * 60)
    print(f"Mean:   ${df['Billing Amount'].mean():.2f}")
    print(f"Median: ${df['Billing Amount'].median():.2f}")

    print("\nTop 10 Anomalous Cases")
    print("-" * 60)
    print(
        anomalies[['Name', 'Age', 'Medical Condition', 'Billing Amount']]
        .sort_values(by='Billing Amount', ascending=False)
        .head(10)
        .to_string(index=False)
    )


def save_results(anomalies):
    anomalies[['Name', 'Age', 'Gender', 'Medical Condition',
               'Billing Amount', 'Admission Type', 'Anomaly_Label']
              ].to_csv("detected_anomalies.csv", index=False)
    print("\nResults saved to detected_anomalies.csv")


if __name__ == "__main__":

    df = load_data("healthcare_dataset.csv")
    df = detect_anomalies(df)

    anomalies = df[df['Anomaly'] == -1]
    normal = df[df['Anomaly'] == 1]

    print_summary(df, normal, anomalies)
    generate_plots(df, normal, anomalies)
    save_results(anomalies)
