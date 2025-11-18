"""
MAIN SCRIPT - Healthcare Patient Risk Analysis
Runs all tasks sequentially:
  1. Supervised Learning (Test Result Prediction)
  2. Unsupervised Learning (Anomaly Detection)
  3. AI Doctor Recommendation Generator

Note:
  - EDA is available separately in healthcare_analysis.ipynb
"""

import subprocess
import sys
import os


def print_header(title: str):
    """Display formatted task headers."""
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80 + "\n")


def run_task(script_name: str, task_name: str) -> bool:
    """Execute a Python script and handle errors."""
    print_header(task_name)

    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✓ {task_name} completed successfully!\n")
        return True

    except subprocess.CalledProcessError as error:
        print(f"\n✗ ERROR: {task_name} failed!")
        print(f"Reason: {error}\n")
        return False

    except FileNotFoundError:
        print(f"\n✗ ERROR: Script '{script_name}' not found!")
        return False


def main():
    print_header("HEALTHCARE PATIENT RISK ANALYSIS - AI/ML INTERN ASSIGNMENT")

    # Check dataset availability
    dataset_file = "healthcare_dataset.csv"
    if not os.path.exists(dataset_file):
        print("✗ ERROR: Required dataset 'healthcare_dataset.csv' is missing!")
        print("Please place the dataset in the current directory and try again.\n")
        return
    else:
        print(f"✓ Dataset found: {dataset_file}\n")

    print("Starting task pipeline...\n")

    # List of tasks to run
    tasks = [
        ("task2_supervised.py", "TASK 2: SUPERVISED LEARNING"),
        ("task3_anomaly_detection.py", "TASK 3: ANOMALY DETECTION"),
        ("task4_ai_recommendation.py", "TASK 4: AI DOCTOR RECOMMENDATION"),
    ]

    completed_count = 0

    # Execute tasks one-by-one
    for script, task_name in tasks:
        success = run_task(script, task_name)
        if success:
            completed_count += 1
        else:
            print("\nExecution halted due to error.\n")
            break

    # Summary Section
    print("\n" + "=" * 80)
    print(f"EXECUTION SUMMARY: {completed_count}/{len(tasks)} tasks completed")
    print("=" * 80 + "\n")

    if completed_count == len(tasks):
        print("✓ All tasks executed successfully!")
        print("\nGenerated Output Files:")
        print("  • predictions.csv               (Task 2)")
        print("  • detected_anomalies.csv        (Task 3)")
        print("  • ai_doctor_recommendation.txt  (Task 4)")
        print("  • confusion_matrix.png          (Task 2)")
        print("  • feature_importance.png        (Task 2)")
        print("  • anomaly_detection.png         (Task 3)")
        print("\nFor EDA, open the Jupyter Notebook:")
        print("  → healthcare_analysis.ipynb\n")
    else:
        print("⚠ Some tasks failed. Please review the errors above.\n")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
