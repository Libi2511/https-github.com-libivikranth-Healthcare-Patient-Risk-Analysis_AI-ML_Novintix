# Task 4: AI Doctor Recommendation Generator

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def generate_doctor_recommendation(age, medical_condition, medication, test_result):
    """
    Generate AI-powered doctor-style health recommendation based on patient inputs.
    """

    # Report Header
    recommendation = f"""
================================================================================
                    AI DOCTOR RECOMMENDATION REPORT
================================================================================

PATIENT PROFILE:
  - Age: {age} years
  - Medical Condition: {medical_condition}
  - Current Medication: {medication}
  - Test Result: {test_result}

"""

    # Condition-specific medical advice
    condition_advice = {
        'Diabetes': {
            'Normal': (
                "Continue diabetes management plan. Monitor blood glucose regularly. "
                "Maintain a balanced, low-sugar diet."
            ),
            'Abnormal': (
                "URGENT: Blood glucose abnormal. Medication adjustment required. "
                "Meet endocrinologist within 48 hours."
            ),
            'Inconclusive': (
                "Repeat HbA1c and fasting glucose tests. Track sugar levels 3 times daily."
            ),
        },
        'Hypertension': {
            'Normal': (
                "Blood pressure is stable. Continue medication and reduce sodium intake."
            ),
            'Abnormal': (
                "ALERT: Elevated BP detected. Increase monitoring. Medication review needed."
            ),
            'Inconclusive': (
                "Recheck BP for 24 hours. Assess medication adherence and lifestyle factors."
            ),
        },
        'Cancer': {
            'Normal': (
                "Condition stable. Continue ongoing treatment and follow-up with oncology."
            ),
            'Abnormal': (
                "CRITICAL: Possible disease progression. Immediate oncology consultation required."
            ),
            'Inconclusive': (
                "Further diagnostic imaging and tumor marker evaluation recommended."
            ),
        },
        'Asthma': {
            'Normal': (
                "Respiratory function stable. Continue preventer inhaler. Avoid triggers."
            ),
            'Abnormal': (
                "WARNING: Asthma flare-up. Increase controller dose. Keep rescue inhaler ready."
            ),
            'Inconclusive': (
                "Spirometry test needed. Check inhaler usage technique."
            ),
        },
        'Arthritis': {
            'Normal': (
                "Joint health stable. Continue anti-inflammatory medication. "
                "Light exercise recommended."
            ),
            'Abnormal': (
                "Inflammation increased. Pain management review necessary."
            ),
            'Inconclusive': (
                "Repeat CRP/ESR tests. Consider imaging to assess joint damage."
            ),
        },
        'Obesity': {
            'Normal': (
                "Metabolic levels normal. Maintain weight-loss routine and balanced diet."
            ),
            'Abnormal': (
                "CONCERN: Complications detected. Begin intensive weight control program."
            ),
            'Inconclusive': (
                "Further blood tests required (thyroid, metabolic profile)."
            ),
        },
    }

    # Age-based recommendations
    if age < 30:
        age_advice = "Young adult: Focus on preventive health and active lifestyle."
    elif age < 50:
        age_advice = "Middle-aged: Regular screenings recommended to prevent chronic diseases."
    elif age < 70:
        age_advice = "Senior: Regular monitoring for age-related risks is essential."
    else:
        age_advice = "Elderly: Geriatric assessment advised. Extra safety precautions required."

    # Extract condition advice
    specific_advice = condition_advice.get(medical_condition, {}).get(
        test_result,
        "Continue regular treatment and monitor health periodically."
    )

    # Medication-specific information
    medication_notes = {
        'Aspirin': "Take with food to avoid stomach irritation. Watch for bleeding symptoms.",
        'Ibuprofen': "Avoid long-term use. Can affect kidneys.",
        'Paracetamol': "Do not exceed 4g/day. Safer for most patients.",
        'Penicillin': "Take full course. Watch for allergies.",
        'Lipitor': "Take at night for best effect. Monitor liver function.",
    }

    med_note = medication_notes.get(
        medication,
        "Use medication as prescribed. Follow dosage instructions strictly."
    )

    # Add all sections to report
    recommendation += f"""CLINICAL ASSESSMENT:
{specific_advice}

AGE-RELATED CONSIDERATIONS:
{age_advice}

MEDICATION GUIDANCE:
{med_note}

GENERAL HEALTH ADVICE:
  - Maintain 7–8 hours of sleep daily
  - Drink adequate water (6–8 glasses/day)
  - Eat fruits, vegetables, and high-fiber foods
  - Exercise at least 30 minutes daily
  - Avoid smoking; limit alcohol consumption
  - Manage stress and mental well-being

FOLLOW-UP PLAN:
"""

    # Follow-up recommendations
    if test_result == 'Abnormal':
        recommendation += (
            "  - Immediate follow-up within 3–5 days\n"
            "  - Repeat clinical tests as advised\n"
            "  - Daily symptom monitoring required\n"
        )
    elif test_result == 'Inconclusive':
        recommendation += (
            "  - Follow-up within 1–2 weeks\n"
            "  - Additional diagnostic tests needed\n"
            "  - Keep a symptom record\n"
        )
    else:
        recommendation += (
            "  - Routine follow-up in 3–6 months\n"
            "  - Continue current treatment plan\n"
        )

    # Emergency instructions
    recommendation += """
WARNING - SEEK IMMEDIATE MEDICAL HELP IF:
  - Severe chest pain or difficulty breathing
  - Sudden confusion, fainting, or severe headache
  - Uncontrollable bleeding
  - Signs of severe allergic reaction

================================================================================
Note: This is an AI-generated recommendation for informational purposes only.
Always consult a licensed healthcare professional for medical decisions.
================================================================================
"""
    return recommendation


# Main Execution
if __name__ == "__main__":

    print("=" * 80)
    print("TASK 4: AI DOCTOR RECOMMENDATION GENERATOR")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv("healthcare_dataset.csv")
    df = df.dropna(subset=['Test Results', 'Billing Amount'])

    # Features used for prediction
    features = [
        'Age', 'Gender', 'Blood Type', 'Medical Condition',
        'Billing Amount', 'Room Number', 'Admission Type', 'Medication'
    ]

    X = df[features].copy()
    y = df["Test Results"]

    # Encoding categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include='object'):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Pick a sample patient
    idx = X_test.index[0]
    sample = df.loc[idx]
    sample_features = X_test.iloc[0:1]

    # Predict test result
    predicted = le_target.inverse_transform(model.predict(sample_features))[0]

    print("\n" + "=" * 80)
    print("SAMPLE AI DOCTOR RECOMMENDATION")
    print("=" * 80)

    # Generate recommendation
    output = generate_doctor_recommendation(
        age=int(sample['Age']),
        medical_condition=sample['Medical Condition'],
        medication=sample['Medication'],
        test_result=predicted
    )

    print(output)

    # Save to file
    with open("ai_doctor_recommendation.txt", "w") as f:
        f.write(output)

    print("\nRecommendation saved successfully.")

    # Generate for 3 patients
    print("\n" + "=" * 80)
    print("GENERATING RECOMMENDATIONS FOR 3 SAMPLE PATIENTS")
    print("=" * 80)

    for i in range(min(3, len(X_test))):
        idx = X_test.index[i]
        sample = df.loc[idx]
        sample_features = X_test.iloc[i:i+1]
        predicted = le_target.inverse_transform(model.predict(sample_features))[0]

        print(f"\n{'-'*80}")
        print(f"PATIENT {i+1}: {sample.get('Name', 'Unknown')}")
        print(f"Age: {sample['Age']} | Condition: {sample['Medical Condition']}")
        print(f"Predicted Test Result: {predicted}")
        print(f"{'-'*80}")
