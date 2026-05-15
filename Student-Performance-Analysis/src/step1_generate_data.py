"""
Step 1: Generate Synthetic Student Performance Dataset
=======================================================
Creates a realistic dataset of student records for performance analysis.
Features: Study Hours, Attendance, Quiz Scores, Sleep Hours,
          Social Media Usage, GPA, Pass/Fail Status
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

STUDENT_COUNT = 500

def generate_student_data(n_students=STUDENT_COUNT):
    data = {
        'Student_ID': [f'S{1000+i:03d}' for i in range(1, n_students+1)],
        'Study_Hours_Per_Week': np.random.normal(20, 10, n_students).clip(0, 60).round(1),
        'Attendance_Percent': np.random.normal(80, 15, n_students).clip(0, 100).round(1),
        'Quiz_Avg_Score': np.random.normal(65, 20, n_students).clip(0, 100).round(1),
        'Sleep_Hours_Per_Night': np.random.normal(7, 1.5, n_students).clip(3, 12).round(1),
        'Social_Media_Hours_Per_Day': np.random.exponential(3, n_students).clip(0, 15).round(1),
        'Previous_GPA': np.random.normal(2.8, 0.8, n_students).clip(0, 4.0).round(2),
    }

    df = pd.DataFrame(data)

    df['Extracurricular_Activities'] = np.random.choice([0, 1], n_students, p=[0.4, 0.6])
    df['Parent_Education_Level'] = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n_students, p=[0.3, 0.4, 0.2, 0.1]
    )

    df['Current_GPA'] = (
        0.30 * (df['Study_Hours_Per_Week'] / 60) * 4.0 +
        0.25 * (df['Attendance_Percent'] / 100) * 4.0 +
        0.20 * (df['Quiz_Avg_Score'] / 100) * 4.0 +
        0.10 * (df['Previous_GPA']) +
        0.10 * (df['Sleep_Hours_Per_Night'] / 12) * 4.0 -
        0.05 * (df['Social_Media_Hours_Per_Day'] / 15) * 4.0 +
        np.random.normal(0, 0.3, n_students)
    ).clip(0, 4.0).round(2)

    df['Final_Score'] = (
        0.35 * (df['Study_Hours_Per_Week'] / 60) * 100 +
        0.20 * (df['Attendance_Percent'] / 100) * 100 +
        0.25 * (df['Quiz_Avg_Score']) +
        0.10 * (df['Sleep_Hours_Per_Night'] / 12) * 100 -
        0.10 * (df['Social_Media_Hours_Per_Day'] / 15) * 100 +
        np.random.normal(0, 8, n_students)
    ).clip(0, 100).round(1)

    df['Pass_Fail'] = np.where(df['Final_Score'] >= 50, 'Pass', 'Fail')
    df['Grade'] = pd.cut(
        df['Final_Score'],
        bins=[0, 50, 65, 75, 85, 100],
        labels=['F', 'D', 'C', 'B', 'A']
    )
    df['Risk_Level'] = pd.cut(
        df['Final_Score'],
        bins=[0, 40, 60, 100],
        labels=['High Risk', 'Medium Risk', 'Low Risk']
    )

    return df


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_student_data()
    output_path = output_dir / 'student_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPass/Fail Distribution:\n{df['Pass_Fail'].value_counts()}")
    print(f"\nGrade Distribution:\n{df['Grade'].value_counts().sort_index()}")
    print(f"\nSample Records:\n{df.head()}")
