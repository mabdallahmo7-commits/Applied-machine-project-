"""
Step 2: Data Preprocessing & Feature Engineering
=================================================
Cleans raw data, handles missing values, encodes categorical variables,
and engineers features for ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data():
    raw_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'student_performance.csv'
    return pd.read_csv(raw_path)


def preprocess(df):
    df = df.copy()

    print(f"Initial shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates()

    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            print(f"{col}: {outliers} outliers capped")
            df[col] = df[col].clip(lower, upper)

    label_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    df['Parent_Education_Level_Encoded'] = df['Parent_Education_Level'].map(label_map)

    le_pass = LabelEncoder()
    df['Pass_Fail_Encoded'] = le_pass.fit_transform(df['Pass_Fail'])

    le_grade = LabelEncoder()
    df['Grade_Encoded'] = le_grade.fit_transform(df['Grade'])

    le_risk = LabelEncoder()
    df['Risk_Level_Encoded'] = le_risk.fit_transform(df['Risk_Level'])

    df['Study_Efficiency'] = (df['Quiz_Avg_Score'] / (df['Study_Hours_Per_Week'] + 1)).round(2)
    df['Sleep_Study_Ratio'] = (df['Sleep_Hours_Per_Night'] / (df['Study_Hours_Per_Week'] / 7 + 0.1)).round(2)
    df['Social_Media_Risk'] = (df['Social_Media_Hours_Per_Day'] * (100 - df['Attendance_Percent']) / 100).round(2)
    df['Attendance_Study_Product'] = (df['Attendance_Percent'] * df['Study_Hours_Per_Week'] / 100).round(2)

    feature_cols = [
        'Study_Hours_Per_Week', 'Attendance_Percent', 'Quiz_Avg_Score',
        'Sleep_Hours_Per_Night', 'Social_Media_Hours_Per_Day',
        'Previous_GPA', 'Extracurricular_Activities',
        'Parent_Education_Level_Encoded',
        'Study_Efficiency', 'Sleep_Study_Ratio',
        'Social_Media_Risk', 'Attendance_Study_Product',
        'Current_GPA', 'Final_Score'
    ]

    X = df[feature_cols].copy()
    y_pass = df['Pass_Fail_Encoded']
    y_grade = df['Grade_Encoded']
    y_risk = df['Risk_Level_Encoded']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    df_processed = df.copy()
    for i, col in enumerate(feature_cols):
        df_processed[f'{col}_scaled'] = X_scaled.iloc[:, i]

    df_processed['Feature_Set'] = [list(row) for row in X_scaled.values]

    print(f"\nProcessed shape: {df_processed.shape}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    return df_processed, X_scaled, y_pass, y_grade, y_risk, scaler, feature_cols


if __name__ == "__main__":
    df = load_data()
    df_processed, X_scaled, y_pass, y_grade, y_risk, scaler, features = preprocess(df)

    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'student_performance_processed.csv'
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed data saved: {output_path}")
