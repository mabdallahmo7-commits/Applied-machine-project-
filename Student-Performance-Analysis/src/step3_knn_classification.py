"""
Step 3: KNN Classification - Student Performance Prediction
============================================================
Uses K-Nearest Neighbors to predict student pass/fail status
and grade categories based on behavioral and academic features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


def load_processed():
    path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'student_performance_processed.csv'
    df = pd.read_csv(path)
    raw_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'student_performance.csv'
    df_raw = pd.read_csv(raw_path)

    feature_cols = [
        'Study_Hours_Per_Week', 'Attendance_Percent', 'Quiz_Avg_Score',
        'Sleep_Hours_Per_Night', 'Social_Media_Hours_Per_Day',
        'Previous_GPA', 'Extracurricular_Activities',
        'Parent_Education_Level_Encoded',
        'Study_Efficiency', 'Sleep_Study_Ratio',
        'Social_Media_Risk', 'Attendance_Study_Product',
        'Current_GPA', 'Final_Score'
    ]

    scaled_cols = [f'{col}_scaled' for col in feature_cols]
    X = df[scaled_cols].values
    y_pass = df['Pass_Fail_Encoded'].values
    y_grade = df['Grade_Encoded'].values

    return X, y_pass, y_grade, df_raw, df, feature_cols


def find_best_k(X_train, y_train, X_test, y_test, max_k=30):
    results = []
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'k': k, 'accuracy': acc})

    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df['accuracy'].idxmax(), 'k']
    return int(best_k), results_df


def train_knn(X_train, y_train, X_test, y_test, k, task_name="Pass/Fail"):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy')

    report = {
        'task': task_name,
        'k': k,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'model': knn,
        'y_pred': y_pred,
    }

    if len(np.unique(y_train)) == 2:
        try:
            y_prob = knn.predict_proba(X_test)[:, 1]
            report['roc_auc'] = roc_auc_score(y_test, y_prob)
            report['y_prob'] = y_prob
        except Exception:
            report['roc_auc'] = None
            report['y_prob'] = None

    return report


def print_report(report):
    print(f"\n{'='*60}")
    print(f"Task: {report['task']}")
    print(f"{'='*60}")
    print(f"Best K: {report['k']}")
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1-Score:  {report['f1_score']:.4f}")
    print(f"CV Mean:   {report['cv_mean']:.4f} (+/- {report['cv_std']:.4f})")
    if report.get('roc_auc') is not None:
        print(f"ROC-AUC:   {report['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{report['confusion_matrix']}")
    print(f"\nClassification Report:\n{report['classification_report']}")


if __name__ == "__main__":
    X, y_pass, y_grade, df_raw, df_processed, features = load_processed()

    print("="*60)
    print("KNN CLASSIFICATION - STUDENT PERFORMANCE ANALYSIS")
    print("="*60)

    results = {}

    # --- Pass/Fail Classification ---
    print("\n\n--- PASS/FAIL CLASSIFICATION ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_pass, test_size=0.25, random_state=42, stratify=y_pass
    )
    best_k, k_results = find_best_k(X_train, y_train, X_test, y_test)
    report_pass = train_knn(X_train, y_train, X_test, y_test, best_k, "Pass/Fail Prediction")
    print_report(report_pass)
    results['pass_fail'] = report_pass

    # --- Grade Classification ---
    print("\n\n--- GRADE CLASSIFICATION (A/B/C/D/F) ---")
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X, y_grade, test_size=0.25, random_state=42, stratify=y_grade
    )
    best_k_g, k_results_g = find_best_k(X_train_g, y_train_g, X_test_g, y_test_g, max_k=30)
    report_grade = train_knn(X_train_g, y_train_g, X_test_g, y_test_g, best_k_g, "Grade Prediction (A/B/C/D/F)")
    print_report(report_grade)
    results['grade'] = report_grade

    print(f"\nBest K for Pass/Fail: {best_k}")
    print(f"Best K for Grade: {best_k_g}")

    output_dir = Path(__file__).resolve().parent.parent / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        f"KNN Classification Results\n"
        f"{'='*50}\n\n"
        f"Pass/Fail: Accuracy={report_pass['accuracy']:.4f}, "
        f"F1={report_pass['f1_score']:.4f}, "
        f"K={report_pass['k']}\n"
        f"Grade:     Accuracy={report_grade['accuracy']:.4f}, "
        f"F1={report_grade['f1_score']:.4f}, "
        f"K={report_grade['k']}\n"
    )
    (output_dir / 'knn_results.txt').write_text(summary)
    print(f"\nResults saved to reports/knn_results.txt")
