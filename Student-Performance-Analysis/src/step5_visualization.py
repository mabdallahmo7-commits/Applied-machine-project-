"""
Step 5: Visualization & Reporting
==================================
Generates comprehensive visualizations for both KNN and DBSCAN results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def setup_dirs():
    base = Path(__file__).resolve().parent.parent
    reports = base / 'reports' / 'charts'
    reports.mkdir(parents=True, exist_ok=True)
    return base, reports


def load_knn_results():
    reports_dir = Path(__file__).resolve().parent.parent / 'reports'
    path = reports_dir / 'knn_results.txt'
    if path.exists():
        return path.read_text()
    return ""


def plot_pass_fail_distribution(df_raw, reports_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_raw['Pass_Fail'].value_counts().plot(
        kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'],
        edgecolor='black', linewidth=1.2
    )
    axes[0].set_title('Pass/Fail Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Status')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    for i, v in enumerate(df_raw['Pass_Fail'].value_counts().values):
        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

    df_raw['Grade'].value_counts().sort_index().plot(
        kind='bar', ax=axes[1], color=sns.color_palette("viridis", 5),
        edgecolor='black', linewidth=1.2
    )
    axes[1].set_title('Grade Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Grade')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(df_raw['Grade'].value_counts().sort_index().values):
        axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(reports_dir / '01_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_distributions.png")


def plot_feature_correlations(df_raw, reports_dir):
    numeric_cols = ['Study_Hours_Per_Week', 'Attendance_Percent', 'Quiz_Avg_Score',
                    'Sleep_Hours_Per_Night', 'Social_Media_Hours_Per_Day',
                    'Previous_GPA', 'Current_GPA', 'Final_Score']
    corr = df_raw[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Feature Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(reports_dir / '02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_correlation_heatmap.png")


def plot_knn_accuracy_vs_k(X, y_pass, reports_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_pass, test_size=0.25, random_state=42, stratify=y_pass
    )

    k_range = range(1, 31)
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k_range, accuracies, marker='o', linestyle='-', color='#3498db',
            markersize=6, linewidth=2, markerfacecolor='#e74c3c')
    best_k = k_range[np.argmax(accuracies)]
    best_acc = max(accuracies)
    ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7,
               label=f'Best K={best_k} (Acc={best_acc:.3f})')
    ax.set_title('KNN: Accuracy vs K Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Neighbors (K)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / '03_knn_accuracy_vs_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_knn_accuracy_vs_k.png")


def plot_feature_importance_by_correlation(df_raw, reports_dir):
    numeric_cols = ['Study_Hours_Per_Week', 'Attendance_Percent', 'Quiz_Avg_Score',
                    'Sleep_Hours_Per_Night', 'Social_Media_Hours_Per_Day',
                    'Previous_GPA', 'Extracurricular_Activities']
    target = 'Final_Score'

    correlations = []
    valid_cols = [c for c in numeric_cols if c in df_raw.columns]
    for col in valid_cols:
        corr_val = df_raw[col].corr(df_raw[target])
        correlations.append({'Feature': col, 'Correlation': corr_val})

    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)

    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr_df['Correlation']]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, edgecolor='black', linewidth=1.2)
    ax.set_title(f'Feature Correlation with {target}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pearson Correlation')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    for bar, val in zip(bars, corr_df['Correlation']):
        ax.text(val + 0.02 if val > 0 else val - 0.06, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(reports_dir / '04_feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_feature_correlations.png")


def plot_confusion_matrix(model, X_test, y_test, title, filename, reports_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(reports_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_dbscan_clusters(df_clustered, reports_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cluster_colors = {0: '#3498db', 1: '#2ecc71', 2: '#e74c3c', 3: '#f39c12',
                      4: '#9b59b6', 5: '#1abc9c', 6: '#e67e22', 7: '#34495e'}
    palette = [cluster_colors.get(c, '#95a5a6') for c in df_clustered['Cluster']]

    scatter1 = axes[0].scatter(
        df_clustered['PCA1'], df_clustered['PCA2'],
        c=palette, s=50, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    axes[0].set_title('DBSCAN Clusters (PCA Projection)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    cluster_labels = sorted(df_clustered['Cluster'].unique())
    legend_elements = []
    for cl in cluster_labels:
        label = 'Outliers' if cl == -1 else f'Cluster {cl}'
        color = cluster_colors.get(cl, '#95a5a6')
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8, label=label))
    axes[0].legend(handles=legend_elements, loc='best')

    scatter2 = axes[1].scatter(
        df_clustered['PCA1'], df_clustered['PCA2'],
        c=df_clustered['Final_Score'], s=50, alpha=0.7,
        cmap='RdYlGn', edgecolors='black', linewidth=0.5
    )
    axes[1].set_title('Students by Final Score (PCA Projection)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    cbar = plt.colorbar(scatter2, ax=axes[1])
    cbar.set_label('Final Score', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(reports_dir / '05_dbscan_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_dbscan_clusters.png")


def plot_outlier_analysis(df_clustered, reports_dir):
    outliers = df_clustered[df_clustered['Cluster'] == -1]
    normal = df_clustered[df_clustered['Cluster'] != -1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('Study_Hours_Per_Week', 'Study Hours/Week'),
        ('Attendance_Percent', 'Attendance %'),
        ('Social_Media_Hours_Per_Day', 'Social Media Hours/Day'),
        ('Sleep_Hours_Per_Night', 'Sleep Hours/Night'),
        ('Current_GPA', 'Current GPA'),
        ('Final_Score', 'Final Score')
    ]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        ax.boxplot([normal[col].dropna(), outliers[col].dropna()],
                   labels=['Normal', 'Outliers'], widths=0.6)
        ax.set_ylabel(label)
        ax.set_title(f'{label}: Normal vs Outliers', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Outlier Analysis: Normal Students vs DBSCAN Outliers',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(reports_dir / '06_outlier_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_outlier_analysis.png")


def plot_knn_roc_curve(model, X_test, y_test, reports_dir):
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#3498db', lw=2,
                label=f'KNN (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - KNN Pass/Fail', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / '07_knn_roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: 07_knn_roc_curve.png")
    except Exception as e:
        print(f"ROC plot skipped: {e}")


def plot_cluster_profiles(cluster_profiles, reports_dir):
    if len(cluster_profiles) == 0:
        return
    perf_cols = ['Study_Hours_Per_Week_mean', 'Attendance_Percent_mean',
                 'Quiz_Avg_Score_mean', 'Current_GPA_mean', 'Final_Score_mean']
    existing_cols = [c for c in perf_cols if c in cluster_profiles.columns]
    if len(existing_cols) == 0:
        return

    plot_df = cluster_profiles.set_index('Cluster')[existing_cols]
    short_names = [c.replace('_mean', '').replace('_Percent', '')
                   .replace('_Per_Week', '').replace('_Per_Night', '')
                   .replace('_Hours', '') for c in existing_cols]
    plot_df.columns = short_names

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.T.plot(kind='bar', ax=ax, edgecolor='black', linewidth=1.2)
    ax.set_title('Cluster Profiles - Mean Feature Values', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Mean Value')
    ax.legend(title='Group', loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / '08_cluster_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_cluster_profiles.png")


if __name__ == "__main__":
    print("Generating Visualizations...")
    base_dir, reports_dir = setup_dirs()

    raw_path = base_dir / 'data' / 'raw' / 'student_performance.csv'
    df_raw = pd.read_csv(raw_path)

    processed_path = base_dir / 'data' / 'processed' / 'student_performance_processed.csv'
    df_processed = pd.read_csv(processed_path)

    cluster_path = base_dir / 'data' / 'processed' / 'student_clusters.csv'
    df_clustered = pd.read_csv(cluster_path) if cluster_path.exists() else None

    feature_cols = [
        'Study_Hours_Per_Week_scaled', 'Attendance_Percent_scaled',
        'Quiz_Avg_Score_scaled', 'Sleep_Hours_Per_Night_scaled',
        'Social_Media_Hours_Per_Day_scaled', 'Previous_GPA_scaled',
        'Extracurricular_Activities_scaled', 'Parent_Education_Level_Encoded_scaled',
        'Study_Efficiency_scaled', 'Sleep_Study_Ratio_scaled',
        'Social_Media_Risk_scaled', 'Attendance_Study_Product_scaled',
        'Current_GPA_scaled', 'Final_Score_scaled'
    ]
    available_features = [c for c in feature_cols if c in df_processed.columns]
    X = df_processed[available_features].values if available_features else df_processed.filter(regex='_scaled$').values

    y_pass_path = base_dir / 'data' / 'processed' / 'student_performance_processed.csv'
    y_pass = pd.read_csv(y_pass_path)['Pass_Fail_Encoded'].values if 'Pass_Fail_Encoded' in pd.read_csv(y_pass_path).columns else None

    plot_pass_fail_distribution(df_raw, reports_dir)
    plot_feature_correlations(df_raw, reports_dir)

    if X.size > 0 and y_pass is not None:
        plot_knn_accuracy_vs_k(X, y_pass, reports_dir)

    plot_feature_importance_by_correlation(df_raw, reports_dir)

    if df_clustered is not None:
        plot_dbscan_clusters(df_clustered, reports_dir)
        plot_outlier_analysis(df_clustered, reports_dir)

        cluster_profiles_path = base_dir / 'data' / 'processed' / 'student_clusters.csv'
        cluster_profiles_df = pd.read_csv(cluster_profiles_path)
        profiles = cluster_profiles_df.groupby('Cluster').agg({
            'Study_Hours_Per_Week': 'mean', 'Attendance_Percent': 'mean',
            'Quiz_Avg_Score': 'mean', 'Current_GPA': 'mean', 'Final_Score': 'mean'
        }).reset_index()
        profiles['Cluster'] = profiles['Cluster'].apply(lambda x: 'Outliers' if x == -1 else f'Cluster {x}')
        plot_cluster_profiles(profiles, reports_dir)

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    if X.size > 0 and y_pass is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_pass, test_size=0.25, random_state=42, stratify=y_pass
        )
        best_k_path = base_dir / 'reports' / 'knn_results.txt'
        best_k = 5
        if best_k_path.exists():
            content = best_k_path.read_text()
            for line in content.split('\n'):
                if 'Pass/Fail' in line and 'K=' in line:
                    try:
                        best_k = int(line.split('K=')[-1].strip())
                    except:
                        pass

        knn_model = KNeighborsClassifier(n_neighbors=best_k)
        knn_model.fit(X_train, y_train)
        plot_confusion_matrix(knn_model, X_test, y_test,
                            'KNN Pass/Fail', '09_knn_confusion_matrix.png', reports_dir)
        plot_knn_roc_curve(knn_model, X_test, y_test, reports_dir)

    print(f"\nAll charts saved to: {reports_dir}")
