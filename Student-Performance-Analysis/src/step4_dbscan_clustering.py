"""
Step 4: DBSCAN Clustering - Student Group Discovery
====================================================
Uses DBSCAN (with PCA dimensionality reduction) to discover natural
student groups and identify outliers (exceptional or struggling).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


def load_data():
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

    return X, df, df_raw, feature_cols


def reduce_with_pca(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    return X_pca, pca


def find_eps_knee(X, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])
    return k_distances


def find_best_dbscan(X, eps_range=None, min_samples_range=None):
    if eps_range is None:
        eps_range = np.arange(0.1, 2.0, 0.05)
    if min_samples_range is None:
        min_samples_range = range(3, 15)

    best_score = -1
    best_params = {'eps': 0.5, 'min_samples': 5}
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_pct = n_noise / len(labels)

            if n_clusters >= 2 and noise_pct < 0.5:
                mask = labels != -1
                n_clustered = mask.sum()
                if n_clustered > n_clusters:
                    try:
                        sil = silhouette_score(X[mask], labels[mask])
                        combined = sil * (1 - noise_pct)
                        results.append({
                            'eps': round(eps, 2),
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'noise_pct': round(noise_pct * 100, 1),
                            'silhouette': round(sil, 4),
                            'combined': round(combined, 4)
                        })
                        if combined > best_score:
                            best_score = combined
                            best_params = {'eps': eps, 'min_samples': min_samples}
                    except:
                        pass

    return best_params, best_score, pd.DataFrame(results)


def analyze_clusters(df, labels):
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\n{'='*60}")
    print("DBSCAN CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points (outliers): {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    cluster_profiles = []
    perf_cols = ['Study_Hours_Per_Week', 'Attendance_Percent', 'Quiz_Avg_Score',
                 'Sleep_Hours_Per_Night', 'Social_Media_Hours_Per_Day',
                 'Current_GPA', 'Final_Score']

    for cluster_id in sorted(set(labels)):
        mask = df_clustered['Cluster'] == cluster_id
        count = mask.sum()
        label_name = f"Cluster {cluster_id}" if cluster_id >= 0 else "Outliers"

        profile = {'Cluster': label_name, 'Count': count,
                   'Percentage': f"{count/len(labels)*100:.1f}%"}
        for col in perf_cols:
            profile[f'{col}_mean'] = round(df_clustered.loc[mask, col].mean(), 2)

        profile['Pass_Percent'] = round(
            (df_clustered.loc[mask, 'Pass_Fail'] == 'Pass').mean() * 100, 1
        )
        profile['Top_Grade'] = df_clustered.loc[mask, 'Grade'].mode().iloc[0] if count > 0 else '-'

        cluster_profiles.append(profile)

        print(f"\n--- {label_name} (n={count}) ---")
        for col in perf_cols:
            print(f"  {col}: {profile[f'{col}_mean']}")
        print(f"  Pass Rate: {profile['Pass_Percent']}%")
        print(f"  Most Common Grade: {profile['Top_Grade']}")

    return df_clustered, pd.DataFrame(cluster_profiles)


def identify_outliers(df_clustered, labels):
    outlier_mask = labels == -1
    outliers = df_clustered[outlier_mask].copy()

    print(f"\n{'='*60}")
    print("OUTLIER ANALYSIS")
    print(f"{'='*60}")
    if len(outliers) == 0:
        print("No outliers detected by DBSCAN.")
        return outliers

    high_performers = outliers[outliers['Final_Score'] >= 75]
    low_performers = outliers[outliers['Final_Score'] < 40]

    print(f"\nExceptional Students (Outliers with Score >= 75): {len(high_performers)}")
    if len(high_performers) > 0:
        print(f"  Avg Study Hours: {high_performers['Study_Hours_Per_Week'].mean():.1f}")
        print(f"  Avg Attendance: {high_performers['Attendance_Percent'].mean():.1f}%")
        print(f"  Avg GPA: {high_performers['Current_GPA'].mean():.2f}")

    print(f"\nAt-Risk Students (Outliers with Score < 40): {len(low_performers)}")
    if len(low_performers) > 0:
        print(f"  Avg Social Media: {low_performers['Social_Media_Hours_Per_Day'].mean():.1f}h/day")
        print(f"  Avg Sleep: {low_performers['Sleep_Hours_Per_Night'].mean():.1f}h")
        print(f"  Avg GPA: {low_performers['Current_GPA'].mean():.2f}")

    print(f"\nOther Outliers: {len(outliers) - len(high_performers) - len(low_performers)}")

    return outliers


if __name__ == "__main__":
    X, df, df_raw, features = load_data()

    X_pca, pca_model = reduce_with_pca(X, n_components=3)

    best_params, best_score, search_results = find_best_dbscan(X_pca)
    print(f"Best DBSCAN params on PCA data: eps={best_params['eps']}, "
          f"min_samples={best_params['min_samples']}, combined_score={best_score:.4f}")

    if not search_results.empty:
        top_configs = search_results.sort_values('combined', ascending=False).head(5)
        print(f"\nTop 5 DBSCAN configurations (by combined score):")
        print(top_configs.to_string(index=False))

    dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    labels = dbscan.fit_predict(X_pca)

    df_clustered, cluster_profiles = analyze_clusters(df, labels)
    outliers = identify_outliers(df_clustered, labels)

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X)
    df_clustered['PCA1'] = X_pca_2d[:, 0]
    df_clustered['PCA2'] = X_pca_2d[:, 1]

    output_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / 'student_clusters.csv'
    df_clustered.to_csv(cluster_path, index=False)
    print(f"\nCluster data saved: {cluster_path}")

    report_dir = Path(__file__).resolve().parent.parent / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"DBSCAN Clustering Results",
        f"{'='*50}",
        f"PCA components: 3 (variance ratio: {pca_model.explained_variance_ratio_.sum():.3f})",
        f"Best eps: {best_params['eps']}",
        f"Best min_samples: {best_params['min_samples']}",
        f"Silhouette Score: {best_score:.4f}",
        f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}",
        f"Noise points: {list(labels).count(-1)} ({list(labels).count(-1)/len(labels)*100:.1f}%)",
        f"\nOutliers Analysis:",
        f"  Exceptional (score>=75): {len(outliers[outliers['Final_Score']>=75]) if len(outliers)>0 else 0}",
        f"  At-Risk (score<40): {len(outliers[outliers['Final_Score']<40]) if len(outliers)>0 else 0}",
    ]
    (report_dir / 'dbscan_results.txt').write_text('\n'.join(report_lines))
    print(f"Results saved to reports/dbscan_results.txt")
