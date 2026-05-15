"""
Smart Student Performance Analysis
===================================
Master script to run the complete pipeline:
1. Generate synthetic student dataset
2. Preprocess & feature engineering
3. KNN Classification (predict pass/fail & grade)
4. DBSCAN Clustering (discover groups & outliers)
5. Visualizations
"""

import sys
import time
from pathlib import Path


def run_step(step_num, step_name, script_path):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*60}")

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    start = time.time()
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start
        print(f"\n[OK] Step {step_num} completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Step {step_num} failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        return False


def main():
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / 'src'

    steps = [
        (1, "Generate Dataset", src_dir / 'step1_generate_data.py'),
        (2, "Preprocessing", src_dir / 'step2_preprocessing.py'),
        (3, "KNN Classification", src_dir / 'step3_knn_classification.py'),
        (4, "DBSCAN Clustering", src_dir / 'step4_dbscan_clustering.py'),
        (5, "Visualization", src_dir / 'step5_visualization.py'),
    ]

    print("\n")
    print("=" * 60)
    print("  SMART STUDENT PERFORMANCE ANALYSIS")
    print("  KNN Classification + DBSCAN Clustering")
    print("=" * 60)

    overall_start = time.time()
    all_success = True

    for step_num, step_name, script_path in steps:
        success = run_step(step_num, step_name, script_path)
        if not success:
            all_success = False
            print(f"\n[WARN] Pipeline stopped at Step {step_num}")
            break

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    if all_success:
        print(f"  PIPELINE COMPLETED SUCCESSFULLY in {overall_elapsed:.2f}s")
        print(f"\n  Outputs:")
        print(f"  - Data:     data/raw/student_performance.csv")
        print(f"  - Data:     data/processed/student_performance_processed.csv")
        print(f"  - Data:     data/processed/student_clusters.csv")
        print(f"  - Reports:  reports/knn_results.txt")
        print(f"  - Reports:  reports/dbscan_results.txt")
        print(f"  - Charts:   reports/charts/*.png")
    else:
        print(f"  PIPELINE FAILED after {overall_elapsed:.2f}s")
        print(f"  Check errors above and fix before re-running.")
    print(f"{'='*60}\n")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
