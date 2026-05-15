# Student Performance Analysis

A complete data analysis and machine learning project for analyzing student performance using Python. This project includes data preprocessing, visualization, classification using K-Nearest Neighbors (KNN), and clustering using DBSCAN.

---

## 📌 Project Overview

The goal of this project is to analyze student academic performance data and apply machine learning techniques to:

* Understand patterns in student performance
* Clean and preprocess educational datasets
* Classify student outcomes using KNN
* Discover hidden groups using DBSCAN clustering
* Visualize insights using charts and graphs

This project demonstrates a complete machine learning workflow from raw data to final analysis and reporting.

---

## 📂 Project Structure

```bash
Student-Performance-Analysis/
│
├── data/
│   ├── raw/
│   │   └── student_performance.csv
│   │
│   └── processed/
│       ├── student_performance_processed.csv
│       └── student_clusters.csv
│
├── reports/
│   ├── charts/
│   │   ├── 01_distributions.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_knn_accuracy_vs_k.png
│   │   ├── 04_feature_correlations.png
│   │   ├── 05_dbscan_clusters.png
│   │   ├── 06_outlier_analysis.png
│   │   ├── 07_knn_roc_curve.png
│   │   └── 09_knn_confusion_matrix.png
│   │
│   ├── knn_results.txt
│   └── dbscan_results.txt
│
├── src/
│   ├── step1_generate_data.py
│   ├── step2_preprocessing.py
│   ├── step3_knn_classification.py
│   ├── step4_dbscan_clustering.py
│   └── step5_visualization.py
│
├── run_all_steps.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* DBSCAN Clustering
* K-Nearest Neighbors (KNN)

---

## 🧠 Machine Learning Techniques

### 1. K-Nearest Neighbors (KNN)

Used for classification tasks to predict student performance categories.

Features:

* Accuracy evaluation
* ROC curve generation
* Confusion matrix visualization
* K-value comparison

### 2. DBSCAN Clustering

Used for discovering hidden patterns and groups in the dataset.

Features:

* Noise detection
* Cluster visualization
* Outlier analysis

---

## 📊 Visualizations Included

The project generates multiple visual reports including:

* Data distribution charts
* Correlation heatmaps
* KNN accuracy comparison
* Feature correlation analysis
* DBSCAN cluster visualization
* Outlier detection plots
* ROC curve
* Confusion matrix

---

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run all project steps using:

```bash
python run_all_steps.py
```

Or execute each step separately:

```bash
python src/step1_generate_data.py
python src/step2_preprocessing.py
python src/step3_knn_classification.py
python src/step4_dbscan_clustering.py
python src/step5_visualization.py
```

---

## 📈 Output

After running the project:

* Processed datasets will be saved in `data/processed/`
* Charts and visual reports will be saved in `reports/charts/`
* Model evaluation results will be stored in:

  * `reports/knn_results.txt`
  * `reports/dbscan_results.txt`

---

## 🎯 Learning Outcomes

This project helps in understanding:

* Data preprocessing techniques
* Exploratory Data Analysis (EDA)
* Classification algorithms
* Clustering algorithms
* Data visualization
* Machine learning workflow

