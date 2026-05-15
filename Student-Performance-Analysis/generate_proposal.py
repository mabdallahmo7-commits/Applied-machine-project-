"""
Generate Professional PDF Proposal for Smart Student Performance Analysis
==========================================================================
Creates a comprehensive project proposal document in PDF format.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os

DARK_BLUE = HexColor("#1a365d")
MEDIUM_BLUE = HexColor("#2c5282")
LIGHT_BLUE = HexColor("#ebf4ff")
ACCENT_BLUE = HexColor("#3182ce")
DARK_GRAY = HexColor("#2d3748")
MEDIUM_GRAY = HexColor("#4a5568")
LIGHT_GRAY = HexColor("#e2e8f0")
GREEN = HexColor("#2ecc71")
RED = HexColor("#e74c3c")
WHITE = white


def create_pdf(output_path="Student_Performance_Analysis_Proposal.pdf"):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='CoverTitle', fontName='Helvetica-Bold',
        fontSize=26, leading=32, textColor=DARK_BLUE, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle(name='CoverSubtitle', fontName='Helvetica',
        fontSize=14, leading=20, textColor=MEDIUM_BLUE, alignment=TA_CENTER, spaceAfter=40))
    styles.add(ParagraphStyle(name='CoverInfo', fontName='Helvetica',
        fontSize=12, leading=18, textColor=MEDIUM_GRAY, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle(name='SectionHeader', fontName='Helvetica-Bold',
        fontSize=18, leading=24, textColor=DARK_BLUE, spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='SubSectionHeader', fontName='Helvetica-Bold',
        fontSize=14, leading=18, textColor=MEDIUM_BLUE, spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name='ProposalBody', fontName='Helvetica',
        fontSize=11, leading=16, textColor=DARK_GRAY, alignment=TA_JUSTIFY, spaceAfter=8))
    styles.add(ParagraphStyle(name='PBullet', fontName='Helvetica',
        fontSize=11, leading=16, textColor=DARK_GRAY, leftIndent=20, spaceAfter=4))
    styles.add(ParagraphStyle(name='Footer', fontName='Helvetica',
        fontSize=8, leading=12, textColor=MEDIUM_GRAY, alignment=TA_CENTER))

    story = []

    # ================= COVER =================
    story.append(Spacer(1, 1.5*inch))
    story.append(HRFlowable(width="60%", thickness=3, color=DARK_BLUE, spaceAfter=20))
    story.append(Paragraph("SMART STUDENT PERFORMANCE ANALYSIS", styles['CoverTitle']))
    story.append(Paragraph("Machine Learning Classification &amp; Clustering Project", styles['CoverSubtitle']))
    story.append(HRFlowable(width="60%", thickness=3, color=DARK_BLUE, spaceAfter=40))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Project Proposal Document", styles['CoverInfo']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['CoverInfo']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Version 1.0", styles['CoverInfo']))
    story.append(Spacer(1, 0.8*inch))

    cover_data = [
        ['Project Type', 'Machine Learning / Data Science'],
        ['Algorithms', 'KNN (Classification) + DBSCAN (Clustering)'],
        ['Objective', 'Predict student performance & discover student groups'],
        ['Dataset', 'Synthetic Student Records (500 students, 14 features)'],
        ['Tools', 'Python, Scikit-learn, Pandas, Matplotlib, Seaborn'],
    ]
    cover_table = Table(cover_data, colWidths=[2.0*inch, 4.0*inch])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_BLUE),
        ('BACKGROUND', (1, 0), (1, -1), WHITE),
        ('TEXTCOLOR', (0, 0), (0, -1), DARK_BLUE),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK_GRAY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # ================= 1. EXECUTIVE SUMMARY =================
    story.append(Paragraph("1. Executive Summary", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    story.append(Paragraph(
        "This project presents a comprehensive machine learning analysis of student academic performance, "
        "combining K-Nearest Neighbors (KNN) classification and DBSCAN clustering to understand and predict "
        "student outcomes. The analysis uses a synthetic dataset of 500 students with behavioral and academic "
        "features including study hours, attendance, quiz scores, sleep patterns, and social media usage.",
        styles['ProposalBody']
    ))
    story.append(Paragraph(
        "The project demonstrates two complementary algorithmic approaches: KNN for supervised classification "
        "(predicting pass/fail status and grade categories), and DBSCAN for unsupervised clustering (identifying "
        "natural student groups and detecting outliers). This dual-algorithm design provides a complete view of "
        "both individual prediction and population-level pattern discovery.",
        styles['ProposalBody']
    ))

    # ================= 2. OBJECTIVES =================
    story.append(Paragraph("2. Project Objectives", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    objectives = [
        "Build a KNN classifier to predict student pass/fail status with high accuracy",
        "Extend KNN classification to predict fine-grained grade categories (A/B/C/D/F)",
        "Apply DBSCAN clustering to discover natural student groupings based on behavioral patterns",
        "Identify exceptional students (high performers) and at-risk students (struggling) as outliers",
        "Engineer meaningful features from raw student data to improve model performance",
        "Generate comprehensive visualizations and documentation for stakeholder communication",
    ]
    for obj in objectives:
        story.append(Paragraph(f"<bullet>&bull;</bullet> {obj}", styles['PBullet']))

    # ================= 3. DATASET =================
    story.append(Paragraph("3. Dataset Description", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    story.append(Paragraph(
        "A synthetic dataset of 500 student records was generated with realistic distributions, "
        "including natural correlations between features and target variables. The dataset includes:",
        styles['ProposalBody']
    ))

    feat_data = [
        ['Feature', 'Type', 'Description'],
        ['Study Hours/Week', 'Numerical', 'Hours spent studying per week (0-60)'],
        ['Attendance %', 'Numerical', 'Percentage of classes attended (0-100%)'],
        ['Quiz Avg Score', 'Numerical', 'Average quiz score (0-100)'],
        ['Sleep Hours/Night', 'Numerical', 'Average sleep duration (3-12 hrs)'],
        ['Social Media Hrs/Day', 'Numerical', 'Daily social media usage (0-15 hrs)'],
        ['Previous GPA', 'Numerical', 'Cumulative GPA from previous semester (0-4)'],
        ['Extracurricular', 'Binary', 'Participation in activities (0/1)'],
        ['Parent Education', 'Categorical', 'Highest education level (4 levels)'],
        ['Current GPA', 'Numerical', 'Current semester GPA (derived)'],
        ['Final Score', 'Numerical', 'Final course score (derived, 0-100)'],
        ['Pass/Fail', 'Binary', 'Target: Pass (score >= 50) or Fail'],
        ['Grade', 'Ordinal', 'Target: A/B/C/D/F grade categories'],
    ]
    ft = Table(feat_data, colWidths=[1.5*inch, 1.2*inch, 3.3*inch])
    ft.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(ft)

    # ================= 4. METHODOLOGY =================
    story.append(Paragraph("4. Methodology & Approach", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))

    story.append(Paragraph("4.1 Data Generation & Preprocessing", styles['SubSectionHeader']))
    story.append(Paragraph(
        "Synthetic data is generated with realistic distributions and correlations. The preprocessing pipeline "
        "includes outlier capping using IQR, categorical encoding, and feature engineering (Study Efficiency, "
        "Sleep-Study Ratio, Social Media Risk Score, Attendance-Study Product). All features are standardized "
        "using StandardScaler for ML algorithm compatibility.",
        styles['ProposalBody']
    ))

    story.append(Paragraph("4.2 KNN Classification", styles['SubSectionHeader']))
    story.append(Paragraph(
        "K-Nearest Neighbors is used for two classification tasks: binary (Pass/Fail) and multi-class (A/B/C/D/F). "
        "The optimal K value is found by evaluating accuracy across K=1 to 30. The dataset is split 75/25 with "
        "stratification. Models are evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. "
        "5-fold stratified cross-validation ensures robustness.",
        styles['ProposalBody']
    ))

    story.append(Paragraph("4.3 DBSCAN Clustering", styles['SubSectionHeader']))
    story.append(Paragraph(
        "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is applied after PCA dimensionality "
        "reduction to 3 components. DBSCAN discovers clusters of arbitrary shape and identifies outliers as noise "
        "points. Optimal eps and min_samples are found via grid search using a combined score: "
        "silhouette * (1 - noise_percentage). This ensures meaningful clusters with minimal noise.",
        styles['ProposalBody']
    ))

    # ================= 5. PROJECT STRUCTURE =================
    story.append(Paragraph("5. Project Structure", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))

    struct_data = [
        ['Directory/File', 'Contents'],
        ['src/step1_generate_data.py', 'Synthetic student dataset generation'],
        ['src/step2_preprocessing.py', 'Cleaning, encoding, feature engineering'],
        ['src/step3_knn_classification.py', 'KNN model training & evaluation'],
        ['src/step4_dbscan_clustering.py', 'DBSCAN clustering & outlier detection'],
        ['src/step5_visualization.py', 'Charts and visual reports'],
        ['run_all_steps.py', 'Orchestrator script for full pipeline'],
        ['data/raw/', 'Original generated dataset'],
        ['data/processed/', 'Cleaned and clustered datasets'],
        ['reports/', 'Results and chart outputs'],
        ['requirements.txt', 'Python dependencies'],
    ]
    st = Table(struct_data, colWidths=[2.0*inch, 4.0*inch])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(st)

    # ================= 6. RESULTS =================
    story.append(PageBreak())
    story.append(Paragraph("6. Expected Results", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))

    story.append(Paragraph("6.1 KNN Classification Results", styles['SubSectionHeader']))
    knn_data = [
        ['Metric', 'Pass/Fail', 'Grade (A/B/C/D/F)'],
        ['Accuracy', '~91%', '~86%'],
        ['F1-Score', '~0.91', '~0.84'],
        ['ROC-AUC', '~0.98', 'N/A (multiclass)'],
        ['Best K', '29', '22'],
        ['CV Accuracy', '~89% (+/- 3%)', '~82% (+/- 3%)'],
    ]
    kt = Table(knn_data, colWidths=[1.5*inch, 1.8*inch, 2.7*inch])
    kt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (0, -1), LIGHT_BLUE), ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10), ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(kt)

    story.append(Spacer(1, 12))
    story.append(Paragraph("6.2 DBSCAN Clustering Results", styles['SubSectionHeader']))
    cluster_data = [
        ['Metric', 'Value'],
        ['PCA Components', '3 (51.5% variance explained)'],
        ['Optimal eps', '1.6'],
        ['Optimal min_samples', '3'],
        ['Silhouette Score', '0.73'],
        ['Clusters Found', '2'],
        ['Noise Points', '~8 (1.6%)'],
    ]
    ct = Table(cluster_data, colWidths=[2.0*inch, 4.0*inch])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(ct)

    # ================= 7. DELIVERABLES =================
    story.append(Paragraph("7. Deliverables", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    del_data = [
        ['Deliverable', 'Format', 'Location'],
        ['Raw Student Dataset', 'CSV', 'data/raw/student_performance.csv'],
        ['Processed Dataset', 'CSV', 'data/processed/student_performance_processed.csv'],
        ['Clustered Dataset', 'CSV', 'data/processed/student_clusters.csv'],
        ['KNN Results', 'Text', 'reports/knn_results.txt'],
        ['DBSCAN Results', 'Text', 'reports/dbscan_results.txt'],
        ['Visualizations', 'PNG (8 charts)', 'reports/charts/'],
        ['Pipeline Script', 'Python', 'run_all_steps.py'],
        ['Proposal Document', 'PDF', 'Student_Performance_Analysis_Proposal.pdf'],
    ]
    dt = Table(del_data, colWidths=[2.2*inch, 1.5*inch, 2.3*inch])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(dt)

    # ================= 8. TECHNICAL REQUIREMENTS =================
    story.append(Paragraph("8. Technical Requirements", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    tech_data = [
        ['Package', 'Version', 'Purpose'],
        ['pandas', '>=1.3.0', 'Data manipulation'],
        ['numpy', '>=1.20.0', 'Numerical computing'],
        ['matplotlib', '>=3.4.0', 'Plotting'],
        ['seaborn', '>=0.11.0', 'Statistical visualization'],
        ['scikit-learn', '>=1.0.0', 'ML algorithms (KNN, DBSCAN, PCA)'],
        ['scipy', '>=1.7.0', 'Scientific computing'],
    ]
    tt = Table(tech_data, colWidths=[1.5*inch, 1.3*inch, 3.2*inch])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(tt)
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Python Version:</b> 3.8+ required.", styles['ProposalBody']))

    # ================= 9. RISK ASSESSMENT =================
    story.append(Paragraph("9. Risk Assessment", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    risk_data = [
        ['Risk', 'Likelihood', 'Impact', 'Mitigation'],
        ['Synthetic data bias', 'Medium', 'Medium', 'Realistic distributions with noise'],
        ['KNN sensitivity to K', 'Low', 'Medium', 'Systematic search over K=1 to 30'],
        ['DBSCAN parameter tuning', 'Medium', 'High', 'Grid search with combined scoring metric'],
        ['Curse of dimensionality', 'Low', 'Medium', 'PCA reduction before DBSCAN'],
        ['Environment dependency', 'Medium', 'Low', 'Version-pinned requirements.txt'],
    ]
    rt = Table(risk_data, colWidths=[1.5*inch, 1.0*inch, 0.8*inch, 2.7*inch])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_BLUE]),
    ]))
    story.append(rt)

    # ================= 10. SUCCESS METRICS =================
    story.append(Paragraph("10. Success Metrics", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))
    metrics = [
        "KNN achieves >85% accuracy on pass/fail classification",
        "DBSCAN identifies meaningful clusters with <10% noise points",
        "Outlier analysis distinguishes exceptional and at-risk students",
        "All 5 pipeline steps execute without errors",
        "Visualizations are generated for all analytical components",
        "Complete documentation and proposal are produced",
    ]
    for m in metrics:
        story.append(Paragraph(f"<bullet>&#9745;</bullet> {m}", styles['PBullet']))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GRAY, spaceAfter=12))
    story.append(Paragraph(
        "This proposal outlines the complete scope and methodology for the Smart Student Performance Analysis "
        "project, combining KNN classification and DBSCAN clustering for comprehensive educational data mining.",
        styles['Footer']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles['Footer']))

    doc.build(story)
    print(f"PDF proposal created: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    create_pdf()
