"""
Customer Insurance Purchase Prediction
Comparative Study of ML Classification Algorithms
=================================================
Dataset features: Age, EstimatedSalary → Purchased (0/1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# STEP 1: LOAD / GENERATE DATASET
# ─────────────────────────────────────────────────────────
# If you have the real CSV, replace this block with:
#   df = pd.read_csv('Social_Network_Ads.csv')
#   df = df[['Age', 'EstimatedSalary', 'Purchased']]

np.random.seed(42)
n = 400
ages    = np.random.randint(18, 65, n)
salaries = np.random.randint(15000, 150000, n)
prob    = 1 / (1 + np.exp(-((ages - 38) * 0.08 + (salaries - 70000) / 30000)))
purchased = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({'Age': ages, 'EstimatedSalary': salaries, 'Purchased': purchased})
print("Dataset shape:", df.shape)
print(df['Purchased'].value_counts())
print(df.head())

# ─────────────────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────────────────
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)
X_all_sc   = sc.transform(X)          # for full-data boundary plots

# ─────────────────────────────────────────────────────────
# STEP 3: DEFINE CLASSIFIERS
# ─────────────────────────────────────────────────────────
classifiers = {
    'Logistic Regression' : LogisticRegression(random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=5),
    'SVM'                 : SVC(kernel='rbf', random_state=42, probability=True),
    'Decision Tree'       : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
}

# ─────────────────────────────────────────────────────────
# STEP 4: TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────
results = {}
trained = {}

print("\n" + "="*65)
print(f"{'Algorithm':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("="*65)

for name, clf in classifiers.items():
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score   (y_test, y_pred, zero_division=0)
    f1   = f1_score       (y_test, y_pred, zero_division=0)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    trained[name] = clf
    print(f"{name:<25} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

results_df = pd.DataFrame(results).T
print("="*65)
best_name = results_df['Accuracy'].idxmax()
best_clf  = trained[best_name]
print(f"\nBest model: {best_name} (Accuracy = {results_df.loc[best_name,'Accuracy']:.3f})")

# ─────────────────────────────────────────────────────────
# HELPER: DECISION BOUNDARY PLOT
# ─────────────────────────────────────────────────────────
def plot_boundary(ax, clf, X_sc, y, title, test_points=None):
    """
    Plots the decision boundary of a classifier in scaled space.
    test_points: list of (raw_point [age, sal], label_str)
    """
    h = 0.02
    x_min, x_max = X_sc[:, 0].min() - 0.5, X_sc[:, 0].max() + 0.5
    y_min, y_max = X_sc[:, 1].min() - 0.5, X_sc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4,
                cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    ax.scatter(X_sc[y==0, 0], X_sc[y==0, 1],
               c='red',   s=10, alpha=0.6, label='No Purchase')
    ax.scatter(X_sc[y==1, 0], X_sc[y==1, 1],
               c='green', s=10, alpha=0.6, label='Purchase')

    if test_points:
        for pt_raw, lbl in test_points:
            pt_sc = sc.transform([pt_raw])[0]
            pred  = clf.predict([pt_sc])[0]
            color = 'blue' if pred == 1 else 'orange'
            ax.scatter(pt_sc[0], pt_sc[1], c=color, s=140,
                       marker='*', edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(lbl, (pt_sc[0], pt_sc[1]), fontsize=6,
                        ha='left', va='bottom')

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Age (scaled)',    fontsize=7)
    ax.set_ylabel('Salary (scaled)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc='upper left')

# ─────────────────────────────────────────────────────────
# FIGURE 1: EDA
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Exploratory Data Analysis', fontsize=13, fontweight='bold')

for cls, color, label in [(0, 'red', 'No Purchase'), (1, 'green', 'Purchase')]:
    mask = y == cls
    axes[0].scatter(X[mask, 0], X[mask, 1]/1000,
                    c=color, alpha=0.5, s=15, label=label)
axes[0].set_xlabel('Age');  axes[0].set_ylabel('Salary ($k)')
axes[0].set_title('Age vs Salary (by Purchase)', fontweight='bold')
axes[0].legend()

age_bins = pd.cut(df['Age'], bins=[17,25,35,45,55,65],
                  labels=['18-25','26-35','36-45','46-55','56-65'])
rate = df.groupby(age_bins, observed=True)['Purchased'].mean()
bars = axes[1].bar(rate.index, rate.values, color='steelblue',
                   alpha=0.8, edgecolor='black')
for bar, v in zip(bars, rate.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f'{v:.2f}', ha='center', fontsize=9)
axes[1].set_ylim(0, 1.1)
axes[1].set_xlabel('Age Group'); axes[1].set_ylabel('Purchase Rate')
axes[1].set_title('Purchase Rate by Age Group', fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_eda.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# FIGURE 2: ALL DECISION BOUNDARIES
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Decision Boundaries – All 5 Classifiers', fontsize=13, fontweight='bold')

for i, (name, clf) in enumerate(classifiers.items()):
    ax = axes[i // 3][i % 3]
    plot_boundary(ax, clf, X_all_sc, y, name)

axes[1][2].axis('off')   # hide empty 6th panel
plt.tight_layout()
plt.savefig('fig2_decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# FIGURE 3: METRICS COMPARISON
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Performance Metrics – All Classifiers', fontsize=13, fontweight='bold')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors  = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
names   = list(classifiers.keys())

for idx, metric in enumerate(metrics):
    ax   = axes[idx // 2][idx % 2]
    vals = [results[c][metric] for c in names]
    bars = ax.bar(names, vals, color=colors[idx], alpha=0.8, edgecolor='black')
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('fig3_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# FIGURE 4: CONFUSION MATRICES
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Confusion Matrices – All Classifiers', fontsize=13, fontweight='bold')

for i, (name, clf) in enumerate(classifiers.items()):
    ax = axes[i // 3][i % 3]
    y_pred = clf.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Buy', 'Buy'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=9, fontweight='bold')

axes[1][2].axis('off')
plt.tight_layout()
plt.savefig('fig4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# QUESTION 1 – PREDICTIONS: SET 1
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("QUESTION 1 – TEST PREDICTIONS: SET 1")
print("="*55)

q1_points = [
    ([30,  87000],  'A30,S87k'),
    ([40,      0],  'A40,NoSal'),
    ([40, 100000],  'A40,S100k'),
    ([50,      0],  'A50,NoSal'),
]

print(f"\n{'Scenario':<14} {'Age':>4} {'Salary':>12}  "
      f"{'LR':>6} {'KNN':>6} {'SVM':>6} {'DT':>6} {'RF':>6}")
print("-"*65)

for pt_raw, lbl in q1_points:
    pt_sc  = sc.transform([pt_raw])
    preds  = {name: clf.predict(pt_sc)[0] for name, clf in classifiers.items()}
    labels = {name: ('BUY' if p else 'NO') for name, p in preds.items()}
    age, sal = pt_raw
    print(f"{lbl:<14} {age:>4} {sal:>12,}  "
          f"{labels['Logistic Regression']:>6} "
          f"{labels['KNN']:>6} "
          f"{labels['SVM']:>6} "
          f"{labels['Decision Tree']:>6} "
          f"{labels['Random Forest']:>6}")

# Plot Q1 on best model boundary
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('Q1 – Test Predictions (Set 1) on All Classifiers\n'
             '★ Blue=Predicted Buy  ★ Orange=Predicted No Buy',
             fontsize=11, fontweight='bold')
for i, (name, clf) in enumerate(classifiers.items()):
    plot_boundary(axes[i], clf, X_all_sc, y, name, q1_points)
plt.tight_layout()
plt.savefig('fig5_q1_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# QUESTION 2 – PREDICTIONS: SET 2  (extreme salary values)
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("QUESTION 2 – TEST PREDICTIONS: SET 2 (Extreme Values)")
print("="*55)

q2_points = [
    ([18,         0], 'A18,NoSal'),
    ([22,    600000], 'A22,S600k'),
    ([35,   2500000], 'A35,S2.5M'),
    ([60, 100000000], 'A60,S100M'),
]

print(f"\n{'Scenario':<12} {'Age':>4} {'Salary':>15}  "
      f"{'LR':>6} {'KNN':>6} {'SVM':>6} {'DT':>6} {'RF':>6}")
print("-"*68)

for pt_raw, lbl in q2_points:
    pt_sc  = sc.transform([pt_raw])
    preds  = {name: clf.predict(pt_sc)[0] for name, clf in classifiers.items()}
    labels = {name: ('BUY' if p else 'NO') for name, p in preds.items()}
    age, sal = pt_raw
    print(f"{lbl:<12} {age:>4} {sal:>15,}  "
          f"{labels['Logistic Regression']:>6} "
          f"{labels['KNN']:>6} "
          f"{labels['SVM']:>6} "
          f"{labels['Decision Tree']:>6} "
          f"{labels['Random Forest']:>6}")

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('Q2 – Test Predictions (Set 2) on All Classifiers\n'
             '★ Blue=Predicted Buy  ★ Orange=Predicted No Buy',
             fontsize=11, fontweight='bold')
for i, (name, clf) in enumerate(classifiers.items()):
    plot_boundary(axes[i], clf, X_all_sc, y, name, q2_points)
plt.tight_layout()
plt.savefig('fig6_q2_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# QUESTION 3 – HYPOTHESES TESTING
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("QUESTION 3 – HYPOTHESES TESTING")
print("="*55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Hypothesis Testing using Best Model: {best_name}',
             fontsize=12, fontweight='bold')

# ── H3: Fix salary → vary age ──
ages_range = np.arange(18, 70, 1)
for sal_val, ls, label in [
    (30000,  '--', 'Low Salary  $30k'),
    (70000,  '-',  'Mid Salary  $70k'),
    (130000, ':',  'High Salary $130k'),
]:
    pts = sc.transform([[a, sal_val] for a in ages_range])
    if hasattr(best_clf, 'predict_proba'):
        probs = best_clf.predict_proba(pts)[:, 1]
    else:
        probs = best_clf.predict(pts)
    axes[0].plot(ages_range, probs, linestyle=ls, label=label, linewidth=2)

axes[0].set_xlabel('Age', fontsize=10)
axes[0].set_ylabel('P(Purchase)', fontsize=10)
axes[0].set_title('Effect of Age at Different Salary Levels', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].set_ylim(-0.05, 1.05)
axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)

# ── H3: Fix age → vary salary ──
sal_range = np.linspace(0, 150000, 300)
for age_val, ls, label in [
    (25, '--', 'Young (25)'),
    (40, '-',  'Middle-aged (40)'),
    (58, ':',  'Older (58)'),
]:
    pts = sc.transform([[age_val, s] for s in sal_range])
    if hasattr(best_clf, 'predict_proba'):
        probs = best_clf.predict_proba(pts)[:, 1]
    else:
        probs = best_clf.predict(pts)
    axes[1].plot(sal_range / 1000, probs, linestyle=ls, label=label, linewidth=2)

axes[1].set_xlabel('Estimated Salary ($ thousands)', fontsize=10)
axes[1].set_ylabel('P(Purchase)', fontsize=10)
axes[1].set_title('Effect of Salary at Different Ages', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].set_ylim(-0.05, 1.05)
axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('fig7_hypotheses.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL METRICS SUMMARY TABLE")
print("="*65)
print(results_df.round(3).to_string())
print("="*65)
print(f"\n✔ Best Overall Model  : {best_name}")
print(f"✔ Highest Precision   : {results_df['Precision'].idxmax()}")
print(f"✔ Highest Recall      : {results_df['Recall'].idxmax()}")
print(f"✔ Highest F1-Score    : {results_df['F1-Score'].idxmax()}")
print("\nOutputs saved: fig1_eda.png  fig2_decision_boundaries.png")
print("  fig3_metrics.png  fig4_confusion_matrices.png")
print("  fig5_q1_predictions.png  fig6_q2_predictions.png")
print("  fig7_hypotheses.png")
