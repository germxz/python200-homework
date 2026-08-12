import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    f1_score,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib
import pandas as pd

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#------ ROC and AUC

#ROC Q1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logi = LogisticRegression(max_iter=1000, random_state=42)
kneighbors = KNeighborsClassifier(n_neighbors=5)


logi.fit(X_train, y_train)
kneighbors.fit(X_train_scaled, y_train)

y_probs_lr = logi.predict_proba(X_test)[:, 1]
kneighbors_probs = kneighbors.predict_proba(X_test_scaled)[:, 1]

logi_auc = roc_auc_score(y_test, y_probs_lr)
kneighbors_auc = roc_auc_score(y_test, kneighbors_probs)

print(f"Q1:Logistic Regression AUC: {logi_auc}")
print(f"Q1:KNN AUC: {kneighbors_auc}")

# KNN has the higher AUC (0.939 vs 0.706). AUC measures how well the model
# ranks positives above negatives across every possible threshold, so KNN
# separates the two classes much better than logistic regression on this data,
# independent of any single cutoff choice.

#ROC Q2
roc_curve_logi = roc_curve(y_test, y_probs_lr)
roc_curve_kneighbors = roc_curve(y_test, kneighbors_probs)

plt.figure()
plt.plot(roc_curve_logi[0], roc_curve_logi[1], label=f"Logistic Regression (AUC = {logi_auc:.3f})")
plt.plot(roc_curve_kneighbors[0], roc_curve_kneighbors[1], label=f"KNN (AUC = {kneighbors_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.title("ROC Curve Comparison on logistic Regression and KNN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("outputs/roc_comparison.png")
plt.close()

# At the point .8 true positive rate, the KNN curve has the lower false positive rate. This means that KNN is better at correctly identifying positive cases while minimizing false positives compared to Logistic Regression at the same TPR.
# If I needed to catch a 80% of positive cases, I would choose KNN as it has only less than 10% false positive rate at that point compared to logistic regression's 60% false positive rate.

#ROC Q3

fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

best_f1 = 0
best_i = 0

for i in range(len(thresholds)):
    y_pred =  (y_probs_lr >= thresholds[i]).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_i = i

print(f"Q3:Best threshold: {thresholds[best_i]:.4f}")
print(f"Q3:TPR at optimum: {tpr[best_i]:.4f}")
print(f"Q3:FPR at optimum: {fpr[best_i]:.4f}")
print(f"Q3:Best F1: {best_f1:.4f}")

# The optimal threshold is 0.28, well below the default 0.5, so the model flags
# many more samples as positive. My TPR jumped to 0.89 but FPR rose to 0.69 --
# I traded a flood of false alarms for catching the most positives. A threshold below
# 0.5 makes sense when missing a positive is the expensive error, like disease
# screening, where a false alarm is just a follow-up test but a miss can be fatal.


# GridSearchCV

# GridSearchCV Q1
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
grid_search = GridSearchCV(pipe, param_grid={'classifier__C':[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, cv=5, scoring='roc_auc').fit(X_train, y_train)

print(f"GSCV Q1: Best C value: {grid_search.best_params_['classifier__C']}")
print(f"GSCV Q1: Best CV AUC: {grid_search.best_score_:.4f}")
print(f"GSCV Q1: Test AUC of the best estimator: {roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]):.4f}")

# the grid picked C=100, not the default 1.0.
# The test AUC with C=100 was 0.7057, vs 0.706 at default -- essentially no change,
# so C was not what limited this model; its linear boundary was.
# The CV AUC (0.7727) was higher than the test AUC because C was chosen by
# maximizing the CV score across six candidates -- the winner of a search is
# always a little lucky, so its CV score runs high. The test set was not used
# to choose anything, so its number is the honest one.


# GridSearchCV Q2
pipe_tree = Pipeline([('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=42))])
grid_tree = GridSearchCV(pipe_tree, param_grid={'classifier__max_depth': [2, 3, 5, 8, None]}, cv=5, scoring='roc_auc').fit(X_train, y_train)

print(f"GSCV Q2: Best max_depth: {grid_tree.best_params_['classifier__max_depth']}")
print(f"GSCV Q2: Best CV AUC: {grid_tree.best_score_:.4f}")
print(f"GSCV Q2: Test AUC: {roc_auc_score(y_test, grid_tree.predict_proba(X_test)[:, 1]):.4f}")

# REWORD: The tree (test AUC 0.9354) beats the tuned logistic regression (0.7057)
# by a wide margin -- the tree can draw a non-linear boundary and this data's
# classes are not linearly separable. I would develop the tree further. AUC is
# not the only consideration: speed, fold-to-fold stability (see Q3), and how
# explainable the model's decisions are all matter too.

# GridSearchCV Q3
res = pd.DataFrame(grid_tree.cv_results_)
res = res[['param_classifier__max_depth', 'mean_test_score', 'std_test_score']]
res = res.sort_values('mean_test_score', ascending=False)
print("GSCV Q3: CV results sorted best to worst:")
print(res.to_string(index=False))

# REWORD: depth=8 (mean 0.881, std 0.026) and depth=None (mean 0.863, std 0.039)
# have similar means but None's spread is much larger. Between similar means I
# would take the lower standard deviation: a score that depends less on which
# data slice it saw is more trustworthy on the next slice. The unlimited tree
# is memorizing fold quirks, which is what the extra variance shows.

# --- joblib ---

# joblib Q1
best_lr_pipe = grid_search.best_estimator_
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")
loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)
assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# If we saved only the LogisticRegression without the scaler, the loaded model
# would still run .predict on raw data without error -- but it would apply
# coefficients learned on scaled features to unscaled features, so the
# predictions would be silently wrong. Saving the whole Pipeline keeps the
# scaler's learned means and standard deviations packaged with the model.

# joblib Q2
# --- Simulated prediction script ---
model_from_disk = joblib.load("models/warmup_model.pkl")

new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

new_preds = model_from_disk.predict(new_samples)
new_probs = model_from_disk.predict_proba(new_samples)[:, 1]
for i in range(len(new_samples)):
    print(f"Sample {i}: predicted class = {new_preds[i]}, P(positive) = {new_probs[i]:.4f}")

# I expected the all-zeros row to sit near 0.5 zeros look like an
# "average" sample, which should be maximally ambiguous. It actually predicts
# positive at about 0.65, because the model's intercept and the training data's
# feature means shift the neutral point: all-zeros in raw space is not exactly
# on the class boundary in scaled space.