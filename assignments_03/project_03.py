import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Task 1: Load and Explore
# ============================================================
# spambase.data has no header row -- the 57 feature names live in
# spambase.names, so we parse them out and attach spam_label as column 58.
names = []
for line in open("spambase.names", encoding="latin-1"):
    m = re.match(r"^([\w_\$\!\#\;\(\[]+):\s*continuous", line.strip())
    if m:
        names.append(m.group(1))

df = pd.read_csv("spambase.data", header=None, names=names + ["spam_label"])

print("Shape:", df.shape)
print("Class counts:\n", df["spam_label"].value_counts())
print("Class proportions:\n", df["spam_label"].value_counts(normalize=True))

# 4601 emails, 60.6% ham / 39.4% spam. Not badly imbalanced, but enough that
# raw accuracy needs context: a model that predicts "ham" for everything scores
# ~61% while catching zero spam, so accuracy below ~0.61 is worse than useless
# and the per-class precision/recall matter more than the single number.

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

for feat in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    plt.figure()
    plt.boxplot([df[df["spam_label"] == 0][feat], df[df["spam_label"] == 1][feat]],
                tick_labels=["Ham", "Spam"])
    plt.ylabel(feat)
    plt.title(f"{feat} by class")
    safe = feat.replace("!", "excl")
    plt.savefig(f"outputs/box_{safe}.png")
    plt.close()

# The boxplots show real but subtle differences: the spam boxes sit higher for
# all three features, but the boxes themselves are squashed near zero and the
# separation lives mostly in the outliers. No single feature cleanly splits
# spam from ham on its own.

print("Zero proportion (top 5 features):\n", (X == 0).mean().sort_values(ascending=False).head())
print(X[["word_freq_free", "char_freq_!", "capital_run_length_total"]].describe().T[["min", "max", "mean"]])

# Most emails have 0 for most word-frequency features -- 73% of emails never use
# the word "free" at all. The data is heavily zero-skewed because any single word
# only appears in a minority of emails. Scales also differ wildly: word/char
# frequencies are small percentages (max ~20-32) while capital_run_length_total
# runs 1 to 15,841. That matters for distance-based models (KNN) and for PCA,
# where the capital-run features would dominate everything unless we standardize.

# ============================================================
# Task 2: Prepare Your Data
# ============================================================
# stratify=y keeps the 61/39 class ratio identical in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaler is fit on the training data only, then applied to both sets --
# fitting on all the data would leak the test set's distribution into the
# transformation and inflate every downstream score.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA preprocessing: scale first (otherwise capital_run_length_total, with a
# range in the thousands, would dominate the variance directions), and fit PCA
# on the training data only for the same leakage reason as the scaler.
pca = PCA()
pca.fit(X_train_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)
n = np.argmax(cumvar >= 0.90) + 1
print("Components needed for 90% variance:", n)

plt.figure()
plt.plot(cumvar)
plt.axhline(0.90, color="red", linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Cumulative Explained Variance (Spambase)")
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

# n = 43 of 57 components to reach 90%. That is a weak reduction -- the features
# are not very redundant, so PCA has little to compress here. This sets the
# expectation that PCA probably won't help the classifiers much in Task 3.

# Keep BOTH the full scaled arrays and the PCA-reduced arrays -- Task 3 compares
# classifiers on each.
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# ============================================================
# Task 3: A Classifier Comparison
# ============================================================
def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(yte, pred))
    print(classification_report(yte, pred))
    return model

evaluate("KNN unscaled", KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test)
evaluate("KNN scaled", KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train, X_test_scaled, y_test)
evaluate("KNN PCA", KNeighborsClassifier(n_neighbors=5), X_train_pca, y_train, X_test_pca, y_test)

# KNN comparison: unscaled 0.799 -> scaled 0.908 -> PCA 0.907. Scaling is worth
# +11 points because KNN is pure distance -- unscaled, capital_run_length_total
# drowns out every word frequency. PCA on top of scaling changes nothing
# (0.908 vs 0.907), which matches the Task 2 finding: with 43 of 57 components
# needed, there was almost no redundancy for PCA to exploit.

# Decision Tree depth exploration
for d in [3, 5, 10, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    print(f"depth={d}: train={t.score(X_train, y_train):.4f}  test={t.score(X_test, y_test):.4f}")

# depth=3:    train 0.897  test 0.885
# depth=5:    train 0.923  test 0.899
# depth=10:   train 0.967  test 0.909
# depth=None: train 0.9997 test 0.911
# As depth increases, train accuracy races toward memorizing the training set
# (99.97% at unlimited depth) while test accuracy barely moves past 0.91.
# The gap between train and test is the overfitting: extra depth buys the model
# nothing on unseen data. I would use max_depth=10 in production -- it captures
# essentially all of the generalizable signal (test 0.909 vs 0.911 unlimited)
# while keeping the tree small enough to inspect and less brittle to retraining.
CHOSEN_DEPTH = 10

tree = evaluate(f"Decision Tree (max_depth={CHOSEN_DEPTH})",
                DecisionTreeClassifier(max_depth=CHOSEN_DEPTH, random_state=42),
                X_train, y_train, X_test, y_test)

rf = evaluate("Random Forest",
              RandomForestClassifier(n_estimators=100, random_state=42),
              X_train, y_train, X_test, y_test)

evaluate("LogReg scaled",
         LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
         X_train_scaled, y_train, X_test_scaled, y_test)
evaluate("LogReg PCA",
         LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
         X_train_pca, y_train, X_test_pca, y_test)

# --- Task 3 summary ---
# Final accuracies: KNN unscaled 0.799, KNN scaled 0.908, KNN PCA 0.907,
# Decision Tree (d=10) 0.909, Random Forest 0.945, LogReg scaled 0.929,
# LogReg PCA 0.919.
#
# Random Forest wins clearly. For both PCA comparisons, PCA either did nothing
# (KNN) or slightly hurt (LogReg 0.929 -> 0.919) -- consistent with the Task 2
# hypothesis: needing 43/57 components meant this data had little redundant
# structure to reduce, so PCA mostly just threw away a bit of signal.
#
# Is accuracy the right metric for a spam filter? No. The two error types have
# very different costs: a false negative means one junk email reaches the inbox
# and gets deleted in two seconds; a false positive means a legitimate email --
# a job offer, a bank alert -- silently disappears into a spam folder nobody
# checks. I would optimize to minimize false positives (maximize precision on
# the spam class), accepting slightly more spam getting through as the price of
# never losing real mail.

# Feature importances
feat_names = X.columns
tree_imp = pd.Series(tree.feature_importances_, index=feat_names).sort_values(ascending=False)
rf_imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTop 10 Decision Tree importances:\n", tree_imp.head(10))
print("\nTop 10 Random Forest importances:\n", rf_imp.head(10))

plt.figure(figsize=(10, 5))
rf_imp.head(10).plot(kind="bar")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances (Top 10)")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# The two models agree on the core signals -- 6 of each top-10 overlap, and both
# put char_freq_$, char_freq_!, and word_freq_remove at or near the top. The
# difference is concentration: the single tree loads 39% of its importance onto
# char_freq_$ alone, while the forest spreads importance across many features
# (its top score is only 11%). That's the averaging across 100 trees at work.
# The results match intuition about spam: dollar signs, exclamation points,
# "remove" (as in "click to remove"), "free", and long runs of capital letters.

# Confusion matrix for the best model (Random Forest)
best_model = rf
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
print("\nRandom Forest confusion matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

# 18 false positives (ham marked spam) vs 33 false negatives (spam let through).
# The model makes more false negatives -- given the costs above, that is the
# better direction to err: it loses real mail less often than it lets junk in.

# ============================================================
# Task 4: Cross-Validation
# ============================================================
# All classifiers from Task 3, each cross-validated on the data variant it was
# trained on. (Note: CV inside this section still leaks slightly for the scaled/
# PCA variants because the transforms were fit on all of X_train -- the pipeline
# approach in Task 5 is the clean fix for that.)
cv_models = {
    "KNN unscaled":  (KNeighborsClassifier(n_neighbors=5), X_train.values),
    "KNN scaled":    (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "KNN PCA":       (KNeighborsClassifier(n_neighbors=5), X_train_pca),
    "Decision Tree": (DecisionTreeClassifier(max_depth=CHOSEN_DEPTH, random_state=42), X_train.values),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train.values),
    "LogReg scaled": (LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"), X_train_scaled),
    "LogReg PCA":    (LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"), X_train_pca),
}
for name, (m, data) in cv_models.items():
    scores = cross_val_score(m, data, y_train, cv=5)
    print(f"{name}: mean={scores.mean():.4f}  std={scores.std():.4f}")

# Random Forest is both the most accurate (mean ~0.954) and among the most
# stable. The single Decision Tree has the highest variance across folds
# (std ~0.019, roughly double most other models) -- exactly the fragility the
# lesson describes: small changes in training data reshape a single tree, while
# the forest's internal averaging over 100 trees smooths that out. The CV
# ranking matches the single train/test split: Random Forest first, LogReg
# second, KNN scaled / tree in the middle, KNN unscaled last.

# ============================================================
# Task 5: Building a Prediction Pipeline
# ============================================================
# Best tree-based classifier: Random Forest. Trees split on per-feature
# thresholds, so scaling and PCA add nothing -- the pipeline is just the model.
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
tree_pipeline.fit(X_train, y_train)
print("\nTree pipeline (Random Forest):")
print(classification_report(y_test, tree_pipeline.predict(X_test)))

# Best non-tree classifier: Logistic Regression on scaled data. Task 3 showed
# PCA slightly HURT LogReg (0.929 -> 0.919), so per the prompt's condition it
# is NOT included as a step -- just scaler + classifier.
nontree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])
nontree_pipeline.fit(X_train, y_train)
print("Non-tree pipeline (scaler + LogReg):")
print(classification_report(y_test, nontree_pipeline.predict(X_test)))

print("Tree pipeline accuracy:    ", tree_pipeline.score(X_test, y_test))
print("Non-tree pipeline accuracy:", nontree_pipeline.score(X_test, y_test))

# Both pipelines reproduce the Task 3 numbers exactly (RF 0.945, LogReg 0.929),
# confirming the pipeline applies the same train-only fitting as the manual
# steps. The two pipelines do NOT share a structure, and that's the point:
# the LogReg pipeline needs a scaler because regularized coefficients live on
# the feature scale, while the forest splits on thresholds and is scale-blind,
# so preprocessing would be dead weight. The practical value of packaging it
# this way: whoever receives the pipeline calls fit/predict and cannot forget
# a preprocessing step, apply steps out of order, or accidentally fit the
# scaler on test data -- the leakage protection is built into the object
# instead of depending on the person's bookkeeping.