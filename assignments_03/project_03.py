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

# TODO: replace with the loading code from the logistic regression lesson
df = pd.read_csv("spambase.csv")
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

print("Shape:", df.shape)
print("Class counts:\n", y.value_counts())
print("Class proportions:\n", y.value_counts(normalize=True))

for feat in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    plt.figure()
    plt.boxplot([df[df["spam_label"] == 0][feat], df[df["spam_label"] == 1][feat]], labels=["Ham", "Spam"])
    plt.ylabel(feat)
    plt.title(f"{feat} by class")
    plt.savefig(f"outputs/box_{feat.replace('!','excl').replace('/','_')}.png")
    plt.close()

print(X.describe().T[["min", "max", "mean"]].head())
print("Proportion of zeros per feature:\n", (X == 0).mean().sort_values(ascending=False).head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n = np.argmax(cumvar >= 0.90) + 1
print("Components for 90% variance:", n)

plt.figure()
plt.plot(cumvar)
plt.axhline(0.90, color="red", linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

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

for d in [3, 5, 10, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    print(f"depth={d}: train={t.score(X_train, y_train):.4f} test={t.score(X_test, y_test):.4f}")

CHOSEN_DEPTH = 10
tree = evaluate(f"Decision Tree (depth={CHOSEN_DEPTH})", DecisionTreeClassifier(max_depth=CHOSEN_DEPTH, random_state=42), X_train, y_train, X_test, y_test)
rf = evaluate("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, X_test, y_test)
evaluate("LogReg scaled", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"), X_train_scaled, y_train, X_test_scaled, y_test)
evaluate("LogReg PCA", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"), X_train_pca, y_train, X_test_pca, y_test)

feat_names = X.columns
print("\nTop 10 tree:\n", pd.Series(tree.feature_importances_, index=feat_names).sort_values(ascending=False).head(10))
rf_imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTop 10 RF:\n", rf_imp.head(10))

plt.figure(figsize=(10, 5))
rf_imp.head(10).plot(kind="bar")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

best_model = rf
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
print("\nConfusion matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

cv_models = {
    "KNN scaled": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "KNN PCA": (KNeighborsClassifier(n_neighbors=5), X_train_pca),
    "Decision Tree": (DecisionTreeClassifier(max_depth=CHOSEN_DEPTH, random_state=42), X_train),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "LogReg scaled": (LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"), X_train_scaled),
}
for name, (m, data) in cv_models.items():
    s = cross_val_score(m, data, y_train, cv=5)
    print(f"{name}: mean={s.mean():.4f} std={s.std():.4f}")

tree_pipeline = Pipeline([("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])
tree_pipeline.fit(X_train, y_train)
print("\nTree pipeline:\n", classification_report(y_test, tree_pipeline.predict(X_test)))

nontree_pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))])
nontree_pipeline.fit(X_train, y_train)
print("Non-tree pipeline:\n", classification_report(y_test, nontree_pipeline.predict(X_test)))
print("DONE")