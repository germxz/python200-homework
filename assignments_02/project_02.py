# The file uses semicolons as separators, not commas, so pd.read_csv needs sep=";"
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "student_performance_math.csv"), sep=";")

print(df.shape)
print(df.head())
print(df.dtypes)

plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig(os.path.join(OUT, "g3_distribution.png"))
plt.close()

# --- Task 2: Preprocess ---
print("shape before filtering:", df.shape)
df_clean = df[df["G3"] > 0].copy()
print("shape after filtering: ", df_clean.shape)

# G3=0 means the student never sat the final exam, not that they earned a zero.
# Keeping those rows would train the model to predict "0" from ordinary background
# features, dragging every prediction down and modeling absence rather than performance.

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])
print(f"absences vs G3 (original): {corr_original:.4f}")
print(f"absences vs G3 (filtered): {corr_filtered:.4f}")

# Students with G3=0 had stopped attending, so they carried high absences AND a zero
# grade. Those points sat in the bottom-right of the scatter and pulled the correlation
# toward zero, hiding the real negative relationship. Once they're removed, the
# expected pattern shows up: more absences, lower grade.

# --- Task 3: EDA ---
numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc"]

correlations = df_clean[numeric_cols].corrwith(df_clean["G3"]).sort_values()
print(correlations)

# failures has the strongest relationship with G3 (most negative). Medu and studytime
# come out positive. goout and Walc being negative was mildly surprising, and absences
# is weaker than expected even after filtering.

plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.5)
plt.title("Past Failures vs Final Grade")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig(os.path.join(OUT, "failures_vs_g3.png"))
plt.close()

# Clear downward trend. Students with zero prior failures span the full grade range,
# while anyone with 2+ failures rarely clears the midpoint.

plt.figure()
plt.bar(correlations.index, correlations.values)
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=45, ha="right")
plt.title("Correlation of Each Feature with G3")
plt.xlabel("Feature")
plt.ylabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_correlations.png"))
plt.close()

# Puts every feature on one axis. Nothing is strongly correlated -- the largest
# magnitude is well under 0.4 -- which sets expectations low for the R² in Task 5.

# --- Task 4: Baseline Model ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

X_base = df_clean[["failures"]].values
y = df_clean["G3"].values

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

base_model = LinearRegression()
base_model.fit(Xb_train, yb_train)
yb_pred = base_model.predict(Xb_test)

print(f"Task 4 slope: {base_model.coef_[0]:.4f}")
print(f"Task 4 RMSE:  {np.sqrt(np.mean((yb_pred - yb_test) ** 2)):.4f}")
print(f"Task 4 R²:    {base_model.score(Xb_test, yb_test):.4f}")

# Slope of about -1.43 means each additional past failure predicts roughly 1.4 fewer
# points on the 0-20 final grade. RMSE near 3.0 means a typical prediction misses by
# about 3 points, which is a lot on a 20-point scale -- roughly a letter grade and a half.
# R² of about 0.09 means failures alone explains under 10% of the variation in G3.
# That is lower than the -0.29 correlation might suggest, but correlation squared is
# about 0.086, so it lines up exactly. One feature is not enough.

# --- Task 5: Full Model ---
feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

full_model = LinearRegression()
full_model.fit(X_train, y_train)
y_pred = full_model.predict(X_test)

print(f"Task 5 train R²: {full_model.score(X_train, y_train):.4f}")
print(f"Task 5 test R²:  {full_model.score(X_test, y_test):.4f}")
print(f"Task 5 RMSE:     {np.sqrt(np.mean((y_pred - y_test) ** 2)):.4f}")

for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
    
# --- Task 6: Evaluate and Summarize ---
plt.figure()
plt.scatter(y_pred, y_test, alpha=0.6)

lims = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
plt.plot(lims, lims, "k--")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig(os.path.join(OUT, "predicted_vs_actual_g3.png"))
plt.close()

# The predictions bunch into a narrow band around 10-13 while the actual grades spread
# across roughly 4-20. That is the signature of a weak model: it hedges toward the mean
# instead of committing. Error is worst at both extremes -- high scorers are badly
# underpredicted, low scorers overpredicted -- and tightest in the middle.
# A point above the diagonal means the student scored higher than predicted; below the
# diagonal means they scored lower than predicted.

# --- Summary ---
# After dropping the 38 students with G3=0, the dataset holds 357 rows, split into
# 285 for training and 72 for testing.
#
# The full model reaches a test R² of about 0.26 with an RMSE near 2.66. On a 0-20
# scale that means a typical prediction is off by roughly 2.7 points -- around a letter
# grade and a half. The model explains about a quarter of why students score differently,
# and the rest comes from things this data does not capture.
#
# Largest positive coefficient: internet at about +1.04, home internet access predicting
# roughly a point higher. It likely stands in for household resources generally rather
# than the connection itself.
# Largest negative coefficient: schoolsup at about -2.26. Receiving school support
# predicts a lower grade because support is assigned to students already struggling --
# the flag reports prior difficulty, it does not cause it.
#
# Most surprising result: absences collapsed from a -0.21 standalone correlation to a
# coefficient near -0.06 once other features were present. A variable can matter on its
# own and still contribute nothing unique to a model.

# --- Neglected Feature: The Power of G1 ---
feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

g1_model = LinearRegression()
g1_model.fit(Xg_train, yg_train)
print(f"Test R² with G1: {g1_model.score(Xg_test, yg_test):.4f}")

# R² jumps from ~0.26 to ~0.76. High R² does not mean G1 causes G3 -- both are measures
# of the same underlying ability in the same class, taken months apart. G1 predicts G3
# for the same reason your height last year predicts your height this year.
#
# As an early-warning tool this model is close to useless. By the time G1 exists, the
# first grading period is already over and teachers can see who is struggling without a
# model. Anything G1 tells you, a gradebook tells you sooner.
#
# To intervene before G1, educators would need signals available at enrollment:
# past failures, prior-year attendance, whether the student is already flagged for
# support, and family education background. Those are exactly the weak-but-real
# predictors in the Task 5 model. A 0.26 R² model that works in September beats a
# 0.76