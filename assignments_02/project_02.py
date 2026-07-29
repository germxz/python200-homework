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