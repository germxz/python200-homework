import requests 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import sys
import json
import sklearn
import joblib

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.47,
    "longitude": -97.52,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Chicago",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print(df.shape)
print(df.head())
print(df.describe())

# --- Step 2: Engineer Labels ---
TEMP_MAX_LOW, TEMP_MAX_HIGH = 7, 26
TEMP_MIN_FLOOR = 0
PRECIP_MAX = 3.0
WIND_MAX = 35   


df["good_run"] = (
    (df["temperature_2m_max"].between(TEMP_MAX_LOW, TEMP_MAX_HIGH))
    & (df["temperature_2m_min"] >= TEMP_MIN_FLOOR)
    & (df["precipitation_sum"] < PRECIP_MAX)
    & (df["wind_speed_10m_max"] < WIND_MAX)).astype(int)

print("Class distribution:\n", df["good_run"].value_counts())
print("Fraction good:", round(df["good_run"].mean(), 3))

# --- Step 3: Train and Tune ---
feature_names = ["temperature_2m_max", "temperature_2m_min",
                 "precipitation_sum", "wind_speed_10m_max"]
X = df[feature_names]
y = df["good_run"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
grid = GridSearchCV(
    pipe,
    param_grid={"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    cv=5, scoring="roc_auc",
).fit(X_train, y_train)

y_pred = grid.predict(X_test)
y_probs = grid.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs)

print("Best C:", grid.best_params_["classifier__C"])
print(f"Best CV AUC: {grid.best_score_:.4f}")
print(classification_report(y_test, y_pred))
print(f"Test AUC: {test_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Weather Classifier ROC")
plt.legend()
plt.savefig("outputs/weather_roc.png")
plt.close()

# --- Step 4: Reflect on Evaluation ---

# The AUC of 0.89 is strong but lower than you might expect given that the label
# was constructed entirely from these four features. This is not surprising:
# logistic regression is limited to a linear decision boundary, so each feature
# can only push in one direction -- it cannot express that temperature is bad
# # both below 7C and above 26C -- it learned "colder is better" from OKC's hot
# summers, so it ends up confusing cold-side failures with good days.

# The more common error is false negatives: recall on good days was only 0.54,
# meaning the model recommended just 13 of the 24 truly good days, while its
# precision of 0.72 means that when it does say "run," it is usually right.
# For a running app I would rather it under-recommend like this than
# over-recommend -- missing a nice day costs nothing, but sending someone out
# into a storm breaks trust in the app.

# Even so, 0.54 recall wastes too many good days, so instead of the default 0.5
# I would lower the threshold to around 0.4 to recover some of the missed good
# days, accepting a few more bad recommendations as the price.



# --- Step 5: Save the Model ---

joblib.dump(grid.best_estimator_, "models/weather_classifier.pkl")   

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "feature_names": feature_names,
    "best_params": grid.best_params_,
    "test_auc": round(test_auc, 4),
    "city": {"name": "Oklahoma City", "latitude": 35.47, "longitude": -97.52},
    "label_thresholds": "good_run = temp_max 7-26C, temp_min >= 0C, precip < 3.0mm, wind < 35 km/h (wind raised from 30 for OKC climate)",
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved models/weather_classifier.pkl and models/weather_classifier_metadata.json")