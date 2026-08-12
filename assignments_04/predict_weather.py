
import json
import joblib
import pandas as pd

# --- Task 1: Load and Verify ---
model = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Test AUC:", metadata["test_auc"])

# --- Task 2: Predict on New Data ---
feature_names = metadata["feature_names"]

new_days = pd.DataFrame(
    [
        [20.0, 12.0, 0.0, 15.0],   # clearly good: mild, dry, calm
        [24.0, 14.0, 0.5, 20.0],   # clearly good: warm, near-dry
        [38.0, 26.0, 0.0, 25.0],   # clearly bad: OKC summer heat
        [2.0, -6.0, 0.0, 30.0],    # clearly bad: freezing and windy
        [15.0, 8.0, 6.0, 22.0],    # clearly bad: rained out
        [26.5, 13.0, 1.0, 33.0],   # borderline: right at the temp and wind edges
    ],
    columns=feature_names,
)

preds = model.predict(new_days)
probs = model.predict_proba(new_days)[:, 1]

for i in range(len(new_days)):
    label = "good" if preds[i] == 1 else "skip"
    print(f"Day {i}: {new_days.iloc[i].to_dict()}")
    print(f"        prediction = {label}, P(good for running) = {probs[i]:.4f}")

# --- Task 3: Reflect ---
# My borderline day (26.5C, 33 km/h wind) came out at P = 0.2065 -- more
# confident than I expected; both features sitting just past their thresholds
# stacked up to a clear skip rather than a coin flip. The most revealing
# prediction was Day 3: a freezing, windy day rated 0.74 "good," because the
# linear model learned "colder = better" from OKC's hot summers and cannot
# also penalize the cold end. For a day where the model said 0.52 I would not
# treat the output as a real answer -- the app should surface the probability
# or say "borderline" instead of a hard yes/no.
# If someone ran this script before the training script, joblib.load would
# crash with FileNotFoundError on models/weather_classifier.pkl. A more helpful
# version would catch that error and print "Model file not found -- run
# train_weather_classifier.py first," then exit cleanly.
# To classify tomorrow's weather daily in production, this script would need to
# fetch tomorrow's forecast from the Open-Meteo forecast API instead of using
# hand-typed rows, build the DataFrame with the same four feature columns from
# the metadata, and run on a schedule (like a daily cron job), logging or
# sending the prediction somewhere useful.