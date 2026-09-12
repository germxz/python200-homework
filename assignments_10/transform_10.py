import os
import json
from dotenv import load_dotenv
import joblib
from supabase import create_client
import pandas as pd
import joblib
from openai import OpenAI

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Step 1: Incremental read 

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
    
feature_names = metadata["feature_names"]
raw_rows = supabase.table("weather_raw").select("*").execute().data
enriched_rows = supabase.table("weather_enriched").select("date").execute().data
already_done = {row["date"] for row in enriched_rows}
to_process = [row for row in raw_rows if row["date"] not in already_done]

print(f"Raw records in weather_raw: {len(raw_rows)}")
print(f"Already enriched: {len(already_done)}")
print(f"To process this run: {len(to_process)}")

# Summary: There are 370 raw records and zero enriched records. All 370 records are ready to be processed this run.

# Step 2: ML transform

if not to_process:
    print("No new records to process. Exiting.")
    raise SystemExit(0)

model = joblib.load("models/weather_classifier.pkl")


# features 
features_df = pd.DataFrame(to_process)[feature_names]
predict = model.predict(features_df)
predict_proba = model.predict_proba(features_df)

enrichment_records = []
for date_row,prediction, probability in zip(to_process, predict, predict_proba):
    enrichment_records.append({
        "date": date_row["date"],
        "good_for_running": bool(prediction),
        "confidence": round(float(probability.max()), 4),
    })

good_for_running = sum(1 for record in enrichment_records if record["good_for_running"])
confidence = [record["confidence"] for record in enrichment_records]

print(f"Classified good for running: {good_for_running} of {len(enrichment_records)}")
print(f"Confidence range: {min(confidence):.4f} to {max(confidence):.4f}")

print("Prediction Summary: 99 days out of 370 days were classified as good for running, with confidence ranging from 0.5003 to 1.0.")

# Step 3: LLM transform
# ---------- Step 3: LLM Transform ----------


SYSTEM_PROMPT = (
    "You write running recommendations for a daily weather app. You will receive one day's "
    "weather conditions and a machine learning model's prediction about whether that day is "
    "good for running. "
    "Write exactly one sentence stating whether the day is good for running and why. "
    "Be direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)


def make_user_message(date_row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {date_row['date']}\n"
        f"High: {date_row['temperature_2m_max']}°C, Low: {date_row['temperature_2m_min']}°C\n"
        f"Precipitation: {date_row['precipitation_sum']} mm\n"
        f"Max wind speed: {date_row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )


for i, record in enumerate(enrichment_records):
    raw_row = to_process[i]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(
                    raw_row, record["good_for_running"], record["confidence"]
                )},
            ],
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        record["llm_summary"] = summary or "Recommendation unavailable."
    except Exception as e:
        print(f"  API error on {record['date']}: {e}")
        record["llm_summary"] = "Recommendation unavailable."

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1} / {len(enrichment_records)}")