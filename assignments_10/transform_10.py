# Video link:

import os
import json
from dotenv import load_dotenv
import joblib
from supabase import create_client
import pandas as pd
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
    print("No new records to process.")
else:
    model = joblib.load("models/weather_classifier.pkl")

    # features
    features_df = pd.DataFrame(to_process)[feature_names]
    predict = model.predict(features_df)
    predict_proba = model.predict_proba(features_df)

    enrichment_records = []
    for date_row, prediction, probability in zip(to_process, predict, predict_proba):
        enrichment_records.append({
            "date": date_row["date"],
            "good_for_running": bool(prediction),
            "confidence": round(float(probability.max()), 4),
        })

    good_for_running = sum(1 for record in enrichment_records if record["good_for_running"])
    confidence = [record["confidence"] for record in enrichment_records]

    print(f"Classified good for running: {good_for_running} of {len(enrichment_records)}")
    print(f"Confidence range: {min(confidence):.4f} to {max(confidence):.4f}")

    print(f"Prediction Summary: {good_for_running} days out of {len(enrichment_records)} days were classified as good for running, with confidence ranging from {min(confidence)} to {max(confidence)}.")

    # Step 3: LLM transform

    SYSTEM_PROMPT = (
        "You write running recommendations for a daily weather app. You will receive one day's "
        "weather conditions and a machine learning model's prediction about whether that day is "
        "good for running. "
        "Always agree with the model's prediction - your job is to explain it, not to second-guess it. "
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

    # Step 4: Load

    result = supabase.table("weather_enriched").upsert(enrichment_records).execute()
    print(f"Rows upserted: {len(result.data)}")

# Step 5  Validation

all_rows = supabase.table("weather_enriched").select("*").execute().data

print(f"Total rows in weather_enriched: {len(all_rows)}")

print("\nFive sample rows:")
for row in all_rows[:5]:
    print(f"{row['date']} | good={row['good_for_running']} | conf={row['confidence']} | {row['llm_summary']}")

good_count = sum(1 for row in all_rows if row["good_for_running"])
print(f"\nDays classified as good for running: {good_count}")

# Looking at the summaries, 2023-01-02 is the strongest one. The model said not ideal with
# 66% confidence and the sentence names the actual reason - high wind speed affecting
# stability - instead of listing every feature. The weakest is 2023-01-01. The model was only
# 52% confident, which is nearly a coin flip, but the sentence says the day is suitable with
# no hesitation at all. It also calls a January day in Oklahoma City "mild." The cause is my
# prompt: I pass the confidence into the user message but my system prompt never tells the
# model to do anything with it, so the model treats 52% and 99% identically. 2023-01-04 and
# 2023-01-05 have the opposite problem - they are nearly word for word the same sentence,
# because when nothing about a day stands out the model falls back on listing all three
# features.


# Step 6: Reflect

# 1. My metadata says the classifier was trained on Oklahoma City, not Charlotte, and the
# label thresholds note says the wind cutoff was already raised from 30 to 35 km/h for the
# OKC climate. So my Week 9 city and my training city match, and I expect the predictions to
# hold up. If they had not matched, I would expect the model to do poorly, because "good for
# running" is not the same thing in every climate. A 26C day is ordinary in Oklahoma and
# unusual in a cooler city, so a model trained on one would draw the line in the wrong place
# for the other. The features themselves would still be valid numbers, but the labels they
# were trained against would encode the wrong idea of a good day.
#
# 2. The LLM is purely additive. It receives the prediction as a finished fact and writes a
# sentence around it, and nothing in my pipeline lets its output change the good_for_running
# column. That is the right design, because the classifier is the part that was actually
# trained on labeled data, while the LLM has never seen my weather set. The risk is that the
# LLM can still contradict the prediction in words while the column stays put. I saw this
# concretely; before I added "always agree with the model's prediction" to my system prompt,
# some rows came back with good_for_running set to True while the sentence said the day was
# not good for running. The LLM cannot change the boolean, but it can disagree with it in
# words, which makes the row useless either way.
#
# 3. Latency, not cost. At gpt-4o-mini prices 50,000 summaries is only a couple of dollars,
# but my loop makes one request at a time and waits for each one. At roughly a second per
# call that is close to fourteen hours for a single run, and any crash in the middle wastes
# everything generated so far. I would fix it by sending requests concurrently instead of one
# by one, and by writing to Supabase in chunks as I go rather than holding all 50,000 records
# in memory for one upsert at the end. Incremental processing matters much more at that size
# too, since it means a rerun after a crash only pays for what is actually missing.